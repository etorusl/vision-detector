import json
import os
import argparse

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

torch.backends.cudnn.enabled = False


LABELING_PROMPT = (
    "Look at the image. A user asked a question about it and a model produced the answer below. "
    "Your task: check if the answer contains hallucinations (factual errors, miscounting, "
    "or invented details inconsistent with what's visible in the image).\n\n"
    "Output a JSON array of hallucinated phrases found in the answer. "
    'Each entry: {"phrase": "exact substring from the answer", "label": "mischaracterization"|"miscounting"|"invention"}. '
    "Quote the phrase EXACTLY as it appears — copy-paste characters. "
    "If the answer is fully correct, output [].\n"
    "Output ONLY the JSON array, nothing else."
)


def char_iou(gold_spans, pred_spans, response_len):
    def to_set(spans):
        s = set()
        for sp in spans:
            a = max(0, min(sp["start"], response_len))
            b = max(0, min(sp["end"], response_len))
            for i in range(a, b):
                s.add(i)
        return s
    g = to_set(gold_spans)
    p = to_set(pred_spans)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    inter = g & p
    union = g | p
    return len(inter) / len(union) if union else 0.0


def parse_output(raw_output):
    raw_output = raw_output.strip()
    if raw_output.startswith("```"):
        lines = raw_output.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw_output = "\n".join(lines).strip()
    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def phrases_to_spans(phrases, response_text):
    spans = []
    for entry in phrases:
        if not isinstance(entry, dict):
            continue
        phrase = entry.get("phrase", "")
        label = entry.get("label", "unknown")
        idx = response_text.find(phrase)
        if idx == -1:
            continue
        spans.append({"start": idx, "end": idx + len(phrase), "label": label})
    spans.sort(key=lambda s: s["start"])
    return spans


def load_model_and_processor(model_id):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    print(f"[MODEL] {model_id}  config.model_type={model.config.model_type}", flush=True)
    return model, processor


INSPECT_PROMPT = (
    "Look at this image. Below is a question someone asked about it, and an answer "
    "a model gave. Your task: point out any factual errors, miscounts, contradictions, "
    "or invented details in the answer that don't match what you see in the image.\n"
    "Be specific — quote the problematic part of the answer. "
    "If the answer is fully correct, say so."
)


def generate_response(model, processor, image, text, temperature=0.5, max_pixels=512*512):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": text},
        ]
    }]
    template = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=template, images=image, return_tensors="pt",
        max_pixels=max_pixels,
    ).to(model.device)

    prompt_tokens = inputs.input_ids.shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=temperature,
            top_p=0.95,
            top_k=64,
        )

    gen_tokens = outputs.shape[-1] - prompt_tokens
    generated = outputs[0][prompt_tokens:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def label_one(model, processor, image_path, prompt, response, temperature, debug):
    image = Image.open(image_path).convert("RGB")
    text = (f"{LABELING_PROMPT}\n\n"
            f'Question: "{prompt}"\n'
            f'Answer: "{response}"\n'
            "Output:")
    raw = generate_response(model, processor, image, text, temperature,
                            max_pixels=512*512)
    if debug:
        print(f"[DEBUG] len(raw)={len(raw)}", flush=True)
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--image_dir", required=True, help="Directory with images")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--model_id", default="google/gemma-4-E2B-it")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--inspect", action="store_true",
                        help="Inspect mode: free-form QA, print all responses for manual review")
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_id)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if args.inspect:
        print(f"\n{'='*80}\nINSPECT MODE — free-form hallucination analysis\n{'='*80}\n",
              flush=True)
        for idx, line in enumerate(lines):
            if args.max_samples and idx >= args.max_samples:
                break
            item = json.loads(line.strip())
            img_path = os.path.join(args.image_dir, item["image_name"])
            img_path = os.path.realpath(img_path)
            if not os.path.exists(img_path):
                continue

            image = Image.open(img_path).convert("RGB")
            text = (f'{INSPECT_PROMPT}\n\n'
                    f'Question: "{item["prompt"]}"\n'
                    f'Answer: "{item["response"]}"\n\n'
                    f'Your analysis:')

            analysis = generate_response(model, processor, image, text, args.temperature)

            print(f"[{idx+1}] {item['image_name']}")
            print(f"    ID: {item['id']}")
            print(f"    Gold labels: {item['labels']}")
            print(f"    Question: {item['prompt']}")
            print(f"    Answer: {item['response'][:200]}{'...' if len(item['response'])>200 else ''}")
            print(f"    Model says:\n    {analysis}")
            print(f"    {'-'*70}")
            print(flush=True)

        print(f"\n{'='*80}\nINSPECT DONE\n{'='*80}\n", flush=True)
        return

    if args.max_samples:
        lines = lines[:args.max_samples]

    results = []
    iou_total = 0.0
    iou_count = 0
    parse_errors = 0
    empty_preds = 0
    report_every = 20

    image_dir_real = os.path.realpath(args.image_dir) + os.sep

    for idx, line in enumerate(tqdm(lines, desc="Labeling")):
        item = json.loads(line.strip())

        raw_path = os.path.join(args.image_dir, item["image_name"])
        image_path = os.path.realpath(raw_path)

        if not image_path.startswith(image_dir_real):
            item["pred_labels"] = None
            item["error"] = f"Path traversal: {item['image_name']}"
            results.append(item)
            continue

        if not os.path.exists(image_path):
            item["pred_labels"] = None
            item["error"] = f"Image not found: {image_path}"
            results.append(item)
            continue

        debug = idx < 3
        raw = label_one(
            model, processor, image_path,
            item["prompt"], item["response"],
            args.temperature, debug,
        )
        parsed = parse_output(raw)
        if parsed is not None:
            parsed = phrases_to_spans(parsed, item.get("response", ""))

        item["pred_labels"] = parsed
        item["raw_output"] = raw

        if parsed is None:
            item["parse_error"] = True
            parse_errors += 1
        else:
            if len(parsed) == 0:
                empty_preds += 1
            gold = item.get("labels", [])
            rlen = len(item.get("response", ""))
            iou = char_iou(gold, parsed, rlen)
            iou_total += iou
            iou_count += 1
            if iou_count <= 3:
                print(f"[IoU #{iou_count}] {item['id']} gold={gold} pred={parsed} iou={iou:.4f}",
                      flush=True)

        results.append(item)

        if (idx + 1) % report_every == 0 and iou_count > 0:
            avg = iou_total / iou_count
            print(f"\n[RUNNING {idx+1:5d}] Char-IoU={avg:.4f}  "
                  f"parse_err={parse_errors}  empty={empty_preds}  valid={iou_count}",
                  flush=True)

    if iou_count > 0:
        print(f"\n=== FINAL ===\n"
              f"IoU: {iou_total/iou_count:.4f} ({iou_count} samples)\n"
              f"Parse errors: {parse_errors}\n"
              f"Empty: {empty_preds}\n", flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
