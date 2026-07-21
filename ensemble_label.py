import json
import os
import argparse

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

torch.backends.cudnn.enabled = False


INSPECT_PROMPT = (
    "Look at this image. Below is a question someone asked about it, and an answer "
    "a model gave. Your task: point out any factual errors, miscounts, contradictions, "
    "or invented details in the answer that don't match what you see in the image.\n"
    "Be specific — quote the problematic part of the answer. "
    "If the answer is fully correct, say so."
)

CONSENSUS_PROMPT = (
    "Below are {k} independent analyses of whether an AI answer contains "
    "hallucinations compared to an image. Extract the common findings that appear "
    "in most analyses. Be specific: quote the hallucinated phrases.\n"
    "If all analyses agree there are no hallucinations, say 'NO_HALLUCINATIONS'.\n\n"
    "Answer being analyzed: \"\"\"{answer}\"\"\"\n\n"
)

LABEL_PROMPT = (
    "Based on this hallucination analysis, output a JSON array of hallucinated "
    "phrases from the answer. Each entry: "
    '{{"phrase": "exact substring from the answer", "label": "mischaracterization"|"miscounting"|"invention"}}. '
    "Quote the phrase EXACTLY as it appears. If no hallucinations, output [].\n"
    "Output ONLY the JSON array, nothing else.\n\n"
    "Answer: \"\"\"{answer}\"\"\"\n"
    "Analysis: \"\"\"{consensus}\"\"\"\n"
    "Output:"
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
    print(f"[MODEL] {model_id}  type={model.config.model_type}", flush=True)
    return model, processor


def generate(model, processor, messages, images=None, temperature=0.5, max_pixels=512*512):
    template = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    kwargs = dict(text=template, return_tensors="pt", max_pixels=max_pixels)
    if images:
        kwargs["images"] = images
    inputs = processor(**kwargs).to(model.device)

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
    return processor.decode(generated, skip_special_tokens=True).strip(), prompt_tokens, gen_tokens


def image_msg(image, text):
    return {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}


def text_msg(text):
    return {"role": "user", "content": text}


def ensemble_label(model, processor, image, question, answer, temperature, k=3):
    analyses = []
    for _ in range(k):
        raw, _, _ = generate(
            model, processor,
            [image_msg(image, f"{INSPECT_PROMPT}\n\nQuestion: \"{question}\"\nAnswer: \"{answer}\"\n\nYour analysis:")],
            images=[image],
            temperature=temperature,
        )
        analyses.append(raw)

    consensus_parts = []
    for i, a in enumerate(analyses, 1):
        consensus_parts.append(f"Analysis {i}: {a}")
    consensus_text = CONSENSUS_PROMPT.format(k=k, answer=answer) + "\n".join(consensus_parts) + "\n\nConsensus:"

    consensus, _, _ = generate(
        model, processor,
        [text_msg(consensus_text)],
        temperature=0.3,
    )

    label_text = LABEL_PROMPT.format(answer=answer, consensus=consensus)
    raw_labels, _, _ = generate(
        model, processor,
        [text_msg(label_text)],
        temperature=0.3,
    )

    parsed = parse_output(raw_labels)
    if parsed is not None:
        parsed = phrases_to_spans(parsed, answer)

    return parsed, analyses, consensus, raw_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--ensemble", type=int, default=1,
                        help="Number of independent analyses per sample (default: 1)")
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_id)
    print(f"Ensemble K={args.ensemble}", flush=True)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if args.max_samples:
        lines = lines[:args.max_samples]

    results = []
    iou_total = 0.0
    iou_count = 0
    parse_errors = 0
    empty_preds = 0
    report_every = 10
    image_dir_real = os.path.realpath(args.image_dir) + os.sep

    for idx, line in enumerate(tqdm(lines, desc="Ensemble")):
        item = json.loads(line.strip())
        img_path = os.path.realpath(os.path.join(args.image_dir, item["image_name"]))
        if not img_path.startswith(image_dir_real) or not os.path.exists(img_path):
            item["pred_labels"] = None
            item["error"] = "Image not found"
            results.append(item)
            continue

        image = Image.open(img_path).convert("RGB")
        pred, analyses, consensus, raw = ensemble_label(
            model, processor, image,
            item["prompt"], item["response"],
            args.temperature, k=args.ensemble,
        )

        item["pred_labels"] = pred
        item["analyses"] = analyses
        item["consensus"] = consensus
        item["raw_labels"] = raw

        if pred is None:
            parse_errors += 1
        else:
            if len(pred) == 0:
                empty_preds += 1
            iou = char_iou(item.get("labels", []), pred, len(item.get("response", "")))
            iou_total += iou
            iou_count += 1
            if iou_count <= 3:
                print(f"[IoU #{iou_count}] {item['id']} gold={item['labels']} pred={pred} iou={iou:.4f}",
                      flush=True)
        results.append(item)

        if (idx + 1) % report_every == 0 and iou_count > 0:
            avg = iou_total / iou_count
            print(f"\n[RUNNING {idx+1:5d}] IoU={avg:.4f}  "
                  f"parse_err={parse_errors}  empty={empty_preds}  valid={iou_count}",
                  flush=True)

    if iou_count > 0:
        print(f"\n=== FINAL K={args.ensemble} ===\n"
              f"IoU: {iou_total/iou_count:.4f} ({iou_count} samples)\n"
              f"Parse errors: {parse_errors}\n"
              f"Empty: {empty_preds}\n", flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
