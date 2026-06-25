import json
import os
import argparse
import sys

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


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

FEWSHOT_IDS = ["train-en-412", "train-en-415"]


def load_model_and_processor(model_id):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    print(f"[MODEL] {model_id}  config.model_type={model.config.model_type}", flush=True)
    return model, processor


def load_fewshot_data(data_path, image_dir):
    with open(data_path, "r", encoding="utf-8") as f:
        all_items = {item["id"]: item for line in f
                     if (item := json.loads(line.strip()))["id"] in FEWSHOT_IDS}
    fewshot = []
    for fid in FEWSHOT_IDS:
        item = all_items[fid]
        img_path = os.path.join(image_dir, item["image_name"])
        if not os.path.exists(img_path):
            print(f"WARNING: few-shot image not found: {img_path}", flush=True)
            continue
        img = Image.open(img_path).convert("RGB")
        rtext = item["response"]
        phrases = []
        for lb in item.get("labels", []):
            phrase = rtext[lb["start"]:lb["end"]]
            phrases.append({"phrase": phrase, "label": lb["label"]})
        fewshot.append({
            "image": img,
            "prompt": item["prompt"],
            "response": rtext,
            "output": phrases,
        })
    return fewshot


def label_sample(model, processor, image_path, prompt, response, temperature=0.5,
                 debug=False, image_only=False, fewshot_data=None):
    image = Image.open(image_path).convert("RGB")
    messages = []
    all_images = []

    if image_only:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe what you see in this image in 2-3 sentences. Be specific."},
            ]
        })
        all_images.append(image)
    else:
        if fewshot_data:
            for i, fs in enumerate(fewshot_data, 1):
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text",
                         "text": f"Example {i}:\nQuestion: \"{fs['prompt']}\"\nAnswer: \"{fs['response']}\"\nOutput:"},
                    ]
                })
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(fs["output"]),
                })
                all_images.append(fs["image"])

        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text",
                 "text": (f"{LABELING_PROMPT}\n\n"
                          f'Question: "{prompt}"\n'
                          f'Answer: "{response}"\n'
                          "Output:")},
            ]
        })
        all_images.append(image)

    template_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=template_text, images=all_images, return_tensors="pt"
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

    if debug:
        img_keys = [k for k in inputs.keys() if "pixel" in k.lower() or "image" in k.lower() or "img" in k.lower()]
        pixel_shapes = {k: tuple(inputs[k].shape) for k in img_keys}
        has_img_token = "<image>" in template_text
        print(f"[DEBUG] prompt={prompt_tokens}, gen={gen_tokens}, imgs={len(all_images)}, "
              f"pixels={pixel_shapes}, has_<image>={has_img_token}",
              flush=True)

    generated_tokens = outputs[0][prompt_tokens:]
    raw_output = processor.decode(generated_tokens, skip_special_tokens=True).strip()
    return raw_output


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--image_dir", required=True, help="Directory with images")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--model_id", default="google/gemma-4-E2B-it")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--debug_images", type=int, default=0,
                        help="Describe first N images to verify vision works, then exit")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Generation temperature (default: 0.5)")
    parser.add_argument("--no_fewshot", action="store_true",
                        help="Disable few-shot, single image only")
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_id)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if args.no_fewshot:
        fewshot_data = []
        print("Few-shot disabled", flush=True)
    else:
        fewshot_data = load_fewshot_data(args.input, args.image_dir)
        print(f"Loaded {len(fewshot_data)} few-shot examples", flush=True)

    if args.debug_images > 0:
        print(f"\n=== VISION DEBUG: describing first {args.debug_images} images ===\n", flush=True)
        for i in range(min(args.debug_images, len(lines))):
            item = json.loads(lines[i].strip())
            img_path = os.path.join(args.image_dir, item["image_name"])
            img_path = os.path.realpath(img_path)
            if not os.path.exists(img_path):
                print(f"[{i}] {item['image_name']} — NOT FOUND", flush=True)
                continue
            desc = label_sample(model, processor, img_path, "", "",
                                debug=True, image_only=True)
            print(f"[{i}] {item['image_name']}", flush=True)
            print(f"    Prompt: {item['prompt']}", flush=True)
            print(f"    Model sees: {desc[:300]}", flush=True)
            print(flush=True)
        print("=== VISION DEBUG DONE ===\n", flush=True)
        print("If descriptions are reasonable → vision works, problem is in labeling prompt.")
        print("If descriptions are garbage → vision input is broken.\n")
        return

    if args.max_samples:
        lines = lines[: args.max_samples]

    results = []
    fewshot_ids = set(FEWSHOT_IDS) if not args.no_fewshot else set()
    debug_count = 0
    iou_total = 0.0
    iou_count = 0
    parse_errors = 0
    empty_preds = 0
    report_every = 20

    for idx, line in enumerate(tqdm(lines, desc="Labeling")):
        item = json.loads(line.strip())

        if item["id"] in fewshot_ids:
            continue

        raw_path = os.path.join(args.image_dir, item["image_name"])
        image_path = os.path.realpath(raw_path)
        image_dir_real = os.path.realpath(args.image_dir) + os.sep

        if not image_path.startswith(image_dir_real):
            item["pred_labels"] = None
            item["error"] = f"Path traversal rejected: {item['image_name']}"
            results.append(item)
            continue

        if not os.path.exists(image_path):
            item["pred_labels"] = None
            item["error"] = f"Image not found: {image_path}"
            results.append(item)
            continue

        debug = debug_count < 3
        raw = label_sample(
            model, processor, image_path,
            item["prompt"], item["response"],
            temperature=args.temperature,
            debug=debug,
            fewshot_data=fewshot_data or None,
        )
        debug_count += 1
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
                print(f"[IoU DEBUG #{iou_count}] id={item['id']} gold={gold} pred={parsed} iou={iou:.4f}",
                      flush=True)

        results.append(item)

        if (idx + 1) % report_every == 0 and iou_count > 0:
            avg = iou_total / iou_count
            print(f"\n[RUNNING {idx+1:5d}] Char-IoU={avg:.4f}  "
                  f"parse_err={parse_errors}  empty={empty_preds}  "
                  f"valid={iou_count}",
                  flush=True)

    result = (
        f"\n=== FINAL ===\n"
        f"Total IoU: {iou_total/iou_count:.4f} ({iou_count} samples)\n"
        f"Parse errors: {parse_errors}\n"
        f"Empty predictions ([]): {empty_preds}\n"
    )
    print(result, flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
