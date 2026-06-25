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
    "Look at the image. A user asked a question about it and a model produced an answer. "
    "Your task: check if the answer contains hallucinations (factual errors, miscounting, "
    "or invented details inconsistent with what's visible in the image).\n\n"
    "Output a JSON array of hallucination spans. Each span: "
    '{"start": int, "end": int, "label": "mischaracterization"|"miscounting"|"invention"}. '
    "Indices are 0-based, end is exclusive. If no hallucinations, output [].\n"
    "Output ONLY the JSON array, nothing else."
)


def load_model_and_processor(model_id):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    return model, processor


def label_sample(model, processor, image_path, prompt, response, debug=False):
    image = Image.open(image_path).convert("RGB")

    user_content = [
        {"type": "image"},
        {
            "type": "text",
            "text": (
                f"{LABELING_PROMPT}\n\n"
                f'Question: "{prompt}"\n'
                f'Answer: "{response}"\n'
                "Output:"
            ),
        },
    ]

    messages = [{"role": "user", "content": user_content}]

    template_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=template_text, images=image, return_tensors="pt"
    ).to(model.device)

    prompt_tokens = inputs.input_ids.shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )

    gen_tokens = outputs.shape[-1] - prompt_tokens

    if debug:
        print(f"[DEBUG] prompt={prompt_tokens}, chars={len(template_text)}, gen={gen_tokens}",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--image_dir", required=True, help="Directory with images")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--model_id", default="google/gemma-4-E2B-it")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_id)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if args.max_samples:
        lines = lines[: args.max_samples]

    results = []
    debug_count = 0
    iou_total = 0.0
    iou_count = 0
    parse_errors = 0
    empty_preds = 0
    report_every = 20

    for idx, line in enumerate(tqdm(lines, desc="Labeling")):
        item = json.loads(line.strip())

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
            debug=debug,
        )
        debug_count += 1
        parsed = parse_output(raw)

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
