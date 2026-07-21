import json
import os
import argparse
import re

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

torch.backends.cudnn.enabled = False


def partition(text, k):
    """Split into fixed k-char chunks (fallback)."""
    chunks = []
    spans = []
    for i in range(0, len(text), k):
        chunk = text[i:i + k]
        chunks.append(chunk)
        spans.append((i, min(i + k, len(text))))
    return chunks, spans


def sentence_partition(text, max_len=120):
    """Split by sentences, then by commas/colons within long sentences (>max_len)."""
    raw_sents = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    spans = []
    for sent in raw_sents:
        sent = sent.strip()
        if not sent:
            continue
        start = text.find(sent, spans[-1][1] if spans else 0)
        if start == -1:
            start = spans[-1][1] if spans else 0
        if len(sent) <= max_len:
            chunks.append(sent)
            spans.append((start, start + len(sent)))
        else:
            sub_parts = re.split(r'(?<=[,;:—–-])\s+', sent)
            for part in sub_parts:
                part = part.strip()
                if not part:
                    continue
                pstart = text.find(part, start)
                if pstart == -1:
                    pstart = start
                chunks.append(part)
                spans.append((pstart, pstart + len(part)))
                start = pstart + len(part)
    return chunks, spans


CHUNK_PROMPT = (
    "Look at this image. Below is a model's answer about this image, "
    "split into numbered chunks. "
    "For each chunk, check if it contains any hallucination "
    "(factual error, miscount, or invented detail) that contradicts the image. "
    "Output ONLY a JSON array of chunk indices that contain hallucinations. "
    "If no chunks contain hallucinations, output [].\n\n"
    "Question: \"{question}\"\n\n"
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


def extract_json(raw_text):
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw_text = "\n".join(lines).strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    matches = list(re.finditer(r"\[[^\]]*\]", raw_text))
    if matches:
        for m in reversed(matches):
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    return None


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


def generate(model, processor, messages, images=None, temperature=0.3, max_pixels=512*512):
    template = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    kwargs = dict(text=template, return_tensors="pt", max_pixels=max_pixels)
    if images:
        kwargs["images"] = images
    inputs = processor(**kwargs).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=temperature,
            top_p=0.95,
            top_k=64,
        )

    prompt_tokens = inputs.input_ids.shape[-1]
    generated = outputs[0][prompt_tokens:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--chunk_k", type=int, default=30,
                        help="Char chunk size when --sentences is not set (default: 30)")
    parser.add_argument("--sentences", action="store_true",
                        help="Split by sentences (then by commas for long ones)")
    parser.add_argument("--sent_max_len", type=int, default=120,
                        help="Max sentence length before further splitting (default: 120)")
    parser.add_argument("--no_merge", action="store_true",
                        help="Keep each chunk as independent span, don't merge adjacent ones")
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_id)

    if args.sentences:
        print(f"Chunk mode: sentences (max_len={args.sent_max_len})  merge={'no' if args.no_merge else 'yes'}",
              flush=True)
    else:
        print(f"Chunk mode: fixed k={args.chunk_k}  merge={'no' if args.no_merge else 'yes'}", flush=True)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if args.max_samples:
        lines = lines[:args.max_samples]

    results = []
    iou_total = 0.0
    iou_count = 0
    parse_errors = 0
    report_every = 20
    image_dir_real = os.path.realpath(args.image_dir) + os.sep

    for idx, line in enumerate(tqdm(lines, desc="Chunk classify")):
        item = json.loads(line.strip())
        img_path = os.path.realpath(os.path.join(args.image_dir, item["image_name"]))
        if not img_path.startswith(image_dir_real) or not os.path.exists(img_path):
            item["pred_labels"] = None
            results.append(item)
            continue

        image = Image.open(img_path).convert("RGB")
        answer = item["response"]

        if args.sentences:
            chunks, chunk_spans = sentence_partition(answer, args.sent_max_len)
        else:
            chunks, chunk_spans = partition(answer, args.chunk_k)

        chunk_lines = "\n".join(f"[{i}] \"{c}\"" for i, c in enumerate(chunks))
        prompt = CHUNK_PROMPT.format(question=item["prompt"])
        prompt += f"Answer chunks:\n{chunk_lines}\n\nHallucinated chunks:"

        raw = generate(
            model, processor,
            [{"role": "user",
              "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            images=[image],
            temperature=args.temperature,
        )

        chunk_indices = extract_json(raw)
        item["raw_output"] = raw
        item["chunk_indices"] = chunk_indices

        if chunk_indices is None:
            item["pred_labels"] = None
            item["iou"] = None
            parse_errors += 1
            if parse_errors <= 3:
                print(f"[PARSE ERR] {item['id']} raw={raw[:200]}", flush=True)
        else:
            pred_spans = []
            if isinstance(chunk_indices, list):
                for ci in chunk_indices:
                    if isinstance(ci, int) and 0 <= ci < len(chunk_spans):
                        pred_spans.append({
                            "start": chunk_spans[ci][0],
                            "end": chunk_spans[ci][1],
                            "label": "hallucination",
                        })
            if args.no_merge:
                merged = sorted(pred_spans, key=lambda x: x["start"])
            else:
                merged = []
                for s in sorted(pred_spans, key=lambda x: x["start"]):
                    if merged and merged[-1]["end"] >= s["start"]:
                        merged[-1]["end"] = max(merged[-1]["end"], s["end"])
                    else:
                        merged.append(s)
            item["pred_labels"] = merged

            iou = char_iou(item.get("labels", []), merged, len(answer))
            item["iou"] = iou
            item["gold_labels"] = item.get("labels", [])
            iou_total += iou
            iou_count += 1
            if iou_count <= 3:
                print(f"[IoU #{iou_count}] {item['id']} gold={item['gold_labels']} pred={merged} iou={iou:.4f}",
                      flush=True)

        results.append(item)

        if (idx + 1) % report_every == 0 and iou_count > 0:
            avg = iou_total / iou_count
            print(f"\n[RUNNING {idx+1:5d}] IoU={avg:.4f}  parse_err={parse_errors}  valid={iou_count}",
                  flush=True)

    if iou_count > 0:
        avg = iou_total / iou_count
        mode = f"sentences" if args.sentences else f"k={args.chunk_k}"
        print(f"\n{'='*60}\n"
              f"FINAL  mode={mode}  IoU={avg:.4f}  samples={iou_count}  parse_err={parse_errors}\n"
              f"{'='*60}", flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
