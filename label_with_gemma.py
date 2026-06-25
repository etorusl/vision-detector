import json
import os
import argparse

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


FEWSHOT_IDS = {"train-en-412", "train-en-415", "train-en-416"}

SYSTEM_PROMPT = (
    "You are an expert at detecting hallucinations in vision-language model responses. "
    "Given an image, a user prompt about the image, and a model's response, you must "
    "output a JSON array of character-level hallucination spans. Each span has "
    '"start" (0-based inclusive index), "end" (0-based exclusive index), and '
    '"label" (one of: "mischaracterization", "miscounting", "invention"). '
    "If there are no hallucinations, output an empty array []. "
    "Output ONLY the JSON array, nothing else. Do not wrap in markdown code blocks."
)


def load_fewshot_examples(data_path, exclude_ids=None):
    if exclude_ids is None:
        exclude_ids = set()
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if item["id"] in FEWSHOT_IDS and item["id"] not in exclude_ids:
                examples.append(item)
    return examples


def build_fewshot_prompt(examples):
    parts = [SYSTEM_PROMPT, "", "Here are labeled examples:"]
    for i, ex in enumerate(examples, 1):
        parts.append(f"\nExample {i}:")
        parts.append(f'Prompt: "{ex["prompt"]}"')
        parts.append(f'Response: "{ex["response"]}"')
        clean_labels = [
            {"start": lb["start"], "end": lb["end"], "label": lb["label"]}
            for lb in ex["labels"]
        ]
        parts.append(f"Labels: {json.dumps(clean_labels)}")
    parts.append(
        "\nNow analyze the image below and the given prompt/response. "
        "Output ONLY the JSON array of hallucination spans (or [] if none)."
    )
    return "\n".join(parts)


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


def label_sample(model, processor, image_path, prompt, response, fewshot_prompt):
    image = Image.open(image_path).convert("RGB")

    user_content = [
        {"type": "image"},
        {
            "type": "text",
            "text": (
                f"{fewshot_prompt}\n\n"
                f'Prompt: "{prompt}"\n'
                f'Response: "{response}"\n'
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
    print(f"[DEBUG] prompt tokens: {prompt_tokens}, template chars: {len(template_text)}",
          flush=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.8,
            top_p=0.95,
            top_k=64,
        )

    gen_tokens = outputs.shape[-1] - prompt_tokens
    print(f"[DEBUG] generated tokens: {gen_tokens}", flush=True)

    if gen_tokens == 0:
        print("[DEBUG] zero tokens — trying direct processor chat mode", flush=True)
        inputs2 = processor(
            images=image,
            text=messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)
        pm2 = inputs2.input_ids.shape[-1]
        with torch.no_grad():
            outputs2 = model.generate(**inputs2, max_new_tokens=1024, temperature=0.8, top_p=0.95, top_k=64)
        gen2 = outputs2.shape[-1] - pm2
        print(f"[DEBUG] fallback: prompt={pm2}, generated={gen2}", flush=True)
        generated_tokens = outputs2[0][pm2:]
    else:
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
    fewshot_examples = load_fewshot_examples(args.input)
    fewshot_prompt = build_fewshot_prompt(fewshot_examples)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if args.max_samples:
        lines = lines[: args.max_samples]

    results = []
    exclude_ids = {ex["id"] for ex in fewshot_examples}

    for line in tqdm(lines, desc="Labeling"):
        item = json.loads(line.strip())

        if item["id"] in exclude_ids:
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

        raw = label_sample(
            model, processor, image_path,
            item["prompt"], item["response"], fewshot_prompt,
        )
        parsed = parse_output(raw)

        item["pred_labels"] = parsed
        item["raw_output"] = raw
        if parsed is None:
            item["parse_error"] = True
        results.append(item)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
