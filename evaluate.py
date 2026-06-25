import json
import argparse
from collections import defaultdict


def spans_to_char_set(spans, response_len):
    char_set = set()
    for span in spans:
        start = span["start"]
        end = span["end"]
        start = max(0, min(start, response_len))
        end = max(0, min(end, response_len))
        for i in range(start, end):
            char_set.add(i)
    return char_set


def char_iou(gold_spans, pred_spans, response_len):
    gold_set = spans_to_char_set(gold_spans, response_len)
    pred_set = spans_to_char_set(pred_spans, response_len)

    if not gold_set and not pred_set:
        return 1.0, 0, 0, 0
    if not gold_set or not pred_set:
        return 0.0, len(gold_set), len(pred_set), 0

    intersection = gold_set & pred_set
    union = gold_set | pred_set

    iou = len(intersection) / len(union) if union else 0.0
    return iou, len(gold_set), len(pred_set), len(intersection)


def per_label_char_sets(gold_spans, response_len):
    label_sets = defaultdict(set)
    for span in gold_spans:
        label = span.get("label", "unknown")
        start = max(0, min(span["start"], response_len))
        end = max(0, min(span["end"], response_len))
        for i in range(start, end):
            label_sets[label].add(i)
    return label_sets


def label_char_iou(gold_spans, pred_spans, response_len):
    gold_label_sets = per_label_char_sets(gold_spans, response_len)
    pred_label_sets = per_label_char_sets(pred_spans, response_len)

    all_labels = set(gold_label_sets.keys()) | set(pred_label_sets.keys())
    results = {}

    for label in all_labels:
        gold_set = gold_label_sets.get(label, set())
        pred_set = pred_label_sets.get(label, set())

        if not gold_set and not pred_set:
            results[label] = {"iou": 1.0, "gold_chars": 0, "pred_chars": 0, "intersection": 0}
        elif not gold_set or not pred_set:
            results[label] = {"iou": 0.0, "gold_chars": len(gold_set), "pred_chars": len(pred_set), "intersection": 0}
        else:
            inter = gold_set & pred_set
            union = gold_set | pred_set
            results[label] = {
                "iou": len(inter) / len(union) if union else 0.0,
                "gold_chars": len(gold_set),
                "pred_chars": len(pred_set),
                "intersection": len(inter),
            }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="Gold standard JSONL file")
    parser.add_argument("--pred", required=True, help="Predicted JSONL file")
    args = parser.parse_args()

    with open(args.gold, "r", encoding="utf-8") as f:
        gold_lines = f.readlines()

    with open(args.pred, "r", encoding="utf-8") as f:
        pred_lines = f.readlines()

    gold_by_id = {}
    for line in gold_lines:
        item = json.loads(line.strip())
        gold_by_id[item["id"]] = item

    ious = []
    per_label_ious = defaultdict(list)
    parse_errors = 0
    missing = 0

    for line in pred_lines:
        pred_item = json.loads(line.strip())
        item_id = pred_item["id"]

        if item_id not in gold_by_id:
            missing += 1
            continue

        gold_item = gold_by_id[item_id]
        gold_labels = gold_item.get("labels", [])
        pred_labels = pred_item.get("pred_labels")

        if pred_labels is None:
            parse_errors += 1
            continue

        response_len = len(gold_item.get("response", ""))

        iou, gold_chars, pred_chars, inter = char_iou(gold_labels, pred_labels, response_len)
        ious.append(iou)

        label_results = label_char_iou(gold_labels, pred_labels, response_len)
        for label, metrics in label_results.items():
            per_label_ious[label].append(metrics["iou"])

    total = len(ious)
    if total == 0:
        print("No valid samples to evaluate.")
        return

    avg_iou = sum(ious) / total
    print(f"=== Character-level IoU Evaluation ===")
    print(f"Total samples: {total}")
    print(f"Parse errors / missing preds: {parse_errors}")
    print(f"Missing gold IDs: {missing}")
    print(f"Average Char-IoU: {avg_iou:.4f}")
    print()

    print("--- Per-label Char-IoU ---")
    for label in sorted(per_label_ious.keys()):
        vals = per_label_ious[label]
        print(f"  {label}: {sum(vals)/len(vals):.4f}  (n={len(vals)})")

    print()
    print("--- IoU distribution ---")
    buckets = [(0.0, 0.0), (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.0)]
    for lo, hi in buckets:
        if lo == hi:
            count = sum(1 for iou in ious if iou == lo)
        else:
            count = sum(1 for iou in ious if lo <= iou < hi)
        pct = 100 * count / total
        bar = "#" * int(pct / 2)
        if lo == 1.0 and hi == 1.0:
            print(f"  IoU = 1.0:    {count:5d} ({pct:5.1f}%) {bar}")
        elif lo == 0.0 and hi == 0.0:
            print(f"  IoU = 0.0:    {count:5d} ({pct:5.1f}%) {bar}")
        else:
            print(f"  IoU [{lo:.1f}, {hi:.1f}): {count:5d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
