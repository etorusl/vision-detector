#!/bin/bash
# Cluster run script for Gemma 4 E2B hallucination labeling
# Usage: bash run_cluster.sh [max_samples]

set -eu

MAX_SAMPLES="${1:-}"
INPUT_JSONL="shroom-vision.train.en.labeled.jsonl"
IMAGE_DIR="/userspace/srm/shroom-vis-images"
OUTPUT_PRED="preds_gemma.jsonl"
OUTPUT_EVAL="eval_results.txt"

echo "=== Setting up environment ==="
mkdir -p /userspace/srm/.triton-cache /userspace/srm/.torch-cache
export HOME=/userspace/srm
export HF_HOME=/userspace/srm/.hf-cache
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR=/userspace/srm/.mpl-cache
export TRITON_CACHE_DIR=/userspace/srm/.triton-cache
export TORCH_HOME=/userspace/srm/.torch-cache
export XDG_CACHE_HOME=/userspace/srm/.xdg-cache

eval "$(/userspace/srm/miniconda3/bin/conda shell.bash hook)"
conda activate ragognizer

echo "=== Step 1: Labeling with Gemma 4 E2B ==="
ARGS=(--input "$INPUT_JSONL" --image_dir "$IMAGE_DIR" --output "$OUTPUT_PRED")
if [ -n "$MAX_SAMPLES" ]; then
    ARGS+=(--max_samples "$MAX_SAMPLES")
fi
python label_with_gemma.py "${ARGS[@]}"

echo "=== Step 2: Evaluating with Char-IoU ==="
python evaluate.py --gold "$INPUT_JSONL" --pred "$OUTPUT_PRED" | tee "$OUTPUT_EVAL"

echo "=== Done ==="
echo "Predictions: $OUTPUT_PRED"
echo "Evaluation:  $OUTPUT_EVAL"
