#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: bash scripts/evaluate.sh {mvtec|visa} DATA_ROOT CHECKPOINT [extra evaluation arguments]" >&2
    exit 2
fi
dataset=$1
data_root=$2
checkpoint=$3
shift 3
case "$dataset" in
    mvtec) scoring=(--map-aggregation product --sigma 4 --kernel-size 5 --top-ratio 0.01) ;;
    visa) scoring=(--map-aggregation sum --sigma 0 --kernel-size 0 --top-ratio 0) ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac
data_root=$(cd "$data_root" && pwd)
checkpoint=$(cd "$(dirname "$checkpoint")" && pwd)/$(basename "$checkpoint")
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
python_bin=${PYTHON_BIN:-python}
weights=${DINOV2_WEIGHTS:-$HOME/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth}

"$python_bin" run_decad.py evaluate --dataset "$dataset" --data-root "$data_root" \
    --weights "$weights" --checkpoint "$checkpoint" --output "${OUTPUT_ROOT:-outputs}/eval_$dataset" \
    --layers 3 6 9 --micro-batch 16 "${scoring[@]}" "$@"
