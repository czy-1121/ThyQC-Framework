#!/usr/bin/env bash
set -euo pipefail

# Set ROOT to the directory containing the external data/model paths.
ROOT="${ROOT:-/root}"
PYTHON="${PYTHON:-python}"
CODE="$(cd "$(dirname "$0")" && pwd)/code/train_g2d_uot_gt_qdm_seed42_20260919.py"
MANIFEST="${MANIFEST:?Set MANIFEST to student_global_ordered_k4_manifest.jsonl}"
TEACHER="${TEACHER:?Set TEACHER to the teacher labels JSONL}"
OUT_ROOT="${OUT_ROOT:-$(cd "$(dirname "$0")" && pwd)/inference_outputs}"

for SEED in 41 42 43; do
  VAR="CACHE_${SEED}"
  CACHE="${!VAR:-${CACHE_DIR:?Set CACHE_DIR or CACHE_41/CACHE_42/CACHE_43}}"
  STATE="$(cd "$(dirname "$0")" && pwd)/weights/seed${SEED}_best_gt_qdm_state.pt"
  "$PYTHON" "$CODE" \
    --manifest "$MANIFEST" --prob-cache "$CACHE" --teacher-jsonl "$TEACHER" \
    --semantic-cost-mode fixed_jaccard --out-dir "$OUT_ROOT/seed${SEED}" \
    --init-state "$STATE" --seed "$SEED" --epochs 0 \
    --lambda-prob 0.06 --lambda-g2d 1.8 --uot-loss-mode transport
done
