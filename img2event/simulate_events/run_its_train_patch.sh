#!/usr/bin/env bash
# Generate synthetic event streams for the ITS training set (GPU patch version).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-data}"
INPUT_DIR="${INPUT_DIR:-${DATA_ROOT}/ITS/train/clear_pair}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/ITS/train/event_pair}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-100}"

python "$SCRIPT_DIR/simulate_events_multigpu_patch.py" \
  --input_dir="$INPUT_DIR" \
  --output_root="$OUTPUT_ROOT" \
  --gpu_ids="$GPU_IDS" \
  --batch_size="$BATCH_SIZE"
