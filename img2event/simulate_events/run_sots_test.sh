#!/usr/bin/env bash
# Generate synthetic event streams for the SOTS (nyuhaze500) test set.
#
# Default layout (override with DATA_ROOT, INPUT_DIR or OUTPUT_ROOT):
#   ${DATA_ROOT}/SOTS/nyuhaze500/gt_pair  ->  ${DATA_ROOT}/SOTS/nyuhaze500/event_pair
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-data}"
INPUT_DIR="${INPUT_DIR:-${DATA_ROOT}/SOTS/nyuhaze500/gt_pair}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/SOTS/nyuhaze500/event_pair}"
NUM_THREADS="${NUM_THREADS:-32}"

python "$SCRIPT_DIR/simulate_events_multithread.py" \
    --input_dir="$INPUT_DIR" \
    --output_root="$OUTPUT_ROOT" \
    --num_threads="$NUM_THREADS"
