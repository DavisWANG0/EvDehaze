#!/usr/bin/env bash
# Generate synthetic event streams for the ITS training set.
#
# Default layout (override with DATA_ROOT, INPUT_DIR or OUTPUT_ROOT):
#   ${DATA_ROOT}/ITS/train/clear_pair  ->  ${DATA_ROOT}/ITS/train/event_pair
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-data}"
INPUT_DIR="${INPUT_DIR:-${DATA_ROOT}/ITS/train/clear_pair}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/ITS/train/event_pair}"
NUM_THREADS="${NUM_THREADS:-32}"

python "$SCRIPT_DIR/simulate_events_multithread.py" \
    --input_dir="$INPUT_DIR" \
    --output_root="$OUTPUT_ROOT" \
    --num_threads="$NUM_THREADS"
