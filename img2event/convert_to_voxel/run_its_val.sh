#!/usr/bin/env bash
# Merge the four motion event streams into a single voxel file for ITS val.
#
# Env overrides: DATA_ROOT, EVENT_BASE_DIR, OUTPUT_DIR, NUM_BINS, HEIGHT, WIDTH,
# NUM_PROCESSES
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-data}"
EVENT_BASE_DIR="${EVENT_BASE_DIR:-${DATA_ROOT}/ITS/val/event_pair}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/ITS/val/event_preprocessed}"
NUM_BINS="${NUM_BINS:-4}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-256}"
NUM_PROCESSES="${NUM_PROCESSES:-32}"

echo "=========================================="
echo "预处理验证集 Event 数据"
echo "=========================================="
echo "输入目录: $EVENT_BASE_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "Bins: $NUM_BINS, 尺寸: ${HEIGHT}x${WIDTH}"
echo "并行进程数: $NUM_PROCESSES"
echo "=========================================="
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

python "$SCRIPT_DIR/convert_events_to_voxel_parallel.py" \
    --event_base_dir "$EVENT_BASE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_bins "$NUM_BINS" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num_processes "$NUM_PROCESSES"

echo ""
echo "=========================================="
echo "预处理完成！"
echo "=========================================="
echo ""
echo "验证输出:"
echo "  ls $OUTPUT_DIR | head -10"
ls $OUTPUT_DIR | head -10
echo ""
echo "文件数量:"
echo "  $(ls $OUTPUT_DIR | wc -l) 个预处理文件"
echo "=========================================="

