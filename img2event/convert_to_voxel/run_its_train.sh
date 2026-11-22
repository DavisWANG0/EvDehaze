#!/usr/bin/env bash
# Merge the four motion event streams into a single voxel file for ITS train.
#
# Env overrides: DATA_ROOT, EVENT_BASE_DIR, OUTPUT_DIR, HEIGHT, WIDTH, NUM_BINS
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-data}"
EVENT_BASE_DIR="${EVENT_BASE_DIR:-${DATA_ROOT}/ITS/train/event_pair}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/ITS/train/event_preprocessed}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-256}"
NUM_BINS="${NUM_BINS:-4}"

echo "=========================================="
echo "预处理 Event 数据"
echo "=========================================="
echo "输入目录: $EVENT_BASE_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "目标尺寸: ${HEIGHT}x${WIDTH}"
echo "时间 Bins: $NUM_BINS"
echo "=========================================="
echo ""
echo "这个过程可能需要 10-30 分钟..."
echo "但之后训练速度会提升 5-10 倍！"
echo ""

# 运行预处理
python "$SCRIPT_DIR/convert_events_to_voxel.py" \
    --event_base_dir "$EVENT_BASE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_bins "$NUM_BINS" \
    --height "$HEIGHT" \
    --width "$WIDTH"

echo ""
echo "=========================================="
echo "预处理完成！"
echo "=========================================="
echo ""
echo "现在修改训练脚本，使用预处理的数据:"
echo "  1. 修改 EVENT_DIR 为: $OUTPUT_DIR"
echo "  2. 在训练命令中添加: --use_preprocessed_events"
echo ""
echo "预期加速: 5-10倍"
echo "=========================================="

