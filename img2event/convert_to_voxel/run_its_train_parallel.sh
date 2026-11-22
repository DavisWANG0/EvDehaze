#!/usr/bin/env bash
# Parallel version of the ITS train voxel conversion.
#
# Env overrides: DATA_ROOT, EVENT_BASE_DIR, OUTPUT_DIR, HEIGHT, WIDTH, NUM_BINS,
# NUM_PROCESSES
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-data}"
EVENT_BASE_DIR="${EVENT_BASE_DIR:-${DATA_ROOT}/ITS/train/event_pair}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/ITS/train/event_preprocessed}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-256}"
NUM_BINS="${NUM_BINS:-4}"

NUM_PROCESSES="${NUM_PROCESSES:-}"  # empty = auto-detect

echo "=========================================="
echo "并行预处理 Event 数据"
echo "=========================================="
echo "输入目录: $EVENT_BASE_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "目标尺寸: ${HEIGHT}x${WIDTH}"
echo "时间 Bins: $NUM_BINS"
if [ -z "$NUM_PROCESSES" ]; then
    echo "进程数: 自动检测 (CPU核心数-2)"
else
    echo "进程数: $NUM_PROCESSES"
fi
echo "=========================================="
echo ""
echo "⚡ 使用多进程并行处理"
echo "⚡ 预计加速: 10-20倍"
echo "⚡ 预计时间: 2-5分钟（vs 原来的30-40分钟）"
echo ""
echo "=========================================="

# 构建命令
CMD="python \"$SCRIPT_DIR/convert_events_to_voxel_parallel.py\" \
    --event_base_dir \"$EVENT_BASE_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --num_bins \"$NUM_BINS\" \
    --height \"$HEIGHT\" \
    --width \"$WIDTH\""

# 如果指定了进程数，添加参数
if [ ! -z "$NUM_PROCESSES" ]; then
CMD="$CMD --num_processes \"$NUM_PROCESSES\""
fi

# 运行预处理
$CMD

echo ""
echo "=========================================="
echo "预处理完成！"
echo "=========================================="
echo ""
echo "📊 统计信息:"
echo "  预处理数据目录: $OUTPUT_DIR"
du -sh $OUTPUT_DIR 2>/dev/null && echo "  数据大小: $(du -sh $OUTPUT_DIR | awk '{print $1}')" || echo "  数据大小: 计算中..."
ls $OUTPUT_DIR/*.npz 2>/dev/null | wc -l | xargs echo "  文件数量:"

echo ""
echo "🚀 下一步："
echo "  使用预处理数据进行快速训练:"
echo "  bash scripts/train_multimodal_fast.sh"
echo ""
echo "  或者在训练脚本中使用:"
echo "  EVENT_DIR=\"$OUTPUT_DIR\""
echo "  --use_preprocessed_events"
echo ""
echo "=========================================="

