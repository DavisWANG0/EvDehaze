"""
预处理 Event 数据（多进程版本）
将 4 个 event 类型融合并转换为 voxel grid，保存为单个文件
使用多进程加速处理，充分利用多核 CPU
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import os
from multiprocessing import Pool, cpu_count
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.event_utils import load_and_merge_event_files


def process_single_event_file(args):
    """
    处理单个 event 文件（用于多进程）
    
    Args:
        args: (event_dirs, filename, output_dir, num_bins, target_height, target_width)
    
    Returns:
        (filename, success, error_msg)
    """
    event_dirs, filename, output_dir, num_bins, target_height, target_width = args
    
    output_file = output_dir / filename
    
    # 如果已经处理过，跳过
    if output_file.exists():
        return (filename, True, "already_exists")
    
    try:
        # 加载并融合 4 个 event 类型
        merged_voxel = load_and_merge_event_files(
            event_dirs,
            filename,
            num_bins=num_bins,
            height=target_height,
            width=target_width
        )  # (T, P, H, W)
        
        # 保存为压缩的 npz 文件
        np.savez_compressed(
            output_file,
            voxel=merged_voxel.astype(np.float16)  # 使用 float16 减小文件大小
        )
        
        return (filename, True, None)
        
    except Exception as e:
        return (filename, False, str(e))


def preprocess_events_parallel(
    event_base_dir,
    output_dir,
    num_bins=4,
    target_height=256,
    target_width=256,
    num_processes=None
):
    """
    并行预处理所有 event 数据
    
    Args:
        event_base_dir: Event 数据基础目录
        output_dir: 输出目录
        num_bins: 时间 bins 数量
        target_height: 目标高度
        target_width: 目标宽度
        num_processes: 进程数（None = 自动检测）
    """
    event_base_dir = Path(event_base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Event 类型目录
    event_dirs = [
        event_base_dir / 'translation_x',
        event_base_dir / 'translation_y',
        event_base_dir / 'rotation',
        event_base_dir / 'radial'
    ]
    
    # 获取所有 event 文件
    event_files = sorted(list(event_dirs[0].glob('*.npz')))
    
    # 自动检测 CPU 核心数
    if num_processes is None:
        num_processes = max(1, cpu_count() - 2)  # 留 2 个核心给系统
    
    print(f"找到 {len(event_files)} 个 event 文件")
    print(f"输出目录: {output_dir}")
    print(f"目标尺寸: {target_height}x{target_width}")
    print(f"时间 bins: {num_bins}")
    print(f"CPU 总核心数: {cpu_count()}")
    print(f"使用进程数: {num_processes}")
    print(f"预计加速: {num_processes}x")
    print()
    
    # 准备参数列表
    args_list = [
        (event_dirs, f.name, output_dir, num_bins, target_height, target_width)
        for f in event_files
    ]
    
    # 使用多进程处理
    print("开始并行处理...")
    print("=" * 60)
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_event_file, args_list),
            total=len(args_list),
            desc="预处理 Event 数据",
            unit="files"
        ))
    
    print("=" * 60)
    
    # 统计结果
    success_count = sum(1 for _, success, _ in results if success)
    already_exists = sum(1 for _, success, msg in results if success and msg == "already_exists")
    new_processed = success_count - already_exists
    error_count = len(results) - success_count
    
    print(f"\n处理完成!")
    print(f"  总文件数: {len(results)}")
    print(f"  成功处理: {success_count}")
    print(f"    - 新处理: {new_processed}")
    print(f"    - 已存在: {already_exists}")
    print(f"  失败: {error_count}")
    
    if error_count > 0:
        print(f"\n失败的文件:")
        for filename, success, error in results:
            if not success and error != "already_exists":
                print(f"  {filename}: {error}")
    
    # 统计文件大小
    processed_files = list(output_dir.glob('*.npz'))
    if len(processed_files) > 0:
        processed_size = sum(f.stat().st_size for f in processed_files) / (1024**3)
        print(f"\n输出数据统计:")
        print(f"  文件数量: {len(processed_files)}")
        print(f"  总大小: {processed_size:.2f} GB")
        print(f"  平均大小: {processed_size / len(processed_files) * 1024:.2f} MB/file")


def main():
    parser = argparse.ArgumentParser(description='预处理 Event 数据（多进程版本）')
    parser.add_argument('--event_base_dir', type=str, required=True,
                       help='Event 数据基础目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--num_bins', type=int, default=4,
                       help='时间 bins 数量')
    parser.add_argument('--height', type=int, default=256,
                       help='目标高度')
    parser.add_argument('--width', type=int, default=256,
                       help='目标宽度')
    parser.add_argument('--num_processes', type=int, default=None,
                       help='进程数（默认自动检测 = CPU核心数-2）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Event 数据并行预处理")
    print("=" * 60)
    print()
    
    preprocess_events_parallel(
        args.event_base_dir,
        args.output_dir,
        args.num_bins,
        args.height,
        args.width,
        args.num_processes
    )
    
    print("\n" + "=" * 60)
    print("预处理完成！现在可以开始快速训练了")
    print("=" * 60)
    print("\n使用方法:")
    print("  bash scripts/train_multimodal_fast.sh")
    print("\n或者在训练命令中添加:")
    print("  --event_base_dir", args.output_dir)
    print("  --use_preprocessed_events")
    print()


if __name__ == '__main__':
    main()

