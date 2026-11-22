"""
Event data processing utilities for multimodal image restoration
处理 event stream 数据并转换为网络输入格式
"""

import numpy as np
import torch
from pathlib import Path


def events_to_voxel_grid(events, num_bins=4, height=None, width=None, normalize=True):
    """
    将 event stream (t, x, y, p) 转换为 voxel grid 表示
    
    Args:
        events: dict with keys ['t', 'x', 'y', 'p']
        num_bins: 时间维度的 bin 数量
        height: 输出高度，如果为 None 则从 events 推断
        width: 输出宽度，如果为 None 则从 events 推断
        normalize: 是否归一化
        
    Returns:
        voxel_grid: numpy array of shape (num_bins, 2, H, W)
                   2 表示正负极性分开存储
    """
    t = events['t'].astype(np.float64)
    x = events['x'].astype(np.int32)
    y = events['y'].astype(np.int32)
    p = events['p'].astype(np.int32)
    
    # 确定图像尺寸
    if height is None:
        height = y.max() + 1
    if width is None:
        width = x.max() + 1
    
    # 过滤超出边界的 events
    valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    t = t[valid_mask]
    x = x[valid_mask]
    y = y[valid_mask]
    p = p[valid_mask]
    
    if len(t) == 0:
        # 如果没有 events，返回全零的 voxel grid
        return np.zeros((num_bins, 2, height, width), dtype=np.float32)
    
    # 时间归一化到 [0, num_bins)
    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_norm = (t - t_min) / (t_max - t_min) * (num_bins - 1e-6)
    else:
        t_norm = np.zeros_like(t)
    
    t_idx = t_norm.astype(np.int32)
    t_idx = np.clip(t_idx, 0, num_bins - 1)
    
    # 创建 voxel grid: (T, 2, H, W) - 2 个通道分别表示正负极性
    voxel = np.zeros((num_bins, 2, height, width), dtype=np.float32)
    
    # 使用双线性插值（简化版本：直接累加）
    for i in range(len(t)):
        time_bin = t_idx[i]
        polarity = 0 if p[i] <= 0 else 1  # 0: 负极性, 1: 正极性
        voxel[time_bin, polarity, y[i], x[i]] += 1.0
    
    # 归一化
    if normalize and voxel.max() > 0:
        voxel = voxel / (voxel.max() + 1e-6)
    
    return voxel


def load_and_merge_event_files(event_dirs, filename, num_bins=4, height=None, width=None):
    """
    加载四个 event 文件夹中的同名文件并平均
    
    Args:
        event_dirs: list of Path objects, 四个 event 文件夹路径
                   [translation_x_dir, translation_y_dir, rotation_dir, radial_dir]
        filename: event 文件名 (例如 '9999_10_0.8892.npz')
        num_bins: 时间 bin 数量
        height: 目标高度
        width: 目标宽度
        
    Returns:
        merged_voxel: numpy array of shape (num_bins * 2, H, W) = (8, H, W)
                     或 (num_bins, 2, H, W) 根据需求
    """
    voxel_grids = []
    
    for event_dir in event_dirs:
        event_file = event_dir / filename
        
        if not event_file.exists():
            print(f"Warning: {event_file} does not exist, using zeros")
            if height is not None and width is not None:
                voxel = np.zeros((num_bins, 2, height, width), dtype=np.float32)
            else:
                raise ValueError(f"Cannot create zero voxel without height and width")
        else:
            # 加载 event 数据
            events = np.load(event_file)
            
            # 转换为 voxel grid
            voxel = events_to_voxel_grid(
                events, 
                num_bins=num_bins, 
                height=height, 
                width=width,
                normalize=True
            )
        
        voxel_grids.append(voxel)
    
    # 平均四个 event 类型
    merged_voxel = np.mean(voxel_grids, axis=0)  # (num_bins, 2, H, W)
    
    return merged_voxel


def prepare_event_input_for_network(merged_voxel, target_format='4d'):
    """
    准备 event 数据用于网络输入
    
    Args:
        merged_voxel: shape (T, 2, H, W) 或 (T, P, H, W)
        target_format: '4d' for (B, T*P, H, W) or '5d' for (B, T, P, H, W)
        
    Returns:
        event_tensor: torch.Tensor
    """
    if target_format == '4d':
        # Reshape to (T*P, H, W) then add batch dimension
        T, P, H, W = merged_voxel.shape
        event_tensor = merged_voxel.reshape(T * P, H, W)
        event_tensor = torch.from_numpy(event_tensor).unsqueeze(0)  # (1, T*P, H, W)
    elif target_format == '5d':
        # Add batch dimension: (T, P, H, W) -> (1, T, P, H, W)
        event_tensor = torch.from_numpy(merged_voxel).unsqueeze(0)
    else:
        raise ValueError(f"Unknown target_format: {target_format}")
    
    return event_tensor


def events_to_representation(events, representation_type='voxel', 
                             num_bins=4, height=None, width=None):
    """
    将 event stream 转换为不同的表示形式
    
    Args:
        events: dict with keys ['t', 'x', 'y', 'p']
        representation_type: 'voxel', 'time_surface', 'event_frame'
        num_bins: 时间 bin 数量（用于 voxel）
        height: 图像高度
        width: 图像宽度
        
    Returns:
        representation: numpy array
    """
    if representation_type == 'voxel':
        return events_to_voxel_grid(events, num_bins, height, width)
    
    elif representation_type == 'event_frame':
        # 简单的 event frame: 正负极性分开统计
        t = events['t']
        x = events['x'].astype(np.int32)
        y = events['y'].astype(np.int32)
        p = events['p'].astype(np.int32)
        
        if height is None:
            height = y.max() + 1
        if width is None:
            width = x.max() + 1
        
        # 过滤超出边界的 events
        valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x = x[valid_mask]
        y = y[valid_mask]
        p = p[valid_mask]
        
        event_frame = np.zeros((2, height, width), dtype=np.float32)
        for i in range(len(x)):
            polarity = 0 if p[i] <= 0 else 1
            event_frame[polarity, y[i], x[i]] += 1.0
        
        # 归一化
        if event_frame.max() > 0:
            event_frame = event_frame / event_frame.max()
        
        return event_frame
    
    else:
        raise NotImplementedError(f"Representation type {representation_type} not implemented")


if __name__ == '__main__':
    """测试 event 数据处理"""
    
    # 测试路径
    base_dir = Path('/hpc2hdd/home/lwang851/Project/PromptIR/ResShift/reside/ITS/ITS/train/ITS_event_pair')
    event_dirs = [
        base_dir / 'translation_x',
        base_dir / 'translation_y',
        base_dir / 'rotation',
        base_dir / 'radial'
    ]
    
    # 测试文件
    test_filename = '9999_10_0.8892.npz'
    
    print("=" * 60)
    print("测试 Event 数据处理")
    print("=" * 60)
    
    # 测试单个 event 文件
    print(f"\n[测试 1] 加载单个 event 文件")
    events = np.load(event_dirs[0] / test_filename)
    print(f"Event stream keys: {list(events.keys())}")
    print(f"Number of events: {len(events['t'])}")
    print(f"Time range: [{events['t'].min()}, {events['t'].max()}]")
    print(f"Spatial range: x[{events['x'].min()}, {events['x'].max()}], "
          f"y[{events['y'].min()}, {events['y'].max()}]")
    print(f"Polarity: {np.unique(events['p'])}")
    
    # 转换为 voxel grid
    print(f"\n[测试 2] 转换为 voxel grid")
    voxel = events_to_voxel_grid(events, num_bins=4, height=407, width=541)
    print(f"Voxel grid shape: {voxel.shape}")  # (4, 2, 407, 541)
    print(f"Voxel grid dtype: {voxel.dtype}")
    print(f"Voxel grid range: [{voxel.min():.4f}, {voxel.max():.4f}]")
    print(f"Non-zero ratio: {(voxel > 0).sum() / voxel.size:.4f}")
    
    # 加载并合并四个 event 文件
    print(f"\n[测试 3] 加载并合并四个 event 类型")
    merged_voxel = load_and_merge_event_files(
        event_dirs, 
        test_filename, 
        num_bins=4, 
        height=407, 
        width=541
    )
    print(f"Merged voxel shape: {merged_voxel.shape}")  # (4, 2, 407, 541)
    print(f"Merged voxel range: [{merged_voxel.min():.4f}, {merged_voxel.max():.4f}]")
    
    # 准备网络输入
    print(f"\n[测试 4] 准备网络输入格式")
    event_4d = prepare_event_input_for_network(merged_voxel, target_format='4d')
    print(f"4D format shape: {event_4d.shape}")  # (1, 8, 407, 541)
    
    event_5d = prepare_event_input_for_network(merged_voxel, target_format='5d')
    print(f"5D format shape: {event_5d.shape}")  # (1, 4, 2, 407, 541)
    
    print("\n" + "=" * 60)
    print("Event 数据处理测试完成!")
    print("=" * 60)

