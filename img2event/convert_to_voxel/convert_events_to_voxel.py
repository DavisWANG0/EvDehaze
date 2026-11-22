#!/usr/bin/env python3
"""
预处理脚本：将 4 个运动类型目录的原始 event (t,x,y,p) 合并为 voxel grid 并保存。

输入:  clear_events/{translation_x,translation_y,rotation,radial}/{id}.npz  (keys: t,x,y,p)
输出:  clear_events_preprocessed/{id}.npz  (key: voxel, shape: (num_bins, 2, H, W))

Usage:
    python preprocess_events.py \
        --event_base_dir /home/lw/dataset/reside/ITS_v2/clear_events \
        --output_dir /home/lw/dataset/reside/ITS_v2/clear_events_preprocessed \
        --num_bins 4 \
        --num_workers 8
"""

import argparse
import os
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

EVENT_TYPES = ['translation_x', 'translation_y', 'rotation', 'radial']


def events_to_voxel_grid_fast(events, num_bins=4, height=None, width=None):
    """
    向量化版本：用 np.add.at 替代 Python for 循环，速度提升 100x+
    """
    t = events['t'].astype(np.float64)
    x = events['x'].astype(np.int64)
    y = events['y'].astype(np.int64)
    p = events['p'].astype(np.int64)

    if height is None:
        height = int(y.max()) + 1
    if width is None:
        width = int(x.max()) + 1

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    t, x, y, p = t[valid], x[valid], y[valid], p[valid]

    if len(t) == 0:
        return np.zeros((num_bins, 2, height, width), dtype=np.float32)

    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_idx = ((t - t_min) / (t_max - t_min) * (num_bins - 1e-6)).astype(np.int64)
    else:
        t_idx = np.zeros(len(t), dtype=np.int64)
    t_idx = np.clip(t_idx, 0, num_bins - 1)

    pol = (p > 0).astype(np.int64)  # 0: negative, 1: positive

    voxel = np.zeros((num_bins, 2, height, width), dtype=np.float32)
    np.add.at(voxel, (t_idx, pol, y, x), 1.0)

    vmax = voxel.max()
    if vmax > 0:
        voxel /= vmax

    return voxel


def process_one_image(args_tuple):
    """处理单张图像：加载 4 种 event → 转 voxel → 取平均 → 保存"""
    image_id, event_dirs, output_dir, num_bins, target_height, target_width = args_tuple
    out_path = output_dir / f"{image_id}.npz"
    if out_path.exists():
        return image_id, True, "skipped"

    voxels = []
    height, width = None, None

    for event_dir in event_dirs:
        npz_path = event_dir / f"{image_id}.npz"
        if not npz_path.exists():
            return image_id, False, f"missing {npz_path}"

        events = np.load(npz_path)

        if height is None and target_height is None:
            height = int(events['y'].max()) + 1
            width = int(events['x'].max()) + 1
        elif target_height is not None:
            height = target_height
            width = target_width if target_width is not None else target_height

        voxel = events_to_voxel_grid_fast(events, num_bins=num_bins,
                                           height=height, width=width)
        voxels.append(voxel)

    merged = np.mean(voxels, axis=0).astype(np.float32)
    np.savez_compressed(str(out_path), voxel=merged)
    return image_id, True, "ok"


def main():
    parser = argparse.ArgumentParser(description="Preprocess: merge 4 event types into voxel grid")
    parser.add_argument("--event_base_dir", type=str, required=True,
                        help="Base dir containing translation_x/ translation_y/ rotation/ radial/")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output dir for preprocessed .npz (key='voxel', shape=(T,2,H,W))")
    parser.add_argument("--num_bins", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--height", type=int, default=None,
                        help="输出 voxel 高度（默认按事件坐标推断）")
    parser.add_argument("--width", type=int, default=None,
                        help="输出 voxel 宽度（默认按事件坐标推断）")
    args = parser.parse_args()

    base = Path(args.event_base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    event_dirs = [base / et for et in EVENT_TYPES]
    for d in event_dirs:
        assert d.is_dir(), f"Event dir not found: {d}"

    ref_dir = event_dirs[0]
    image_ids = sorted([f.stem for f in ref_dir.iterdir() if f.suffix == '.npz'])
    print(f"Found {len(image_ids)} images to process")
    print(f"Event dirs: {[str(d) for d in event_dirs]}")
    print(f"Output: {output_dir}")

    tasks = [(img_id, event_dirs, output_dir, args.num_bins, args.height, args.width)
             for img_id in image_ids]

    ok, skip, fail = 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_one_image, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            img_id, success, msg = future.result()
            if not success:
                fail += 1
                print(f"  FAIL {img_id}: {msg}")
            elif msg == "skipped":
                skip += 1
            else:
                ok += 1

    print(f"\nDone! ok={ok}, skipped={skip}, failed={fail}")
    print(f"Output format: .npz, key='voxel', shape=({args.num_bins}, 2, H, W)")


if __name__ == "__main__":
    main()
