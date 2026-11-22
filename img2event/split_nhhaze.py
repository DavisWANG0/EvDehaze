#!/usr/bin/env python3
"""
把官方 NH-HAZE（55 对 NN_hazy.png / NN_GT.png）切分成 50 train / 5 test，
并重组成 ResShift PairedDataAligned 需要的「hazy 与 clear 同名」目录结构：

    <out_root>/train/hazy/NN.png      (来自 NN_hazy.png)
    <out_root>/train/clear/NN.png     (来自 NN_GT.png，stem 与 hazy 一致)
    <out_root>/test/hazy/NN.png
    <out_root>/test/clear/NN.png

clear 同时单独软链/拷贝到 <out_root>/{train,test}/clear（即可直接作为 vid2e 的 input_dir，
从 clear/GT 生成 event）。

默认 split：01-50 train，51-55 test（NH-HAZE 常用最后 5 张作 test）。
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/NH-HAZE",
                    help="解压后的 NH-HAZE 目录（含 NN_GT.png / NN_hazy.png）")
    ap.add_argument("--out", default="datasets/NH-HAZE",
                    help="输出根目录")
    ap.add_argument("--num_test", type=int, default=5)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    ids = sorted({p.name.split("_")[0] for p in src.glob("*_GT.png")})
    assert len(ids) == 55, f"expected 55 ids, got {len(ids)}"

    test_ids = ids[-args.num_test:]
    train_ids = ids[:-args.num_test]
    print(f"train={len(train_ids)} ({train_ids[0]}..{train_ids[-1]}), "
          f"test={len(test_ids)} ({test_ids})")

    for split, split_ids in [("train", train_ids), ("test", test_ids)]:
        hazy_d = out / split / "hazy"
        clear_d = out / split / "clear"
        hazy_d.mkdir(parents=True, exist_ok=True)
        clear_d.mkdir(parents=True, exist_ok=True)
        for i in split_ids:
            shutil.copy(src / f"{i}_hazy.png", hazy_d / f"{i}.png")
            shutil.copy(src / f"{i}_GT.png", clear_d / f"{i}.png")
        print(f"[{split}] hazy/clear -> {hazy_d.parent}  ({len(split_ids)} pairs)")


if __name__ == "__main__":
    main()
