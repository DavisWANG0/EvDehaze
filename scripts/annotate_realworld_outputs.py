#!/usr/bin/env python
"""Add event_vis + RGB channel histograms to existing real-world outputs.

Does not re-run diffusion; reads ``*_restored.png`` and pairs hazy RGB + events
from ``--data_dir`` (same layout as visualize_realworld.py).

Example
-------
python scripts/annotate_realworld_outputs.py \\
  --out_dir outputs/evdehaze_realworld_flex35k_sots480 \\
  --data_dir /path/to/realworld_capture \\
  --capture_prefix 20250415 --event_time_crop --sots_frame
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.realworld_vis import (
    event_tensor_to_vis,
    save_channel_histogram,
    save_triptych,
)
from utils.event_voxel import SOTS_EVENT_WINDOW_MS
from visualize_realworld import (
    _apply_sots_frame,
    _hazy_bgr,
    _load_event,
    collect_pairs,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Annotate real-world output folder")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_dir", default="")
    ap.add_argument("--prepared_dir", default="")
    ap.add_argument("--capture_prefix", default="*")
    ap.add_argument("--session", default="")
    ap.add_argument("--max_sessions", type=int, default=0)
    ap.add_argument("--session_start_idx", type=int, default=0)
    ap.add_argument("--frames_per_session", type=int, default=0)
    ap.add_argument("--event_time_crop", action="store_true")
    ap.add_argument("--event_window_ms", type=int, default=SOTS_EVENT_WINDOW_MS)
    ap.add_argument("--event_modulo", action="store_true")
    ap.add_argument("--include_misaligned", action="store_true")
    ap.add_argument("--sots_frame", action="store_true")
    ap.add_argument("--no_event", action="store_true")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip if event_vis and channel_hist already exist")
    return ap.parse_args()


def _pair_by_stem(pairs, stem):
    for p in pairs:
        if Path(p["_rgb_abs"]).stem == stem:
            return p
    return None


def main():
    args = parse_args()
    if not args.data_dir and not args.prepared_dir:
        raise SystemExit("set --data_dir or --prepared_dir")

    pairs = collect_pairs(args)
    by_stem = {Path(p["_rgb_abs"]).stem: p for p in pairs}

    out_dir = Path(args.out_dir)
    restored_paths = sorted(
        p for p in out_dir.glob("*_restored.png")
        if not p.stem.endswith("_hazy")
    )
    if not restored_paths:
        raise SystemExit(f"no *_restored.png in {out_dir}")

    for i, rp in enumerate(restored_paths):
        stem = rp.stem.replace("_restored", "")
        pair = by_stem.get(stem)
        if pair is None:
            print(f"  [{i+1}] skip {stem}: no RGB/event pair", flush=True)
            continue

        ev_out = out_dir / f"{stem}_event_vis.png"
        hist_out = out_dir / f"{stem}_channel_hist.png"
        trip_out = out_dir / f"{stem}_triptych.png"
        if args.skip_existing and ev_out.is_file() and hist_out.is_file():
            print(f"  [{i+1}] skip existing {stem}", flush=True)
            continue

        restored_bgr = np.array(Image.open(rp).convert("RGB"))[..., ::-1]
        restored_rgb = restored_bgr[..., ::-1]

        hazy_bgr = _hazy_bgr(pair["_rgb_abs"], sots_frame=args.sots_frame)
        if hazy_bgr.shape[:2] != restored_bgr.shape[:2]:
            h, w = restored_bgr.shape[:2]
            hazy_bgr = np.array(
                Image.fromarray(hazy_bgr[..., ::-1]).resize((w, h), Image.BICUBIC)
            )[..., ::-1]
        hazy_rgb = hazy_bgr[..., ::-1]

        ev_vis_rgb = None
        if not args.no_event:
            ev = _load_event(pair, pair.get("_root"), args)
            if args.sots_frame:
                lq_dummy = torch.zeros(1, 3, hazy_rgb.shape[0], hazy_rgb.shape[1])
                _, ev, _, _ = _apply_sots_frame(lq_dummy, ev)
            oh, ow = restored_rgb.shape[:2]
            ev_vis_rgb = event_tensor_to_vis(ev, oh, ow)
            Image.fromarray(ev_vis_rgb).save(ev_out)

        save_channel_histogram(
            hazy_rgb, restored_rgb, str(hist_out), title=f"{stem} — channel histogram"
        )
        if ev_vis_rgb is not None:
            save_triptych(hazy_rgb, ev_vis_rgb, restored_rgb, str(trip_out))

        print(f"  [{i+1}/{len(restored_paths)}] {stem} -> event_vis, channel_hist, triptych",
              flush=True)

    print(f"annotated -> {out_dir}")


if __name__ == "__main__":
    main()
