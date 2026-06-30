#!/usr/bin/env python
"""Add event_vis + RGB channel histograms to existing real-world outputs.

Does not re-run diffusion; reads ``*_restored.png`` and pairs hazy RGB + events
from ``--data_dir`` (same layout as visualize_realworld.py).

Example
-------
python scripts/annotate_realworld_outputs.py \\
  --out_dir outputs/evdehaze_realworld_flex35k_sots480 \\
  --data_dir /path/to/realworld_capture \\
  --capture_prefix 20250415 --event_time_crop --sots_frame --max_side 384
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.realworld_vis import (
    event_tensor_to_vis,
    save_combined_panel,
)
from utils.event_voxel import SOTS_EVENT_WINDOW_MS
from visualize_realworld import collect_pairs, prepare_display_pair


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
    ap.add_argument("--max_side", type=int, default=256,
                    help="must match the inference run (384 typical with --sots_frame)")
    ap.add_argument("--no_event", action="store_true")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip if panel png already exists")
    args = ap.parse_args()
    if args.sots_frame and args.max_side == 256:
        args.max_side = 384
    return args


def main():
    args = parse_args()
    if not args.data_dir and not args.prepared_dir:
        raise SystemExit("set --data_dir or --prepared_dir")

    pairs = collect_pairs(args)
    by_stem = {Path(p["_rgb_abs"]).stem: p for p in pairs}

    out_dir = Path(args.out_dir)
    restored_paths = sorted(
        p for p in out_dir.glob("*_restored.png")
        if "_hazy_restored" not in p.name
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
        panel_out = out_dir / f"{stem}_panel.png"
        if args.skip_existing and panel_out.is_file():
            print(f"  [{i+1}] skip existing {stem}", flush=True)
            continue

        restored_rgb = np.array(Image.open(rp).convert("RGB"))
        oh, ow = restored_rgb.shape[:2]

        hazy_rgb, ev = prepare_display_pair(
            pair["_rgb_abs"], pair, args, oh, ow, root=pair.get("_root"),
        )

        ev_vis_rgb = None
        if ev is not None:
            ev_vis_rgb = event_tensor_to_vis(ev, oh, ow)
            Image.fromarray(ev_vis_rgb).save(ev_out)
            save_combined_panel(
                hazy_rgb, ev_vis_rgb, restored_rgb, str(panel_out),
                title=f"{stem} — dehazing panel",
            )
        else:
            save_combined_panel(
                hazy_rgb, np.zeros_like(hazy_rgb), restored_rgb, str(panel_out),
                title=f"{stem} — dehazing panel",
            )

        print(f"  [{i+1}/{len(restored_paths)}] {stem} aligned {ow}x{oh} -> "
              f"event_vis, panel", flush=True)

    print(f"annotated -> {out_dir}")


if __name__ == "__main__":
    main()
