#!/usr/bin/env python
"""Build RGB–event pairing index and optional voxel cache for real-world captures.

Real-world folders use:
  RGB:   <prefix>_<session>_<frame_ms>.png   (~30 frames per session)
  Event: <prefix>_<session>_<chunk>_event.npy (one ~1s stream per session)

Example
-------
python scripts/preprocess_realworld_events.py \\
  --data_dir /path/to/realworld_capture \\
  --out_dir datasets/realworld_prepared \\
  --capture_prefix 20250415 \\
  --max_sessions 10
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.event_voxel import (
    load_event_voxel_for_rgb,
    parse_frame_ms,
    parse_event_chunk_ms,
    is_frame_alignable,
    SOTS_EVENT_WINDOW_MS,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--capture_prefix", default="*",
                    help="filename prefix for RGB/event glob (e.g. 20250415 or *)")
    ap.add_argument("--num_bins", type=int, default=4)
    ap.add_argument("--max_sessions", type=int, default=0, help="0 = all sessions")
    ap.add_argument("--frames_per_session", type=int, default=0, help="0 = all RGB per session")
    ap.add_argument("--event_time_crop", action="store_true",
                    help="per-RGB temporal crop aligned to frame_ms and event chunk offset")
    ap.add_argument("--event_window_ms", type=int, default=SOTS_EVENT_WINDOW_MS,
                    help=f"0=frame gap, else ms (SOTS ~{SOTS_EVENT_WINDOW_MS})")
    ap.add_argument("--skip_cache", action="store_true",
                    help="only write pairing index, do not save per-session npz")
    return ap.parse_args()


def session_from_rgb(path):
    parts = Path(path).stem.split("_")
    return parts[1] if len(parts) >= 3 else None


def find_event_file(data_dir, session, prefix):
    evs = sorted(data_dir.glob(f"{prefix}_{session}_*_event.npy"))
    return evs[0] if evs else None


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    prefix = args.capture_prefix
    hazy_dir = out_dir / "hazy"
    ev_dir = out_dir / "events_preprocessed"
    hazy_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_cache:
        ev_dir.mkdir(parents=True, exist_ok=True)

    rgbs = sorted(
        p for p in data_dir.glob(f"{prefix}_*.png")
        if "event_vis" not in p.name
    )
    by_sess = {}
    for p in rgbs:
        s = session_from_rgb(p)
        if s:
            by_sess.setdefault(s, []).append(p)

    sessions = sorted(by_sess)
    if args.max_sessions:
        sessions = sessions[: args.max_sessions]

    pairs = []
    for sess in tqdm(sessions, desc="sessions"):
        ev_path = find_event_file(data_dir, sess, prefix)
        if ev_path is None:
            continue
        event_chunk_ms = parse_event_chunk_ms(ev_path)
        window_ms = args.event_window_ms if args.event_window_ms > 0 else SOTS_EVENT_WINDOW_MS
        ref_rgb = sorted(by_sess[sess], key=lambda x: x.stem)
        if args.event_time_crop:
            ref_rgb = [
                p for p in ref_rgb
                if is_frame_alignable(parse_frame_ms(p), event_chunk_ms, window_ms)
            ]
        rgb_list = ref_rgb
        if args.frames_per_session and args.frames_per_session < len(ref_rgb):
            idx = np.linspace(0, len(ref_rgb) - 1, args.frames_per_session, dtype=int)
            rgb_list = [ref_rgb[i] for i in idx]
        if not rgb_list:
            continue
        im = Image.open(rgb_list[0])
        w, h = im.size
        for rgb_path in rgb_list:
            frame_ms = parse_frame_ms(rgb_path)
            cache_name = f"{sess}_{frame_ms:04d}.npz" if args.event_time_crop else f"{sess}.npz"
            npz_path = ev_dir / cache_name
            if not args.skip_cache and not npz_path.exists():
                voxel = load_event_voxel_for_rgb(
                    ev_path, h, w, num_bins=args.num_bins,
                    frame_ms=frame_ms,
                    event_chunk_ms=event_chunk_ms,
                    window_ms=window_ms,
                    time_crop=args.event_time_crop,
                )
                np.savez_compressed(npz_path, voxel=voxel.reshape(args.num_bins, 2, h, w))
            out_name = rgb_path.name
            out_rgb = hazy_dir / out_name
            if not out_rgb.exists():
                Image.open(rgb_path).save(out_rgb)
            pairs.append({
                "rgb": str(out_rgb.relative_to(out_dir)),
                "session": sess,
                "frame_ms": frame_ms,
                "event_chunk_ms": event_chunk_ms,
                "window_ms": window_ms,
                "time_crop": args.event_time_crop,
                "event_npy": str(ev_path),
                "event_npz": str(npz_path.relative_to(out_dir)) if not args.skip_cache else None,
                "size_wh": [w, h],
            })

    with open(out_dir / "pairs.json", "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"wrote {len(pairs)} pairs across {len(sessions)} sessions -> {out_dir}")


if __name__ == "__main__":
    main()
