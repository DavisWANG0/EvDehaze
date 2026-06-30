#!/usr/bin/env python
"""EvDehaze real-world dehazing (no GT, qualitative only).

Loads hazy RGB + paired event (raw .npy or preprocessed .npz from
scripts/preprocess_realworld_events.py) and runs the SOTS-trained checkpoint.

Example
-------
# batch: sessions with temporal event crop
python visualize_realworld.py \\
  --data_dir /path/to/realworld_capture \\
  --max_sessions 20 --frames_per_session 3 --event_time_crop --max_side 256

# SOTS-aligned frame (4:3 center crop -> 640x480, no upscale to native)
python visualize_realworld.py \\
  --data_dir /path/to/realworld_capture \\
  --sots_frame --max_side 384 --event_time_crop \\
  --max_sessions 5 --frames_per_session 2

# after preprocess_realworld_events.py
python visualize_realworld.py \\
  --prepared_dir datasets/realworld_prepared --max_images 20 --tta
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

os.environ.setdefault("LOCAL_RANK", "0")

from utils.util_common import get_obj_from_str
from utils import util_image
from utils.realworld_vis import (
    event_tensor_to_vis,
    save_combined_panel,
)
from utils.event_voxel import (
    load_event_voxel_for_rgb,
    parse_frame_ms,
    parse_event_chunk_ms,
    is_frame_alignable,
    SOTS_EVENT_WINDOW_MS,
)

# SOTS clear GT native size (H, W)
SOTS_CLEAR_H, SOTS_CLEAR_W = 480, 640


def parse_args():
    ap = argparse.ArgumentParser(description="EvDehaze real-world visualization")
    ap.add_argument("--config", default="configs/evdehaze_sots_fullres.yaml")
    ap.add_argument("--diffusion_ckpt", default="checkpoints/evdehaze_diffusion_ema.pth")
    ap.add_argument("--resample_ckpt", default="checkpoints/evdehaze_resample.pth")
    ap.add_argument("--data_dir", default="", help="raw real-world folder")
    ap.add_argument("--prepared_dir", default="", help="output of preprocess_realworld_events.py")
    ap.add_argument("--capture_prefix", default="*",
                    help="filename prefix for RGB/event glob (e.g. 20250415 or *)")
    ap.add_argument("--session", default="", help="only this session id, e.g. 083338")
    ap.add_argument("--max_sessions", type=int, default=0, help="limit number of sessions")
    ap.add_argument("--session_start_idx", type=int, default=0,
                    help="skip first N sessions in sorted order (use with max_sessions)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip frames whose restored png already exists in out_dir")
    ap.add_argument("--frames_per_session", type=int, default=0,
                    help="sample this many RGB frames per session (0 = all)")
    ap.add_argument("--event_time_crop", action="store_true",
                    help="crop events to frame_ms window (recommended for real-world)")
    ap.add_argument("--event_window_ms", type=int, default=SOTS_EVENT_WINDOW_MS,
                    help=f"temporal window in ms (SOTS synthetic ~{SOTS_EVENT_WINDOW_MS}ms; 0=frame gap)")
    ap.add_argument("--event_modulo", action="store_true",
                    help="legacy: wrap frame_ms into 1s clip (usually wrong)")
    ap.add_argument("--include_misaligned", action="store_true",
                    help="keep RGB frames without a full 83ms event window")
    ap.add_argument("--out_dir", default="outputs/evdehaze_realworld_vis")
    ap.add_argument("--mode", choices=["deterministic", "stochastic"], default="deterministic")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--num_flips", type=int, default=8)
    ap.add_argument("--pad_to", type=int, default=128)
    ap.add_argument("--max_side", type=int, default=256,
                    help="cap longest side before inference (0=native; 480 with --sots_frame)")
    ap.add_argument("--sots_frame", action="store_true",
                    help="center 4:3 crop then resize to 640x480 (SOTS clear); "
                         "event bilinear to match; do not upscale to native 720p")
    ap.add_argument("--no_upscale", action="store_true",
                    help="save at inference resolution (auto-enabled with --sots_frame)")
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--no_event", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_diagnostics", action="store_true",
                    help="skip event_vis, channel histogram, and triptych outputs")
    args = ap.parse_args()
    if args.sots_frame:
        args.no_upscale = True
        if args.max_side == 256:
            args.max_side = 384  # safer default at 640x480 SOTS frame size
    return args


def build_cfg(args):
    cfg = OmegaConf.load(args.config)
    cfg.cfg_path = args.config
    cfg.save_dir = "outputs/evdehaze_realworld"
    cfg.resume = ""
    cfg.model.ckpt_path = args.diffusion_ckpt
    cfg.resample.ckpt_path = args.resample_ckpt
    cfg.train.use_ema_val = False
    return cfg


def _fwd(x, k, flip):
    x = torch.rot90(x, k, dims=[-2, -1])
    return torch.flip(x, dims=[-1]) if flip else x


def _inv(x, k, flip):
    if flip:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, -k, dims=[-2, -1])


def _transforms(n):
    combos = [(k, f) for f in (False, True) for k in (0, 1, 2, 3)]
    return combos[:n]


def _pad_to(x, m):
    h, w = x.shape[-2:]
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (h, w)


@torch.no_grad()
def sample(bd, model, ae, y, model_kwargs, deterministic, device):
    z_y = bd.encode_first_stage(y, ae, up_sample=True)
    z = z_y.clone() if deterministic else bd.prior_sample(z_y, torch.randn_like(z_y))
    for ti in reversed(range(bd.num_timesteps)):
        tt = torch.tensor([ti] * y.shape[0], dtype=torch.int64, device=device)
        out = bd.p_mean_variance(model, z, z_y, tt, clip_denoised=False, model_kwargs=model_kwargs)
        z = out["mean"] if (deterministic or ti == 0) else (
            out["mean"] + torch.exp(0.5 * out["log_variance"]) * torch.randn_like(z))
    return bd.decode_first_stage(z, ae).clamp(-1.0, 1.0)


def _crop_box_center_43(h, w):
    """Pixel box (y0, x0, y1, x1) for a centered 4:3 crop."""
    ar = w / h
    target = 4 / 3
    if ar > target:
        new_w = int(round(h * target))
        x0 = (w - new_w) // 2
        return 0, x0, h, x0 + new_w
    new_h = int(round(w / target))
    y0 = (h - new_h) // 2
    return y0, 0, y0 + new_h, w


def _apply_sots_frame(lq, ev, out_h=SOTS_CLEAR_H, out_w=SOTS_CLEAR_W):
    """Native frame -> center 4:3 crop -> SOTS clear size; event follows with bilinear."""
    h, w = lq.shape[-2:]
    y0, x0, y1, x1 = _crop_box_center_43(h, w)
    lq = lq[..., y0:y1, x0:x1]
    lq = F.interpolate(lq, size=(out_h, out_w), mode="bicubic", align_corners=False)
    if ev is not None:
        ev = ev[..., y0:y1, x0:x1]
        ev = F.interpolate(ev, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return lq, ev, out_h, out_w


def apply_spatial_pipeline(lq, ev, args, H_native, W_native):
    """Match inference: optional SOTS crop, then max_side downscale (before pad)."""
    if args.sots_frame:
        lq, ev, H0, W0 = _apply_sots_frame(lq, ev)
    else:
        H0, W0 = H_native, W_native
    if args.max_side and max(H0, W0) > args.max_side:
        s = args.max_side / max(H0, W0)
        wh = (max(1, round(H0 * s)), max(1, round(W0 * s)))
        lq = F.interpolate(lq, size=wh, mode="bicubic", align_corners=False)
        if ev is not None:
            ev = F.interpolate(ev, size=wh, mode="bilinear", align_corners=False)
    return lq, ev


def _lq_tensor_to_rgb(lq):
    """Normalized NCHW lq in [-1,1] -> HxWx3 uint8 RGB."""
    x = lq.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return np.clip((x * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def prepare_display_pair(rgb_path, pair, args, target_h, target_w, root=None):
    """Hazy RGB + event voxel aligned to the restored image grid."""
    lq, H_native, W_native = _norm_rgb(rgb_path)
    ev = None if args.no_event else _load_event(pair, root, args)
    lq, ev = apply_spatial_pipeline(lq, ev, args, H_native, W_native)
    dh, dw = lq.shape[-2:]
    if (dh, dw) != (target_h, target_w):
        lq = F.interpolate(lq, size=(target_h, target_w), mode="bicubic", align_corners=False)
        if ev is not None:
            ev = F.interpolate(ev, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return _lq_tensor_to_rgb(lq), ev


def _hazy_bgr(rgb_path, sots_frame=False):
    im = np.array(Image.open(rgb_path).convert("RGB"))
    if sots_frame:
        y0, x0, y1, x1 = _crop_box_center_43(im.shape[0], im.shape[1])
        im = im[y0:y1, x0:x1]
        im = np.array(Image.fromarray(im).resize((SOTS_CLEAR_W, SOTS_CLEAR_H), Image.BICUBIC))
    return im[..., ::-1]


def _norm_rgb(path):
    im = np.array(Image.open(path).convert("RGB"))
    t = util_image.ToTensor(max_value=255)(im)
    return ((t - 0.5) / 0.5).unsqueeze(0), im.shape[0], im.shape[1]


def _load_event(pair, data_dir=None, args=None):
    if pair.get("event_npz"):
        p = Path(pair["event_npz"])
        if not p.is_absolute() and data_dir:
            p = Path(data_dir) / p
        npz = np.load(p)
        v = npz["voxel"].reshape(-1, *npz["voxel"].shape[-2:])
        return torch.from_numpy(v).float().unsqueeze(0)
    ev_npy = Path(pair["event_npy"])
    if data_dir and not ev_npy.is_absolute():
        ev_npy = Path(data_dir) / ev_npy
    w, h = pair["size_wh"]
    time_crop = args.event_time_crop if args else pair.get("time_crop", False)
    v = load_event_voxel_for_rgb(
        ev_npy, h, w,
        frame_ms=pair.get("frame_ms"),
        event_chunk_ms=pair.get("event_chunk_ms", 0),
        window_ms=pair.get("window_ms", SOTS_EVENT_WINDOW_MS),
        time_crop=time_crop,
        use_modulo=bool(args and args.event_modulo),
    )
    return torch.from_numpy(v).unsqueeze(0)


def _sample_session_frames(paths, frames_per_session, alignable_ms=None):
    paths = sorted(paths, key=lambda x: int(x.stem.split("_")[-1]))
    if alignable_ms is not None:
        paths = [p for p in paths if int(p.stem.split("_")[-1]) in alignable_ms]
    if not paths:
        return []
    if not frames_per_session or frames_per_session >= len(paths):
        return paths
    if frames_per_session == 1:
        return [paths[len(paths) // 2]]
    idx = np.linspace(0, len(paths) - 1, frames_per_session, dtype=int)
    return [paths[i] for i in idx]


def collect_pairs(args):
    if args.prepared_dir:
        root = Path(args.prepared_dir)
        with open(root / "pairs.json") as f:
            pairs = json.load(f)
        for p in pairs:
            p["_rgb_abs"] = str(root / p["rgb"])
            p["_root"] = str(root)
        if args.session:
            pairs = [p for p in pairs if p["session"] == args.session]
        return pairs

    root = Path(args.data_dir)
    prefix = args.capture_prefix
    rgbs = sorted(
        p for p in root.glob(f"{prefix}_*.png")
        if "event_vis" not in p.name
    )
    if args.session:
        rgbs = [p for p in rgbs if f"_{args.session}_" in p.name]
    by_sess = {}
    for p in rgbs:
        sess = p.stem.split("_")[1]
        by_sess.setdefault(sess, []).append(p)
    pairs = []
    sessions = sorted(by_sess)
    if args.session_start_idx:
        sessions = sessions[args.session_start_idx:]
    if args.max_sessions:
        sessions = sessions[: args.max_sessions]
    for sess in sessions:
        evs = sorted(root.glob(f"{prefix}_{sess}_*_event.npy"))
        if not evs:
            continue
        ev_path = evs[0]
        event_chunk_ms = parse_event_chunk_ms(ev_path)
        all_ms = sorted(int(p.stem.split("_")[-1]) for p in by_sess[sess])
        window_ms = args.event_window_ms if args.event_window_ms > 0 else SOTS_EVENT_WINDOW_MS
        alignable_ms = None
        if args.event_time_crop and not args.include_misaligned:
            alignable_ms = {
                m for m in all_ms
                if is_frame_alignable(
                    m, event_chunk_ms, window_ms, use_modulo=args.event_modulo
                )
            }
        paths = _sample_session_frames(
            by_sess[sess], args.frames_per_session, alignable_ms=alignable_ms
        )
        if not paths:
            print(f"  [skip] session {sess}: no RGB with aligned event window", flush=True)
            continue
        im = Image.open(paths[0])
        w, h = im.size
        for rgb_path in paths:
            frame_ms = parse_frame_ms(rgb_path)
            pairs.append({
                "_rgb_abs": str(rgb_path),
                "event_npy": str(ev_path),
                "size_wh": [w, h],
                "session": sess,
                "frame_ms": frame_ms,
                "event_chunk_ms": event_chunk_ms,
                "window_ms": window_ms,
            })
    return pairs


def main():
    args = parse_args()
    if not args.data_dir and not args.prepared_dir:
        raise SystemExit("set --data_dir or --prepared_dir")

    pairs = collect_pairs(args)
    if args.max_images:
        pairs = pairs[: args.max_images]
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = build_cfg(args)
    trainer = get_obj_from_str(cfg.trainer.target)(cfg)
    trainer.configs = cfg
    trainer.init_logger()
    trainer.build_model()
    model = trainer.model; model.eval()
    ae = trainer.autoencoder; ae.eval()
    bd = trainer.base_diffusion
    device = args.device
    deterministic = args.mode == "deterministic"
    tfs = _transforms(args.num_flips) if args.tta else [(0, False)]

    for i, pair in enumerate(pairs):
        rgb_path = pair["_rgb_abs"]
        name = Path(rgb_path).stem
        out_restored = os.path.join(args.out_dir, f"{name}_restored.png")
        if args.skip_existing and os.path.isfile(out_restored):
            print(f"  [{i+1}/{len(pairs)}] skip existing {name}", flush=True)
            continue
        lq, H_native, W_native = _norm_rgb(rgb_path)
        lq = lq.to(device)
        ev = None if args.no_event else _load_event(pair, pair.get("_root"), args).to(device)

        lq, ev = apply_spatial_pipeline(lq, ev, args, H_native, W_native)

        lq_p, (H, W) = _pad_to(lq, args.pad_to)
        ev_p = None
        if ev is not None:
            ev_p, _ = _pad_to(ev, args.pad_to)

        out_sum = None
        for k, flip in tfs:
            mk = None
            if cfg.model.params.cond_lq:
                mk = {"lq": _fwd(lq_p, k, flip)}
                if ev_p is not None:
                    mk["event"] = _fwd(ev_p, k, flip)
            o = _inv(sample(bd, model, ae, _fwd(lq_p, k, flip), mk, deterministic, device), k, flip)
            out_sum = o if out_sum is None else out_sum + o
        out = (out_sum / len(tfs))[..., :H, :W]
        out_h, out_w = out.shape[-2:]
        if not args.no_upscale and (out_h, out_w) != (H_native, W_native):
            out = F.interpolate(
                out, size=(H_native, W_native), mode="bicubic", align_corners=False
            ).clamp(-1.0, 1.0)
            out_h, out_w = H_native, W_native

        hazy_rgb = _lq_tensor_to_rgb(lq)
        hazy_bgr = hazy_rgb[..., ::-1]
        restored = util_image.tensor2img(out * 0.5 + 0.5, rgb2bgr=True, min_max=(0, 1))
        trip = np.concatenate([hazy_bgr, restored], axis=1)
        util_image.imwrite(trip, os.path.join(args.out_dir, f"{name}_hazy_restored.png"),
                           chn="bgr", dtype_in="uint8")
        util_image.imwrite(restored, os.path.join(args.out_dir, f"{name}_restored.png"),
                           chn="bgr", dtype_in="uint8")

        if not args.no_diagnostics:
            restored_rgb = restored[..., ::-1]
            diag_base = os.path.join(args.out_dir, name)
            if ev is not None:
                ev_vis = event_tensor_to_vis(ev, out_h, out_w)
                Image.fromarray(ev_vis).save(f"{diag_base}_event_vis.png")
                save_combined_panel(
                    hazy_rgb, ev_vis, restored_rgb, f"{diag_base}_panel.png",
                    title=f"{name}",
                    event_tensor=ev,
                )

        frame_tag = f"sots640x480" if args.sots_frame else f"{W_native}x{H_native}"
        print(f"  [{i+1}/{len(pairs)}] {name} native={W_native}x{H_native} "
              f"infer={out_w}x{out_h} frame={frame_tag} "
              f"sess={pair.get('session','')} frame_ms={pair.get('frame_ms','')} "
              f"chunk={pair.get('event_chunk_ms','')}",
              flush=True)

    print(f"saved {len(pairs)} results -> {args.out_dir}")


if __name__ == "__main__":
    main()
