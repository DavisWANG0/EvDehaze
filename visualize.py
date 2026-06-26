#!/usr/bin/env python
"""EvDehaze original-resolution visualization.

Runs the released event-conditioned ResShift dehazer on the *whole* image at its
original resolution (no tiling / no squeeze to 128) and writes the restored
images. This is the qualitative counterpart of inference.py (which reports 128px
tile metrics). The pipeline is identical to the release -- same bicubic VAE
bridge, diffusion UNet, event encoder and latent correction -- except the UNet
matches event features relative to the input latent size so event guidance stays
active at native resolution (EvDehazeFlexibleResolution). Weights load unchanged.

Each input is reflect-padded so the VAE latent is a multiple of the UNet window
(`--pad_to`), restored, then cropped back to the original size. Large images are
handled by downscaling the longest side to ``--max_side`` before restoration and
upscaling the result back to native (default 256; caps VAE / attention memory).

Example
-------
python visualize.py --config configs/evdehaze_sots_fullres.yaml \
    --out_dir outputs/evdehaze_fullres_vis --tta --num_flips 8 --max_images 10
# default --max_side 256; set --max_side 0 to attempt true native (may OOM on large images)
"""
import os
import argparse

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

os.environ.setdefault("LOCAL_RANK", "0")

from utils.util_common import get_obj_from_str
from utils import util_image
from datapipe.datasets import create_dataset


def parse_args():
    ap = argparse.ArgumentParser(description="EvDehaze original-resolution visualization")
    ap.add_argument("--config", default="configs/evdehaze_sots_fullres.yaml")
    ap.add_argument("--diffusion_ckpt", default="checkpoints/evdehaze_diffusion_ema.pth")
    ap.add_argument("--resample_ckpt", default="checkpoints/evdehaze_resample.pth")
    ap.add_argument("--out_dir", default="outputs/evdehaze_fullres_vis")
    ap.add_argument("--mode", choices=["deterministic", "stochastic"], default="deterministic")
    ap.add_argument("--tta", action="store_true", help="dihedral test-time augmentation")
    ap.add_argument("--num_flips", type=int, default=8, help="TTA transforms to average (<=8)")
    ap.add_argument("--pad_to", type=int, default=128,
                    help="reflect-pad H/W up to a multiple of this. 128 keeps the Swin middle block "
                         "aligned: image/128 -> latent/64 -> middle/8 (== window_size)")
    ap.add_argument("--max_side", type=int, default=256,
                    help="downscale longest side to this before restore, then upscale back to native "
                         "(default 256). 0 = no cap (true native; large images may OOM)")
    ap.add_argument("--max_images", type=int, default=0, help="0 = all val images")
    ap.add_argument("--no_event", action="store_true", help="ablation: drop event conditioning")
    ap.add_argument("--save_triplet", action="store_true",
                    help="also save hazy/restored/gt side-by-side")
    return ap.parse_args()


def build_cfg(args):
    cfg = OmegaConf.load(args.config)
    cfg.cfg_path = args.config
    cfg.save_dir = "outputs/evdehaze_fullres"
    cfg.resume = ""
    cfg.model.ckpt_path = args.diffusion_ckpt
    cfg.resample.ckpt_path = args.resample_ckpt
    cfg.train.use_ema_val = False
    return cfg


# ----- dihedral transforms (work for non-square tensors) -----------------------
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
    """reflect-pad (right/bottom) so H and W are multiples of m. Returns padded x, (H, W)."""
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
        if deterministic or ti == 0:
            z = out["mean"]
        else:
            z = out["mean"] + torch.exp(0.5 * out["log_variance"]) * torch.randn_like(z)
    return bd.decode_first_stage(z, ae).clamp(-1.0, 1.0)


def _norm(im_uint8):
    """HxWx3 uint8 -> (1,3,H,W) in [-1,1]."""
    t = util_image.ToTensor(max_value=255)(im_uint8)   # (3,H,W) [0,1]
    return ((t - 0.5) / 0.5).unsqueeze(0)


def main():
    args = parse_args()
    cfg = build_cfg(args)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda:0"

    trainer = get_obj_from_str(cfg.trainer.target)(cfg)
    trainer.configs = cfg
    trainer.init_logger()
    trainer.build_model()
    model = trainer.model; model.eval()
    ae = trainer.autoencoder; ae.eval()
    bd = trainer.base_diffusion
    deterministic = args.mode == "deterministic"
    tfs = _transforms(args.num_flips) if args.tta else [(0, False)]

    ds = create_dataset(cfg.data.val)
    n = len(ds.file_paths)
    if args.max_images:
        n = min(n, args.max_images)

    for idx in range(n):
        im_hazy, im_gt, event = ds._load_full(idx)            # uint8 HxWx3, uint8 HxWx3, (C,H,W)
        name = os.path.splitext(os.path.basename(ds.file_paths[idx]))[0]

        lq = _norm(im_hazy).to(device)
        ev = None if args.no_event else event.unsqueeze(0).to(device)
        H0, W0 = lq.shape[-2:]

        # optionally cap the working resolution (VAE mid-block attention is
        # quadratic in spatial size); the result is upscaled back to native.
        if args.max_side and max(H0, W0) > args.max_side:
            s = args.max_side / max(H0, W0)
            wh = (max(1, round(H0 * s)), max(1, round(W0 * s)))
            lq = F.interpolate(lq, size=wh, mode="bicubic", align_corners=False)
            if ev is not None:
                ev = F.interpolate(ev, size=wh, mode="bilinear", align_corners=False)

        lq_p, (H, W) = _pad_to(lq, args.pad_to)
        ev_p = None
        if ev is not None:
            ev_p, _ = _pad_to(ev, args.pad_to)

        out_sum = None
        for (k, flip) in tfs:
            mk = None
            if cfg.model.params.cond_lq:
                mk = {"lq": _fwd(lq_p, k, flip)}
                if ev_p is not None:
                    mk["event"] = _fwd(ev_p, k, flip)
            o = _inv(sample(bd, model, ae, _fwd(lq_p, k, flip), mk, deterministic, device), k, flip)
            out_sum = o if out_sum is None else out_sum + o
        out = (out_sum / len(tfs))[..., :H, :W]                # crop padding back
        if out.shape[-2:] != (H0, W0):                          # upscale back to native
            out = F.interpolate(out, size=(H0, W0), mode="bicubic", align_corners=False).clamp(-1.0, 1.0)

        restored = util_image.tensor2img(out * 0.5 + 0.5, rgb2bgr=True, min_max=(0, 1))
        util_image.imwrite(restored, os.path.join(args.out_dir, f"{name}_restored.png"),
                           chn='bgr', dtype_in='uint8')
        if args.save_triplet:
            import numpy as np
            trip = np.concatenate([im_hazy[..., ::-1], restored, im_gt[..., ::-1]], axis=1)
            util_image.imwrite(trip, os.path.join(args.out_dir, f"{name}_triplet.png"),
                               chn='bgr', dtype_in='uint8')
        print(f"  [{idx + 1}/{n}] {name}  ({W0}x{H0})", flush=True)

    print(f"saved {n} images -> {args.out_dir}")


if __name__ == "__main__":
    main()
