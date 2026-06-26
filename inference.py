#!/usr/bin/env python
"""EvDehaze inference / evaluation.

Runs the released event-conditioned ResShift dehazer on the validation set and
reports PSNR / SSIM / LPIPS. Two inference modes:

  --mode deterministic  (default)  zero initial noise + mean-only reverse steps
                                    (the conditional-mean path; best fidelity).
  --mode stochastic                 standard ResShift sampling (one random sample).

Optional test-time augmentation:
  --tta            average over the dihedral group of geometric transforms
  --num_flips N    number of transforms to average (default 8 = full dihedral)

Optional sharding for multi-GPU eval (each process handles 1/NUM_SHARDS of the
images, then run `--combine` to merge the partial results):
  SHARD / NUM_SHARDS env vars + --shard_out <file>

The checkpoint, config and target are taken from --config / --diffusion_ckpt /
--resample_ckpt. The config already encodes steps=15 and latent_correction=True,
matching the released checkpoint.

Examples
--------
# deterministic, full val set, single GPU
python inference.py --mode deterministic

# deterministic + 8-flip TTA sharded over 3 GPUs
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s SHARD=$s NUM_SHARDS=3 \
    python inference.py --mode deterministic --tta --num_flips 8 \
      --shard_out shard_$s.json &
done; wait
python inference.py --combine --shard_glob 'shard_*.json'
"""
import os
import sys
import json
import glob
import argparse

import torch
import lpips as lpips_pkg
from omegaconf import OmegaConf

os.environ.setdefault("LOCAL_RANK", "0")

from utils.util_common import get_obj_from_str
from utils import util_image


def parse_args():
    ap = argparse.ArgumentParser(description="EvDehaze inference / evaluation")
    ap.add_argument("--config", default="configs/evdehaze_sots.yaml")
    ap.add_argument("--diffusion_ckpt", default="checkpoints/evdehaze_diffusion_ema.pth")
    ap.add_argument("--resample_ckpt", default="checkpoints/evdehaze_resample.pth")
    ap.add_argument("--mode", choices=["deterministic", "stochastic"], default="deterministic")
    ap.add_argument("--tta", action="store_true", help="dihedral test-time augmentation")
    ap.add_argument("--num_flips", type=int, default=8, help="TTA transforms to average (<=8)")
    ap.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"],
                    help="LPIPS backbone for the reported metric (default alex)")
    ap.add_argument("--max_tiles", type=int, default=0, help="0 = full val set")
    ap.add_argument("--no_event", action="store_true", help="ablation: drop event conditioning")
    ap.add_argument("--shard_out", default="", help="write partial sums json (for sharded multi-GPU eval)")
    ap.add_argument("--combine", action="store_true", help="combine shard jsons and print final metric")
    ap.add_argument("--shard_glob", default="shard_*.json")
    return ap.parse_args()


def build_cfg(args):
    cfg = OmegaConf.load(args.config)
    cfg.cfg_path = args.config
    cfg.save_dir = f"outputs/evdehaze-eval-shard{os.environ.get('SHARD', '0')}"
    cfg.resume = ""
    cfg.model.ckpt_path = args.diffusion_ckpt
    cfg.resample.ckpt_path = args.resample_ckpt
    cfg.train.use_ema_val = False  # the diffusion ckpt is already the EMA weights
    if "length" in cfg.data.val.params:
        cfg.data.val.params.length = None
    return cfg


# ----- dihedral transforms (square tiles) --------------------------------------
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


def combine(args):
    files = sorted(glob.glob(args.shard_glob))
    tot_p = tot_s = tot_l = tot_n = 0.0
    for f in files:
        d = json.load(open(f))
        tot_p += d["psnr_sum"]; tot_n += d["n"]
        tot_s += d.get("ssim_sum", 0.0); tot_l += d.get("lpips_sum", 0.0)
        print(f"  {f}: n={d['n']} PSNR={d['psnr_sum']/d['n']:.4f}")
    print("=" * 70)
    print(f"EvDehaze  ({int(tot_n)} tiles)")
    print(f"  PSNR  = {tot_p/tot_n:.4f}")
    print(f"  SSIM  = {tot_s/tot_n:.4f}")
    print(f"  LPIPS = {tot_l/tot_n:.4f}")
    print("=" * 70)


def main():
    args = parse_args()
    if args.combine:
        combine(args); return

    shard = int(os.environ.get("SHARD", "0"))
    num_shards = int(os.environ.get("NUM_SHARDS", "1"))

    cfg = build_cfg(args)
    trainer = get_obj_from_str(cfg.trainer.target)(cfg)
    trainer.configs = cfg
    trainer.init_logger()
    trainer.build_model()
    trainer.build_dataloader()
    model = trainer.model; model.eval()
    ae = trainer.autoencoder; ae.eval()
    bd = trainer.base_diffusion
    yc = cfg.train.val_y_channel
    device = "cuda:0"
    deterministic = args.mode == "deterministic"
    tfs = _transforms(args.num_flips) if args.tta else [(0, False)]

    # reported LPIPS uses AlexNet (--lpips_net alex, default)
    metric_lpips = lpips_pkg.LPIPS(net=args.lpips_net).to(device).eval()
    for p in metric_lpips.parameters():
        p.requires_grad_(False)

    psnr_sum = ssim_sum = lpips_sum = 0.0
    nimg = 0
    for ii, data in enumerate(trainer.dataloaders["val"]):
        if ii % num_shards != shard:
            continue
        data = trainer.prepare_data(data, phase="val")
        gt, lq = data["gt"], data["lq"]
        ev = None if args.no_event else data.get("event", None)
        out_sum = None
        for (k, flip) in tfs:
            mk = None
            if cfg.model.params.cond_lq:
                mk = {"lq": _fwd(lq, k, flip)}
                if ev is not None:
                    mk["event"] = _fwd(ev, k, flip)
            o = _inv(sample(bd, model, ae, _fwd(lq, k, flip), mk, deterministic, device), k, flip)
            out_sum = o if out_sum is None else out_sum + o
        out = out_sum / len(tfs)
        psnr_sum += util_image.batch_PSNR(out * 0.5 + 0.5, gt * 0.5 + 0.5, ycbcr=yc)
        ssim_sum += util_image.batch_SSIM(out * 0.5 + 0.5, gt * 0.5 + 0.5, ycbcr=yc)
        lpips_sum += metric_lpips(out, gt).sum().item()
        nimg += gt.shape[0]
        if (ii // num_shards) % 25 == 0:
            print(f"  [{nimg}] PSNR={psnr_sum/nimg:.4f}", flush=True)
        if args.shard_out:
            json.dump({"psnr_sum": psnr_sum, "ssim_sum": ssim_sum,
                       "lpips_sum": lpips_sum, "n": nimg}, open(args.shard_out, "w"))
        if args.max_tiles and nimg >= args.max_tiles:
            break

    if args.shard_out:
        json.dump({"psnr_sum": psnr_sum, "ssim_sum": ssim_sum,
                   "lpips_sum": lpips_sum, "n": nimg}, open(args.shard_out, "w"))
    tag = f"{args.mode}{' + ' + str(args.num_flips) + '-flip TTA' if args.tta else ''}"
    print("=" * 70)
    print(f"EvDehaze [{tag}]  ({nimg} tiles)")
    print(f"  PSNR  = {psnr_sum/nimg:.4f}")
    print(f"  SSIM  = {ssim_sum/nimg:.4f}")
    print(f"  LPIPS = {lpips_sum/nimg:.4f}  ({args.lpips_net})")
    print("=" * 70)


if __name__ == "__main__":
    main()
