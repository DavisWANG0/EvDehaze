"""
Bicubic pixel-domain resampling around a frozen VAE, plus a trainable latent
residual correction.

To compare diffusion- and transformer-based dehazers fairly, every method is
trained and evaluated at 128x128. This wrapper bridges the 128px image space and
the VAE's native resolution with plain bicubic interpolation (no learnable
resampler):

    input 128 ──(bicubic Up x2)──▶ 256 ──[Frozen VAE.encode]──▶ latent 64
                                                                  │
                                                             diffusion UNet
                                                                  │
    output 128 ◀──(bicubic Down /2)── 256 ◀──[Frozen VAE.decode]─ latent 64

The VAE is frozen (requires_grad=False). The only trained component is a
zero-initialized residual correction on the predicted clean (x0) latent, applied
right before the frozen VAE decode to remove the systematic VAE-latent bias.

FrozenVAEResampler exposes the same .encode/.decode interface as the raw VAE and
provides ``downsample_factor`` / ``embed_dim`` so the trainer derives the latent
resolution as ``gt_size // downsample_factor`` (128 // 2 = 64).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class _LatentResidual(nn.Module):
    """Zero-initialized residual correction on the predicted x0 latent.

    Applied right before the frozen VAE decode so the model can fix the
    systematic VAE-latent bias of the diffusion x0 prediction. Identity at init
    (last conv zero-init), so warm-start is safe.
    """

    def __init__(self, ch=3, hidden=64, num_blocks=2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            *[_ResBlock(hidden) for _ in range(int(num_blocks))],
            nn.Conv2d(hidden, ch, 3, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, z):
        return z + self.body(z)


class FrozenVAEResampler(nn.Module):
    """
    Frozen VAE wrapped with bicubic 128<->256 resampling and a trainable latent
    residual correction.

    - encode(x_128) -> z_64 :  vae.encode(bicubic_up(x_128))
    - decode(z_64)  -> y_128:  bicubic_down(vae.decode(latent_correction(z_64)))

    Exposes downsample_factor / embed_dim for the trainer to compute the latent
    resolution. The VAE is frozen; only ``latent_correction`` is trainable.
    """

    def __init__(self, vae, embed_dim, latent_correction=False,
                 latent_kwargs=None, scale=2, **_ignored):
        super().__init__()
        self.vae = vae
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.scale = int(scale)
        if latent_correction:
            self.latent_correction = _LatentResidual(ch=embed_dim, **(latent_kwargs or {}))
        else:
            self.latent_correction = None
        # trainer reads these to set the latent resolution: 128 // 2 = 64
        self.downsample_factor = self.scale
        self.embed_dim = embed_dim
        # encode stays differentiable for parity; only latent_correction is trained.
        self.grad_encode = True

    def _up(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)

    def _down(self, x):
        return F.interpolate(x, scale_factor=1.0 / self.scale, mode='bicubic',
                             align_corners=False, antialias=True)

    def encode(self, x):
        return self.vae.encode(self._up(x))            # 128 -> 256 -> latent 64

    def decode(self, z, force_not_quantize=False):
        if self.latent_correction is not None:
            z = self.latent_correction(z)              # fix x0 latent bias before frozen decode
        y = self.vae.decode(z, force_not_quantize=force_not_quantize)  # latent 64 -> 256
        return self._down(y)                            # 256 -> 128

    def forward(self, x):
        return self.decode(self.encode(x))

    def latent_parameters(self):
        return list(self.latent_correction.parameters()) if self.latent_correction is not None else []

    def trainable_parameters(self):
        return self.latent_parameters()

    def resample_state_dict(self):
        sd = {}
        if self.latent_correction is not None:
            sd['latent'] = self.latent_correction.state_dict()
        return sd

    def load_resample_state_dict(self, sd):
        results = {}
        if self.latent_correction is not None and 'latent' in sd:
            current = self.latent_correction.state_dict()
            compatible = {
                k: v for k, v in sd['latent'].items()
                if k in current and tuple(current[k].shape) == tuple(v.shape)
            }
            results['latent'] = self.latent_correction.load_state_dict(compatible, strict=False)
        return results
