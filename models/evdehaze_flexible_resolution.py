"""
Flexible-resolution variant of EvDehaze for native-resolution inference.

EvDehaze fuses event features at fixed latent sizes (64/32/16/8), which is exact
for the 128px training/evaluation setting (latent 64). For qualitative results at
the original image resolution the latent is larger, so this subclass makes the
event cross-attention level matching relative to the input latent size: each
level is matched at ``base/2**level`` instead of the hard-coded {64,32,16,8}.

At latent 64 it reproduces EvDehaze exactly; at any other resolution it keeps the
event guidance active. Only ``forward`` differs and no parameters are added, so a
trained EvDehaze checkpoint loads into this class unchanged.
"""
import torch as th
import torch.nn.functional as F

from .basic_ops import timestep_embedding
from .evdehaze import EvDehaze


class EvDehazeFlexibleResolution(EvDehaze):
    def _match_level(self, current_channels, current_spatial, base_spatial):
        # among levels whose channel count matches, pick the one whose expected
        # spatial size (base/2**level) is closest to the current feature size.
        candidates = [ln for ln, ca in self.event_cross_attns.items()
                      if ca.embed_dim == current_channels]
        if not candidates:
            return None

        def expected(ln):
            level = int(ln.split('_')[1])
            return max(1, round(base_spatial / (2 ** level)))

        return min(candidates, key=lambda ln: abs(expected(ln) - current_spatial))

    def forward(self, x, timesteps, lq=None, mask=None, event=None):
        window_size = 8
        B, C, H, W = x.shape
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            if lq is not None and (lq.shape[-2:] != x.shape[-2:]):
                lq = F.interpolate(lq, size=x.shape[-2:], mode="bilinear", align_corners=False)
            if mask is not None and (mask.shape[-2:] != x.shape[-2:]):
                mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")
            if event is not None and (event.shape[-2:] != x.shape[-2:]):
                event = F.interpolate(event, size=x.shape[-2:], mode="bilinear", align_corners=False)

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)

        if lq is not None:
            assert self.cond_lq
            if mask is not None:
                assert self.cond_mask
                lq = th.cat([lq, mask], dim=1)
            lq = self.feature_extractor(lq.type(self.dtype))
            x = th.cat([x, lq], dim=1)

        h = x.type(self.dtype)
        base_spatial = h.shape[-1]   # latent spatial size at encoder entry

        # Per-level event features (always created so every event module runs;
        # zeros when event is None).
        event_features = {}
        event_gates = {}
        B = h.shape[0]
        if event is not None:
            if len(event.shape) == 5:          # [B, T, M, H, W]
                B, T, M, eh, ew = event.shape
                event = event.view(B, T * M, eh, ew)
            elif len(event.shape) == 3:        # [T*M, H, W]
                event = event.unsqueeze(0)
            for level, _ in enumerate(self.channel_mult):
                level_name = f"level_{level}"
                if level > 0:
                    scale_factor = 1.0 / (2 ** level)
                    current_size = (int(event.shape[-2] * scale_factor),
                                    int(event.shape[-1] * scale_factor))
                    current_event = F.interpolate(event, size=current_size,
                                                  mode="bilinear", align_corners=False)
                else:
                    current_event = event
                event_features[level_name] = self.event_encoders[level_name](current_event)
                event_gates[level_name] = current_event.abs().mean(dim=1, keepdim=True)
        else:
            for level, _ in enumerate(self.channel_mult):
                level_name = f"level_{level}"
                current_res = max(1, base_spatial // (2 ** level))
                dummy_event = th.zeros(B, self.event_in_channels, current_res, current_res,
                                       device=h.device, dtype=h.dtype)
                event_features[level_name] = self.event_encoders[level_name](dummy_event)
                event_gates[level_name] = dummy_event[:, :1]

        # Encoder with multi-level event fusion (relative level matching)
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            if ii > 0:
                level_name = self._match_level(h.shape[1], h.shape[-1], base_spatial)
                if level_name and level_name in event_features:
                    event_feat = event_features[level_name]
                    if event_feat.shape[-2:] != h.shape[-2:]:
                        event_feat = F.interpolate(event_feat, size=h.shape[-2:],
                                                   mode="bilinear", align_corners=False)
                    h = self.event_cross_attns[level_name](h, event_feat, event_gates.get(level_name))
            hs.append(h)

        h = self.middle_block(h, emb)

        # Decoder with event fusion
        for ii, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            level_name = self._match_level(h.shape[1], h.shape[-1], base_spatial)
            if level_name and level_name in event_features:
                event_feat = event_features[level_name]
                if event_feat.shape[-2:] != h.shape[-2:]:
                    event_feat = F.interpolate(event_feat, size=h.shape[-2:],
                                               mode="bilinear", align_corners=False)
                h = self.event_cross_attns[level_name](h, event_feat, event_gates.get(level_name))
        h = h.type(x.dtype)
        return self.out(h)
