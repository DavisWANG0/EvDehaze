from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import glob

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .basic_ops import (
    linear,
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from .swin_transformer import BasicLayer

try:
    import xformers
    import xformers.ops as xop
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

# def visualize_event_feature_heatmap(feature_map, idx=0, method='mean', save_dir=None, prefix='event'):
#     """
#     可视化 [B, C, H, W] 特征图为 heatmap。
#     Args:
#         feature_map: torch.Tensor of shape [B, C, H, W]
#         idx: 选择第 idx 个样本
#         method: 'mean' 或 int 指定通道
#         save_dir: 若指定，则将图片保存至该目录
#         prefix: 输出图片名前缀
#     """
#     fmap = feature_map[idx]  # [C, H, W]
#     if method == 'mean':
#         heatmap = fmap.mean(dim=0).detach().cpu().numpy()
#         title = 'mean'
#     elif isinstance(method, int):
#         heatmap = fmap[method].detach().cpu().numpy()
#         title = f'channel{method}'
#     else:
#         raise ValueError("method must be 'mean' or int")

#     plt.figure(figsize=(4, 4))
#     plt.imshow(heatmap, cmap='hot')
#     plt.axis('off')
#     plt.title(f'{prefix} heatmap ({title})')

#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         path = os.path.join(save_dir, f"{prefix}_{title}.png")
#         plt.savefig(path, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()
def visualize_event_feature_heatmap(feature_map, idx=0, method='mean', save_dir=None, prefix='event'):
    """
    可视化 [B, C, H, W] 特征图为 heatmap。
    Args:
        feature_map: torch.Tensor of shape [B, C, H, W]
        idx: 选择第 idx 个样本
        method: 'mean' 或 int 指定通道
        save_dir: 若指定，则将图片保存至该目录
        prefix: 输出图片名前缀
    """
    fmap = feature_map[idx]  # [C, H, W]
    if method == 'mean':
        heatmap = fmap.mean(dim=0).detach().cpu().numpy()
        title = 'mean'
    elif isinstance(method, int):
        heatmap = fmap[method].detach().cpu().numpy()
        title = f'channel{method}'
    else:
        raise ValueError("method must be 'mean' or int")

    plt.figure(figsize=(4, 4))
    plt.imshow(heatmap, cmap='hot')
    plt.axis('off')
    plt.title(f'{prefix} heatmap ({title})')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # 查找已有文件数量用于自增编号
        existing = glob.glob(os.path.join(save_dir, f"{prefix}_{title}_*.png"))
        file_id = len(existing) + 1
        filename = f"{prefix}_{title}_{file_id:04d}.png"
        path = os.path.join(save_dir, filename)
        plt.savefig(path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        if XFORMERS_IS_AVAILBLE:
            # qkv: b x length x heads x 3ch
            qkv = qkv.reshape(bs, self.n_heads, ch * 3, length).permute(0, 3, 1, 2).to(memory_format=th.contiguous_format)
            q, k, v = qkv.split(ch, dim=3)  # b x length x heads x ch
            a = xop.memory_efficient_attention(q, k, v, p=0.0)  # b x length x heads x ch
            out = a.permute(0, 2, 3, 1).to(memory_format=th.contiguous_format).reshape(bs, -1, length)
        else:
            # q,k, v: (b*heads) x ch x length
            q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
            scale = 1 / math.sqrt(math.sqrt(ch))
            weight = th.einsum(
                "bct,bcs->bts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards     # (b*heads) x M x M
            weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = th.einsum("bts,bcs->bct", weight, v)  # (b*heads) x ch x length
            out = a.reshape(bs, -1, length)
        return out

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        if XFORMERS_IS_AVAILBLE:
            # qkv: b x length x heads x 3ch
            qkv = qkv.reshape(bs, self.n_heads, ch * 3, length).permute(0, 3, 1, 2).to(memory_format=th.contiguous_format)
            q, k, v = qkv.split(ch, dim=3)  # b x length x heads x ch
            a = xop.memory_efficient_attention(q, k, v, p=0.0)  # b x length x heads x length
            out = a.permute(0, 2, 3, 1).to(memory_format=th.contiguous_format).reshape(bs, -1, length)
        else:
            q, k, v = qkv.chunk(3, dim=1)  # b x heads*ch x length
            scale = 1 / math.sqrt(math.sqrt(ch))
            weight = th.einsum(
                "bct,bcs->bts",
                (q * scale).view(bs * self.n_heads, ch, length),
                (k * scale).view(bs * self.n_heads, ch, length),
            )  # More stable with f16 than dividing afterwards
            weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
            out = a.reshape(bs, -1, length)
        return out

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        cond_lq=True,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        self.num_res_blocks = num_res_blocks

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_lq = cond_lq

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = image_size
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds //= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds *= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, y=None, lq=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param lq: an [N x C x ...] Tensor of low quality iamge.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        if lq is not None:
            assert self.cond_lq
            if lq.shape[2:] != x.shape[2:]:
                lq = F.pixel_unshuffle(lq, 2)
            x = th.cat([x, lq], dim=1)

        h = x.type(self.dtype)
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        return out

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

# Event cross-attention (Q = image features, K/V = event features)
def find_divisible_groups(channels, max_groups=8):
    """找到最大的能被通道数整除的分组数"""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1

class EventCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.event_scale = nn.Parameter(th.full((1,), 0.5))
        
        # projection layers
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

        # 使用 GroupNorm 代替 LayerNorm（更适合不同分辨率）
        # 计算合适的分组数（确保能被通道数整除）
        num_groups = find_divisible_groups(embed_dim)
        self.norm1 = nn.GroupNorm(num_groups, embed_dim)
        self.norm2 = nn.GroupNorm(num_groups, embed_dim)

    def forward(self, x, event_feat, event_gate=None):
        B, C, H, W = x.shape
        
        # 保存原始输入用于残差连接
        identity = x
        
        # 先归一化再展平
        x_norm = self.norm1(x)  # [B, C, H, W]
        x_flat = x_norm.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        event_norm = self.norm2(event_feat)  # [B, C, H, W]
        event_flat = event_norm.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        # 交叉注意力
        attn_out, _ = self.attn(x_flat, event_flat, event_flat)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        
        # 投影
        attn_out = self.proj(attn_out)

        if event_gate is not None:
            if event_gate.shape[-2:] != attn_out.shape[-2:]:
                event_gate = F.interpolate(event_gate, size=attn_out.shape[-2:], mode="bilinear", align_corners=False)
            gate = event_gate.to(device=attn_out.device, dtype=attn_out.dtype)
            gate_max = gate.flatten(1).amax(dim=1).view(B, 1, 1, 1).clamp_min(1e-6)
            gate = (gate / gate_max).clamp(0.0, 1.0).sqrt()
            attn_out = attn_out * gate

        # Start from the previous residual behavior, then learn event strength.
        output = identity + self.event_scale.to(attn_out.dtype) * attn_out

        return output


# Event encoder
class EventEncoderCNN(nn.Module):
    def __init__(self, in_channels=8, out_channels=192, encoder_type='default'):  # ⭐ 8 = 4 bins × 2 polarities (预处理格式)
        super().__init__()
        hidden = min(out_channels // 2, 176)
        self.encoder_type = encoder_type
        if encoder_type == 'pyramid':
            # Multi-scale event encoder: three convolutional layers + pooling. The
            # spatial downsample is recovered by the interpolate-back below so the
            # fused map keeps the level resolution expected by EventCrossAttention.
            h = min(out_channels // 2, 126)
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, h, kernel_size=3, padding=1),
                nn.PReLU(num_parameters=h, init=0.1),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(h, h, kernel_size=3, padding=1),
                nn.PReLU(num_parameters=h, init=0.1),
                nn.Conv2d(h, out_channels, kernel_size=3, padding=1),
                nn.PReLU(num_parameters=out_channels, init=0.1),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        # x should be [B, C, H, W] where C = T * P (e.g., 8 = 4 bins * 2 polarities)
        if len(x.shape) == 5:  # If still [B, T, P, H, W]
            B, T, P, H, W = x.shape
            x = x.view(B, T * P, H, W)  # [B, C, H, W]

        in_hw = x.shape[-2:]
        out = self.encoder(x)
        if self.encoder_type == 'pyramid' and out.shape[-2:] != in_hw:
            out = F.interpolate(out, size=in_hw, mode='bilinear', align_corners=False)
        return out     # [B, out_channels, H, W]

class EvDehaze(nn.Module):
    """
    EvDehaze backbone: a Swin-augmented UNet diffusion denoiser conditioned on the
    hazy image (lq) and an event-voxel branch. Event features are encoded per scale
    (EventEncoderCNN) and fused into the encoder/decoder via gated EventCrossAttention
    (Q = image features, K/V = event features).
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :patch_norm: patch normalization in swin transformer
    :swin_embed_norm: embed_dim in swin transformer
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        swin_depth=2,
        swin_embed_dim=96,
        window_size=8,
        mlp_ratio=2.0,
        patch_norm=False,
        cond_lq=True,
        cond_mask=False,
        lq_size=256,
        event_in_channels=8,
        event_encoder_type='default',
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        if num_heads == -1:
            assert swin_embed_dim % num_head_channels == 0 and num_head_channels > 0
        self.num_res_blocks = num_res_blocks

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_lq = cond_lq
        self.cond_mask = cond_mask
        self.event_in_channels = event_in_channels
        self.event_encoder_type = event_encoder_type

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if cond_lq and lq_size == image_size:
            self.feature_extractor = nn.Identity()
            base_chn = 4 if cond_mask else 3
        else:
            feature_extractor = []
            feature_chn = 4 if cond_mask else 3
            base_chn = 16
            for ii in range(int(math.log(lq_size / image_size) / math.log(2))):
                feature_extractor.append(nn.Conv2d(feature_chn, base_chn, 3, 1, 1))
                feature_extractor.append(nn.SiLU())
                feature_extractor.append(Downsample(base_chn, True, out_channels=base_chn*2))
                base_chn *= 2
                feature_chn = base_chn
            self.feature_extractor = nn.Sequential(*feature_extractor)

        ch = input_ch = int(channel_mult[0] * model_channels)
        in_channels += base_chn
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = image_size
        for level, mult in enumerate(channel_mult):
            for jj in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions and jj==0:
                    layers.append(
                        BasicLayer(
                                in_chans=ch,
                                embed_dim=swin_embed_dim,
                                num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                                window_size=window_size,
                                depth=swin_depth,
                                img_size=ds,
                                patch_size=1,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=True,
                                qk_scale=None,
                                drop=dropout,
                                attn_drop=0.,
                                drop_path=0.,
                                use_checkpoint=False,
                                norm_layer=normalization,
                                patch_norm=patch_norm,
                                 )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds //= 2

        # Event processing modules for multi-scale fusion
        self.event_encoders = nn.ModuleDict()
        self.event_cross_attns = nn.ModuleDict()
        
        # Create event modules for different resolution levels
        current_ch = int(channel_mult[0] * model_channels)
        for level, mult in enumerate(channel_mult):
            ch_level = int(mult * model_channels)
            level_name = f"level_{level}"
            
            # Event encoder for this level
            # ⭐ 修改为 8 通道输入（4 bins × 2 polarities）
            self.event_encoders[level_name] = EventEncoderCNN(
                in_channels=self.event_in_channels,
                out_channels=ch_level,
                encoder_type=self.event_encoder_type,
            )
            
            # Event cross attention for this level
            self.event_cross_attns[level_name] = EventCrossAttention(
                embed_dim=ch_level, 
                num_heads=min(4, ch_level // 64)  # Adaptive num_heads
            )

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            BasicLayer(
                    in_chans=ch,
                    embed_dim=swin_embed_dim,
                    num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                    window_size=window_size,
                    depth=swin_depth,
                    img_size=ds,
                    patch_size=1,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=dropout,
                    attn_drop=0.,
                    drop_path=0.,
                    use_checkpoint=False,
                    norm_layer=normalization,
                    patch_norm=patch_norm,
                     ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions and i==0:
                    layers.append(
                        BasicLayer(
                                in_chans=ch,
                                embed_dim=swin_embed_dim,
                                num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                                window_size=window_size,
                                depth=swin_depth,
                                img_size=ds,
                                patch_size=1,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=True,
                                qk_scale=None,
                                drop=dropout,
                                attn_drop=0.,
                                drop_path=0.,
                                use_checkpoint=False,
                                norm_layer=normalization,
                                patch_norm=patch_norm,
                                 )
                    )
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds *= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, lq=None, mask=None, event=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param lq: an [N x C x ...] Tensor of low quality iamge.
        :return: an [N x C x ...] Tensor of outputs.
        """
        #Add
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
            # if lq is not None:
            #     lq = F.pad(lq, (0, pad_w, 0, pad_h), mode="reflect")
            # if mask is not None:
            #     mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="reflect")
            # if event is not None:
            #     event = F.pad(event, (0, pad_w, 0, pad_h), mode="reflect")

        # For Swin Transformer
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

        # Process event features at different scales - always created so every
        # event module participates (zeros when event is None).
        event_features = {}
        event_gates = {}
        B = h.shape[0]
        
        if event is not None:
            # Check event shape and reshape if necessary
            if len(event.shape) == 5:  # [B, T, M, H, W]
                B, T, M, H, W = event.shape
                event = event.view(B, T * M, H, W)  # [B, 24, H, W]
            elif len(event.shape) == 3:  # [T*M, H, W] - missing batch dimension
                event = event.unsqueeze(0)  # [1, T*M, H, W]
            
            current_event = event
            for level, mult in enumerate(self.channel_mult):
                level_name = f"level_{level}"
                # Downsample event to match current resolution
                if level > 0:
                    scale_factor = 1.0 / (2 ** level)
                    current_size = (int(event.shape[-2] * scale_factor), 
                                  int(event.shape[-1] * scale_factor))
                    current_event = F.interpolate(event, size=current_size, 
                                                mode="bilinear", align_corners=False)
                
                # Encode event features for this level
                event_features[level_name] = self.event_encoders[level_name](current_event)
                event_gates[level_name] = current_event.abs().mean(dim=1, keepdim=True)
        else:
            # 当 event=None 时，创建零张量确保事件模块被使用
            for level, mult in enumerate(self.channel_mult):
                level_name = f"level_{level}"
                
                # 计算当前级别的分辨率
                current_res = self.image_size // (2 ** level)
                
                # ⭐ 创建虚拟事件输入 [B, 8, H, W] (8 = 4 bins × 2 polarities)
                dummy_event = th.zeros(B, self.event_in_channels, current_res, current_res,
                                     device=h.device, dtype=h.dtype)
                
                # 确保事件编码器被调用
                event_features[level_name] = self.event_encoders[level_name](dummy_event)
                event_gates[level_name] = dummy_event[:, :1]
        
        # Encoder with multi-level event fusion
        current_level = 0
        ds = self.image_size
        block_idx = 0
        
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            
            # Add event cross attention at appropriate levels
            if ii > 0:  # Skip first conv layer
                # Find matching level based on channel count and spatial resolution
                current_channels = h.shape[1]
                current_spatial_size = h.shape[-1]  # height or width (they're equal)
                level_name = None
                
                # Define expected spatial sizes for each level
                level_spatial_mapping = {
                    'level_0': 64,   # Original size (64x64 in latent space)
                    'level_1': 32,   # 1/2 downsample  
                    'level_2': 16,   # 1/4 downsample (this was working)
                    'level_3': 8     # 1/8 downsample
                }
                
                # Map based on both channel count and spatial size to handle duplicate channels
                for ln in self.event_cross_attns.keys():
                    cross_attn = self.event_cross_attns[ln]
                    expected_spatial = level_spatial_mapping[ln]
                    
                    # Match on both channel count and spatial resolution
                    if (cross_attn.embed_dim == current_channels and 
                        abs(current_spatial_size - expected_spatial) <= 2):  # Small tolerance for rounding
                        level_name = ln
                        break
                
                # Always apply cross attention if level exists
                if level_name and level_name in event_features:
                    # Resize event features to match current feature map
                    event_feat = event_features[level_name]
                    if event_feat.shape[-2:] != h.shape[-2:]:
                        event_feat = F.interpolate(event_feat, size=h.shape[-2:], 
                                                 mode="bilinear", align_corners=False)
                    
                    # Apply cross attention
                    h = self.event_cross_attns[level_name](h, event_feat, event_gates.get(level_name))
            
            hs.append(h)
        
        h = self.middle_block(h, emb)
        
        # Decoder with event fusion
        for ii, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            
            # Add event cross attention in decoder
            # Find matching level based on channel count and spatial resolution
            current_channels = h.shape[1]
            current_spatial_size = h.shape[-1]  # height or width (they're equal)
            level_name = None
            
            # Define expected spatial sizes for each level
            level_spatial_mapping = {
                'level_0': 64,   # Original size (64x64 in latent space)
                'level_1': 32,   # 1/2 downsample  
                'level_2': 16,   # 1/4 downsample
                'level_3': 8     # 1/8 downsample
            }
            
            # Map based on both channel count and spatial size to handle duplicate channels
            for ln in self.event_cross_attns.keys():
                cross_attn = self.event_cross_attns[ln]
                expected_spatial = level_spatial_mapping[ln]
                
                # Match on both channel count and spatial resolution
                if (cross_attn.embed_dim == current_channels and 
                    abs(current_spatial_size - expected_spatial) <= 2):  # Small tolerance for rounding
                    level_name = ln
                    break
            
            # Always apply cross attention if level exists
            if level_name and level_name in event_features:
                # Resize event features to match current feature map
                event_feat = event_features[level_name]
                if event_feat.shape[-2:] != h.shape[-2:]:
                    event_feat = F.interpolate(event_feat, size=h.shape[-2:], 
                                             mode="bilinear", align_corners=False)
                
                # Apply cross attention
                h = self.event_cross_attns[level_name](h, event_feat, event_gates.get(level_name))
        h = h.type(x.dtype)
        out = self.out(h)
        return out

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.feature_extractor.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.event_encoders.apply(convert_module_to_f16)
        self.event_cross_attns.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        self.event_encoders.apply(convert_module_to_f32)
        self.event_cross_attns.apply(convert_module_to_f32)

class ResBlockConv(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class UNetModelConv(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        cond_lq=True,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_fp16=False,
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.dtype = th.float16 if use_fp16 else th.float32

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.cond_lq = cond_lq

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks[level]):
                layers = [
                    ResBlockConv(
                        ch,
                        time_embed_dim,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConv(
                            ch,
                            time_embed_dim,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)

        self.middle_block = TimestepEmbedSequential(
            ResBlockConv(
                ch,
                time_embed_dim,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            ResBlockConv(
                ch,
                time_embed_dim,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlockConv(
                        ch + ich,
                        time_embed_dim,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlockConv(
                            ch,
                            time_embed_dim,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, lq=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param lq: an [N x C x ...] Tensor of low quality iamge.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if lq is not None:
            assert self.cond_lq
            if lq.shape[2:] != x.shape[2:]:
                lq = F.pixel_unshuffle(lq, 2)
            x = th.cat([x, lq], dim=1)

        h = x.type(self.dtype)
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        return out
