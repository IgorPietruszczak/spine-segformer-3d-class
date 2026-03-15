"""
model_segformer3d.py
--------------------
3D SegFormer (MiT-B1 encoder + all-MLP decoder) for spine CT segmentation.

Input  : (B, 1, D, H, W)  — single-channel CT volume
Output : (B, num_seg_classes, D, H, W)  — raw logits

Axis convention throughout this file: (B, C, D, H, W)
MONAI loads NIfTI as (C, H, W, D); the caller must permute before passing here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt


# ─────────────────────────────────────────────────────────────────────────────
#  Primitives
# ─────────────────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Stochastic depth (drop whole residual path during training)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.p = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * noise / keep


class OverlapPatchEmbed3D(nn.Module):
    """
    Overlapping patch embedding via strided Conv3D.
    Stride < kernel so adjacent patches share context — important for
    segmentation where boundary detail matters.
    """
    def __init__(self, in_ch: int, embed_dim: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, embed_dim,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int,int]]:
        x = self.proj(x)                          # (B, C, D, H, W)
        B, C, D, H, W = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))   # (B, N, C)
        return x.transpose(1, 2).reshape(B, C, D, H, W), (D, H, W)


class EfficientSelfAttn3D(nn.Module):
    """
    Efficient self-attention with spatial reduction (SR).
    SR ratio R reduces key/value sequence length by R³, saving memory.
    """
    def __init__(self, dim: int, num_heads: int,
                 sr_ratio: int = 1, drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.nh       = num_heads
        self.hd       = dim // num_heads
        self.scale    = self.hd ** -0.5
        self.q        = nn.Linear(dim, dim)
        self.kv       = nn.Linear(dim, dim * 2)
        self.attn_drop = nn.Dropout(drop)
        self.proj     = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr   = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = self.norm = None

    def forward(self, x: torch.Tensor,
                spatial: Tuple[int,int,int]) -> torch.Tensor:
        B, N, C = x.shape
        D, H, W = spatial
        q = self.q(x).reshape(B, N, self.nh, self.hd).permute(0, 2, 1, 3)
        if self.sr is not None:
            xr = self.sr(x.transpose(1,2).reshape(B, C, D, H, W))
            xr = self.norm(xr.flatten(2).transpose(1, 2))
        else:
            xr = x
        kv = self.kv(xr).reshape(B, -1, 2, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = self.attn_drop((q @ k.transpose(-2, -1)) * self.scale)
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class MixFFN3D(nn.Module):
    """
    MixFFN: Linear → depth-wise Conv3D (captures local context) → GELU → Linear.
    The DWConv is the key difference from a vanilla FFN.
    """
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1   = nn.Linear(dim, hidden_dim)
        self.dw    = nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.act   = nn.GELU()
        self.fc2   = nn.Linear(hidden_dim, dim)
        self.drop  = nn.Dropout(drop)

    def forward(self, x: torch.Tensor,
                spatial: Tuple[int,int,int]) -> torch.Tensor:
        B, N, C = x.shape
        D, H, W = spatial
        x = self.drop(self.fc1(x))
        x = self.dw(x.transpose(1,2).reshape(B, -1, D, H, W))
        x = self.drop(self.fc2(self.drop(self.act(x.flatten(2).transpose(1, 2)))))
        return x


class TransformerBlock3D(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float,
                 sr_ratio: int, drop: float, drop_path: float):
        super().__init__()
        self.norm1     = nn.LayerNorm(dim)
        self.attn      = EfficientSelfAttn3D(dim, num_heads, sr_ratio, drop)
        self.drop_path = DropPath(drop_path)
        self.norm2     = nn.LayerNorm(dim)
        self.ffn       = MixFFN3D(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor,
                spatial: Tuple[int,int,int]) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), spatial))
        x = x + self.drop_path(self.ffn(self.norm2(x), spatial))
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  MiT-B1 Encoder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MiT3DConfig:
    in_chans:       int   = 1
    embed_dims:     tuple = (64, 128, 320, 512)
    depths:         tuple = (2, 2, 2, 2)
    num_heads:      tuple = (1, 2, 5, 8)
    mlp_ratio:      float = 4.0
    sr_ratios:      tuple = (8, 4, 2, 1)
    drop:           float = 0.0
    drop_path:      float = 0.1
    use_checkpoint: bool  = True


class MiT3DEncoder(nn.Module):
    def __init__(self, cfg: MiT3DConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.embed_dims

        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed3D(cfg.in_chans, D[0], kernel_size=7, stride=4, padding=3),
            OverlapPatchEmbed3D(D[0], D[1], kernel_size=3, stride=2, padding=1),
            OverlapPatchEmbed3D(D[1], D[2], kernel_size=3, stride=2, padding=1),
            OverlapPatchEmbed3D(D[2], D[3], kernel_size=3, stride=2, padding=1),
        ])

        # Stochastic depth decay rule — linearly increase drop_path rate
        dpr = torch.linspace(0, cfg.drop_path, sum(cfg.depths)).tolist()
        cur = 0
        self.stages = nn.ModuleList()
        self.norms  = nn.ModuleList()
        for i in range(4):
            stage = nn.ModuleList([
                TransformerBlock3D(
                    D[i], cfg.num_heads[i], cfg.mlp_ratio,
                    cfg.sr_ratios[i], cfg.drop, dpr[cur + j]
                )
                for j in range(cfg.depths[i])
            ])
            self.stages.append(stage)
            self.norms.append(nn.LayerNorm(D[i]))
            cur += cfg.depths[i]

    def _run_stage(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        x, (D, H, W) = self.patch_embeds[idx](x)
        B, C, D, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)             # (B, N, C)
        for blk in self.stages[idx]:
            if self.cfg.use_checkpoint and self.training:
                # Use non-reentrant checkpoint to avoid issues with newer PyTorch
                seq = grad_ckpt(blk, seq, (D, H, W), use_reentrant=False)
            else:
                seq = blk(seq, (D, H, W))
        seq = self.norms[idx](seq)
        return seq.transpose(1, 2).reshape(B, C, D, H, W)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for i in range(4):
            x = self._run_stage(x, i)
            feats.append(x)
        return feats   # [f0 f1 f2 f3], resolutions: /4, /8, /16, /32


# ─────────────────────────────────────────────────────────────────────────────
#  All-MLP Decoder
# ─────────────────────────────────────────────────────────────────────────────

class SegFormerDecoder3D(nn.Module):
    """
    Lightweight all-MLP decoder:
    1. Project each encoder stage to a common embed_dim with 1×1×1 conv
    2. Upsample all to the coarsest-stage resolution (¼ of input)
    3. Concatenate and fuse with two 3×3×3 convs
    4. Final 1×1×1 conv → class logits
    """
    def __init__(self, in_dims: Tuple[int,...], embed_dim: int = 256,
                 num_classes: int = 18, drop: float = 0.0):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv3d(d, embed_dim, kernel_size=1) for d in in_dims
        ])
        self.fuse = nn.Sequential(
            nn.Conv3d(embed_dim * 4, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
            nn.Dropout3d(drop),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
        )
        self.head = nn.Conv3d(embed_dim, num_classes, kernel_size=1)

    def forward(self, feats: List[torch.Tensor],
                target_size: Tuple[int,int,int]) -> torch.Tensor:
        # Reference resolution = coarsest (feats[0] = /4)
        ref = feats[0].shape[-3:]
        projected = []
        for i, f in enumerate(feats):
            p = self.proj[i](f)
            if p.shape[-3:] != ref:
                p = F.interpolate(p, size=ref, mode="trilinear", align_corners=False)
            projected.append(p)
        x = self.fuse(torch.cat(projected, dim=1))
        x = self.head(x)
        return F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Top-level model
# ─────────────────────────────────────────────────────────────────────────────

class SegFormer3D(nn.Module):
    """
    3D SegFormer for multi-class spine CT segmentation.

    Args:
        in_chans        : input channels (1 for CT)
        num_seg_classes : total classes including background
                          (default 18 = background + T1–T12 + L1–L5)
        drop            : dropout rate in FFN / decoder
        drop_path       : stochastic depth rate
        use_checkpoint  : gradient checkpointing (saves VRAM during training)

    forward(x) → (B, num_seg_classes, D, H, W) raw logits
    Compatible with MONAI sliding_window_inference out of the box.
    """
    def __init__(
        self,
        in_chans:        int   = 1,
        num_seg_classes: int   = 18,
        drop:            float = 0.0,
        drop_path:       float = 0.1,
        use_checkpoint:  bool  = True,
    ):
        super().__init__()
        cfg = MiT3DConfig(
            in_chans=in_chans,
            drop=drop,
            drop_path=drop_path,
            use_checkpoint=use_checkpoint,
        )
        self.encoder = MiT3DEncoder(cfg)
        self.decoder = SegFormerDecoder3D(
            in_dims    = cfg.embed_dims,
            embed_dim  = 256,
            num_classes = num_seg_classes,
            drop       = drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.decoder(feats, x.shape[-3:])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
