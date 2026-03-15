"""
transforms.py
-------------
MONAI-based transform pipelines for multi-class vertebra segmentation.

Label convention after remap_labels.py:
  0=background, 1=T1 ... 12=T12, 13=L1 ... 17=L5

MONAI axis order: (C, H, W, D)
Model axis order: (B, C, D, H, W)  ← permuted in train.py via to_model()
"""
from __future__ import annotations

import numpy as np
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, EnsureTyped,
    SpatialPadd, RandGaussianNoised, RandAdjustContrastd, RandShiftIntensityd,
)
from monai.transforms.transform import MapTransform, RandomizableTransform


# ─────────────────────────────────────────────────────────────────────────────
#  Custom transforms
# ─────────────────────────────────────────────────────────────────────────────

class CastLabeld(MapTransform):
    """
    Cast segmentation label to int16.
    Required so MONAI does not accidentally treat integer labels as float
    and interpolate them with bilinear mode.
    """
    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for k in self.keys:
            x = d[k]
            if isinstance(x, np.ndarray):
                d[k] = x.astype(np.int16)
            elif isinstance(x, torch.Tensor):
                d[k] = x.to(torch.int16)
        return d


class RandomCTWindowd(RandomizableTransform, MapTransform):
    """
    Clip and normalise CT HU values to [lo, hi] → [0, 1].
    With jitter > 0 the window shifts randomly per sample during training,
    which acts as intensity augmentation and improves robustness to
    scanner differences.
    """
    def __init__(self, keys, base_lo: float = -1000.0, base_hi: float = 2000.0,
                 jitter: float = 200.0, prob: float = 1.0):
        RandomizableTransform.__init__(self, prob)
        MapTransform.__init__(self, keys)
        self.base_lo = float(base_lo)
        self.base_hi = float(base_hi)
        self.jitter  = float(jitter)
        self._lo = self.base_lo
        self._hi = self.base_hi

    def randomize(self) -> None:
        self._lo = self.base_lo + self.R.uniform(-self.jitter, self.jitter)
        self._hi = self.base_hi + self.R.uniform(-self.jitter, self.jitter)
        self._hi = max(self._hi, self._lo + 1.0)

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        self.randomize()
        for k in self.keys:
            x = d[k]
            if isinstance(x, np.ndarray):
                x = np.clip(x, self._lo, self._hi)
                d[k] = ((x - self._lo) / (self._hi - self._lo)).astype(np.float32)
            else:
                x = x.clamp(self._lo, self._hi)
                d[k] = ((x - self._lo) / (self._hi - self._lo)).float()
        return d


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline builders
# ─────────────────────────────────────────────────────────────────────────────

def build_train_transforms(
    isotropic_mm:    float,
    patch_size:      tuple,
    pos:             int,
    neg:             int,
    num_samples:     int,
    base_lo:         float,
    base_hi:         float,
    jitter:          float,
    reorient:        str = "RAS",
) -> Compose:
    """
    Training pipeline:
      Load → Orient → Resample → Window → Pad → RandomCrop → Augment
    Augmentations are conservative and anatomy-preserving:
      - Flip LR (most common)
      - Rotate 90° in axial plane (less common — avoid for small datasets)
      - Gaussian noise (simulates scanner noise)
      - Contrast/intensity jitter (handles inter-scanner variability)
    """
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes=reorient),
        Spacingd(
            keys=["image", "label"],
            pixdim=(isotropic_mm, isotropic_mm, isotropic_mm),
            mode=("bilinear", "nearest"),
        ),
        CastLabeld(keys=["label"]),
        RandomCTWindowd(keys=["image"], base_lo=base_lo, base_hi=base_hi,
                        jitter=jitter),
        EnsureTyped(keys=["image", "label"], track_meta=True),
        # Pad so patch_size always fits — important for small volumes
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        ),
        # ── Spatial augmentation ────────────────────────────────────────────
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),  # L-R
        # ── Intensity augmentation ──────────────────────────────────────────
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.03),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),
        RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.3),
    ])


def build_val_transforms(
    isotropic_mm: float,
    base_lo:      float,
    base_hi:      float,
    reorient:     str = "RAS",
) -> Compose:
    """
    Validation pipeline: identical preprocessing, no augmentation.
    Full volume is kept — sliding window inference is used at eval time.
    """
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes=reorient),
        Spacingd(
            keys=["image", "label"],
            pixdim=(isotropic_mm, isotropic_mm, isotropic_mm),
            mode=("bilinear", "nearest"),
        ),
        CastLabeld(keys=["label"]),
        RandomCTWindowd(keys=["image"], base_lo=base_lo, base_hi=base_hi,
                        jitter=0.0),
        EnsureTyped(keys=["image", "label"], track_meta=True),
    ])


def build_infer_transforms(
    isotropic_mm: float,
    base_lo:      float,
    base_hi:      float,
    reorient:     str = "RAS",
) -> Compose:
    """Inference pipeline — no label keys."""
    return Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes=reorient),
        Spacingd(
            keys=["image"],
            pixdim=(isotropic_mm, isotropic_mm, isotropic_mm),
            mode=("bilinear",),
        ),
        RandomCTWindowd(keys=["image"], base_lo=base_lo, base_hi=base_hi,
                        jitter=0.0),
        EnsureTyped(keys=["image"], track_meta=True),
    ])
