"""
metrics.py
----------
Metric utilities for multi-class vertebra segmentation.

Provides:
  - build_metrics()        → (DiceMetric, HausdorffDistanceMetric)
  - MetricTracker          → accumulates per-class Dice across batches,
                             prints a summary table, logs to TensorBoard
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Optional, TYPE_CHECKING
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

from config import VERTEBRA_CLASSES, NUM_SEG_CLASSES


def build_metrics() -> tuple[DiceMetric, HausdorffDistanceMetric]:
    """
    Returns metric objects configured for multi-class segmentation.
    Both metrics expect one-hot inputs: (B, C, D, H, W).

    reduction="mean_batch"  → per-class mean over the batch (not collapsed).
    include_background=False → background class excluded from mean Dice.
    """
    dice = DiceMetric(
        include_background=False,
        reduction="mean_batch",      # returns per-class value
        get_not_nans=True,
    )
    hd95 = HausdorffDistanceMetric(
        include_background=False,
        reduction="mean",
        percentile=95,
    )
    return dice, hd95


# ── One-hot post-processing ────────────────────────────────────────────────────
# Instantiate once and reuse to avoid repeated object creation
_to_onehot = AsDiscrete(argmax=True, to_onehot=NUM_SEG_CLASSES)
_lab_onehot = AsDiscrete(to_onehot=NUM_SEG_CLASSES)


def pred_to_onehot(logits: torch.Tensor) -> torch.Tensor:
    """(B, C, D, H, W) logits → (B, C, D, H, W) one-hot"""
    return _to_onehot(logits)


def label_to_onehot(label: torch.Tensor) -> torch.Tensor:
    """(B, 1, D, H, W) int label → (B, C, D, H, W) one-hot"""
    return _lab_onehot(label)


# ─────────────────────────────────────────────────────────────────────────────
#  MetricTracker
# ─────────────────────────────────────────────────────────────────────────────

class MetricTracker:
    """
    Accumulates Dice and HD95 across validation batches, then summarises.

    Usage:
        tracker = MetricTracker()
        for batch in val_loader:
            pred_oh = pred_to_onehot(logits)
            lab_oh  = label_to_onehot(labels)
            tracker.update(pred_oh, lab_oh)
        result = tracker.compute()
        tracker.log(writer, epoch)
        tracker.print_table()
        tracker.reset()
    """

    def __init__(self):
        self.dice_metric, self.hd95_metric = build_metrics()

    def reset(self) -> None:
        self.dice_metric.reset()
        self.hd95_metric.reset()

    def update(self, pred_onehot: torch.Tensor,
               label_onehot: torch.Tensor) -> None:
        """pred and label must already be one-hot (B, C, D, H, W)."""
        self.dice_metric(pred_onehot, label_onehot)
        self.hd95_metric(pred_onehot, label_onehot)

    def compute(self) -> dict:
        """
        Returns:
            mean_dice     : float  — mean over all present vertebra classes
            mean_hd95     : float
            per_class_dice: dict {class_name: dice_value}
        """
        # per_class shape: (num_classes - 1,)  [background excluded]
        per_class_raw, not_nan = self.dice_metric.aggregate()
        per_class = per_class_raw.cpu().numpy()
        not_nan   = not_nan.cpu().numpy().astype(bool)

        # Map back to class names (index 0 here = class 1 = T1)
        per_class_named: dict[str, float] = {}
        for i, (val, valid) in enumerate(zip(per_class, not_nan)):
            cls_id   = i + 1                      # offset because background excluded
            cls_name = VERTEBRA_CLASSES.get(cls_id, f"cls_{cls_id}")
            per_class_named[cls_name] = float(val) if valid else float("nan")

        valid_vals = [v for v in per_class_named.values() if not np.isnan(v)]
        mean_dice  = float(np.mean(valid_vals)) if valid_vals else 0.0

        try:
            mean_hd95 = float(self.hd95_metric.aggregate().item())
        except Exception:
            mean_hd95 = float("nan")

        return {
            "mean_dice":      mean_dice,
            "mean_hd95":      mean_hd95,
            "per_class_dice": per_class_named,
        }

    def log(self, writer: "SummaryWriter", epoch: int,
            prefix: str = "val") -> None:
        """Write all metrics to TensorBoard."""
        if writer is None:
            return
        result = self.compute()
        writer.add_scalar(f"{prefix}/mean_dice", result["mean_dice"], epoch)
        writer.add_scalar(f"{prefix}/mean_hd95", result["mean_hd95"], epoch)
        for name, val in result["per_class_dice"].items():
            if not np.isnan(val):
                writer.add_scalar(f"{prefix}/dice_{name}", val, epoch)

    def print_table(self, epoch: Optional[int] = None) -> None:
        """Print a neat per-class Dice table to stdout."""
        result = self.compute()
        header = f"  Epoch {epoch} — " if epoch is not None else "  "
        header += f"mean Dice: {result['mean_dice']:.4f}  |  HD95: {result['mean_hd95']:.2f} mm"
        print(header)
        print("  " + "─" * 52)

        # Group by region for readability
        regions = {
            "Thoracic": [f"T{i}" for i in range(1, 13)],
            "Lumbar":   [f"L{i}" for i in range(1, 6)],
        }
        for region, names in regions.items():
            vals = [result["per_class_dice"].get(n, float("nan")) for n in names]
            valid = [(n, v) for n, v in zip(names, vals) if not np.isnan(v)]
            if not valid:
                continue
            print(f"  {region}:")
            row = "  " + "  ".join(
                f"{n}:{v:.3f}" for n, v in valid
            )
            print(row)
        print()
