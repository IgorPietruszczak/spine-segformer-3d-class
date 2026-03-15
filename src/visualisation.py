"""
visualisation.py
----------------
TensorBoard visualisation helpers for vertebra segmentation training.

Three things are provided:

1. seg_overlay_grid()
   Renders axial / coronal / sagittal slices with GT and prediction
   segmentation overlaid on the CT in separate columns.
   → logged with writer.add_image()

2. ConfusionAccumulator  +  confusion_matrix_figure()
   Accumulates per-voxel (gt, pred) pairs across the validation set,
   then renders a normalised recall heatmap via matplotlib.
   Rows = GT class, Cols = predicted class.
   Dark diagonal = good. Off-diagonal = where the model confuses vertebrae.
   → logged with writer.add_figure()

3. loss_curve_image() — not used externally; per-batch logging is in train.py.

All functions are CPU-only and safe to call in the val loop.
They import matplotlib lazily so startup time is unaffected.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from config import VERTEBRA_CLASSES, NUM_SEG_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
#  Consistent class colour map
# ─────────────────────────────────────────────────────────────────────────────

def _build_colormap(n: int) -> np.ndarray:
    """
    Returns (n, 3) uint8 array.
    Background = black.
    T1-T12     = blue → cyan gradient.
    L1-L5      = orange → red gradient.
    """
    cmap = np.zeros((n, 3), dtype=np.uint8)
    # Thoracic: deep blue (T1) → cyan (T12)
    for i, cls_id in enumerate(range(1, 13)):       # 12 thoracic
        t = i / 11.0
        cmap[cls_id] = [
            int(20  + t * 20),   # R  20→40
            int(80  + t * 160),  # G  80→240
            int(220 - t * 80),   # B  220→140
        ]
    # Lumbar: orange (L1) → red (L5)
    for i, cls_id in enumerate(range(13, 18)):      # 5 lumbar
        t = i / 4.0
        cmap[cls_id] = [
            int(230 - t * 30),   # R  230→200
            int(130 - t * 110),  # G  130→20
            int(20),             # B  constant dark
        ]
    return cmap


CLASS_COLORS = _build_colormap(NUM_SEG_CLASSES)   # (18, 3)  uint8


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """
    Convert integer mask (H, W) → RGB image (H, W, 3) uint8
    using the class colormap.
    """
    out = CLASS_COLORS[np.clip(mask, 0, NUM_SEG_CLASSES - 1)]
    return out.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  Segmentation overlay grid
# ─────────────────────────────────────────────────────────────────────────────

def _best_slice(mask: np.ndarray, axis: int) -> int:
    """Return the index along `axis` with the most foreground voxels."""
    sums = (mask > 0).sum(
        axis=tuple(i for i in range(mask.ndim) if i != axis)
    )
    idx = int(np.argmax(sums))
    # Fall back to mid-slice if all zeros
    return idx if sums[idx] > 0 else mask.shape[axis] // 2


def _overlay_slice(
    ct_slice:   np.ndarray,   # (H, W) float [0, 1]
    seg_slice:  np.ndarray,   # (H, W) int
    alpha:      float = 0.45,
) -> np.ndarray:
    """
    Blend coloured segmentation over greyscale CT.
    Returns (H, W, 3) uint8.
    """
    # Greyscale CT → RGB
    ct_norm  = np.clip(ct_slice, 0.0, 1.0)
    ct_rgb   = (np.stack([ct_norm] * 3, axis=-1) * 255).astype(np.uint8)

    seg_rgb  = mask_to_rgb(seg_slice)
    fg       = seg_slice > 0

    blended        = ct_rgb.copy()
    blended[fg]    = (
        (1 - alpha) * ct_rgb[fg].astype(np.float32)
        + alpha     * seg_rgb[fg].astype(np.float32)
    ).astype(np.uint8)

    return blended



def seg_overlay_grid(
    ct:       torch.Tensor,    # (1, 1, D, H, W)  float32  [0, 1]
    pred:     torch.Tensor,    # (1, C, D, H, W)  logits  or  (1, 1, D, H, W) int
    label:    torch.Tensor,    # (1, 1, D, H, W)  int
    alpha:    float = 0.45,
) -> np.ndarray:
    """
    Build a (3, grid_H, grid_W) uint8 array for writer.add_image().

    Layout (3 rows, 2 columns):
        row 0: axial     — [GT | PRED]
        row 1: coronal   — [GT | PRED]
        row 2: sagittal  — [GT | PRED]

    Works with:
      - pred as raw logits (B, C, D, H, W)  → argmax taken internally
      - pred as integer mask (B, 1, D, H, W) → used directly
    """
    # Convert to numpy cpu
    ct_np = ct[0, 0].cpu().float().numpy()     # (D, H, W)

    if pred.shape[1] > 1:
        pred_np = pred[0].argmax(dim=0).cpu().numpy().astype(np.int16)
    else:
        pred_np = pred[0, 0].cpu().numpy().astype(np.int16)

    lab_np = label[0, 0].cpu().numpy().astype(np.int16)  # (D, H, W)

    views = [
        ("axial",    0),   # axis=0 → slice along D
        ("coronal",  1),   # axis=1 → slice along H
        ("sagittal", 2),   # axis=2 → slice along W
    ]

    rows = []
    for view_name, axis in views:
        idx_gt   = _best_slice(lab_np,  axis)
        idx_pred = _best_slice(pred_np, axis)
        # Use GT slice index for both so the same anatomy is shown
        idx = idx_gt

        ct_sl   = np.take(ct_np,   idx, axis=axis)
        gt_sl   = np.take(lab_np,  idx, axis=axis)
        pred_sl = np.take(pred_np, idx, axis=axis)

        gt_img   = _overlay_slice(ct_sl, gt_sl,   alpha)
        pred_img = _overlay_slice(ct_sl, pred_sl, alpha)

        # Separator line between GT and PRED
        sep = np.full((gt_img.shape[0], 2, 3), 200, dtype=np.uint8)

        # Small white header bar with label
        H, W = gt_img.shape[:2]
        header_h = max(12, H // 20)
        header = np.zeros((header_h, W * 2 + 2, 3), dtype=np.uint8)

        # Assemble row: [GT | sep | PRED]
        row = np.concatenate([gt_img, sep, pred_img], axis=1)   # (H, W*2+2, 3)
        rows.append(row)

    # Pad all rows to the same width before stacking
    max_w = max(r.shape[1] for r in rows)
    max_h = max(r.shape[0] for r in rows)
    padded = []
    for r in rows:
        h, w = r.shape[:2]
        pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        pad[:h, :w] = r
        padded.append(pad)

    # Separator between rows
    row_sep = np.full((2, max_w, 3), 128, dtype=np.uint8)
    grid = np.concatenate(
        [padded[0], row_sep, padded[1], row_sep, padded[2]], axis=0
    )   # (total_H, max_W, 3)

    # TensorBoard expects (C, H, W)
    return grid.transpose(2, 0, 1).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

class ConfusionAccumulator:
    """
    Accumulates (gt, pred) voxel pairs across multiple validation volumes.
    Only foreground voxels (gt > 0 OR pred > 0) are counted to keep the
    matrix interpretable — the background would otherwise dominate.

    Usage:
        acc = ConfusionAccumulator()
        for each val volume:
            acc.update(pred_mask, gt_mask)  # (D,H,W) int tensors
        fig = acc.plot()
        writer.add_figure("val/confusion", fig, epoch)
        acc.reset()
    """

    def __init__(self, num_classes: int = NUM_SEG_CLASSES):
        self.n = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.matrix[:] = 0

    def update(
        self,
        pred: torch.Tensor,   # (D, H, W)  int  (already argmaxed)
        gt:   torch.Tensor,   # (D, H, W)  int
    ) -> None:
        p = pred.cpu().numpy().ravel().astype(np.int32)
        g = gt.cpu().numpy().ravel().astype(np.int32)

        # Only keep foreground voxels
        fg = (g > 0) | (p > 0)
        p, g = p[fg], g[fg]

        p = np.clip(p, 0, self.n - 1)
        g = np.clip(g, 0, self.n - 1)

        # Accumulate
        np.add.at(self.matrix, (g, p), 1)

    def plot(self) -> "matplotlib.figure.Figure":
        """
        Returns a matplotlib Figure of the recall-normalised confusion matrix.
        Rows = GT classes, Cols = predicted classes.
        Only classes that appear in the data are shown.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        mat = self.matrix.copy().astype(np.float64)

        # Only show classes that actually appear
        present = np.where(mat.sum(axis=1) > 0)[0]
        if len(present) == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        mat = mat[np.ix_(present, present)]

        # Normalise row-wise → recall matrix
        row_sums = mat.sum(axis=1, keepdims=True)
        norm_mat = np.where(row_sums > 0, mat / row_sums, 0.0)

        labels = [VERTEBRA_CLASSES.get(int(i), str(i)) for i in present]
        n      = len(labels)

        fig_size = max(6, n * 0.55)
        fig, ax  = plt.subplots(figsize=(fig_size, fig_size))

        im = ax.imshow(norm_mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")

        # Annotate cells
        for r in range(n):
            for c in range(n):
                v = norm_mat[r, c]
                if v > 0.01:
                    color = "white" if v > 0.6 else "black"
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=max(5, 9 - n // 5), color=color)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("GT (recall normalised)", fontsize=9)
        ax.set_title("Vertebra Confusion Matrix\n(row = GT class, diagonal = recall)", fontsize=9)

        # Colour-code tick labels by region
        for tick, cls_id in zip(ax.get_xticklabels(), present):
            tick.set_color(_tick_color(int(cls_id)))
        for tick, cls_id in zip(ax.get_yticklabels(), present):
            tick.set_color(_tick_color(int(cls_id)))

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall")
        fig.tight_layout()
        return fig


def _tick_color(cls_id: int) -> str:
    if 1 <= cls_id <= 12:
        return "#1a6faf"    # thoracic → blue
    if 13 <= cls_id <= 17:
        return "#c04a00"    # lumbar → orange-red
    return "black"
