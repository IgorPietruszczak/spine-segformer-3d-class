"""
losses.py
---------
Loss function for multi-class vertebra segmentation.

DiceCELoss combines:
  - Dice loss:           catches class imbalance (vertebrae are small vs background)
  - Cross-entropy loss:  stable gradients early in training

include_background=False:
  Background (class 0) is excluded from Dice to prevent it from
  dominating. Background is still included in the CE part, which
  provides gradient signal for the full volume.
"""
from monai.losses import DiceCELoss


def build_loss(num_seg_classes: int) -> DiceCELoss:
    """
    Returns the segmentation loss function.

    Args:
        num_seg_classes: total number of classes including background (e.g. 18)

    The loss expects:
        logits  : (B, C, D, H, W)  float — raw model output
        targets : (B, 1, D, H, W)  long  — integer class labels

    to_onehot_y=True handles the integer→one-hot conversion internally.
    softmax=True applies softmax to logits before computing Dice.
    """
    return DiceCELoss(
        softmax=True,
        to_onehot_y=True,
        include_background=False,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )
