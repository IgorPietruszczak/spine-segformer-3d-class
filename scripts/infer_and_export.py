"""
infer_and_export.py
-------------------
Step 2 of the iterative labeling workflow:

  1. Run this on a folder of NEW (unlabeled) CT scans
  2. Open the output *_pred.nii files in 3D Slicer as segmentations
  3. Fix mistakes, save as *_segmentation.nii + *label.json
  4. Run remap_labels.py on the corrected files
  5. Add them to the dataset and retrain

Usage:
  python infer_and_export.py \
      --input_dir  "D:/Skany/Nowe skany" \
      --output_dir "D:/Skany/Nowe skany/predictions" \
      --ckpt       "runs/segformer3d_spine/20250314-120000/best.pt"

  # Use ensemble of all LOOCV checkpoints for better predictions:
  python infer_and_export.py \
      --input_dir "..." --output_dir "..." \
      --ckpt_dir "runs/segformer3d_spine/20250314-120000"

Output per case:
  <output_dir>/<case_name>/
    proc_ct.nii          — preprocessed CT (what the model saw)
    pred_vertebrae.nii   — integer mask: 1=T1 ... 17=L5  (open in Slicer as seg)
    pred_label.json      — class names (for reference)
    confidence.nii       — per-voxel max softmax score  (low = uncertain regions)
"""

import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from monai.data import Dataset
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference

from config import TrainConfig, VERTEBRA_CLASSES
from model_segformer3d import SegFormer3D
from transforms import build_infer_transforms


def _to_bcdhw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 1, 4, 2, 3).contiguous()


def load_model(cfg: TrainConfig, ckpt_path: str, device: str) -> SegFormer3D:
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model = SegFormer3D(
        in_chans        = 1,
        variant         = cfg.model_variant,
        num_seg_classes = cfg.num_seg_classes,
        num_cls_classes = 0,
        drop            = cfg.dropout,
        drop_path       = cfg.drop_path,
        use_checkpoint  = False,
    ).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)
    print(f"  loaded checkpoint: {ckpt_path}  "
          f"(epoch {ckpt.get('epoch','?')}, "
          f"dice {ckpt.get('best_dice', '?'):.4f})")
    return model


def load_ensemble(cfg: TrainConfig, ckpt_dir: str, device: str):
    """Load all fold checkpoints from a run directory."""
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "fold_*", "best.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No fold_*/best.pt found in {ckpt_dir}")
    models = [load_model(cfg, p, device) for p in ckpts]
    print(f"  ensemble of {len(models)} models")

    def predictor(x):
        logits = [m(x) for m in models]
        return torch.stack(logits, dim=0).mean(0)

    return predictor


def infer_one(
    cfg:       TrainConfig,
    predictor,
    batch:     dict,
    device:    str,
    out_dir:   str,
):
    img = _to_bcdhw(batch["image"].to(device))   # (1,1,D,H,W)

    with torch.no_grad():
        logits = sliding_window_inference(
            img,
            roi_size      = (cfg.patch_size[2], cfg.patch_size[0], cfg.patch_size[1]),
            sw_batch_size = cfg.sw_batch_size,
            predictor     = predictor,
            overlap       = cfg.sw_overlap,
            mode          = "gaussian",
        )
        probs = F.softmax(logits, dim=1)   # (1, C, D, H, W)

    pred_class  = probs.argmax(dim=1)[0]                    # (D, H, W)
    confidence  = probs.max(dim=1).values[0]                # (D, H, W)

    # Back to HWD (MONAI/Slicer convention)
    pred_hwd  = pred_class.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    conf_hwd  = confidence.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    ct_hwd    = batch["image"][0, 0].cpu().numpy().astype(np.float32)

    # Affine
    if "image_meta_dict" in batch and "affine" in batch["image_meta_dict"]:
        aff = batch["image_meta_dict"]["affine"][0]
        aff = aff.cpu().numpy() if hasattr(aff, "cpu") else np.asarray(aff)
    else:
        aff = np.eye(4, dtype=np.float32)

    os.makedirs(out_dir, exist_ok=True)

    nib.save(nib.Nifti1Image(ct_hwd,   aff), os.path.join(out_dir, "proc_ct.nii"))
    nib.save(nib.Nifti1Image(pred_hwd, aff), os.path.join(out_dir, "pred_vertebrae.nii"))
    nib.save(nib.Nifti1Image(conf_hwd, aff), os.path.join(out_dir, "confidence.nii"))

    # JSON label map — useful when importing into Slicer
    present_ids   = sorted(set(np.unique(pred_hwd).tolist()) - {0})
    label_json    = {str(i): VERTEBRA_CLASSES[i] for i in present_ids
                     if i in VERTEBRA_CLASSES}
    with open(os.path.join(out_dir, "pred_label.json"), "w") as f:
        json.dump(label_json, f, indent=2)

    return present_ids


def main():
    ap = argparse.ArgumentParser(description="SpineSegFormer3D — batch inference")
    ap.add_argument("--input_dir",  required=True,
                    help="Folder of new CT .nii files to segment")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write predictions")

    # Checkpoint: either single file or run dir (ensemble)
    ck = ap.add_mutually_exclusive_group(required=True)
    ck.add_argument("--ckpt",     help="Path to single best.pt checkpoint")
    ck.add_argument("--ckpt_dir", help="Run directory — uses ensemble of all fold_*/best.pt")

    ap.add_argument("--overlap",   type=float, default=None,
                    help="Sliding window overlap (default: from config = 0.5)")
    args = ap.parse_args()

    cfg    = TrainConfig(dataset_dir="dummy")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    if args.overlap:
        cfg.sw_overlap = args.overlap

    # ── Load model(s) ─────────────────────────────────────────────────────────
    if args.ckpt:
        model     = load_model(cfg, args.ckpt, device)
        predictor = model
    else:
        predictor = load_ensemble(cfg, args.ckpt_dir, device)

    # ── Build dataset ─────────────────────────────────────────────────────────
    images = sorted(glob.glob(os.path.join(args.input_dir, "*.nii")))
    # Exclude any files that look like segmentations
    images = [p for p in images
              if not any(x in os.path.basename(p).lower()
                         for x in ["segmentation", "remapped", "pred", "label"])]
    if not images:
        print("No .nii image files found in input_dir. Exiting.")
        return

    print(f"\nFound {len(images)} image(s) to process.\n")

    tf     = build_infer_transforms(cfg.isotropic_mm, cfg.base_lo, cfg.base_hi,
                                    reorient=cfg.reorient)
    ds     = Dataset([{"image": p} for p in images], transform=tf)
    loader = DataLoader(ds, batch_size=1, num_workers=0)

    # ── Inference loop ────────────────────────────────────────────────────────
    for i, batch in enumerate(loader):
        src      = batch["image_meta_dict"]["filename_or_obj"][0]
        case_name = os.path.splitext(os.path.basename(src))[0]
        out_dir   = os.path.join(args.output_dir, case_name)

        print(f"[{i+1}/{len(images)}]  {case_name}")
        present = infer_one(cfg, predictor, batch, device, out_dir)
        detected = [VERTEBRA_CLASSES.get(i, f"cls_{i}") for i in present]
        print(f"   detected: {', '.join(detected)}")
        print(f"   saved to: {out_dir}\n")

    print("Done.")
    print("\nNext steps:")
    print("  1. Open proc_ct.nii + pred_vertebrae.nii in 3D Slicer")
    print("  2. Fix segmentation errors")
    print("  3. Export corrected mask as *_segmentation.nii + *label.json")
    print("  4. Run:  python remap_labels.py --dataset_dir <your_dataset>")
    print("  5. Retrain:  python run_train.py --dataset_dir <your_dataset> ...")


if __name__ == "__main__":
    main()
