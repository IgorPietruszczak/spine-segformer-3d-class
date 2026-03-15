"""
run_train.py
------------
Terminal entrypoint for SpineSegFormer3D training.

Usage examples
--------------
# First iteration — 6 cases, LOOCV
python run_train.py --dataset_dir "D:/Skany/Anonimizowane dane/DATASET" --strategy loocv

# Second iteration — 18 cases, fixed val set
python run_train.py \
    --dataset_dir "D:/Skany/Anonimizowane dane/DATASET" \
    --strategy fixed \
    --val_cases "5_kosci_1.0,7_vol_kr_kosci" \
    --epochs 300

# Resume / tweak LR
python run_train.py --dataset_dir "..." --strategy fixed --val_cases "..." --lr 5e-5

TensorBoard
-----------
tensorboard --logdir runs/segformer3d_spine --port 6006
"""

import argparse
import time
import os

from config import TrainConfig
from train import run_training


def parse_args():
    ap = argparse.ArgumentParser(description="SpineSegFormer3D — training")

    # Required
    ap.add_argument("--dataset_dir", required=True,
                    help="Folder with *_segmentation_remapped.nii files")

    # Validation strategy
    ap.add_argument("--strategy", default="loocv", choices=["loocv", "fixed"],
                    help="'loocv' for small datasets, 'fixed' once you have 10+ cases")
    ap.add_argument("--val_cases", default="",
                    help="Comma-separated case_ids for fixed val set, e.g. 'case1,case2'")

    # Training hyperparams (all optional — defaults in TrainConfig)
    ap.add_argument("--epochs",      type=int,   default=None)
    ap.add_argument("--lr",          type=float, default=None)
    ap.add_argument("--batch_size",  type=int,   default=None)
    ap.add_argument("--patch_size",  type=int,   default=None,
                    help="Cubic patch size (single int, e.g. 96)")
    ap.add_argument("--num_workers", type=int,   default=None)
    ap.add_argument("--no_tb",       action="store_true",
                    help="Disable TensorBoard logging")

    # Output
    ap.add_argument("--out_dir", default=None,
                    help="Override output directory (default: runs/segformer3d_spine/<timestamp>)")

    # Auto-test after training
    ap.add_argument("--test_case_id", default=None,
                    help="case_id to run test export on after training")

    return ap.parse_args()


def main():
    args = args = parse_args()

    run_name = time.strftime("%Y%m%d-%H%M%S")
    out_dir  = args.out_dir or os.path.join("runs", "segformer3d_spine", run_name)

    cfg = TrainConfig(
        dataset_dir   = args.dataset_dir,
        out_dir       = out_dir,
        val_strategy  = args.strategy,
        val_case_ids  = [v.strip() for v in args.val_cases.split(",") if v.strip()]
                        or None,
        use_tensorboard = not args.no_tb,
        test_case_id  = args.test_case_id,
    )

    # Apply optional overrides
    if args.epochs:      cfg.num_epochs   = args.epochs
    if args.lr:          cfg.lr           = args.lr
    if args.batch_size:  cfg.batch_size   = args.batch_size
    if args.num_workers: cfg.num_workers  = args.num_workers
    if args.patch_size:  cfg.patch_size   = (args.patch_size,) * 3

    # ── Print run summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SpineSegFormer3D — Training Run")
    print("=" * 60)
    print(f"  dataset     : {cfg.dataset_dir}")
    print(f"  strategy    : {cfg.val_strategy}"
          + (f"  val_cases: {cfg.val_case_ids}" if cfg.val_strategy == "fixed" else ""))
    print(f"  classes     : {cfg.num_seg_classes}  (background + T1–T12 + L1–L5)")
    print(f"  epochs      : {cfg.num_epochs}  |  lr: {cfg.lr}  |  patch: {cfg.patch_size}")
    print(f"  out_dir     : {cfg.out_dir}")
    if cfg.use_tensorboard:
        print(f"  TensorBoard : tensorboard --logdir \"{cfg.out_dir}\" --port 6006")
    print("=" * 60 + "\n")

    run_training(cfg)


if __name__ == "__main__":
    main()
