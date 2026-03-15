"""
train.py
--------
Core training engine for SpineSegFormer3D.

Speed optimisations vs naïve baseline:
  1. CacheDataset  — preprocesses every volume once, keeps in RAM.
                     Eliminates per-epoch disk I/O + resampling.
  2. torch.compile — PyTorch 2.x graph compiler on the model (~15% faster).
  3. Warmup LR     — linear ramp then cosine, lets us train for 200 epochs
                     instead of 300 without losing convergence quality.
  4. val_sw_overlap=0.25 — 3x faster sliding window during training-time
                     validation (full 0.5 overlap used by infer_and_export.py).
  5. val_interval=10 — halves the number of full-volume val passes per fold.

TensorBoard logging:
  train/loss_step   — per-batch smoothed loss (every cfg.log_every_n_steps steps)
  train/loss_epoch  — mean epoch loss
  train/lr          — current LR (shows warmup ramp + cosine tail)
  gpu/peak_mem_mb   — peak VRAM per epoch
  val/mean_dice     — mean Dice across all vertebrae
  val/mean_hd95     — HD95 in mm
  val/dice_T1 ...   — per-class Dice for every vertebra present in val case
  val/seg_overlay   — axial / coronal / sagittal GT vs PRED side-by-side
  val/confusion     — recall-normalised confusion matrix heatmap
"""
from __future__ import annotations

import os
import math
import time
from collections import deque
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from monai.data import CacheDataset, Dataset, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import KeepLargestConnectedComponent
from monai.utils import set_determinism

from config import TrainConfig
from data import list_pairs, split_pairs
from model_segformer3d import SegFormer3D
from losses import build_loss
from metrics import MetricTracker, pred_to_onehot, label_to_onehot
from transforms import build_train_transforms, build_val_transforms
from utils import ensure_dir, seed_everything, count_parameters, save_checkpoint
from visualisation import seg_overlay_grid, ConfusionAccumulator


# ─────────────────────────────────────────────────────────────────────────────
#  Axis helpers  (MONAI: C,H,W,D  →  model: B,C,D,H,W)
# ─────────────────────────────────────────────────────────────────────────────

def to_model(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:          # single item after decollate_batch
        x = x.unsqueeze(0)
    return x.permute(0, 1, 4, 2, 3).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
#  LCC post-processing
# ─────────────────────────────────────────────────────────────────────────────

_lcc = KeepLargestConnectedComponent(
    applied_labels=list(range(1, 18)),
    independent=True,
)


def apply_lcc(pred_onehot: torch.Tensor) -> torch.Tensor:
    return _lcc(pred_onehot)


# ─────────────────────────────────────────────────────────────────────────────
#  LR scheduler with linear warmup + cosine decay
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Linear warmup for `warmup_epochs`, then cosine decay to `eta_min`.

    Why warmup?
      Transformer models have many normalisation layers that are sensitive to
      large gradient updates at the start of training. Warming up the LR for
      the first 10 epochs prevents early instability and lets you safely use
      200 epochs instead of 300 — saving ~33% training time.
    """
    def __init__(
        self,
        optimizer:     torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs:  int,
        eta_min_ratio: float = 0.01,   # eta_min = lr * eta_min_ratio
    ):
        self.warmup  = warmup_epochs
        self.total   = total_epochs
        self.min_r   = eta_min_ratio

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                # Linear ramp: 0 → 1
                return (epoch + 1) / max(warmup_epochs, 1)
            # Cosine tail: 1 → eta_min_ratio
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_r + (1.0 - self.min_r) * cosine

        super().__init__(optimizer, lr_lambda=lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset builder  (CacheDataset when cache_rate > 0)
# ─────────────────────────────────────────────────────────────────────────────

def _build_dataset(pairs, transform, cache_rate: float, num_workers: int):
    """
    CacheDataset runs the full preprocessing pipeline (Load → Orient →
    Resample → Window) once per case and stores the result in RAM.
    On subsequent epochs the DataLoader reads directly from the cache —
    no disk I/O, no resampling.

    cache_rate=1.0  → cache 100% of the dataset (recommended if RAM ≥ 8 GB)
    cache_rate=0.0  → fallback to plain Dataset (original behaviour)

    num_workers is used only for the initial cache-fill pass; normal epoch
    iteration uses num_workers=0 since data is already in RAM.
    """
    if cache_rate > 0.0:
        return CacheDataset(
            data         = pairs,
            transform    = transform,
            cache_rate   = cache_rate,
            num_workers  = num_workers,   # parallel cache fill
            progress     = True,
        )
    return Dataset(data=pairs, transform=transform)


# ─────────────────────────────────────────────────────────────────────────────
#  Single fold
# ─────────────────────────────────────────────────────────────────────────────

def run_one_fold(
    cfg:         TrainConfig,
    train_pairs: List[Dict],
    val_pairs:   List[Dict],
    fold_dir:    str,
    device:      str,
    fold_label:  str = "train",
) -> Dict:

    ensure_dir(fold_dir)
    ckpt_path = os.path.join(fold_dir, "best.pt")

    # ── TensorBoard ──────────────────────────────────────────────────────────
    writer = None
    if cfg.use_tensorboard:
        tb_dir = cfg.tb_dir or os.path.join(fold_dir, "tb")
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_text("config", str(cfg.__dict__))

    # ── Transforms ───────────────────────────────────────────────────────────
    train_tf = build_train_transforms(
        isotropic_mm = cfg.isotropic_mm,
        patch_size   = cfg.patch_size,
        pos          = cfg.pos_neg_ratio[0],
        neg          = cfg.pos_neg_ratio[1],
        num_samples  = cfg.num_samples_per_volume,
        base_lo      = cfg.base_lo,
        base_hi      = cfg.base_hi,
        jitter       = cfg.jitter,
        reorient     = cfg.reorient,
    )
    val_tf = build_val_transforms(
        isotropic_mm = cfg.isotropic_mm,
        base_lo      = cfg.base_lo,
        base_hi      = cfg.base_hi,
        reorient     = cfg.reorient,
    )

    # ── Datasets  ────────────────────────────────────────────────────────────
    print(f"  Building cache (cache_rate={cfg.cache_rate}) …")
    t0 = time.time()

    train_ds = _build_dataset(train_pairs, train_tf, cfg.cache_rate, cfg.num_workers)
    val_ds   = _build_dataset(val_pairs,   val_tf,   cfg.cache_rate, 0)

    elapsed = time.time() - t0
    print(f"  Cache ready in {elapsed:.1f}s")

    # When using CacheDataset the data is already in RAM — workers only add
    # process-spawn overhead (especially bad on Windows).
    iter_workers = 0 if cfg.cache_rate > 0 else cfg.num_workers

    train_loader = DataLoader(
        train_ds,
        batch_size         = cfg.batch_size,
        shuffle            = True,
        num_workers        = iter_workers,
        pin_memory         = device.startswith("cuda") and iter_workers == 0,
        collate_fn         = list_data_collate,
        persistent_workers = iter_workers > 0,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────────
    model = SegFormer3D(
        in_chans        = 1,
        num_seg_classes = cfg.num_seg_classes,
        drop            = cfg.dropout,
        drop_path       = cfg.drop_path,
        use_checkpoint  = cfg.use_checkpoint,
    ).to(device)

    # torch.compile — wraps the model in TorchDynamo/Inductor for ~15% speedup.
    # Falls back gracefully if the PyTorch version doesn't support it.
    if cfg.use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile ✓")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")

    print(f"  Parameters: {count_parameters(model)}")

    # ── Loss / optimiser / scheduler ─────────────────────────────────────────
    loss_fn   = build_loss(cfg.num_seg_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs = cfg.warmup_epochs,
        total_epochs  = cfg.num_epochs,
        eta_min_ratio = 0.01,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    # ── Metric / confusion trackers ───────────────────────────────────────────
    metric_tracker = MetricTracker()
    confusion_acc  = ConfusionAccumulator(cfg.num_seg_classes)

    best_dice   = -1.0
    best_hd95   = float("inf")
    global_step = 0
    loss_window: deque[float] = deque(maxlen=cfg.log_every_n_steps)

    fold_start = time.time()

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_steps    = 0

        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            desc  = f"[{fold_label}] {epoch:03d}/{cfg.num_epochs}",
            leave = False,
            ncols = 88,
        )

        for raw_batch in pbar:
            items = decollate_batch(raw_batch)
            for item in items:
                img = to_model(item["image"].to(device))
                lab = to_model(item["label"].to(device))

                with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
                    logits = model(img)
                    loss   = loss_fn(logits, lab.long()) / cfg.grad_accum

                scaler.scale(loss).backward()
                raw_loss    = loss.item() * cfg.grad_accum
                epoch_loss += raw_loss
                n_steps    += 1
                global_step += 1
                loss_window.append(raw_loss)

                if writer and global_step % cfg.log_every_n_steps == 0:
                    writer.add_scalar("train/loss_step",
                                      float(np.mean(loss_window)), global_step)

                if n_steps % cfg.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                pbar.set_postfix(loss=f"{epoch_loss/n_steps:.4f}")

        # Flush remaining grads
        if n_steps % cfg.grad_accum != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        avg_loss = epoch_loss / max(n_steps, 1)
        if writer:
            writer.add_scalar("train/loss_epoch", avg_loss, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
            if device.startswith("cuda"):
                writer.add_scalar(
                    "gpu/peak_mem_mb",
                    torch.cuda.max_memory_allocated() / 1024 ** 2,
                    epoch,
                )

        # ── Validation ────────────────────────────────────────────────────────
        if epoch % cfg.val_interval != 0 and epoch != cfg.num_epochs:
            continue

        model.eval()
        metric_tracker.reset()
        confusion_acc.reset()
        first_img_cpu = first_logits_cpu = first_lab_cpu = None

        with torch.no_grad():
            for v_idx, val_batch in enumerate(val_loader):
                img = to_model(val_batch["image"].to(device))
                lab = to_model(val_batch["label"].to(device))

                logits = sliding_window_inference(
                    img,
                    roi_size      = (cfg.patch_size[2],
                                     cfg.patch_size[0],
                                     cfg.patch_size[1]),
                    sw_batch_size = cfg.sw_batch_size,
                    predictor     = model,
                    overlap       = cfg.val_sw_overlap,   # ← fast during training
                    mode          = "gaussian",
                )

                pred_oh = pred_to_onehot(logits)
                lab_oh  = label_to_onehot(lab.long())

                if cfg.keep_lcc:
                    pred_oh = apply_lcc(pred_oh)

                metric_tracker.update(pred_oh, lab_oh)

                pred_mask = logits.argmax(dim=1)[0]
                gt_mask   = lab[0, 0].long()
                confusion_acc.update(pred_mask, gt_mask)

                if v_idx == 0:
                    first_img_cpu    = img.cpu()
                    first_logits_cpu = logits.cpu()
                    first_lab_cpu    = lab.cpu()

        result = metric_tracker.compute()
        dice   = result["mean_dice"]
        hd95   = result["mean_hd95"]

        metric_tracker.log(writer, epoch)
        metric_tracker.print_table(epoch)

        # ── Segmentation overlay ──────────────────────────────────────────────
        if writer and first_img_cpu is not None:
            try:
                grid = seg_overlay_grid(first_img_cpu, first_logits_cpu,
                                        first_lab_cpu, alpha=0.45)
                writer.add_image("val/seg_overlay",
                                 torch.from_numpy(grid), epoch)
            except Exception as exc:
                print(f"  [vis] overlay skipped: {exc}")

        # ── Confusion matrix ──────────────────────────────────────────────────
        if writer:
            try:
                import matplotlib.pyplot as plt
                fig = confusion_acc.plot()
                writer.add_figure("val/confusion", fig, epoch)
                plt.close(fig)
            except Exception as exc:
                print(f"  [vis] confusion skipped: {exc}")

        writer and writer.flush()

        # ── ETA estimate ─────────────────────────────────────────────────────
        elapsed_min = (time.time() - fold_start) / 60
        eta_min     = elapsed_min / epoch * (cfg.num_epochs - epoch)
        print(f"  elapsed {elapsed_min:.0f}min  |  ETA ~{eta_min:.0f}min")

        # ── Checkpoint ────────────────────────────────────────────────────────
        improved = (dice > best_dice) or (
            dice == best_dice and hd95 < best_hd95
        )
        if improved:
            best_dice = dice
            best_hd95 = hd95
            save_checkpoint(
                ckpt_path, model, cfg,
                fold      = fold_label,
                epoch     = epoch,
                best_dice = best_dice,
                best_hd95 = best_hd95,
                val_cases = [p["case_id"] for p in val_pairs],
            )
            print(f"  ✓ new best  dice={best_dice:.4f}  hd95={best_hd95:.2f}")

    if writer:
        writer.flush()
        writer.close()

    total_h = (time.time() - fold_start) / 3600
    print(f"  Fold done in {total_h:.2f}h")

    return {
        "fold":      fold_label,
        "best_dice": best_dice,
        "best_hd95": best_hd95,
        "val_cases": [p["case_id"] for p in val_pairs],
        "ckpt":      ckpt_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_training(cfg: TrainConfig, device: Optional[str] = None) -> List[Dict]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = list_pairs(cfg.dataset_dir)
    print(f"\n  Found {len(pairs)} cases:")
    for p in pairs:
        tag = "(remapped ✓)" if "remapped" in p["label"] else "(NOT remapped ⚠)"
        print(f"    {p['case_id']}  {tag}")

    # Print speed config summary
    print(f"\n  Speed settings:")
    print(f"    CacheDataset   cache_rate={cfg.cache_rate}")
    print(f"    torch.compile  use_compile={cfg.use_compile}")
    print(f"    epochs         {cfg.num_epochs}  (warmup {cfg.warmup_epochs})")
    print(f"    val_interval   {cfg.val_interval}  (every {cfg.val_interval} epochs)")
    print(f"    val_sw_overlap {cfg.val_sw_overlap}  (train-time val only)\n")

    ensure_dir(cfg.out_dir)
    results = []
    run_start = time.time()

    if cfg.val_strategy == "loocv":
        print(f"  Strategy: LOOCV  ({len(pairs)} folds)")
        for fold_idx in range(len(pairs)):
            train_p, val_p = split_pairs(pairs, "loocv", fold_idx=fold_idx)
            fold_dir   = os.path.join(cfg.out_dir, f"fold_{fold_idx}")
            fold_label = f"fold_{fold_idx}"

            set_determinism(seed=cfg.seed + fold_idx)
            seed_everything(cfg.seed + fold_idx)

            print(f"\n{'='*62}")
            print(f"  FOLD {fold_idx}  —  val: {val_p[0]['case_id']}")
            print(f"  train: {[p['case_id'] for p in train_p]}")
            print(f"{'='*62}\n")

            res = run_one_fold(cfg, train_p, val_p, fold_dir, device, fold_label)
            results.append(res)
            print(f"  → fold {fold_idx}: dice={res['best_dice']:.4f}  "
                  f"hd95={res['best_hd95']:.2f}")

        valid_dice = [r["best_dice"] for r in results if r["best_dice"] > 0]
        valid_hd95 = [r["best_hd95"] for r in results if r["best_hd95"] < 1e8]
        mean_dice  = float(np.mean(valid_dice)) if valid_dice else 0.0
        mean_hd95  = float(np.mean(valid_hd95)) if valid_hd95 else float("nan")

        total_h = (time.time() - run_start) / 3600
        print(f"\n{'='*62}")
        print(f"  LOOCV complete in {total_h:.2f}h")
        print(f"  mean Dice={mean_dice:.4f}  mean HD95={mean_hd95:.2f}")
        for r in results:
            print(f"    {r['fold']}  dice={r['best_dice']:.4f}  "
                  f"hd95={r['best_hd95']:.2f}  val={r['val_cases']}")
        print(f"{'='*62}\n")
        _save_csv(results, cfg.out_dir, mean_dice, mean_hd95)

    elif cfg.val_strategy == "fixed":
        train_p, val_p = split_pairs(pairs, "fixed", val_case_ids=cfg.val_case_ids)
        print(f"  Strategy: fixed val")
        print(f"  train ({len(train_p)}): {[p['case_id'] for p in train_p]}")
        print(f"  val   ({len(val_p)}):   {[p['case_id'] for p in val_p]}\n")

        set_determinism(seed=cfg.seed)
        seed_everything(cfg.seed)

        res = run_one_fold(cfg, train_p, val_p, cfg.out_dir, device, "fixed")
        results.append(res)

        total_h = (time.time() - run_start) / 3600
        print(f"\n  Done in {total_h:.2f}h  —  "
              f"best dice={res['best_dice']:.4f}  hd95={res['best_hd95']:.2f}")
        _save_csv(results, cfg.out_dir)

    else:
        raise ValueError(f"Unknown val_strategy='{cfg.val_strategy}'")

    return results


def _save_csv(results, out_dir, mean_dice=None, mean_hd95=None):
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        if mean_dice is not None:
            df.loc[len(df)] = {
                "fold": "mean", "best_dice": mean_dice,
                "best_hd95": mean_hd95, "val_cases": "", "ckpt": "",
            }
        path = os.path.join(out_dir, "results.csv")
        df.to_csv(path, index=False)
        print(f"  Results: {path}")
    except ImportError:
        pass
