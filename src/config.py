"""
config.py — SpineSegFormer3D
Absolute vertebra IDs after running remap_labels.py:
  0=background, 1=T1 ... 12=T12, 13=L1 ... 17=L5
"""
from dataclasses import dataclass
from typing import Optional

# ── Absolute class registry ───────────────────────────────────────────────────
VERTEBRA_CLASSES: dict[int, str] = {
    0:  "background",
    1:  "T1",  2:  "T2",  3:  "T3",  4:  "T4",
    5:  "T5",  6:  "T6",  7:  "T7",  8:  "T8",
    9:  "T9",  10: "T10", 11: "T11", 12: "T12",
    13: "L1",  14: "L2",  15: "L3",  16: "L4",  17: "L5",
}
NUM_SEG_CLASSES = 18   # 0..17


@dataclass
class TrainConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    dataset_dir: str = ""

    # Resampling
    isotropic_mm: float = 1.5   # 1.5mm: 3.4x fewer voxels than 1.0mm, fits in 32GB RAM
    reorient:     str   = "RAS"

    # Patch sampling
    patch_size:             tuple = (96, 96, 96)   # (H, W, D) — MONAI convention
    pos_neg_ratio:          tuple = (7, 3)
    num_samples_per_volume: int   = 4   # 8->4: halves patch memory per epoch

    # CT windowing (bone window — works well for spine CT)
    base_lo: float =  -1000.0
    base_hi: float =   2000.0
    jitter:  float =    200.0    # ± range applied randomly during training

    # ── Segmentation ─────────────────────────────────────────────────────────
    num_seg_classes: int = NUM_SEG_CLASSES   # 18

    # ── Training ─────────────────────────────────────────────────────────────
    num_epochs:   int   = 200   # 200 + warmup converges as well as 300 without it
    val_interval:       int   = 20   # validate every 20 epochs — reduces RAM pressure
    log_every_n_steps:  int   = 20   # per-batch loss logging frequency

    # ── Speed ─────────────────────────────────────────────────────────────
    # CacheDataset: preprocess every volume ONCE and keep in RAM.
    # With 6 cases this uses ~2-4 GB RAM and eliminates per-epoch disk I/O.
    # Set cache_rate=0.0 to disable (e.g. if RAM is limited).
    cache_rate:         float = 0.0   # disabled - saves ~8GB RAM; volumes load fast enough

    # torch.compile() — PyTorch 2.x graph compiler, ~15% faster forward/backward.
    # Disable if you hit compatibility issues (e.g. older CUDA drivers).
    use_compile:        bool  = False  # torch.compile requires Triton — not available on Windows

    # Sliding window overlap during VALIDATION (not test-time export).
    # 0.25 is ~3x faster than 0.5 with negligible metric difference during training.
    # infer_and_export.py always uses the full sw_overlap=0.5 for best results.
    val_sw_overlap:     float = 0.25

    # LR warmup: ramp LR linearly from 0 to cfg.lr over this many epochs,
    # then apply cosine decay. Allows the model to stabilise before the
    # full learning rate hits — lets you safely reduce num_epochs to 200.
    warmup_epochs:      int   = 10
    lr:           float = 2e-4
    weight_decay: float = 1e-4
    batch_size:   int   = 1
    grad_accum:   int   = 2
    num_workers:  int   = 2
    seed:         int   = 42

    # ── Model ─────────────────────────────────────────────────────────────────
    model_variant:  str   = "b1"
    dropout:        float = 0.0
    drop_path:      float = 0.1
    use_checkpoint: bool  = True

    # ── Inference ─────────────────────────────────────────────────────────────
    sw_overlap:    float = 0.5
    sw_batch_size: int   = 1   # keep at 1 - already minimal
    keep_lcc:      bool  = False  # disabled during training to save RAM; used in infer_and_export.py

    # ── Output ───────────────────────────────────────────────────────────────
    out_dir: str = "runs/segformer3d_spine"

    # ── TensorBoard ───────────────────────────────────────────────────────────
    use_tensorboard: bool = True
    tb_dir:          str  = ""    # "" → <fold_dir>/tb

    # ── Validation strategy ───────────────────────────────────────────────────
    # "loocv"  — leave-one-out (good for ≤8 cases)
    # "fixed"  — fixed val set defined by val_case_ids (better for larger datasets)
    val_strategy: str       = "loocv"
    val_case_ids: object = None  # list[str] | None — e.g. ["5_kosci_1.0", "7_vol_kr_kosci"]

    # ── Auto-test + NIfTI export after training ───────────────────────────────
    test_case_id:    Optional[str] = None
    test_image:      Optional[str] = None
    test_label:      Optional[str] = None
    test_export_dir: str           = ""
    test_use_ensemble: bool        = False

    # ── Helpers ───────────────────────────────────────────────────────────────
    @property
    def binary_mode(self) -> bool:
        return self.num_seg_classes == 2