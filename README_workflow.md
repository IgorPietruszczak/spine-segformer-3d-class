# SpineSegFormer3D — Iterative Training Workflow

## Absolute class map
| Model ID | Vertebra | Model ID | Vertebra |
|----------|----------|----------|----------|
| 0        | background | 9     | T9       |
| 1        | T1       | 10       | T10      |
| 2        | T2       | 11       | T11      |
| 3        | T3       | 12       | T12      |
| 4        | T4       | 13       | L1       |
| 5        | T5       | 14       | L2       |
| 6        | T6       | 15       | L3       |
| 7        | T7       | 16       | L4       |
| 8        | T8       | 17       | L5       |

---

## One-time setup

```bash
# 1. Remap all existing masks to absolute IDs
python remap_labels.py --dataset_dir "D:/Skany/Anonimizowane dane/DATASET"

# Verify output — should print each case with its detected vertebrae
# e.g. "5_kosci_1.0 → T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, L1"
```

---

## Iteration N — training

### If dataset ≤ 8 cases → LOOCV
```bash
python run_train.py \
    --dataset_dir "D:/Skany/Anonimizowane dane/DATASET" \
    --strategy loocv \
    --epochs 300
```

### If dataset > 8 cases → fixed val set
Pick 2–3 representative cases as a permanent validation set
(ideally one lumbar-only, one full T1–L5, one with challenging anatomy).
Keep these cases in val forever — never add them to training.

```bash
python run_train.py \
    --dataset_dir "D:/Skany/Anonimizowane dane/DATASET" \
    --strategy fixed \
    --val_cases "5_kosci_1.0,7_vol_kr_kosci" \
    --epochs 300
```

TensorBoard:
```bash
tensorboard --logdir runs/segformer3d_spine --port 6006
```

---

## Iteration N — expanding the dataset

```bash
# 1. Run inference on new unlabeled CTs
python infer_and_export.py \
    --input_dir  "D:/Skany/Nowe skany" \
    --output_dir "D:/Skany/Nowe skany/predictions" \
    --ckpt       "runs/segformer3d_spine/<run_timestamp>/best.pt"

# For better predictions use ensemble (all LOOCV folds):
python infer_and_export.py \
    --input_dir  "D:/Skany/Nowe skany" \
    --output_dir "D:/Skany/Nowe skany/predictions" \
    --ckpt_dir   "runs/segformer3d_spine/<run_timestamp>"
```

```
# 2. Open in 3D Slicer:
#    - Load: proc_ct.nii          as Volume
#    - Load: pred_vertebrae.nii   as Segmentation
#    - Fix errors using the segment editor
#    - Export corrected mask as: <case_name>_segmentation.nii
#    - Export label map as:      <case_name>label.json
#      (same format as TotalSegmentator output)

# 3. Copy corrected files to dataset:
copy "D:/Skany/Nowe skany/<case_name>_segmentation.nii" "D:/Skany/.../DATASET/"
copy "D:/Skany/Nowe skany/<case_name>label.json"        "D:/Skany/.../DATASET/"

# 4. Remap the newly added cases
python remap_labels.py --dataset_dir "D:/Skany/Anonimizowane dane/DATASET"

# 5. Retrain (increase epochs slightly if needed)
python run_train.py --dataset_dir "..." --strategy fixed --val_cases "..." --epochs 300
```

---

## When to switch from LOOCV to fixed val

Switch when you have more than ~8 training cases.
LOOCV trains N separate models — slow and wasteful at scale.
Fixed val trains one model — fast, and the val metric is more stable
because the same cases are always evaluated.

---

## Tips for 3D Slicer correction

- Use **confidence.nii** as an overlay (Window/Level → narrow window around 0.5).
  Voxels with low confidence (dark) are where the model is uncertain — check these first.
- The **Segment Editor → Islands** effect is fast for removing stray predictions.
- Save your corrected file with exactly the naming convention:
  `<case_name>_segmentation.nii`  (the same stem as the CT image file, + `_segmentation`)

---

## File naming convention

```
DATASET/
  5 Kosci 1.0.nii                        ← CT image
  5 Kosci 1.0 segmentation.nii           ← original TotalSegmentator mask
  5 Kosci 1.0label.json                  ← TotalSegmentator label names
  5 Kosci 1.0_segmentation_remapped.nii  ← remapped mask (used for training)
```
