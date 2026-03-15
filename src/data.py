"""
data.py
-------
Scans dataset_dir for (image, segmentation) pairs.
Prefers *_segmentation_remapped.nii over *_segmentation.nii.
Supports both LOOCV and fixed validation-set strategies.
"""
import os
import re
from typing import List, Dict, Optional


def _key_from_stem(stem: str) -> str:
    """Normalise a filename stem to a consistent case_id key."""
    k = re.sub(r"[\s_]*segmentation[\s_]*(remapped)?", "", stem, flags=re.IGNORECASE).strip()
    k = re.sub(r"[\s_]+", "_", k)
    return k.lower()


def list_pairs(dataset_dir: str) -> List[Dict[str, str]]:
    """
    Returns a list of dicts: {"image": path, "label": path, "case_id": str}

    Rules:
    - *_segmentation_remapped.nii  is preferred over  *_segmentation.nii
    - Image files are anything that does NOT match segmentation/remapped
    - Raises FileNotFoundError if any image has no matching segmentation
    """
    files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".nii")]

    images:   dict[str, str] = {}
    labels:   dict[str, str] = {}   # key → best available segmentation path

    for fname in files:
        stem = os.path.splitext(fname)[0]
        path = os.path.join(dataset_dir, fname)

        if re.search(r"segmentation", stem, flags=re.IGNORECASE):
            key = _key_from_stem(stem)
            is_remapped = bool(re.search(r"remapped", stem, flags=re.IGNORECASE))
            # Prefer remapped; only fall back to original if remapped absent
            if key not in labels or is_remapped:
                labels[key] = path
        else:
            key = _key_from_stem(stem)
            images[key] = path

    pairs, missing = [], []
    for key, img_path in sorted(images.items()):
        lab_path = labels.get(key)
        if lab_path is None:
            missing.append(f"  {img_path}")
        else:
            using_remapped = "remapped" in os.path.basename(lab_path).lower()
            if not using_remapped:
                print(f"[WARN] Using NON-remapped mask for '{key}'. "
                      f"Run remap_labels.py first!")
            pairs.append({"image": img_path, "label": lab_path, "case_id": key})

    if missing:
        raise FileNotFoundError(
            "No matching segmentation for:\n" + "\n".join(missing)
        )
    if not pairs:
        raise FileNotFoundError(
            f"No image+segmentation pairs found in: {dataset_dir}"
        )
    return pairs


def split_pairs(
    pairs:        List[Dict[str, str]],
    strategy:     str             = "loocv",
    fold_idx:     int             = 0,
    val_case_ids: Optional[list]  = None,
) -> tuple[List[Dict], List[Dict]]:
    """
    Split pairs into (train, val) according to strategy.

    "loocv"  : pairs[fold_idx] is val, rest is train.
    "fixed"  : cases whose case_id is in val_case_ids go to val, rest to train.
               Use this once the dataset grows beyond ~8 cases.
    """
    if strategy == "loocv":
        val   = [pairs[fold_idx]]
        train = [p for i, p in enumerate(pairs) if i != fold_idx]
        return train, val

    if strategy == "fixed":
        if not val_case_ids:
            raise ValueError("val_strategy='fixed' requires val_case_ids list in config.")
        val_ids = set(v.lower() for v in val_case_ids)
        val     = [p for p in pairs if p["case_id"] in val_ids]
        train   = [p for p in pairs if p["case_id"] not in val_ids]
        if not val:
            raise ValueError(
                f"None of val_case_ids {val_case_ids} matched any case. "
                f"Available: {[p['case_id'] for p in pairs]}"
            )
        return train, val

    raise ValueError(f"Unknown val_strategy: '{strategy}'. Use 'loocv' or 'fixed'.")
