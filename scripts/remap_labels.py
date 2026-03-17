"""
remap_labels.py
---------------
Converts per-case relative vertebra IDs to absolute model IDs.

Problem:
  TotalSegmentator + your export script produces labels like:
    1 = topmost vertebra in THIS scan  (could be T1, could be L1)
    2 = next vertebra down
    ...
  The .json file records what each ID actually means.

Solution:
  Re-map every mask so that T1=1, T2=2 ... L5=17, background=0
  regardless of which vertebrae appear in a given scan.

Usage:
  python remap_labels.py --dataset_dir "D:/Skany/Anonimizowane dane/DATASET"

Output:
  Writes <case>_segmentation_remapped.nii next to each original mask.
  Prints a summary table so you can verify before training.

After running this script, point TrainConfig.dataset_dir at the same
folder — data.py will automatically prefer *_segmentation_remapped.nii
over *_segmentation.nii.
"""


# Allow running from project root
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import os
import re
import json
import argparse
import numpy as np
import nibabel as nib

# ── Absolute class map (model IDs) ────────────────────────────────────────────
# 0  = background
# 1  = T1  ... 12 = T12
# 13 = L1  ... 17 = L5
ABSOLUTE_IDS: dict[str, int] = {
    "t1":  1,  "t2":  2,  "t3":  3,  "t4":  4,
    "t5":  5,  "t6":  6,  "t7":  7,  "t8":  8,
    "t9":  9,  "t10": 10, "t11": 11, "t12": 12,
    "l1":  13, "l2":  14, "l3":  15, "l4":  16, "l5":  17,
}
NUM_CLASSES = 18   # 0..17

# Human-readable names for reporting
CLASS_NAMES: dict[int, str] = {v: k.upper() for k, v in ABSOLUTE_IDS.items()}
CLASS_NAMES[0] = "background"


def _parse_label_name(raw: str) -> str | None:
    """
    Extract normalised vertebra key from a JSON label string.
    Examples handled:
      "L1 vertebra"  → "l1"
      "T12 vertebra" → "t12"
      "lumbar 1"     → "l1"
      "thoracic 12"  → "t12"
    Returns None if the string cannot be parsed.
    """
    s = raw.lower().strip()

    # Direct match: "t1", "l5", "t12" etc.
    m = re.search(r'\b([tl])(\d{1,2})\b', s)
    if m:
        key = m.group(1) + m.group(2)
        return key if key in ABSOLUTE_IDS else None

    # "thoracic N" or "lumbar N"
    m = re.search(r'\b(thoracic|lumbar)\s+(\d{1,2})\b', s)
    if m:
        prefix = "t" if m.group(1) == "thoracic" else "l"
        key    = prefix + m.group(2)
        return key if key in ABSOLUTE_IDS else None

    return None


def remap_one_case(
    seg_path:  str,
    json_path: str,
    dry_run:   bool = False,
) -> dict:
    """
    Remap a single case.  Returns a summary dict.
    """
    with open(json_path, encoding="utf-8") as f:
        raw_map: dict[str, str] = json.load(f)

    # Build  {voxel_id (int) → absolute_id (int)}
    #
    # The Slicer JSON export script numbers keys 1,2,3... sequentially,
    # but the actual voxel values in the .nii are the original segment
    # indices which start from a different number.
    # Fix: sort both JSON keys and actual voxel IDs, then match positionally.
    #   JSON key 1 (topmost vertebra) → smallest voxel ID in the volume
    #   JSON key 2                    → next voxel ID
    #   etc.

    img_for_ids = nib.load(seg_path)
    arr_for_ids = np.asarray(img_for_ids.dataobj).astype(np.int16)
    voxel_ids_sorted = sorted(set(np.unique(arr_for_ids).tolist()) - {0})

    json_items = sorted(raw_map.items(), key=lambda x: int(x[0]))

    if len(json_items) != len(voxel_ids_sorted):
        print(f"  [WARN] JSON has {len(json_items)} entries but volume has "
              f"{len(voxel_ids_sorted)} non-zero IDs — will map what we can.")

    id_map   = {}
    skipped  = []
    for i, (rel_str, label_name) in enumerate(json_items):
        if i >= len(voxel_ids_sorted):
            break
        voxel_id = voxel_ids_sorted[i]
        key = _parse_label_name(label_name)
        if key is None:
            skipped.append((voxel_id, label_name))
            continue
        abs_id = ABSOLUTE_IDS[key]
        id_map[voxel_id] = abs_id

    img     = img_for_ids
    arr     = arr_for_ids.copy()
    out     = np.zeros_like(arr, dtype=np.int16)

    present_abs = set()
    for rel_id, abs_id in id_map.items():
        mask            = arr == rel_id
        out[mask]       = abs_id
        if mask.any():
            present_abs.add(abs_id)

    # Sanity: warn about relative IDs in the volume that have no JSON entry
    unique_in_vol = set(np.unique(arr).tolist()) - {0}
    unmapped      = unique_in_vol - set(id_map.keys())

    import re as _re
    out_path = _re.sub(
        r'[\s_]+segmentation\.nii$',
        '_segmentation_remapped.nii',
        seg_path,
        flags=_re.IGNORECASE,
    )
    if not dry_run:
        nib.save(nib.Nifti1Image(out, img.affine, img.header), out_path)

    return {
        "seg_path":    seg_path,
        "out_path":    out_path,
        "mapped":      {k: CLASS_NAMES[v] for k, v in id_map.items()},
        "present":     sorted(CLASS_NAMES[i] for i in present_abs),
        "skipped":     skipped,
        "unmapped_ids": sorted(unmapped),
        "dry_run":     dry_run,
    }


def find_pairs(dataset_dir: str) -> list[tuple[str, str]]:
    """
    Scan dataset_dir for (*_segmentation.nii, *label.json) pairs.
    Skips already-remapped files.
    """
    files = os.listdir(dataset_dir)
    segs  = [f for f in files
             if f.lower().endswith("segmentation.nii")
             and "remapped" not in f.lower()]
    pairs = []
    for seg_fname in sorted(segs):
        stem  = seg_fname.replace(" segmentation.nii", "").replace("_segmentation.nii", "")
        # Look for matching label.json  (case-insensitive "label")
        json_candidates = [f for f in files
                           if f.lower().endswith("label.json")
                           and stem.lower() in f.lower()]
        if not json_candidates:
            print(f"  [WARN] No label.json found for: {seg_fname} (skipping)")
            continue
        pairs.append((
            os.path.join(dataset_dir, seg_fname),
            os.path.join(dataset_dir, json_candidates[0]),
        ))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Remap vertebra labels to absolute IDs")
    ap.add_argument("--dataset_dir", required=True,
                    help="Folder containing *_segmentation.nii + *label.json files")
    ap.add_argument("--dry_run", action="store_true",
                    help="Parse and report only — do not write any files")
    args = ap.parse_args()

    pairs = find_pairs(args.dataset_dir)
    if not pairs:
        print("No (segmentation.nii, label.json) pairs found. Exiting.")
        return

    print(f"\nFound {len(pairs)} case(s).  dry_run={args.dry_run}\n")
    print(f"{'Case':<40}  {'Vertebrae present'}")
    print("-" * 80)

    all_ok = True
    for seg_path, json_path in pairs:
        result = remap_one_case(seg_path, json_path, dry_run=args.dry_run)
        case   = os.path.basename(seg_path)

        if result["skipped"]:
            print(f"  [WARN] Could not parse label names: {result['skipped']}")
            all_ok = False

        if result["unmapped_ids"]:
            print(f"  [WARN] Volume contains IDs not in JSON: {result['unmapped_ids']}")
            all_ok = False

        status = "(dry run)" if args.dry_run else "→ saved"
        print(f"{case:<40}  {', '.join(result['present'])}  {status}")

    print()
    if all_ok:
        print("All cases remapped successfully.")
    else:
        print("Some warnings were raised — check output above before training.")

    print(f"\nAbsolute class map used:")
    for name, abs_id in sorted(ABSOLUTE_IDS.items(), key=lambda x: x[1]):
        print(f"  {abs_id:2d}  {name.upper()}")


if __name__ == "__main__":
    main()