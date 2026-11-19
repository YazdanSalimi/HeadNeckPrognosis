"""
Step 2 - Extract CT radiomic features from GTVs, expanded GTVs, boxes and organs.

For each CT image, features are extracted (PyRadiomics) from:

  * the original and expanded GTVs:   ``<gtv-dir>/GTV*/<case>.nii.gz``
  * the bounding-box masks:           ``<gtv-dir>/Box_*/<case>.nii.gz``
  * organ segmentations (optional):   ``<organ-dir>/<case>*/*.nii.gz``

Masks smaller than 0.001 ml are skipped. One Excel file per case is written
to ``--output-dir`` (one row per mask/label).

Expected input layout (``--gtv-dir``, i.e. the folder used in step 1)::

    <gtv-dir>/
        imageCT/<case>.nii.gz
        GTV-original/<case>.nii.gz
        GTV_1_mm/ ... GTV_34_mm/
        Box_1_mm/ ... Box_34_mm/

Example::

    python step2_extract_radiomics.py --gtv-dir /data/Dataset114_HNGTV \\
        --organ-dir /data/organ-segmentations --output-dir /data/radiomics
"""

import argparse
import os
from functools import partial
from glob import glob

import pandas as pd
from natsort import os_sorted

from hn_tools.image_utils import segment_volume_ml
from hn_tools.parallel import run_parallel
from hn_tools.radiomics_utils import extract_features_all_labels

# Intensity windows (HU) applied to the CT before feature extraction.
GTV_WINDOW = (-600, 600)
BONE_WINDOW = (0, 800)
SOFT_TISSUE_WINDOW = (-300, 300)

BONY_STRUCTURES = (
    "clavicula_left.nii.gz", "clavicula_right.nii.gz", "rib_", "scapula_", "skull.nii.gz",
    "vertebrae_", "cricoid_cartilage.nii.gz", "hyoid.nii.gz", "styloid_process_",
    "thyroid_cartilage.nii.gz", "zygomatic_arch_",
)

MIN_MASK_VOLUME_ML = 0.001


def intensity_window(mask_path):
    """Choose the HU window for a mask: GTV/box, bone, or soft tissue."""
    folder = os.path.basename(os.path.dirname(mask_path))
    name = os.path.basename(mask_path)
    if folder.startswith("GTV") or folder.startswith("Box_"):
        return GTV_WINDOW
    if any(key in name for key in BONY_STRUCTURES):
        return BONE_WINDOW
    return SOFT_TISSUE_WINDOW


def process_case(image_path, gtv_dir, organ_dir, output_dir):
    """Extract features from all masks of one case and save them to Excel."""
    image_name = os.path.basename(image_path)
    case_id = image_name.replace(".nii.gz", "")
    output_path = os.path.join(output_dir, f"{case_id}_vertical.xlsx")
    if os.path.exists(output_path):
        return

    gtv_masks = os_sorted(glob(os.path.join(gtv_dir, "GTV*", image_name)))
    box_masks = os_sorted(glob(os.path.join(gtv_dir, "Box_*", image_name)))
    organ_masks = os_sorted(glob(os.path.join(organ_dir, f"{case_id}*", "*.nii.gz"))) if organ_dir else []

    masks = [m for m in gtv_masks + organ_masks + box_masks if segment_volume_ml(m) > MIN_MASK_VOLUME_ML]

    results = []
    for mask_path in masks:
        if mask_path in organ_masks:
            title = os.path.basename(mask_path).replace(case_id, "").replace(".nii.gz", "")
        else:
            title = os.path.basename(os.path.dirname(mask_path))  # e.g. "GTV_5_mm"
        try:
            results.append(
                extract_features_all_labels(
                    image_path,
                    mask_path,
                    clip_range=intensity_window(mask_path),
                    feature_classes="all",
                    comment=title,
                )
            )
        except Exception as error:  # keep going if one mask fails
            print(f"[warning] {case_id}: failed on {mask_path}: {error}")

    if results:
        pd.concat(results, axis=0).to_excel(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gtv-dir", required=True, help="Folder with imageCT/, GTV*/ and Box_*/ (output of step 1)")
    parser.add_argument("--organ-dir", default=None, help="Optional folder with per-case organ segmentations")
    parser.add_argument("--output-dir", required=True, help="Where the Excel files are written")
    parser.add_argument("--workers", type=int, default=-1, help="Parallel workers, -1 = all cores")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    images = os_sorted(glob(os.path.join(args.gtv_dir, "imageCT", "*.nii.gz")))
    print(f"Found {len(images)} CT images")

    run_parallel(
        partial(process_case, gtv_dir=args.gtv_dir, organ_dir=args.organ_dir, output_dir=args.output_dir),
        images,
        num_workers=args.workers,
        desc="Extracting radiomics",
    )


if __name__ == "__main__":
    main()
