"""
Step 1 - Expand the GTV by isotropic margins and build bounding-box masks.

For every GTV mask and every margin (default 1-34 mm) this script writes:

  * ``GTV_<m>_mm/<case>.nii.gz`` - the GTV dilated by <m> mm, clipped to the body contour
  * ``Box_<m>_mm/<case>.nii.gz`` - the bounding box of each object of that expanded GTV

Expected input layout (``--data-dir``)::

    <data-dir>/
        GTV-original/<case>.nii.gz     # original GTV masks
        body-contour/<case>.nii.gz     # body masks (same file names)

Example::

    python step1_expand_gtv_and_boxes.py --data-dir /data/Dataset114_HNGTV
"""

import argparse
import os
from functools import partial
from glob import glob

import SimpleITK as sitk
from natsort import os_sorted

from hn_tools.image_utils import (
    binarize,
    keep_inside_mask,
    resample_to_reference,
    resample_to_spacing,
    segment_to_boxes,
)
from hn_tools.parallel import run_parallel


def expand_gtv(gtv_path, margin_mm, body_path):
    """
    Dilate a GTV by ``margin_mm`` (isotropic) and clip it to the body contour.

    The dilation is done on a 1 mm isotropic grid so that the kernel radius in
    voxels equals the margin in mm; the result is resampled back onto the
    original GTV grid.
    """
    gtv_1mm = resample_to_spacing(gtv_path, (1, 1, 1), interpolator=sitk.sitkNearestNeighbor)
    gtv_1mm = sitk.Cast(gtv_1mm, sitk.sitkUInt8)

    expanded_1mm = sitk.BinaryDilate(gtv_1mm, (margin_mm,) * 3)
    expanded = resample_to_reference(expanded_1mm, gtv_path, interpolator=sitk.sitkLinear)
    expanded = keep_inside_mask(expanded, body_path)
    return binarize(expanded)


def process_case(gtv_path, data_dir, margins):
    """Create all expanded-GTV and box masks for one case (skips existing outputs)."""
    case_name = os.path.basename(gtv_path)
    body_path = os.path.join(data_dir, "body-contour", case_name)

    for margin in margins:
        gtv_out = os.path.join(data_dir, f"GTV_{margin}_mm", case_name)
        box_out = os.path.join(data_dir, f"Box_{margin}_mm", case_name)
        if os.path.exists(gtv_out) and os.path.exists(box_out):
            continue
        os.makedirs(os.path.dirname(gtv_out), exist_ok=True)
        os.makedirs(os.path.dirname(box_out), exist_ok=True)

        expanded = expand_gtv(gtv_path, margin, body_path)
        sitk.WriteImage(expanded, gtv_out)

        boxes = segment_to_boxes(expanded, min_object_volume_ml=0.001, margin_mm=0)
        sitk.WriteImage(binarize(boxes), box_out)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Folder containing GTV-original/ and body-contour/")
    parser.add_argument("--min-margin", type=int, default=1, help="Smallest margin in mm (default: 1)")
    parser.add_argument("--max-margin", type=int, default=34, help="Largest margin in mm (default: 34)")
    parser.add_argument("--step", type=int, default=1, help="Margin step in mm (default: 1)")
    parser.add_argument("--workers", type=int, default=-1, help="Parallel workers, -1 = all cores")
    return parser.parse_args()


def main():
    args = parse_args()
    margins = list(range(args.min_margin, args.max_margin + 1, args.step))
    gtv_paths = os_sorted(glob(os.path.join(args.data_dir, "GTV-original", "*.nii.gz")))
    print(f"Found {len(gtv_paths)} GTV masks; margins: {margins[0]}-{margins[-1]} mm")

    run_parallel(
        partial(process_case, data_dir=args.data_dir, margins=margins),
        gtv_paths,
        num_workers=args.workers,
        desc="Expanding GTVs",
    )


if __name__ == "__main__":
    main()
