# Head & Neck GTV expansion and radiomics

Code to

1. **expand head-and-neck GTV masks** by isotropic margins (1–34 mm), clip them to
   the body contour, and build the **bounding-box mask** of each expanded GTV;
2. **extract CT radiomic features** (PyRadiomics) from the original GTV, every
   expanded GTV, every box, and (optionally) organ segmentations.

## Repository layout

```
step1_expand_gtv_and_boxes.py   # GTV expansion + box creation
step2_extract_radiomics.py      # radiomic feature extraction
hn_tools/
    image_utils.py              # resampling, masking, bounding boxes, volumes
    radiomics_utils.py          # PyRadiomics wrapper (one row per label)
    parallel.py                 # multiprocessing helper with progress bar
```

## Installation

```bash
pip install -r requirements.txt
```

Python 3.8–3.11 is recommended (PyRadiomics wheels may not be available for newer versions).

## Data layout

All images and masks are NIfTI (`.nii.gz`) and share the same file name per case:

```
Dataset114_HNGTV/
    imageCT/<case>.nii.gz        # CT image
    GTV-original/<case>.nii.gz   # original GTV mask
    body-contour/<case>.nii.gz   # body mask
```

Organ segmentations (optional) are expected as `<organ-dir>/<case>*/<organ>.nii.gz`
(e.g. the output of TotalSegmentator).

## Usage

### Step 1 – expanded GTVs and boxes

```bash
python step1_expand_gtv_and_boxes.py --data-dir /path/to/Dataset114_HNGTV
```

Creates `GTV_<m>_mm/` and `Box_<m>_mm/` folders next to `GTV-original/`.
For each margin *m*:

* the GTV is resampled to 1 mm isotropic voxels, dilated with a ball of radius *m* voxels (= *m* mm),
  resampled back onto the original grid and clipped to the body contour;
* each connected object of the expanded GTV (> 0.001 ml) is replaced by its axis-aligned bounding box.

Options: `--min-margin`, `--max-margin`, `--step`, `--workers`.

### Step 2 – radiomics

```bash
python step2_extract_radiomics.py --gtv-dir /path/to/Dataset114_HNGTV \
    --organ-dir /path/to/organ-segmentations --output-dir /path/to/radiomics
```

Writes one `<case>_vertical.xlsx` file per CT (one row per mask/label).

Main settings (see `hn_tools/radiomics_utils.py`):

| Setting | Value |
|---|---|
| Feature classes | all (first order, shape, GLCM, GLRLM, GLSZM, GLDM, NGTDM) |
| Resampling | 1 × 1 × 1 mm, B-spline |
| Bin width | 10 HU |
| Intensity window | GTV / box: −600 to 600 HU · bone: 0 to 800 HU · soft tissue: −300 to 300 HU |
| Minimum mask volume | 0.001 ml |

## Citation

If you use this code, please cite: .

## License
