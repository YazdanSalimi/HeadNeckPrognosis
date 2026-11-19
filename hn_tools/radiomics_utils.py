"""
Thin wrapper around PyRadiomics to extract features for every label of a mask
and return them as a pandas DataFrame (one row per label).
"""

import logging
import os
import warnings

import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

from .image_utils import read_image, resample_to_reference, unique_labels

logging.getLogger("radiomics").setLevel(logging.ERROR)

ALL_FEATURE_CLASSES = ("firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm")

# Default PyRadiomics settings used in this project.
DEFAULT_SETTINGS = dict(
    binWidth=10,
    resampledPixelSpacing=(1, 1, 1),
    interpolator=sitk.sitkBSpline,
    normalize=False,
    voxelArrayShift=0,
    symmetricalGLCM=True,
    geometryTolerance=0.1,
    additionalInfo=False,
    force2D=False,
    force2Ddimension=0,
    weightingNorm=None,
)


def _name(item):
    return os.path.basename(item) if isinstance(item, str) else "loaded from memory"


def extract_features(
    image,
    segment,
    label=1,
    clip_range=None,
    feature_classes="all",
    comment="",
    shorten_feature_names=True,
    resample_mask_to_image=False,
    **settings,
):
    """
    Extract radiomic features for one label of a segmentation.

    Parameters
    ----------
    image, segment : str or sitk.Image
    label : label value inside ``segment`` to analyse
    clip_range : (min, max) intensity window applied to the image before
        extraction, e.g. (-300, 300) HU; ``None`` disables clipping
    feature_classes : "all" or a sequence such as ("firstorder", "glcm")
    comment : free text stored in the "comment" column
    shorten_feature_names : drop the "original_" / "diagnostics_" prefixes
    resample_mask_to_image : resample the mask onto the image grid first
    **settings : overrides for ``DEFAULT_SETTINGS`` (passed to PyRadiomics)

    Returns
    -------
    pandas.DataFrame with a single row
    """
    image_sitk = read_image(image)
    if clip_range is not None:
        image_sitk = sitk.Clamp(image_sitk, lowerBound=clip_range[0], upperBound=clip_range[1])

    mask = read_image(segment)
    if resample_mask_to_image:
        mask = resample_to_reference(mask, image_sitk, interpolator=sitk.sitkNearestNeighbor)

    params = {**DEFAULT_SETTINGS, **settings, "label": int(label), "voxelBased": False}
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.disableAllFeatures()
    if feature_classes == "all":
        extractor.enableAllFeatures()
    else:
        for feature_class in feature_classes:
            extractor.enableFeatureClassByName(feature_class)

    features = extractor.execute(image_sitk, mask)

    image_name = _name(image).replace(".nii.gz", "")
    info = {
        "image_url": image if isinstance(image, str) else "loaded from memory",
        "segment_url": segment if isinstance(segment, str) else "loaded from memory",
        "image_name": _name(image),
        "segment_name": _name(segment).replace(image_name, ""),
        "segment_label": label,
        "comment": comment,
    }
    row = pd.DataFrame([{**info, **features}])

    if shorten_feature_names:
        row.columns = [c.replace("original_", "").replace("diagnostics_", "") for c in row.columns]
    return row


def extract_features_all_labels(image, segment, **kwargs):
    """
    Run :func:`extract_features` for every non-zero label in ``segment``.

    Keyword arguments are forwarded to :func:`extract_features`.
    Returns a DataFrame with one row per label.
    """
    labels, looks_like_segmentation = unique_labels(segment)
    if not looks_like_segmentation:
        warnings.warn(f"{_name(segment)} does not look like a valid segmentation file.")

    rows = [
        extract_features(image, segment, label=int(label), **kwargs)
        for label in labels
        if label != 0
    ]
    return pd.concat(rows, axis=0) if rows else pd.DataFrame()
