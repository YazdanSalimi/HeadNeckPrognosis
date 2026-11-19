"""
Image and segmentation helpers (SimpleITK) used by the pipeline scripts.

All functions accept either a file path or an already-loaded ``sitk.Image``.
"""

import numpy as np
import SimpleITK as sitk


# --------------------------------------------------------------------------- #
# Reading / geometry
# --------------------------------------------------------------------------- #
def read_image(image, pixel_type=sitk.sitkUnknown):
    """Return a ``sitk.Image``; read it from disk if ``image`` is a path."""
    if isinstance(image, str):
        return sitk.ReadImage(image, pixel_type)
    return image


def copy_geometry(reference_image, image):
    """Copy origin, spacing and direction from ``reference_image`` to ``image``."""
    reference_image = read_image(reference_image)
    image.SetOrigin(reference_image.GetOrigin())
    image.SetSpacing(reference_image.GetSpacing())
    image.SetDirection(reference_image.GetDirection())
    return image


def resample_to_spacing(image, target_spacing, interpolator=sitk.sitkNearestNeighbor):
    """
    Resample ``image`` to a new voxel spacing (in mm), keeping origin and direction.

    Parameters
    ----------
    image : str or sitk.Image
    target_spacing : sequence of float, e.g. ``(1, 1, 1)``
    interpolator : SimpleITK interpolator, default nearest neighbour (for masks)
    """
    image = read_image(image, sitk.sitkFloat64)
    new_size = [
        int(round(size * spacing / target))
        for size, spacing, target in zip(image.GetSize(), image.GetSpacing(), target_spacing)
    ]
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        tuple(target_spacing),
        image.GetDirection(),
    )


def resample_to_reference(image, reference_image, interpolator=sitk.sitkLinear, default_value=0):
    """Resample ``image`` onto the voxel grid (size/spacing/origin/direction) of ``reference_image``."""
    image = read_image(image)
    reference_image = read_image(reference_image)
    return sitk.Resample(image, reference_image, sitk.Transform(), interpolator, default_value)


# --------------------------------------------------------------------------- #
# Segmentation helpers
# --------------------------------------------------------------------------- #
def keep_inside_mask(segment, mask, interpolator=sitk.sitkLinear):
    """
    Set every voxel of ``segment`` that lies outside ``mask`` (mask == 0) to 0.

    Used to clip an expanded GTV to the patient's body contour. The mask is
    first resampled onto the segment grid (a no-op if both share the same grid).
    """
    segment = read_image(segment)
    mask = resample_to_reference(mask, segment, interpolator=interpolator)

    segment_array = sitk.GetArrayFromImage(segment)
    segment_array[sitk.GetArrayFromImage(mask) == 0] = 0

    clipped = sitk.GetImageFromArray(segment_array)
    clipped.CopyInformation(segment)
    return clipped


def unique_labels(segment, max_labels=10):
    """
    Return ``(unique_values, looks_like_segmentation)``.

    ``looks_like_segmentation`` is True when the image has fewer than
    ``max_labels`` distinct values and all of them are integers.
    """
    values = np.unique(sitk.GetArrayFromImage(read_image(segment)))
    looks_like_segmentation = len(values) < max_labels and all(int(v) == v for v in values)
    return values, looks_like_segmentation


def binary_label(segment, label=1):
    """Return a binary image (uint8 0/1) that is 1 where ``segment == label``."""
    segment = read_image(segment, sitk.sitkUInt8)
    binary = sitk.GetImageFromArray((sitk.GetArrayFromImage(segment) == label).astype(np.uint8))
    return copy_geometry(segment, binary)


def segment_volume_ml(segment, label=1):
    """Volume of ``label`` in millilitres (1 ml = 1000 mm^3)."""
    segment = read_image(segment)
    n_voxels = int(sitk.GetArrayFromImage(binary_label(segment, label)).sum())
    return n_voxels * float(np.prod(segment.GetSpacing())) / 1e3


def segment_to_boxes(segment, min_object_volume_ml=0.001, margin_mm=0):
    """
    Replace every connected object of a segmentation by its bounding box.

    Objects with a volume <= ``min_object_volume_ml`` are discarded. Each box
    is optionally enlarged by ``margin_mm`` on every side (clipped to the image).

    Returns a label image with the same geometry as ``segment``: box ``k`` has
    value ``k`` (where boxes overlap, the higher label wins).
    """
    segment = sitk.Cast(read_image(segment, sitk.sitkUInt8), sitk.sitkUInt8)

    boxes = sitk.Image(segment.GetSize(), segment.GetPixelID())
    boxes.CopyInformation(segment)

    # Label connected objects and drop the small ones.
    components = sitk.ConnectedComponent(segment)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(components)

    dim = segment.GetDimension()
    margin_voxels = [int(round(margin_mm / s)) for s in segment.GetSpacing()]

    box_label = 0
    for label in sorted(stats.GetLabels()):
        if stats.GetPhysicalSize(label) <= min_object_volume_ml * 1e3:
            continue
        box_label += 1

        bbox = stats.GetBoundingBox(label)  # (start_x, start_y, start_z, size_x, size_y, size_z)
        start = [max(0, bbox[d] - margin_voxels[d]) for d in range(dim)]
        stop = [min(segment.GetSize()[d], bbox[d] + bbox[dim + d] + margin_voxels[d]) for d in range(dim)]

        boxes[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = box_label

    return boxes


def binarize(image):
    """Return a uint8 image that is 1 where ``image > 0`` and 0 elsewhere."""
    return sitk.Cast(read_image(image) > 0, sitk.sitkUInt8)
