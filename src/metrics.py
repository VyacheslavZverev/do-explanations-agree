"""Agreement between two attribution maps.

Neither metric needs the maps rescaled to a common range. Spearman compares the
order of values, not the values; the IoU threshold is taken inside each map
separately. Normalising was therefore dropped rather than described as a
limitation - one fewer distortion between the methods and the numbers.
"""

import numpy as np
from scipy.stats import spearmanr

TOP_FRACTION = 0.10


def spearman_agreement(first: np.ndarray, second: np.ndarray) -> float:
    """Spearman rank correlation between two maps, over all pixels.

    Tied values get an average rank. This matters here: LIME is constant within
    a segment and half of the positive part of Integrated Gradients is exactly
    zero, so both carry large blocks of ties that pull the correlation towards
    zero on their own.
    """
    correlation, _ = spearmanr(first.ravel(), second.ravel())
    return float(correlation)


def top_mask(attribution: np.ndarray, fraction: float = TOP_FRACTION) -> np.ndarray:
    """Boolean mask of the most important pixels, by a threshold inside the map.

    A threshold keeps tied pixels together, so the mask can cover more or less
    than `fraction` of the image - LIME's segments are indivisible by design.
    Taking exactly that many pixels instead would split a segment at an
    arbitrary point. The real size is reported alongside the results.
    """
    threshold = np.quantile(attribution, 1.0 - fraction)
    return attribution >= threshold


def top_iou(first: np.ndarray, second: np.ndarray,
            fraction: float = TOP_FRACTION) -> tuple[float, float, float]:
    """IoU of the two maps' most important pixels, with both mask sizes."""
    left, right = top_mask(first, fraction), top_mask(second, fraction)
    union = np.logical_or(left, right).sum()
    intersection = np.logical_and(left, right).sum()
    iou = float(intersection / union) if union else float("nan")
    return iou, float(left.mean()), float(right.mean())


def chance_iou(share_first: float, share_second: float) -> float:
    """IoU two independent masks of these sizes would reach by chance.

    The usual floor of 0.053 assumes both masks cover exactly 10% of the image.
    LIME's do not: a threshold cannot split a segment, so its mask runs to 17%
    on some images. Comparing pairs against one shared floor would then flatter
    whichever pair happens to have the larger mask, so each pair gets its own.
    """
    overlap = share_first * share_second
    union = share_first + share_second - overlap
    return float(overlap / union) if union else float("nan")
