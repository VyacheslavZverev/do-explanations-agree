"""95% bootstrap confidence intervals for the numbers the README relies on.

Images are resampled with replacement 10,000 times and the statistic recomputed
each time; the interval is the middle 95% of those values. The bootstrap is used
rather than a textbook formula because correlations are bounded and, here,
long-tailed - zebra_03 sits at -0.57 - so a normal approximation would be
unsafe.

Differences are computed PAIRED: both quantities come from the same resampled
images. An image that is hard for one pair of methods tends to be hard for the
other, and treating them as independent would give an interval that is both
wider and wrong.

The interval covers sampling variability among images like these ones - found
by these twelve search terms on Wikimedia Commons. It says nothing about images
from anywhere else.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import attribution

RESAMPLES = 10_000
RAW = Path("results/results_raw.csv")
CONTROL = Path("results/resolution_control.csv")
OUT_FILE = Path("results/bootstrap_ci.csv")


def interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of one column of per-image values."""
    picks = rng.integers(0, len(values), size=(RESAMPLES, len(values)))
    means = values[picks].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def difference_interval(first: np.ndarray, second: np.ndarray,
                        rng: np.random.Generator) -> tuple[float, float]:
    """Interval for mean(first) - mean(second) from two independent groups of images."""
    a = first[rng.integers(0, len(first), size=(RESAMPLES, len(first)))].mean(axis=1)
    b = second[rng.integers(0, len(second), size=(RESAMPLES, len(second)))].mean(axis=1)
    return float(np.percentile(a - b, 2.5)), float(np.percentile(a - b, 97.5))


def main() -> None:
    rng = np.random.default_rng(attribution.RANDOM_SEED)
    raw = pd.read_csv(RAW)
    control = pd.read_csv(CONTROL).set_index("image_id").loc[raw["image_id"]].reset_index()
    rows = []

    def add(name: str, values: np.ndarray, low_high: tuple[float, float], n: int) -> None:
        rows.append({"quantity": name, "estimate": float(np.mean(values)),
                     "ci_low": low_high[0], "ci_high": low_high[1], "n": n})

    for pair in ("grad_cam_lime", "grad_cam_ig", "ig_lime"):
        spearman = raw[f"{pair}_spearman"].to_numpy()
        above = (raw[f"{pair}_iou"] - raw[f"{pair}_iou_chance"]).to_numpy()
        add(f"{pair} spearman", spearman, interval(spearman, rng), len(raw))
        add(f"{pair} iou above chance", above, interval(above, rng), len(raw))

    coarse = control["cam_ig_coarse_spearman"].to_numpy()
    add("grad_cam_ig spearman, IG coarsened", coarse, interval(coarse, rng), len(raw))

    # Paired: every quantity below is a per-image difference, resampled as one column.
    gain = coarse - control["cam_ig_spearman"].to_numpy()
    add("PAIRED gain from coarsening IG", gain, interval(gain, rng), len(raw))
    gap = coarse - raw["grad_cam_lime_spearman"].to_numpy()
    add("PAIRED coarsened IG minus LIME (vs Grad-CAM)", gap, interval(gap, rng), len(raw))

    # Unpaired: the two strata hold different images.
    for metric, column in (("spearman", "grad_cam_lime_spearman"),
                           ("iou above chance", None)):
        values = (raw["grad_cam_lime_iou"] - raw["grad_cam_lime_iou_chance"]
                  if column is None else raw[column])
        confident = values[raw["stratum"] == "confident"].to_numpy()
        uncertain = values[raw["stratum"] == "uncertain"].to_numpy()
        low, high = difference_interval(confident, uncertain, rng)   # one draw for both ends
        rows.append({"quantity": f"STRATA grad_cam_lime {metric}: confident minus uncertain",
                     "estimate": float(confident.mean() - uncertain.mean()),
                     "ci_low": low, "ci_high": high,
                     "n": f"{len(confident)}+{len(uncertain)}"})

    table = pd.DataFrame(rows)
    table[["estimate", "ci_low", "ci_high"]] = table[["estimate", "ci_low", "ci_high"]].round(4)
    table.to_csv(OUT_FILE, index=False)
    print(table.to_string(index=False))
    print(f"\n{RESAMPLES} resamples, seed {attribution.RANDOM_SEED} -> {OUT_FILE}")


if __name__ == "__main__":
    main()
