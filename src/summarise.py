"""Aggregate results/results_raw.csv into results/summary.csv.

Reports each pair of methods separately for each confidence stratum, and for
both strata together. The column that answers the research question is
`iou_above_chance`: an IoU means nothing until its own chance floor is
subtracted, and that floor differs between pairs.
"""

import itertools
from pathlib import Path

import pandas as pd

RAW = Path("results/results_raw.csv")
OUT = Path("results/summary.csv")
METHODS = ("grad_cam", "ig", "lime")

# Readable names for the report. Deriving them from the column names does not
# work: "grad_cam" already contains an underscore.
LABELS = {"grad_cam": "Grad-CAM", "ig": "IG", "lime": "LIME"}


def describe(frame: pd.DataFrame, stratum: str, pair: str, label: str) -> dict:
    """One summary row: how well these two methods agreed on these images."""
    spearman = frame[f"{pair}_spearman"]
    iou = frame[f"{pair}_iou"]
    chance = frame[f"{pair}_iou_chance"]
    return {
        "stratum": stratum,
        "pair": label,
        "n": len(frame),
        "spearman_mean": spearman.mean(),
        "spearman_median": spearman.median(),
        "spearman_sd": spearman.std(),
        "spearman_min": spearman.min(),
        "spearman_max": spearman.max(),
        "iou_mean": iou.mean(),
        "iou_median": iou.median(),
        "iou_chance_mean": chance.mean(),
        "iou_above_chance": (iou - chance).mean(),
    }


def main() -> None:
    raw = pd.read_csv(RAW)
    pairs = [(f"{a}_{b}", f"{LABELS[a]} vs {LABELS[b]}")
             for a, b in itertools.combinations(METHODS, 2)]

    rows = []
    for stratum in ("confident", "uncertain"):
        subset = raw[raw["stratum"] == stratum]
        if not subset.empty:
            rows += [describe(subset, stratum, pair, label) for pair, label in pairs]
    rows += [describe(raw, "both", pair, label) for pair, label in pairs]

    summary = pd.DataFrame(rows).round(4)
    summary.to_csv(OUT, index=False)
    print(summary.to_string(index=False))
    print(f"\n{len(raw)} images -> {OUT}")


if __name__ == "__main__":
    main()
