"""Robustness check: is 64 integration steps enough for Integrated Gradients?

Two questions, answered separately.

1. Is the integral computed accurately? IG's completeness property says the
   attributions must sum to F(input) - F(baseline). How far the sum misses is a
   direct measure of the approximation error; under 5% is the usual guide.

2. Does the step count change the study's conclusion? Agreement with Grad-CAM
   after coarsening is recomputed at each step count. An accurate integral
   matters less than whether the reported number moves.
"""

import csv
from pathlib import Path

import torch

import attribution
import metrics
import model

STEPS = (32, 64, 128)
CANDIDATES = Path("results/candidates.csv")
OUT_FILE = Path("results/ig_steps_check.csv")


def target_logit(net, image: torch.Tensor, target: int) -> float:
    """The network output IG attributes: the raw logit of the target class."""
    with torch.no_grad():
        return float(net(image.unsqueeze(0))[0, target])


def main() -> None:
    rows = [r for r in csv.DictReader(CANDIDATES.open(encoding="utf-8"))
            if r["stratum"] in ("confident", "uncertain")]
    net = model.load_model()
    results = []

    for index, row in enumerate(rows, 1):
        path = next(Path("data/images").glob(f"{row['image_id']}.*"))
        tensor = model.preprocess()(model.load_image(path))
        target = int(row["predicted_index"])
        cam = attribution.grad_cam(net, tensor, target)
        gap = (target_logit(net, tensor, target)
               - target_logit(net, attribution.baseline_like(tensor, "black"), target))

        record = {"image_id": row["image_id"], "stratum": row["stratum"]}
        for steps in STEPS:
            signed = attribution.integrated_gradients(net, tensor, target, steps=steps)
            record[f"completeness_error_{steps}"] = round(abs(signed.sum() - gap) / abs(gap), 6)
            record[f"cam_ig_coarse_{steps}"] = round(metrics.spearman_agreement(
                cam, attribution.coarsen(attribution.positive_part(signed))), 6)
        results.append(record)
        print(f"  [{index}/{len(rows)}] {row['image_id']:22s} "
              + "  ".join(f"err@{s}={record[f'completeness_error_{s}']:.3f}" for s in STEPS))

    with OUT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n{len(results)} images -> {OUT_FILE}")


if __name__ == "__main__":
    main()
