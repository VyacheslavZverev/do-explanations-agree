"""Is the Grad-CAM/LIME agreement about the network, or about spatial scale?

Grad-CAM and LIME are the two coarse methods, and they are the two that agree.
This re-measures Grad-CAM against Integrated Gradients after forcing IG onto
Grad-CAM's 7x7 grid. A jump would mean the disagreement was resolution.

Coarsening random noise the same way is the control: if that also jumps, the
effect belongs to the procedure rather than to the maps.
"""

import csv
from pathlib import Path

import numpy as np
from torchvision import transforms

import attribution
import metrics
import model

CANDIDATES = Path("results/candidates.csv")
OUT_FILE = Path("results/resolution_control.csv")
COLUMNS = ["image_id", "stratum",
           "cam_ig_spearman", "cam_ig_coarse_spearman",
           "cam_ig_iou", "cam_ig_coarse_iou",
           "cam_noise_spearman", "cam_noise_coarse_spearman"]


def main() -> None:
    rows = [r for r in csv.DictReader(CANDIDATES.open(encoding="utf-8"))
            if r["stratum"] in ("confident", "uncertain")]
    net = model.load_model()
    generator = np.random.default_rng(attribution.RANDOM_SEED)
    results = []

    for index, row in enumerate(rows, 1):
        path = next(Path("data/images").glob(f"{row['image_id']}.*"))
        tensor = model.preprocess()(model.load_image(path))
        target = int(row["predicted_index"])

        cam = attribution.grad_cam(net, tensor, target)
        ig = attribution.positive_part(
            attribution.integrated_gradients(net, tensor, target))
        noise = generator.random(cam.shape)

        results.append({
            "image_id": row["image_id"],
            "stratum": row["stratum"],
            "cam_ig_spearman": round(metrics.spearman_agreement(cam, ig), 6),
            "cam_ig_coarse_spearman": round(
                metrics.spearman_agreement(cam, attribution.coarsen(ig)), 6),
            "cam_ig_iou": round(metrics.top_iou(cam, ig)[0], 6),
            "cam_ig_coarse_iou": round(
                metrics.top_iou(cam, attribution.coarsen(ig))[0], 6),
            "cam_noise_spearman": round(metrics.spearman_agreement(cam, noise), 6),
            "cam_noise_coarse_spearman": round(
                metrics.spearman_agreement(cam, attribution.coarsen(noise)), 6),
        })
        print(f"  [{index}/{len(rows)}] {row['image_id']:22s} "
              f"rho {results[-1]['cam_ig_spearman']:+.3f} -> "
              f"{results[-1]['cam_ig_coarse_spearman']:+.3f}")

    with OUT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n{len(results)} images -> {OUT_FILE}")


if __name__ == "__main__":
    main()
