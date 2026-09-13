"""Compute all three attribution maps and their pairwise agreement.

Writes one row per image to results/results_raw.csv, appending as it goes and
skipping images already present, so an interrupted run continues where it
stopped. Pass a number to process only that many images:

    python src/run_experiment.py 5
"""

import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np
from torchvision import transforms

import attribution
import metrics
import model

CANDIDATES = Path("results/candidates.csv")
OUT_FILE = Path("results/results_raw.csv")
ANALYSED_STRATA = ("confident", "uncertain")
METHODS = ("grad_cam", "ig", "lime")

COLUMNS = (["image_id", "group", "stratum", "predicted_class", "predicted_index",
            "confidence"]
           + [f"{a}_{b}_{metric}"
              for a, b in itertools.combinations(METHODS, 2)
              for metric in ("spearman", "iou")]
           + [f"mask_share_{name}" for name in METHODS]
           + [f"{a}_{b}_iou_chance" for a, b in itertools.combinations(METHODS, 2)]
           + ["seconds"])

CROP = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])


def already_done() -> set[str]:
    """Image ids already present in the output file."""
    if not OUT_FILE.exists():
        return set()
    with OUT_FILE.open(encoding="utf-8") as handle:
        return {row["image_id"] for row in csv.DictReader(handle)}


def attribution_maps(net, image_id: str, target: int) -> dict[str, np.ndarray]:
    """All three maps for one image, reduced to evidence for the target class."""
    path = next(Path("data/images").glob(f"{image_id}.*"))
    picture = model.load_image(path)
    tensor = model.preprocess()(picture)
    array = np.array(CROP(picture)).astype(np.float32) / 255.0
    return {
        "grad_cam": attribution.grad_cam(net, tensor, target),
        "ig": attribution.positive_part(
            attribution.integrated_gradients(net, tensor, target)),
        "lime": attribution.positive_part(
            attribution.lime_map(net, array, target)),
    }


def measure(row: dict, maps: dict[str, np.ndarray], seconds: float) -> dict:
    """Assemble one output row from three maps."""
    result = {key: row[key] for key in
              ("image_id", "group", "stratum", "predicted_class",
               "predicted_index", "confidence")}
    for first, second in itertools.combinations(METHODS, 2):
        result[f"{first}_{second}_spearman"] = round(
            metrics.spearman_agreement(maps[first], maps[second]), 6)
        result[f"{first}_{second}_iou"] = round(
            metrics.top_iou(maps[first], maps[second])[0], 6)
    shares = {name: float(metrics.top_mask(maps[name]).mean()) for name in METHODS}
    for name, share in shares.items():
        result[f"mask_share_{name}"] = round(share, 6)
    for first, second in itertools.combinations(METHODS, 2):
        result[f"{first}_{second}_iou_chance"] = round(
            metrics.chance_iou(shares[first], shares[second]), 6)
    result["seconds"] = round(seconds, 1)
    return result


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    with CANDIDATES.open(encoding="utf-8") as handle:
        todo = [row for row in csv.DictReader(handle)
                if row["stratum"] in ANALYSED_STRATA]

    done = already_done()
    todo = [row for row in todo if row["image_id"] not in done][:limit]
    print(f"{len(done)} already computed | {len(todo)} to do now")

    net = model.load_model()
    is_new_file = not OUT_FILE.exists()
    with OUT_FILE.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if is_new_file:
            writer.writeheader()
        for index, row in enumerate(todo, 1):
            start = time.time()
            maps = attribution_maps(net, row["image_id"], int(row["predicted_index"]))
            record = measure(row, maps, time.time() - start)
            writer.writerow(record)
            handle.flush()          # survive an interrupted run
            print(f"  [{index}/{len(todo)}] {row['image_id']:22s} "
                  f"{record['seconds']:5.1f}s  "
                  f"rho(cam,lime)={record['grad_cam_lime_spearman']:+.3f}")


if __name__ == "__main__":
    main()
