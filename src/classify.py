"""Step 4 - classify every candidate and apply the confidence filter.

Writes results/candidates.csv: one row per downloaded image, including the ones
that fail the filter. Rejected images are kept in the record on purpose - the
selection rule has to be inspectable, not just its outcome.
"""

import collections
import csv
from pathlib import Path

import torch

import model

# Two strata are analysed, not one. The confident stratum follows the original
# selection rule. The uncertain stratum exists because a >=90% filter keeps only
# cases the network already finds easy, which would bias the study towards
# agreement; measuring both turns the result into a dependency, not one number.
CONFIDENT_MIN = 0.90
UNCERTAIN_MIN, UNCERTAIN_MAX = 0.40, 0.70
SOURCES = Path("data/sources.csv")
EXCLUSIONS = Path("data/exclusions.csv")
IMAGE_DIR = Path("data/images")
OUT_FILE = Path("results/candidates.csv")

COLUMNS = ["image_id", "group", "search_term", "expected_class",
           "predicted_class", "predicted_index", "confidence",
           "matches_expected", "stratum", "exclusion_reason"]


def stratum(confidence: float, excluded_because: str) -> tuple[str, str]:
    """Return the analysis stratum for one image and the reason if it is excluded.

    Manual exclusions come from data/exclusions.csv and always carry a written
    reason, so that the selection rule can be read rather than inferred.
    """
    if excluded_because:
        return "excluded", excluded_because
    if confidence >= CONFIDENT_MIN:
        return "confident", ""
    if UNCERTAIN_MIN <= confidence < UNCERTAIN_MAX:
        return "uncertain", ""
    return "excluded", "confidence falls in neither stratum"


def find_image(image_id: str) -> Path:
    """Locate a downloaded file by id, whatever extension it was saved with."""
    matches = sorted(IMAGE_DIR.glob(f"{image_id}.*"))
    if not matches:
        raise FileNotFoundError(f"no image on disk for {image_id}")
    return matches[0]


def classify(net, transform, names, path: Path) -> tuple[int, str, float]:
    """Return the top class index, its name and the model's confidence."""
    batch = transform(model.load_image(path)).unsqueeze(0)
    with torch.no_grad():
        probabilities = net(batch).softmax(dim=1)[0]
    index = int(probabilities.argmax())
    return index, names[index], float(probabilities[index])


def main() -> None:
    net, transform, names = model.load_model(), model.preprocess(), model.class_names()
    excluded = {r["image_id"]: r["reason"]
                for r in csv.DictReader(EXCLUSIONS.open(encoding="utf-8"))}
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for source in csv.DictReader(SOURCES.open(encoding="utf-8")):
        index, predicted, confidence = classify(
            net, transform, names, find_image(source["image_id"]))
        matches = predicted.lower() == source["expected_class"].lower()
        layer, reason = stratum(confidence, excluded.get(source["image_id"], ""))
        rows.append({
            "image_id": source["image_id"],
            "group": source["group"],
            "search_term": source["search_term"],
            "expected_class": source["expected_class"],
            "predicted_class": predicted,
            "predicted_index": index,
            "confidence": round(confidence, 6),
            "matches_expected": int(matches),
            "stratum": layer,
            "exclusion_reason": reason,
        })

    with OUT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    counts = collections.Counter(r["stratum"] for r in rows)
    print(f"{len(rows)} classified -> {OUT_FILE}")
    for name in ("confident", "uncertain", "excluded"):
        print(f"  {name:10s} {counts[name]:3d}")


if __name__ == "__main__":
    main()
