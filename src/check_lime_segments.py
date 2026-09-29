"""Does LIME's segment size move its agreement with Grad-CAM the way the
resolution explanation predicts?

PREDICTION - written and committed before this was run.
The study's main claim is that most disagreement between the methods is a
difference in spatial resolution. If that is right, making LIME coarser should
bring it closer to Grad-CAM (7x7) and making it finer should push it away:

    mean Grad-CAM vs LIME agreement:  40 segments  >  80  >  160

KNOWN CONFOUND, stated in advance. Coarser LIME also means fewer distinct
values and larger blocks of tied ranks, and ties by themselves pull Spearman
towards zero. For Spearman, coarsening LIME therefore pushes in both directions
at once. A rise at 40 segments would have happened despite that; the absence of
a rise would not separate resolution from ties. IoU above chance is less exposed
to ties and is reported alongside.

80 segments is the setting used in the main run and is recomputed here, so that
all three settings come from the same script. Results are appended one row at a
time and an interrupted run resumes.
"""

import csv
from pathlib import Path

import numpy as np
from skimage.segmentation import slic
from torchvision import transforms

import attribution
import metrics
import model

SEGMENT_SETTINGS = (40, 80, 160)
CANDIDATES = Path("results/candidates.csv")
OUT_FILE = Path("results/lime_segments_check.csv")
COLUMNS = ["image_id", "stratum", "segments_requested", "segments_produced",
           "cam_lime_spearman", "cam_lime_iou", "cam_lime_iou_chance",
           "cam_lime_iou_above_chance", "lime_mask_share"]
CROP = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])


def already_done() -> set[tuple[str, int]]:
    """(image, setting) pairs already in the output file."""
    if not OUT_FILE.exists():
        return set()
    with OUT_FILE.open(encoding="utf-8") as handle:
        return {(r["image_id"], int(r["segments_requested"]))
                for r in csv.DictReader(handle)}


def main() -> None:
    with CANDIDATES.open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["stratum"] in ("confident", "uncertain")]
    done = already_done()
    todo = [(r, s) for r in rows for s in SEGMENT_SETTINGS
            if (r["image_id"], s) not in done]
    print(f"{len(done)} already computed | {len(todo)} to do", flush=True)

    net = model.load_model()
    is_new = not OUT_FILE.exists()
    with OUT_FILE.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        for index, (row, segments) in enumerate(todo, 1):
            path = next(Path("data/images").glob(f"{row['image_id']}.*"))
            picture = model.load_image(path)
            tensor = model.preprocess()(picture)
            array = np.array(CROP(picture)).astype(np.float32) / 255.0
            target = int(row["predicted_index"])

            cam = attribution.grad_cam(net, tensor, target)
            lime = attribution.positive_part(
                attribution.lime_map(net, array, target, segments=segments))
            produced = len(np.unique(slic(
                array.astype(np.double), n_segments=segments,
                compactness=attribution.SLIC_COMPACTNESS, start_label=0)))
            iou, share_cam, share_lime = metrics.top_iou(cam, lime)
            chance = metrics.chance_iou(share_cam, share_lime)

            writer.writerow({
                "image_id": row["image_id"],
                "stratum": row["stratum"],
                "segments_requested": segments,
                "segments_produced": produced,
                "cam_lime_spearman": round(metrics.spearman_agreement(cam, lime), 6),
                "cam_lime_iou": round(iou, 6),
                "cam_lime_iou_chance": round(chance, 6),
                "cam_lime_iou_above_chance": round(iou - chance, 6),
                "lime_mask_share": round(share_lime, 6),
            })
            handle.flush()          # survive an interrupted run
            print(f"  [{index}/{len(todo)}] {row['image_id']:22s} "
                  f"{segments:3d} segments", flush=True)


if __name__ == "__main__":
    main()
