"""Figures for the report, in a Russian and an English version.

Nothing is encoded in colour: bars are one grey, points are black, the chance
level is a dashed line. The figures have to survive a projector and a
black-and-white printer, so shape and position carry the meaning.
"""

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW = Path("results/results_raw.csv")
CONTROL = Path("results/resolution_control.csv")
METHODS = ("grad_cam", "ig", "lime")
PAIRS = [f"{a}_{b}" for a, b in itertools.combinations(METHODS, 2)]
BAR_GREY = "0.75"
DPI = 300

TEXT = {
    "en": {"pairs": ["Grad-CAM\nvs LIME", "Grad-CAM\nvs IG", "IG\nvs LIME"],
           "spearman": "Spearman rank correlation",
           "iou": "IoU of the top 10% of pixels",
           "chance": "chance level",
           "caption": "one point per image, n = 44",
           "before": "as computed", "after": "coarsened\nto 7x7",
           "real": "Grad-CAM vs Integrated Gradients",
           "control": "CONTROL: Grad-CAM vs random noise",
           "mean": "mean", "lines": "one line per image, n = 44",
           "input": "input\nnetwork says: {cls} ({conf:.0%})",
           "maps_caption": "three methods, one network, one image, one explained class"},
    "ru": {"pairs": ["Grad-CAM\nи LIME", "Grad-CAM\nи IG", "IG\nи LIME"],
           "spearman": "Ранговая корреляция Спирмена",
           "iou": "IoU для 10 % значимых пикселей",
           "chance": "уровень случайности",
           "caption": "одна точка — одно изображение, n = 44",
           "before": "как есть", "after": "огрублённая\nдо 7×7",
           "real": "Grad-CAM и Integrated Gradients",
           "control": "КОНТРОЛЬ: Grad-CAM и случайный шум",
           "mean": "среднее", "lines": "одна линия — одно изображение, n = 44",
           "input": "изображение\nсеть отвечает: {cls} ({conf:.0%})",
           "maps_caption": "три метода, одна сеть, одно изображение, один объясняемый класс"},
}

ORDER = ["grad_cam_lime", "grad_cam_ig", "ig_lime"]


def scatter_points(axis, position: int, values: np.ndarray, seed: int = 0) -> None:
    """Draw one dot per image, spread sideways so that they do not overlap."""
    offsets = np.random.default_rng(seed).uniform(-0.17, 0.17, len(values))
    axis.scatter(position + offsets, values, s=14, color="black",
                 alpha=0.65, zorder=3, linewidths=0)


def panel(axis, raw: pd.DataFrame, metric: str, words: dict, chance: bool) -> None:
    """One panel: mean bars, per-image points, and optionally the chance level."""
    means = [raw[f"{pair}_{metric}"].mean() for pair in ORDER]
    axis.bar(range(len(ORDER)), means, color=BAR_GREY, edgecolor="black",
             width=0.6, zorder=1)
    for position, pair in enumerate(ORDER):
        scatter_points(axis, position, raw[f"{pair}_{metric}"].to_numpy())
        axis.annotate(f"{means[position]:.2f}", (position, means[position]),
                      xytext=(0, 4), textcoords="offset points", ha="center",
                      fontsize=14, fontweight="bold", zorder=5,
                      bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                edgecolor="none", alpha=0.85))
        if chance:
            level = raw[f"{pair}_iou_chance"].mean()
            axis.plot([position - 0.35, position + 0.35], [level, level],
                      color="black", linestyle="--", linewidth=1.6, zorder=4)
    axis.set_xticks(range(len(ORDER)))
    axis.set_xticklabels(words["pairs"], fontsize=14)
    axis.tick_params(axis="y", labelsize=14)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def build(language: str) -> Path:
    words = TEXT[language]
    raw = pd.read_csv(RAW)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))

    panel(axes[0], raw, "spearman", words, chance=False)
    axes[0].set_ylabel(words["spearman"], fontsize=15)
    panel(axes[1], raw, "iou", words, chance=True)
    axes[1].set_ylabel(words["iou"], fontsize=15)
    axes[1].plot([], [], color="black", linestyle="--", linewidth=1.6,
                 label=words["chance"])
    axes[1].legend(fontsize=13, frameon=False, loc="upper right")

    figure.text(0.5, 0.015, words["caption"], ha="center", fontsize=13)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    out = Path(f"figures/{language}/fig2_agreement.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=DPI)
    plt.close(figure)
    return out




def slope_panel(axis, before: np.ndarray, after: np.ndarray, words: dict,
                title: str) -> None:
    """One line per image, from its value before coarsening to after it.

    The mean alone would hide the fact that the shift happens on nearly every
    image individually, which is the whole point of the comparison.
    """
    for start, end in zip(before, after):
        axis.plot([0, 1], [start, end], color="black", alpha=0.25, linewidth=0.9)
    axis.plot([0, 1], [before.mean(), after.mean()], color="black",
              linewidth=3.2, marker="o", markersize=8, label=words["mean"],
              zorder=5)
    axis.axhline(0, color="black", linewidth=0.8, linestyle=":")
    axis.set_xticks([0, 1])
    axis.set_xticklabels([words["before"], words["after"]], fontsize=14)
    axis.set_xlim(-0.25, 1.25)
    axis.set_ylim(-0.75, 1.0)
    axis.tick_params(axis="y", labelsize=14)
    axis.set_title(title, fontsize=15)
    axis.spines[["top", "right"]].set_visible(False)


def build_resolution(language: str) -> Path:
    words = TEXT[language]
    control = pd.read_csv(CONTROL)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)

    slope_panel(axes[0], control["cam_ig_spearman"].to_numpy(),
                control["cam_ig_coarse_spearman"].to_numpy(), words, words["real"])
    slope_panel(axes[1], control["cam_noise_spearman"].to_numpy(),
                control["cam_noise_coarse_spearman"].to_numpy(), words,
                words["control"])
    axes[0].set_ylabel(words["spearman"], fontsize=15)
    axes[0].legend(fontsize=13, frameon=False, loc="upper left")

    figure.text(0.5, 0.015, words["lines"], ha="center", fontsize=13)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    out = Path(f"figures/{language}/fig3_resolution.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=DPI)
    plt.close(figure)
    return out


def build_maps(language: str, image_id: str = "school_bus_01") -> Path:
    """Figure 1: the same decision explained three ways.

    `inferno` is used rather than a rainbow scale because its lightness
    increases monotonically, so the panels still read correctly once the report
    is printed in black and white.
    """
    import csv

    import attribution
    import model
    from torchvision import transforms

    words = TEXT[language]
    with open("results/candidates.csv", encoding="utf-8") as handle:
        entry = next(r for r in csv.DictReader(handle) if r["image_id"] == image_id)
    target = int(entry["predicted_index"])

    net = model.load_model()
    picture = model.load_image(next(Path("data/images").glob(f"{image_id}.*")))
    tensor = model.preprocess()(picture)
    crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    array = np.array(crop(picture)).astype(np.float32) / 255.0

    panels = [
        (crop(picture), words["input"].format(cls=entry["predicted_class"],
                                              conf=float(entry["confidence"])), None),
        (attribution.grad_cam(net, tensor, target), "Grad-CAM", "inferno"),
        (attribution.positive_part(
            attribution.integrated_gradients(net, tensor, target)),
         "Integrated Gradients", "inferno"),
        (attribution.positive_part(attribution.lime_map(net, array, target)),
         "LIME", "inferno"),
    ]

    figure, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    for axis, (data, title, cmap) in zip(axes, panels):
        axis.imshow(data) if cmap is None else axis.imshow(data, cmap=cmap)
        axis.set_title(title, fontsize=15)
        axis.axis("off")
    figure.text(0.5, 0.02, words["maps_caption"], ha="center", fontsize=13)
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    out = Path(f"figures/{language}/fig1_maps.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=DPI)
    plt.close(figure)
    return out


if __name__ == "__main__":
    for language in ("ru", "en"):
        print("wrote", build(language))
        print("wrote", build_resolution(language))
        print("wrote", build_maps(language))
