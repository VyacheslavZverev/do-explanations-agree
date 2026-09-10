# Do explanation methods agree?

Comparing the agreement of three post-hoc interpretability methods —
**Grad-CAM**, **Integrated Gradients** and **LIME** — applied to the same
pretrained **ResNet-18** and the same images.

**Hypothesis under test:** three methods applied to one model and one image
produce consistent explanations.

**Agreement is measured by**
- Spearman rank correlation between full attribution maps;
- IoU of the top 10% most important pixels.

**Status:** work in progress. Nothing here is a result yet.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | experiment code |
| `data/` | image list and download script (images themselves are not committed) |
| `results/results_raw.csv` | one row per image — raw, unaggregated |
| `results/summary.csv` | aggregated statistics |
| `figures/en/`, `figures/ru/` | figures, English and Russian captions |
| `LIMITATIONS.md` | what this study does **not** show |
| `run_log.md` | date, environment, runtime and library versions of every run |
| `NOTES.md` | working notes (Russian) |

## Reproducing

To be written once the pipeline runs end to end.
