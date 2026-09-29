# Run log

One entry per experiment run: date, environment, wall-clock time, what was
produced. Library versions are pinned in `requirements.txt` and
`requirements-lock.txt` and were identical for every run below.

## Environment

| | |
|---|---|
| Machine | local, Windows 11 Pro (10.0.26200) |
| Python | 3.12.10, virtual environment in `.venv/` |
| Device | CPU only (`torch==2.14.0+cpu`) |
| Model | ResNet-18, `ResNet18_Weights.IMAGENET1K_V1` (`resnet18-f37072fd.pth`) |
| Randomness | seed 0, used by LIME. Two runs with the same seed are bit-identical; everything else is deterministic. |

## Runs

| Date | Script | Input | Wall time | Output |
|---|---|---|---|---|
| 2026-09-10 | `fetch_images.py` | 12 search terms | ~1 min (incl. 28 s of rate-limit backoff) | `data/sources.csv`, 60 candidates |
| 2026-09-10 | `download_images.py` | `data/sources.csv` | ~1 min | `data/images/`, 60 files, 11.1 MB |
| 2026-09-10 | `download_images.py` (re-run) | same | ~30 s | 0 downloaded, 60 already present - resume path verified |
| 2026-09-10 | `classify.py` | 60 images | ~20 s | `results/candidates.csv`: confident 32, uncertain 12, excluded 16 |
| 2026-09-14 | `run_experiment.py` | 44 images | 14.3 min (19.4 s/image) | `results/results_raw.csv`, one row per image |
| 2026-09-14 | `summarise.py` | 44 rows | <1 s | `results/summary.csv` |
| 2026-09-14 | `resolution_control.py` | 44 images | ~3 min | `results/resolution_control.csv` |
| 2026-09-15 | `figures.py` | results tables | ~40 s | six PNGs at 300 dpi in `figures/ru` and `figures/en` |
| 2026-09-29 | `check_ig_steps.py` | 44 images x 3 step counts | ~7 min | `results/ig_steps_check.csv` |
| 2026-09-29 | `bootstrap.py` | existing result tables, 10,000 resamples, seed 0 | ~2 s | `results/bootstrap_ci.csv` |

## Notes

- `fetch_images.py` hit HTTP 429 on the first attempt and now waits between
  requests, retrying with doubling delays. The retry is visible in its output.
- Re-running `download_images.py` is safe and skips existing files.
