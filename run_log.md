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
| Randomness | none yet; inference is deterministic. A seed will be fixed before LIME. |

## Runs

| Date | Script | Input | Wall time | Output |
|---|---|---|---|---|
| 2026-09-10 | `fetch_images.py` | 12 search terms | ~1 min (incl. 28 s of rate-limit backoff) | `data/sources.csv`, 60 candidates |
| 2026-09-10 | `download_images.py` | `data/sources.csv` | ~1 min | `data/images/`, 60 files, 11.1 MB |
| 2026-09-10 | `download_images.py` (re-run) | same | ~30 s | 0 downloaded, 60 already present - resume path verified |
| 2026-09-10 | `classify.py` | 60 images | ~20 s | `results/candidates.csv`: confident 32, uncertain 12, excluded 16 |

## Notes

- `fetch_images.py` hit HTTP 429 on the first attempt and now waits between
  requests, retrying with doubling delays. The retry is visible in its output.
- Re-running `download_images.py` is safe and skips existing files.
