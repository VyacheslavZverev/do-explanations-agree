# Do explanation methods agree?

Three post-hoc interpretability methods — **Grad-CAM**, **Integrated Gradients**
and **LIME** — are applied to the same pretrained **ResNet-18**, the same image,
and the same predicted class. If they disagree, at most one of them can be
describing what the network actually did.

**Hypothesis under test:** the three methods produce consistent explanations.

**Status:** 44 images processed. Results below are preliminary — one
architecture, one seed, a small sample.

## Result

Measured at the resolution each method natively produces, the hypothesis fails.
Two of the three methods agree; the third is close to unrelated to both.

| Pair | Spearman | IoU | chance IoU |
|---|---|---|---|
| Grad-CAM vs LIME | **0.49** | 0.26 | 0.057 |
| Grad-CAM vs Integrated Gradients | 0.06 | 0.10 | 0.053 |
| Integrated Gradients vs LIME | 0.04 | 0.09 | 0.057 |

But the two methods that agree are also the two that are **coarse**: Grad-CAM is
a 7×7 grid stretched to full size, LIME is constant over ~56 segments, while
Integrated Gradients varies pixel by pixel. So a competing explanation is that
the agreement is about spatial scale, not about the network.

That explanation was tested by averaging the Integrated Gradients map onto
Grad-CAM's 7×7 grid and measuring again:

| Comparison | Spearman |
|---|---|
| Grad-CAM vs Integrated Gradients, as computed | 0.06 |
| Grad-CAM vs Integrated Gradients, coarsened to 7×7 | **0.47** |
| Grad-CAM vs LIME (for reference) | 0.49 |
| *control:* Grad-CAM vs random noise | 0.00 |
| *control:* Grad-CAM vs coarsened random noise | 0.01 |

The correlation rose on 38 of 44 images, by 0.42 on average. Coarsening random
noise does **not** produce the same effect, so the jump is not an artefact of
the procedure — though coarsening does inflate the spread of per-image values
(control SD 0.161 against 0.004), which is why only the 44-image mean is quoted.

**So the disagreement was largely a difference of resolution, not of substance.**
Compared at a common scale, Integrated Gradients agrees with Grad-CAM about as
well as LIME does. This is a stronger claim than the raw table above, and a less
comfortable one: it means a study that had stopped at the first table would have
reported a real-looking negative result that was mostly an artefact of how the
maps are rendered.

Six images moved the other way, three of them strongly — `zebra_03` reaches
−0.57 after coarsening. Agreement is not uniform, and the averages hide that.

## What is compared

| Method | Looks inside the network? | Native resolution | Signed? |
|---|---|---|---|
| Grad-CAM | yes — gradients at the last residual block | 7×7, upsampled to 224×224 | no, ReLU by construction |
| Integrated Gradients | yes — gradients along a path from a baseline | 224×224 | yes |
| LIME | no — perturbs segments and fits a linear model | ~56 segments | yes |

The three maps are compared after discarding negative attribution, so that all
of them answer the same question: *what counts as evidence for this class?*

SHAP is deliberately excluded — its cost on convolutional networks does not fit
the scope of this study.

## Selecting images

60 candidates are collected from Wikimedia Commons under CC0 / CC BY / CC BY-SA
/ public-domain licences only. `data/sources.csv` records the licence, author
and URL of every file; the images themselves are not committed.

Candidates are split by the network's own confidence into two strata:

| Stratum | Confidence | Images | Why |
|---|---|---|---|
| `confident` | ≥ 90 % | 32 | the original selection rule |
| `uncertain` | 40–70 % | 12 | a ≥90 % filter alone keeps only what the network already finds easy, which biases the study towards agreement |

Explanations target **the predicted class, not the intended one**. Five
photographs found by searching "Siberian husky" are sled races, and the network
calls them `dogsled` — correctly. The search term is not ground truth.

Images are removed only by a written rule, kept in `data/exclusions.csv` with a
reason. Rejected images stay in `results/candidates.csv` marked `excluded`.

## Measuring agreement

- **Spearman rank correlation** between two full maps.
- **IoU of the top 10 % most important pixels**, thresholded inside each map.

Neither metric needs the maps rescaled to a common range, so no normalisation is
applied — one fewer distortion between the methods and the numbers.

Two sanity checks fix how the numbers should be read:

```
a map against itself      Spearman  1.000    IoU 1.000
a map against random noise Spearman -0.001   IoU 0.053
```

**Chance IoU is not 0.** Two independent masks overlap by
`p1·p2 / (p1 + p2 − p1·p2)` on average - 0.053 when both cover 10 % of the
image. An IoU near that floor means *no* agreement, not *little* agreement.

The floor is computed **per pair**, not once. A quantile threshold cannot split
a LIME segment, so LIME's mask runs to 17 % on some images, which lifts its own
floor to about 0.067. Using one shared floor would flatter whichever pair has
the larger mask. Both mask sizes and the pair's chance IoU are recorded for
every image.

## Choices that change the numbers

Several parameters are decisions rather than properties of the model, and some
of them are hidden in library defaults. They are stated explicitly in the code
and listed in [LIMITATIONS.md](LIMITATIONS.md):

- Grad-CAM explains the last residual block; a different layer gives a different map.
- Integrated Gradients uses a **black** baseline. Black in pixel space is not a
  zero tensor — the network is fed normalised values, where zeros are mid-grey.
  Switching black to grey changed attribution magnitudes roughly threefold.
- LIME fills a switched-off segment with black too. The library default fills it
  with the segment's mean colour, which is LIME's baseline under another name.
- LIME segments with SLIC at a stated segment count rather than the default
  quickshift, so map resolution is comparable across images.

## Reproducing

Python 3.12, CPU only. Randomness is pinned to one seed; two LIME runs with the
same seed are bit-identical.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

.venv/Scripts/python src/fetch_images.py      # build data/sources.csv
.venv/Scripts/python src/download_images.py   # fetch the images (resumable)
.venv/Scripts/python src/classify.py          # predictions and strata
```

`requirements.txt` pins direct dependencies; `requirements-lock.txt` is the full
freeze of the environment the numbers were produced in. `run_log.md` records
every run.

## Layout

| Path | Contents |
|---|---|
| `src/model.py` | the network, its preprocessing, its class names |
| `src/fetch_images.py` | build the candidate list from Wikimedia Commons |
| `src/download_images.py` | fetch and verify the images |
| `src/classify.py` | predictions, confidence strata, exclusions |
| `src/attribution.py` | the three attribution methods |
| `src/metrics.py` | Spearman and top-10 % IoU |
| `data/sources.csv` | provenance and licence of every candidate |
| `results/candidates.csv` | every candidate with its prediction and stratum |
| `LIMITATIONS.md` | what this study does **not** show |
