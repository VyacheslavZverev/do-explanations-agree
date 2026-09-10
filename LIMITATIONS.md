# Limitations

Kept up to date **during** the work, not written at the end.

## Scope
- **Single architecture.** All results come from one pretrained network
  (ResNet-18, ImageNet weights). Agreement between explanation methods may be
  architecture-specific and is not shown to generalise.
- **Small sample.** The number of analysed images is small, so reported
  correlations carry wide uncertainty.
- **Two confidence strata, both arbitrary.** Images are analysed in a confident
  stratum (>=90%) and an uncertain one (40-70%). The second exists because a
  >=90% filter alone keeps only cases the network already finds easy, which
  biases the study towards agreement. Both bounds are chosen, not derived, and
  the 70-90% band is analysed in neither.
- **The explained class is the predicted one, not the true one.** Where the
  network is confidently wrong - five husky photographs are classified as
  `dogsled` at 99.5-99.9% - the explanation targets `dogsled`. This is the
  intended reading of the hypothesis, but it means the study says nothing about
  whether explanations track the correct class.
- **Small strata.** 32 confident and 13 uncertain images. Any difference
  between the two carries wide uncertainty and should not be read as
  established.

## Data
- **A burnt-in watermark in one image.** `eskimo_dog_02` carries a large
  "Copyright (C) Pets Adviser" caption. High-contrast text is a salient feature
  and may attract attribution mass; the image is kept, but any attribution
  landing on the caption is an artefact of the source file.
- **No person class.** ImageNet-1k contains no class for people, so an image
  whose subject is a person forces an arbitrary prediction. Such images are
  excluded by a written rule in `data/exclusions.csv`, not by hand.
- **Search labels are not ground truth.** The `expected_class` column is the
  search term used, not verified annotation. Inspection showed the network is
  often right where the search term is wrong: a photograph found under
  "school bus" is dominated by a church, and photographs found under "Siberian
  husky" are sled races in which `dogsled` is the correct answer.
- **Search-engine provenance.** Candidate images come from a Wikimedia Commons
  keyword search, whose ranking changes over time. `data/sources.csv` pins the
  exact files used; the query itself is not reproducible.
- **Keyword search is noisy.** Searching a class name returns museum objects,
  distribution maps and related-but-wrong species alongside photographs. These
  are deliberately left in the candidate set and removed only by the confidence
  filter, never by hand.
- **Transparency is flattened.** Three source images are RGBA or palette-based
  and are converted to RGB before use. Transparent regions become a solid
  colour that was never in the original photograph, and any attribution mass
  falling there is an artefact of that conversion.

## Method
- **Off-the-shelf implementations.** Grad-CAM, Integrated Gradients and LIME are
  used as provided by `captum` / `lime`, not reimplemented. Library defaults
  (baseline, step count, number of perturbed samples, segmentation algorithm)
  materially affect the resulting maps.
- **Normalisation artefact.** The three methods produce attribution maps on
  different scales and at different resolutions. Making them comparable requires
  resizing and normalisation, which is itself a transformation that can create
  or destroy agreement.
- **Signed vs unsigned attributions.** Integrated Gradients yields signed
  values; Grad-CAM is non-negative by construction. Any reconciliation of the
  two is a modelling choice, not a neutral operation.
- **SHAP not evaluated.** Excluded for computational cost on convolutional
  networks, so conclusions cover three methods, not the field.

## Metrics
- **Spearman correlation** is computed over all pixels, which are spatially
  correlated; the effective sample size is far below the pixel count.
- **IoU of the top 10% pixels** depends on an arbitrary threshold. A different
  threshold can change the ordering of the methods.
- Agreement between two explanations is **not** evidence that either is
  faithful to the model.
