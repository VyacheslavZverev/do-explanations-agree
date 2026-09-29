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
- **The network sees a centre crop, not the photograph.** Every image is
  resized to 256 on the short side and cropped to the central 224x224, so a
  portrait photograph loses about a third of its height before any method
  sees it. All three attribution maps describe that crop. A burnt-in
  "Copyright (C) Pets Adviser" caption on `eskimo_dog_02` was listed here as
  a possible magnet for attribution; checking showed it sits in rows 920-960
  of 1000 while the crop keeps rows 172-828, so the network never sees it.
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

## Choices that change the numbers
- **Integrated Gradients baseline is black**, following the original paper.
  Black in pixel space is not a zero tensor: the network is fed normalised
  values, where zeros are mid-grey. Switching the baseline from black to grey
  shrank attribution values roughly threefold on the image inspected, so this
  is a parameter of the method, not a detail.
- **A black baseline gives dark image regions zero attribution by
  construction**, because attribution is scaled by the difference from the
  baseline. Dark parts of a photograph are structurally disadvantaged.
- **Only the positive part of Integrated Gradients is compared.** Roughly half
  of all pixels carry negative attribution - evidence against the predicted
  class - and Grad-CAM cannot express that at all. Discarding it makes the
  comparison fair but throws away half of what IG computed.
- **Signed attributions nearly cancel.** On the inspected image the positive
  values summed to +1303 and the negative to -1289. This follows from the
  completeness property of IG and means a signed sum cannot serve as a measure
  of importance.
- **Grad-CAM is computed at 7x7 and Integrated Gradients at 224x224.** Grad-CAM
  produces one smooth blob, IG a scatter of points along edges. Some of the
  disagreement this study measures is a difference in native resolution rather
  than a difference of opinion about the image.

- **LIME fills a switched-off segment with black**, matching the Integrated
  Gradients baseline. The library default fills it with the segment mean
  instead. This is LIME's baseline under another name, and it was a default
  rather than a stated choice until it was made explicit.
- **Segmentation is a parameter of the explanation.** SLIC with 80 requested
  segments produced 56 on the inspected image. Different boundaries give a
  different map from the same network.
- **The three methods do not even cover the image comparably.** On the inspected
  image, after discarding negative values, Grad-CAM assigns non-zero importance
  to 100%% of pixels, Integrated Gradients to 50%% and LIME to 71%%. Any metric
  based on a top-10%% threshold is applied to three very different
  distributions.

## Interpreting the main result
- **Resolution explains most of the disagreement, and that was tested.**
  Averaging the Integrated Gradients map onto Grad-CAM's 7x7 grid raised their
  mean Spearman from 0.06 to 0.47, on 38 of 44 images. Coarsening random noise
  did not (0.01), so the jump is not an artefact of the procedure. What the
  headline table measures is therefore partly the rendering of the maps rather
  than the methods themselves.
- **Coarsening inflates per-image variance.** The control rose from SD 0.004 to
  SD 0.161, and coarsened IG ranges from -0.57 to +0.90 across images. Only the
  44-image mean is interpretable; single-image coarse correlations are not.
- **The correction was applied in one direction only.** IG was coarsened to
  match Grad-CAM; Grad-CAM cannot be refined to match IG, because the detail
  was never computed. Saying the methods agree "at a common scale" therefore
  means at the coarser of the two scales, which is a choice.
- **The two metrics disagree about the effect of confidence.** Going from the
  confident to the uncertain stratum, mean Spearman for Grad-CAM vs LIME falls
  (0.52 to 0.42) while the same pair's IoU above chance rises (0.18 to 0.28).
  Any claim about how agreement depends on confidence therefore depends on
  which metric is quoted.
- **The uncertain stratum has 12 images.** Differences between strata are
  suggestive at best.

## Exploratory findings, not tested on held-out data
- **Where the residual disagreement sits.** After coarsening removes the
  fine-scale difference, what remains correlates with how differently the two
  methods split their mass between the centre and the border of the frame:
  Spearman(border gap, coarsened agreement) = -0.58 over 44 images. The gap is
  a difference in centre-periphery emphasis, not a preference for borders - IG
  puts 28.6% of its mass in a ring covering 35.4% of the image, and Grad-CAM
  only 23.7%, so Grad-CAM is the more central of the two.
- **That analysis is exploratory and partly circular.** The hypothesis was
  formed by looking at the worst outlier and then tested on the same 44 images,
  which inflates any significance. It is also not independent of the outcome:
  two maps that divide their mass differently between centre and border must
  correlate less. It localises the disagreement rather than explaining it, and
  `tabby_02` is a clear counterexample.
- **Per-image agreement is unstable under the baseline choice.** Switching
  Integrated Gradients from a black to a grey baseline moved single-image
  agreement with Grad-CAM by up to 0.41 (school_bus_01: 0.90 to 0.49) and in
  both directions. Only sample means should be quoted.
- **A tempting explanation that failed.** The zebra outlier looked like an
  effect of the black baseline, since a zebra is half black stripes and a black
  baseline suppresses dark pixels. It is not: with a grey baseline the
  correlation stays negative (-0.44), and attribution correlates with pixel
  brightness at only 0.11.

## Metrics
- **Spearman correlation** is computed over all pixels, which are spatially
  correlated; the effective sample size is far below the pixel count.
- **Ties depress the correlation on their own.** LIME is constant within a
  segment - 41 distinct values across 50,176 pixels on the image inspected -
  and half of the positive part of Integrated Gradients is exactly zero. Large
  blocks of equal ranks pull Spearman towards zero whether or not the methods
  agree, so a low correlation involving LIME is partly an artefact of its
  resolution.
- **IoU of the top 10% pixels** depends on an arbitrary threshold. A different
  threshold can change the ordering of the methods.
- **Chance IoU is not 0.** Two independent masks overlap by
  p1*p2/(p1+p2-p1*p2) on average - 0.053 when both cover 10% of the image; a
  measured 0.0529 against random noise confirms it. Every reported IoU has to be
  read against that floor, and a value near it means no agreement rather than
  little agreement.
- **The floor differs between pairs.** LIME's mask reaches 17% on some images,
  because a quantile threshold cannot split a segment, which lifts the chance
  IoU for any pair involving LIME to about 0.067. Comparing pairs against one
  shared floor would flatter whichever pair has the larger mask, so the chance
  value is computed per pair and recorded per image.
- **The top-10% mask is not exactly 10%.** The threshold keeps tied pixels
  together, because LIME segments are indivisible by design, so mask sizes vary
  slightly between methods. Both real sizes are recorded per image.
- **Attribution maps are not rescaled to a common range.** Spearman compares
  ranks and the IoU threshold is taken inside each map, so no normalisation is
  needed. This removes a distortion the study originally expected to carry.
- Agreement between two explanations is **not** evidence that either is
  faithful to the model.
