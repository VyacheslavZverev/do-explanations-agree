"""Attribution methods under comparison.

Each function takes a preprocessed image tensor and a target class index, and
returns one 224x224 attribution map as a NumPy array. Keeping the signatures
identical is what makes the three methods comparable at all.
"""

import numpy as np
import torch
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import slic
from captum.attr import IntegratedGradients, LayerAttribution, LayerGradCam

import model

INPUT_SIZE = (224, 224)

# LIME perturbs segments at random. Everything random in this study is pinned to
# this one seed, and the seed is reported with the results.
RANDOM_SEED = 0

# Segmentation parameters are parameters of the explanation, not of the model:
# different segment boundaries produce a different map from the same network.
# SLIC is used instead of the library default (quickshift) because its segment
# count is set directly, which keeps the map resolution comparable across images.
# SLIC itself is deterministic - it grows segments from a regular grid - so the
# only randomness left in LIME is which segments get switched off.
SLIC_SEGMENTS = 80
SLIC_COMPACTNESS = 10.0

# How many perturbed copies of the image the linear model is fitted on.
LIME_SAMPLES = 1000

# What a switched-off segment is filled with. This is LIME's equivalent of the
# Integrated Gradients baseline, and lime hides it in a default: passing None
# fills each segment with its own mean colour. Black is chosen so that both
# methods measure against the same reference.
LIME_HIDE_COLOUR = 0.0


def grad_cam(net: torch.nn.Module, image: torch.Tensor, target: int) -> np.ndarray:
    """Grad-CAM attribution for one image, upsampled to input resolution.

    The explained layer is the last residual block. Grad-CAM was defined for the
    final convolutional layer, where feature maps stand for whole objects rather
    than edges; choosing a different layer is a parameter of the method, not a
    property of the model.

    `relu_attributions=True` follows the original paper, which keeps only
    evidence *for* the class. Integrated Gradients returns signed values
    instead, so the two are not on the same footing - see LIMITATIONS.md.

    The map is produced at 7x7 and stretched to 224x224. The stretching invents
    detail the method never computed.
    """
    explainer = LayerGradCam(net, net.layer4[-1])
    attribution = explainer.attribute(image.unsqueeze(0), target=target,
                                      relu_attributions=True)
    upsampled = LayerAttribution.interpolate(attribution, INPUT_SIZE,
                                             interpolate_mode="bilinear")
    return upsampled.squeeze().detach().numpy()


def baseline_like(image: torch.Tensor, colour: str) -> torch.Tensor:
    """The reference image Integrated Gradients starts from.

    Captum defaults to a tensor of zeros, but the network is fed *normalised*
    values: zeros there are mid-grey (124, 116, 104), not black. Papers that say
    "black baseline" mean black in pixel space, so it has to be stated outright.
    """
    transform = model.preprocess()
    mean = torch.tensor(transform.mean).view(-1, 1, 1)
    std = torch.tensor(transform.std).view(-1, 1, 1)
    if colour == "grey":
        return torch.zeros_like(image)
    if colour == "black":
        return ((0.0 - mean) / std).expand_as(image).clone()
    raise ValueError(f"unknown baseline colour: {colour}")


def integrated_gradients(net: torch.nn.Module, image: torch.Tensor, target: int,
                         colour: str = "black", steps: int = 64) -> np.ndarray:
    """Integrated Gradients for one image, summed over the three colour channels.

    Values keep their sign: positive is evidence for the target class, negative
    is evidence against it. Grad-CAM is non-negative by construction, so the two
    are not directly comparable until that is resolved - see LIMITATIONS.md.
    """
    explainer = IntegratedGradients(net)
    attribution = explainer.attribute(
        image.unsqueeze(0), baselines=baseline_like(image, colour).unsqueeze(0),
        target=target, n_steps=steps)
    return attribution.squeeze(0).sum(dim=0).detach().numpy()


def positive_part(attribution: np.ndarray) -> np.ndarray:
    """Keep only evidence *for* the target class, discarding evidence against.

    Grad-CAM applies a ReLU by construction, so comparing it against signed
    Integrated Gradients would compare answers to two different questions. This
    is applied as a separate, visible step rather than inside the method, so
    that each method still returns what it actually computed.
    """
    return np.clip(attribution, 0.0, None)


def _batch_predictor(net: torch.nn.Module):
    """Wrap the network so LIME can call it on plain RGB arrays.

    LIME hands over unnormalised images in [0, 1]. The normalisation the weights
    were trained with has to be applied here - skipping it does not raise an
    error, it quietly changes the prediction being explained.
    """
    transform = model.preprocess()
    mean = torch.tensor(transform.mean).view(1, -1, 1, 1)
    std = torch.tensor(transform.std).view(1, -1, 1, 1)

    def predict(images: np.ndarray) -> np.ndarray:
        batch = torch.from_numpy(images).float().permute(0, 3, 1, 2)
        with torch.no_grad():
            return net((batch - mean) / std).softmax(dim=1).numpy()

    return predict


def lime_map(net: torch.nn.Module, image: np.ndarray, target: int) -> np.ndarray:
    """LIME attribution for one image, as a map of per-segment weights.

    `image` is the cropped 224x224 picture in [0, 1] - not the normalised tensor
    the other two methods take, because LIME perturbs pixels a human would see.

    The result is piecewise constant: every pixel of a segment carries that
    segment's weight. Its effective resolution is the number of segments, a
    third geometry alongside Grad-CAM's 7x7 grid and IG's per-pixel values.
    """
    explainer = LimeImageExplainer(random_state=RANDOM_SEED)
    explanation = explainer.explain_instance(
        image.astype(np.double), _batch_predictor(net), labels=(target,),
        hide_color=LIME_HIDE_COLOUR,
        top_labels=None, num_samples=LIME_SAMPLES, random_seed=RANDOM_SEED,
        segmentation_fn=lambda img: slic(img, n_segments=SLIC_SEGMENTS,
                                         compactness=SLIC_COMPACTNESS,
                                         start_label=0))

    weights = np.zeros(INPUT_SIZE, dtype=np.float32)
    for segment_id, weight in explanation.local_exp[target]:
        weights[explanation.segments == segment_id] = weight
    return weights
