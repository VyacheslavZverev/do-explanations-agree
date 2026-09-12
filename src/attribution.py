"""Attribution methods under comparison.

Each function takes a preprocessed image tensor and a target class index, and
returns one 224x224 attribution map as a NumPy array. Keeping the signatures
identical is what makes the three methods comparable at all.
"""

import numpy as np
import torch
from captum.attr import IntegratedGradients, LayerAttribution, LayerGradCam

import model

INPUT_SIZE = (224, 224)


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
