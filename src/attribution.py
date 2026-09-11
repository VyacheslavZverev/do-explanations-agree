"""Attribution methods under comparison.

Each function takes a preprocessed image tensor and a target class index, and
returns one 224x224 attribution map as a NumPy array. Keeping the signatures
identical is what makes the three methods comparable at all.
"""

import numpy as np
import torch
from captum.attr import LayerAttribution, LayerGradCam

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
