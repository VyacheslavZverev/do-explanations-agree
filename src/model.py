"""The network under study: loading, preprocessing and class names.

Every part of the experiment goes through this module, so that all stages
see exactly the same model and the same input pipeline.
"""

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

# Pinned to a named weight version on purpose. `ResNet18_Weights.DEFAULT` is an
# alias that torchvision is free to repoint at a newer checkpoint in a future
# release, which would silently change every number in this study.
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


def load_model() -> torch.nn.Module:
    """Return ResNet-18 with ImageNet weights, ready for inference."""
    model = resnet18(weights=WEIGHTS)
    model.eval()
    return model


def preprocess():
    """Return the exact image transform these weights were trained with."""
    return WEIGHTS.transforms()


def class_names() -> list[str]:
    """Return the 1000 ImageNet class names, indexed by model output unit."""
    return WEIGHTS.meta["categories"]


def load_image(path) -> Image.Image:
    """Open one source image as RGB.

    Some Commons files are RGBA or palette-based, while the network expects
    exactly three channels. Converting here, once, keeps every stage of the
    experiment looking at pixel-for-pixel identical input.
    """
    with Image.open(path) as image:
        return image.convert("RGB")
