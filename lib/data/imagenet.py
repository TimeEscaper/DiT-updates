import torch

from typing import Callable
from lib.utils.files import resolve_path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def create_imagenet_dataset(
    image_dir: str,
    transform: Callable[[Image.Image], torch.Tensor]
) -> Dataset:
    """
    Creates an ImageNet-compatible dataset from a directory of images.
    Derived from the original DiT implementation.

    Args:
        image_dir (str): Path to the root directory containing the images.
        transform (Callable[[Image.Image], torch.Tensor]): 
            A function/transform that takes in a PIL image and returns a tensor.

    Returns:
        Dataset: A PyTorch dataset (ImageFolder) for the specified image directory with the given transform applied.
    """
    image_dir = resolve_path(image_dir, "dataset")
    return ImageFolder(root=image_dir, transform=transform)
