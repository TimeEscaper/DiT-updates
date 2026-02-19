import torch

from typing import Callable
from lib.utils.files import resolve_path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def create_imagenet_dataset(image_dir: str, 
                            transform: Callable[[Image.Image], torch.Tensor]) -> Dataset:
    image_dir = resolve_path(image_dir, "dataset")
    return ImageFolder(root=image_dir, transform=transform)
