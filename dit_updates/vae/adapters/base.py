import json
from pathlib import Path
import torch

from typing import Any
from abc import ABC, abstractmethod
from dit_updates.vae.models.normalization import TorchLatentNormalizer


class VAEPreprocessor(ABC):
    """
    Abstract base class for VAE preprocessors.
    Inspired by HuggingFace preprocessors, which are part of the models.
    """

    @abstractmethod
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the input image tensor.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        pass

    @abstractmethod
    def inverse(self, image: torch.Tensor) -> torch.Tensor:
        """
        Inverse the preprocessing operation on the input image tensor.
        Useful to produce proper image samples.

        Args:
            image (torch.Tensor): Preprocessed image tensor.

        Returns:
            torch.Tensor: Image tensor restored to original space.
        """
        pass


class VAEAdapter(ABC):
    """
    Abstract base class for VAE model adapters.
    Main purpose is to provide a unified interface for different VAE models,
    in order to introduce as little changes to original DiT training code
    as possible.

    Conventions:
    - Preprocessors must be produced by model (adapter).
    - Encode and decode should return main tensor (latent/images) and aux dict.
    - Encode and decode should support both batched and single input tensors.
    - Wrapped VAE should be in eval mode.
    """

    def __init__(self,
                 name: str,
                 n_channels: int):
        """
        Initialize the VAE adapter.

        Args:
            name (str): Name of the adapter/model.
            n_channels (int): Number of latent channels.
        """
        self._name = name
        self._n_channels = n_channels

    @property
    def name(self) -> str:
        """
        Get the adapter/model name.

        Returns:
            str: Adapter/model name.
        """
        return self._name

    @property
    def n_channels(self) -> int:
        """
        Get the number of latent channels.

        Returns:
            int: Number of latent channels.
        """
        return self._n_channels

    @property
    @abstractmethod
    def latent_normalizer(self) -> TorchLatentNormalizer:
        """
        Return the latent normalizer instance.

        Returns:
            TorchLatentNormalizer: The latent normalizer.
        """
        pass

    @abstractmethod
    def create_preprocessor(self) -> VAEPreprocessor:
        """
        Create and return a VAEPreprocessor for this adapter.

        Returns:
            VAEPreprocessor: The preprocessor instance.
        """
        pass

    @abstractmethod
    def encode(self,
               images: torch.Tensor,
               normalize: bool = True,
               sample: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Encode input images to latent space.

        Args:
            images (torch.Tensor): Batched images (B, C, H, W) or single image (C, H, W).
            normalize (bool, optional): Whether to normalize latents. Defaults to True.
            sample (bool, optional): Whether to sample from latent distribution. Defaults to True.

        Returns:
            tuple[torch.Tensor, dict[str, Any]]: The encoded latent tensor and info dict.
        """
        pass

    @abstractmethod
    def decode(self,
               latents: torch.Tensor,
               denormalize: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Decode latents to images.

        Args:
            latents (torch.Tensor): Latents (B, C, H, W) or (C, H, W).
            denormalize (bool, optional): Whether to denormalize latents before decoding. Defaults to True.

        Returns:
            tuple[torch.Tensor, dict[str, Any]]: The reconstructed images and info dict.
        """
        pass


def load_latent_stats(cls: type[VAEAdapter], latent_stats: str | Path | None, z_dim: int) -> tuple[list[float], list[float]]:
    """
    Load the latent stats from the file.
    """
    if latent_stats is None:
        mean = [0.] * z_dim
        std = [1.] * z_dim
        return mean, std
    if isinstance(latent_stats, Path) or latent_stats.endswith(".json"):
        with open(latent_stats, "r") as f:
            data = json.load(f)
        mean = data["stats"]["mean"]
        std = data["stats"]["std"]
        return mean, std
    else:
        if latent_stats == "official":
            mean = cls._OFFICIAL_MEAN
            std = cls._OFFICIAL_STD
        elif latent_stats == "imagenet2012_200":
            mean = cls._IMAGENET_2012_200_MEAN
            std = cls._IMAGENET_2012_200_STD
        elif latent_stats == "imagenet2012":
            mean = cls._IMAGENET_2012_MEAN
            std = cls._IMAGENET_2012_STD
        else:
            raise ValueError(f"Invalid latent stats: {latent_stats}")
        return mean, std
