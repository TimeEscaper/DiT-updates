from __future__ import annotations

import enum
import numpy as np
import torch


class NumPyLatentNormalizer:
    """
    Normalizes and denormalizes latent arrays using provided mean and standard deviation.
    """

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray
    ):
        """
        Initialize NumPyLatentNormalizer.

        Args:
            mean (np.ndarray): Mean for normalization.
            std (np.ndarray): Standard deviation for normalization.
        """
        self._mean = mean
        self._std = std

    @property
    def mean(self) -> np.ndarray:
        """
        Get the mean used for normalization.

        Returns:
            np.ndarray: Normalization mean.
        """
        return self._mean

    @property
    def std(self) -> np.ndarray:
        """
        Get the standard deviation used for normalization.

        Returns:
            np.ndarray: Normalization standard deviation.
        """
        return self._std

    def normalize(self, latent: np.ndarray) -> np.ndarray:
        """
        Normalize latent tensor with shape (C, H, W) or (B, C, H, W).

        Args:
            latent (np.ndarray): The latent space to normalize, shape: (C, H, W) or (B, C, H, W).

        Returns:
            np.ndarray: The normalized latent tensor with shape (C, H, W) or (B, C, H, W).
        """
        mean, std = self._reshape(latent.shape)
        return (latent - mean) / std

    def denormalize(self, latent: np.ndarray) -> np.ndarray:
        """
        Denormalize latent tensor with shape (C, H, W) or (B, C, H, W).

        Args:
            latent (np.ndarray): The latent space to denormalize, shape: (C, H, W) or (B, C, H, W).

        Returns:
            np.ndarray: The denormalized latent tensor with shape (C, H, W) or (B, C, H, W).
        """
        mean, std = self._reshape(latent.shape)
        return latent * std + mean

    def _reshape(self, latent_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        """
        Helper to reshape the mean and std for broadcasting over input latent array.

        Args:
            latent_shape (tuple[int, ...]): Shape of the input latent.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean and std reshaped for broadcasting.
        """
        if len(latent_shape) == 3:
            new_shape = (-1, 1, 1)
        elif len(latent_shape) == 4:
            new_shape = (1, -1, 1, 1)
        else:
            raise ValueError(f"Invalid latent shape: {latent_shape}")
        mean = self._mean.reshape(new_shape)
        std = self._std.reshape(new_shape)
        return mean, std


class TorchLatentNormalizer:
    """
    Normalizes and denormalizes latent torch tensors using provided mean and standard deviation.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        device: str = "cuda"
    ):
        """
        Initialize TorchLatentNormalizer.

        Args:
            mean (torch.Tensor): Mean for normalization.
            std (torch.Tensor): Standard deviation for normalization.
            device (str, optional): Torch device. Defaults to "cuda".
        """
        self._mean = mean.to(device)
        self._std = std.to(device)
        self._device = device

    @property
    def mean(self) -> torch.Tensor:
        """
        Get the mean used for normalization.

        Returns:
            torch.Tensor: Normalization mean.
        """
        return self._mean

    @property
    def std(self) -> torch.Tensor:
        """
        Get the standard deviation used for normalization.

        Returns:
            torch.Tensor: Normalization standard deviation.
        """
        return self._std

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Normalize latent tensor with shape (C, H, W) or (B, C, H, W).

        Args:
            latent (torch.Tensor): The latent space to normalize, shape: (C, H, W) or (B, C, H, W).

        Returns:
            torch.Tensor: The normalized latent tensor with shape (C, H, W) or (B, C, H, W).
        """
        mean, std = self._reshape(latent.shape)
        return (latent - mean) / std

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Denormalize latent tensor with shape (C, H, W) or (B, C, H, W).

        Args:
            latent (torch.Tensor): The latent space to denormalize, shape: (C, H, W) or (B, C, H, W).

        Returns:
            torch.Tensor: The denormalized latent tensor with shape (C, H, W) or (B, C, H, W).
        """
        mean, std = self._reshape(latent.shape)
        return latent * std + mean

    def to(self, device: str) -> TorchLatentNormalizer:
        """
        Move mean and std tensors to a different device.

        Args:
            device (str): Device string.

        Returns:
            TorchLatentNormalizer: New instance with tensors on requested device.
        """
        return TorchLatentNormalizer(self._mean.to(device),
                                     self._std.to(device),
                                     device)

    def numpy(self) -> NumPyLatentNormalizer:
        """
        Convert the normalizer to a NumPyLatentNormalizer.

        Returns:
            NumPyLatentNormalizer: Normalizer with tensors converted to numpy arrays.
        """
        return NumPyLatentNormalizer(self._mean.clone().detach().cpu().numpy().astype(np.float32),
                                     self._std.clone().detach().cpu().numpy().astype(np.float32))

    def _reshape(self, latent_shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Helper to reshape the mean and std for broadcasting over input latent tensor.

        Args:
            latent_shape (tuple[int, ...]): Shape of the input latent.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and std reshaped for broadcasting.
        """
        if len(latent_shape) == 3:
            new_shape = (-1, 1, 1)
        elif len(latent_shape) == 4:
            new_shape = (1, -1, 1, 1)
        else:
            raise ValueError(f"Invalid latent shape: {latent_shape}")
        mean = self._mean.view(new_shape)
        std = self._std.view(new_shape)
        return mean, std


class LatentNormalizationType(str, enum.Enum):
    """
    Latent normalization strategies for VAE models.

    Available types:
        - NONE: No normalization.
        - SCALE: Divide by std.
        - SHIFT_SCALE: Subtract mean and divide by std.
    """
    NONE = "none"
    SCALE = "scale"
    SHIFT_SCALE = "shift_scale"

    def make_numpy(self, mean: np.ndarray, std: np.ndarray) -> NumPyLatentNormalizer:
        """
        Create a NumPyLatentNormalizer based on normalization type.

        Args:
            mean (np.ndarray): Mean for normalization.
            std (np.ndarray): Standard deviation for normalization.

        Returns:
            NumPyLatentNormalizer: Normalizer instance configured as per normalization type.

        Raises:
            ValueError: If an unknown normalization type is given.
        """
        if self == LatentNormalizationType.NONE:
            return NumPyLatentNormalizer(mean=np.zeros_like(mean),
                                         std=np.ones_like(std))
        elif self == LatentNormalizationType.SCALE:
            return NumPyLatentNormalizer(mean=np.zeros_like(mean),
                                         std=std)
        elif self == LatentNormalizationType.SHIFT_SCALE:
            return NumPyLatentNormalizer(mean=mean,
                                         std=std)
        else:
            raise ValueError("Unknown latent normalization type")

    def make_torch(self, mean: torch.Tensor, std: torch.Tensor, device: str = "cuda") -> TorchLatentNormalizer:
        """
        Create a TorchLatentNormalizer based on normalization type.

        Args:
            mean (torch.Tensor): Mean for normalization.
            std (torch.Tensor): Standard deviation for normalization.
            device (str, optional): Torch device. Defaults to "cuda".

        Returns:
            TorchLatentNormalizer: Normalizer instance configured as per normalization type.

        Raises:
            ValueError: If an unknown normalization type is given.
        """
        if self == LatentNormalizationType.NONE:
            return TorchLatentNormalizer(mean=torch.zeros_like(mean),
                                         std=torch.ones_like(std),
                                         device=device)
        elif self == LatentNormalizationType.SCALE:
            return TorchLatentNormalizer(mean=torch.zeros_like(mean),
                                         std=std,
                                         device=device)
        elif self == LatentNormalizationType.SHIFT_SCALE:
            return TorchLatentNormalizer(mean=mean,
                                         std=std,
                                         device=device)
        else:
            raise ValueError("Unknown latent normalization type")
