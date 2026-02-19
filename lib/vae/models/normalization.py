from __future__ import annotations

import enum
import numpy as np
import torch


class NumPyLatentNormalizer:

    def __init__(self,
                 mean: np.ndarray,
                 std: np.ndarray):
        self._mean = mean
        self._std = std

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def std(self) -> np.ndarray:
        return self._std

    def normalize(self, latent: np.ndarray) -> np.ndarray:
        """Normalize latent tensor with shape (C, H, W) or (B, C, H, W).
        Args:
            latent: The latent space to normalize, shape: (C, H, W) or (B, C, H, W)
        Returns:
            The normalized latent tensor with shape (C, H, W) or (B, C, H, W).
        """
        mean, std = self._reshape(latent.shape)
        return (latent - mean) / std

    def denormalize(self, latent: np.ndarray) -> np.ndarray:
        """Denormalize latent tensor with shape (C, H, W) or (B, C, H, W).
        Args:
            latent: The latent space to denormalize, shape: (C, H, W) or (B, C, H, W)
        Returns:
            The denormalized latent tensor with shape (C, H, W) or (B, C, H, W).
        """
        mean, std = self._reshape(latent.shape)
        return latent * std + mean

    def _reshape(self, latent_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
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
    def __init__(self,
                 mean: torch.Tensor,
                 std: torch.Tensor,
                 device: str = "cuda"):
        self._mean = mean.to(device)
        self._std = std.to(device)
        self._device = device

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def std(self) -> torch.Tensor:
        return self._std

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        """Normalize latent tensor with shape (C, H, W) or (B, C, H, W).
        Args:
            latent: The latent space to normalize, shape: (C, H, W) or (B, C, H, W)
        Returns:
            The normalized latent tensor with shape (C, H, W) or (B, C, H, W).
        """
        mean, std = self._reshape(latent.shape)
        return (latent - mean) / std

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        """Denormalize latent tensor with shape (C, H, W) or (B, C, H, W).
        Args:
            latent: The latent space to denormalize, shape: (C, H, W) or (B, C, H, W)
        Returns:
            The denormalized latent tensor with shape (C, H, W) or (B, C, H, W).
        """
        mean, std = self._reshape(latent.shape)
        return latent * std + mean
    
    def to(self, device: str) -> TorchLatentNormalizer:
        return TorchLatentNormalizer(self._mean.to(device), 
                                     self._std.to(device), 
                                     device)

    def numpy(self) -> NumPyLatentNormalizer:
        return NumPyLatentNormalizer(self._mean.clone().detach().cpu().numpy().astype(np.float32), 
                                     self._std.clone().detach().cpu().numpy().astype(np.float32))

    def _reshape(self, latent_shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
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
    """Latent normalization strategies for VAE models."""

    """No normalization."""
    NONE = "none"

    """Divide by std."""
    SCALE = "scale"

    """Subtract mean and divide by std."""
    SHIFT_SCALE = "shift_scale"

    def make_numpy(self, mean: np.ndarray, std: np.ndarray) -> NumPyLatentNormalizer:
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

    
