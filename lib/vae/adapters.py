from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.cuda.amp as amp
import torchvision.transforms as T

from lib.utils.files import resolve_path
from lib.vae.models.distributions import DiagonalGaussianDistribution
from lib.vae.models.flux import flux_vae_f8c16
from lib.vae.models.normalization import (LatentNormalizationType,
                                          NumPyLatentNormalizer,
                                          TorchLatentNormalizer)
from lib.vae.models.wan import _video_vae


class VAEPreprocessor(ABC):

    @abstractmethod
    def __call__(image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse(self, image: torch.Tensor) -> torch.Tensor:
        pass


class VAEAdapter(ABC):

    def __init__(self,
                 name: str,
                 n_channels: int):
        self._name = name
        self._n_channels = n_channels

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    @abstractmethod
    def latent_normalizer(self) -> TorchLatentNormalizer:
        pass

    @abstractmethod
    def create_preprocessor(self) -> VAEPreprocessor:
        pass

    @abstractmethod
    def encode(self,
               images: torch.Tensor,
               normalize: bool = True,
               sample: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
        pass

    @abstractmethod
    def decode(self,
               latents: torch.Tensor,
               denormalize: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
        pass


class WANOfficialPreprocessor(VAEPreprocessor):

    def __init__(self):
        super(WANOfficialPreprocessor, self).__init__()
        self._normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5])

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self._normalize(image)

    def inverse(self, image: torch.Tensor) -> torch.Tensor:
        dtype = image.dtype
        mean = torch.as_tensor(self._normalize.mean,
                               dtype=dtype, device=image.device)
        std = torch.as_tensor(self._normalize.std,
                              dtype=dtype, device=image.device)
        if (std == 0).any():
            raise ValueError(
                f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
        if mean.ndim == 1:
            if image.ndim == 3:
                mean = mean.view(-1, 1, 1)
            else:
                mean = mean.view(1, -1, 1, 1)
        if std.ndim == 1:
            if image.ndim == 3:
                std = std.view(-1, 1, 1)
            else:
                std = std.view(1, -1, 1, 1)
        image = image.clone()
        image.mul_(std).add_(mean)
        return image


class WANOfficialAdapter(VAEAdapter):

    # From official code
    _OFFICIAL_MEAN = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ]
    _OFFICIAL_STD = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ]

    _IMAGENET_2012_200_MEAN = [
        -0.2692396938800812,
        0.016344642266631126,
        -0.581970751285553,
        0.13159914314746857,
        -0.33105555176734924,
        0.20614323019981384,
        -0.08000203967094421,
        -0.044480957090854645,
        -0.031538818031549454,
        -0.33836299180984497,
        0.10065454989671707,
        0.670249879360199,
        0.11405237764120102,
        -1.4489037990570068,
        0.8505193591117859,
        0.588209331035614
    ]

    _IMAGENET_2012_200_STD = [
        1.1033473014831543,
        0.6348368525505066,
        1.0813809633255005,
        1.233666181564331,
        0.6216264963150024,
        0.7697148323059082,
        0.9015212655067444,
        0.9805298447608948,
        1.2172343730926514,
        0.8471845984458923,
        0.985538125038147,
        0.6242260932922363,
        0.6484224200248718,
        0.5665448307991028,
        1.212532877922058,
        0.898713231086731
    ]

    _IMAGENET_2012_MEAN = [
        -0.256296306848526,
        0.0077391150407493114,
        -0.5914746522903442,
        0.15141397714614868,
        -0.3327605426311493,
        0.20220015943050385,
        -0.10974644124507904,
        -0.016261115670204163,
        -0.07461243122816086,
        -0.31013739109039307,
        0.14704173803329468,
        0.6520771980285645,
        0.11393015831708908,
        -1.4250420331954956,
        0.8589756488800049,
        0.5744618773460388
    ]

    _IMAGENET_2012_STD = [
        1.0565978288650513,
        0.6282875537872314,
        1.0575528144836426,
        1.238028883934021,
        0.6162665486335754,
        0.764495849609375,
        0.9022724628448486,
        0.9402367472648621,
        1.2197587490081787,
        0.8188892602920532,
        0.9723860621452332,
        0.6259068250656128,
        0.6376462578773499,
        0.5674147009849548,
        1.221549391746521,
        0.9011343121528625
    ]

    def __init__(self,
                 name: str = "wan_2.1_official",
                 checkpoint: str | Path = "Wan2.1-T2V-14B/Wan2.1_VAE.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = "imagenet2012",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        super(WANOfficialAdapter, self).__init__(name=name, n_channels=16)
        latent_norm_type = LatentNormalizationType(latent_norm_type)

        checkpoint = resolve_path(checkpoint, "model")
        model = _video_vae(
            pretrained_path=checkpoint,
            z_dim=16)
        model = model.eval()
        model = model.requires_grad_(False)
        model = model.to(device)
        self._model = model

        if latent_stats is None:
            mean = [0.] * model.z_dim
            std = [1.] * model.z_dim
        elif latent_stats == "official":
            mean = WANOfficialAdapter._OFFICIAL_MEAN
            std = WANOfficialAdapter._OFFICIAL_STD
        elif latent_stats == "imagenet2012_200":
            mean = WANOfficialAdapter._IMAGENET_2012_200_MEAN
            std = WANOfficialAdapter._IMAGENET_2012_200_STD
        elif latent_stats == "imagenet2012":
            mean = WANOfficialAdapter._IMAGENET_2012_MEAN
            std = WANOfficialAdapter._IMAGENET_2012_STD
        else:
            raise ValueError(f"Invalid latent stats: {latent_stats}")

        mean = torch.tensor(mean, dtype=dtype, device=device)
        std = torch.tensor(std, dtype=dtype, device=device)
        self._latent_normalizer = latent_norm_type.make_torch(
            mean, std, device)

        self._dtype = dtype

    @property
    def latent_normalizer(self) -> TorchLatentNormalizer:
        return self._latent_normalizer

    def create_preprocessor(self) -> VAEPreprocessor:
        return WANOfficialPreprocessor()

    @torch.inference_mode()
    def encode(self,
               images: torch.Tensor,
               normalize: bool = True,
               sample: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
        shape = images.shape
        if len(shape) == 3:  # (C, H, w)
            images = images.unsqueeze(0)  # (B, C, H, W)
            single = True
        else:  # (B, C, H, W)
            single = False

        images = images.unsqueeze(2)  # (B, C, T, H, W) for model

        with amp.autocast(dtype=self._dtype):
            distribution = self._model.encode(images, None)
            mean = distribution.mean
            logvar = distribution.logvar

        if sample:
            latents = distribution.sample()
        else:
            latents = distribution.mode()

        if normalize:
            latents = self._latent_normalizer.normalize(latents)

        latents = latents.squeeze(2)  # (B, C, H, W)
        mean = mean.squeeze(2)
        logvar = logvar.squeeze(2)
        if single:
            latents = latents.squeeze(0)  # (C, H, W)
            mean = mean.squeeze(0)
            logvar = logvar.squeeze(0)

        info = {
            "raw_distribution": DiagonalGaussianDistribution(mean, logvar)
        }

        return latents, info

    @torch.inference_mode()
    def decode(self,
               latents: torch.Tensor,
               denormalize: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
        shape = latents.shape
        if len(shape) == 3:  # (C, H, w)
            latents = latents.unsqueeze(0)  # (B, C, H, W)
            single = True
        else:  # (B, C, H, W)
            single = False

        if denormalize:
            latents = self._latent_normalizer.denormalize(latents)

        latents = latents.unsqueeze(2)  # (B, C, T, H, W) for model

        with amp.autocast(dtype=self._dtype):
            images = self._model.decode(latents, None)

        images = images.squeeze(2)  # (B, C, H, W)
        if single:
            images = images.squeeze(0)  # (C, H, W)
        images = images.clamp(-1, 1)  # From official code

        info = {}

        return images, info


class FLUXOfficialAdapter(VAEAdapter):

    # From official code
    _OFFICIAL_MEAN = [
        0.1159
    ]
    _OFFICIAL_STD = [
        0.3611
    ]

    _IMAGENET_2012_200_MEAN = [
        0.0,
    ]
    _IMAGENET_2012_200_STD = [
        1.0
    ]

    _IMAGENET_2012_MEAN = [
        0.0,
    ]
    _IMAGENET_2012_STD = [
        1.0
    ]

    def __init__(self,
                 name: str = "flux_vae_f8c16",
                 checkpoint: str | Path = "flux_vae/flux-vae-f8c16-1.0/ae.safetensors",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = "official",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        super(FLUXOfficialAdapter, self).__init__(name=name, n_channels=16)
        latent_norm_type = LatentNormalizationType(latent_norm_type)

        checkpoint = resolve_path(checkpoint, "model")
        model = flux_vae_f8c16(checkpoint)
        model = model.eval()
        model = model.requires_grad_(False)
        model = model.to(device)
        self._model = model

        if latent_stats is None:
            mean = [0.] * model.params.z_channels
            std = [1.] * model.params.z_channels
        elif latent_stats == "official":
            mean = FLUXOfficialAdapter._OFFICIAL_MEAN
            std = FLUXOfficialAdapter._OFFICIAL_STD
        elif latent_stats == "imagenet2012_200":
            mean = FLUXOfficialAdapter._IMAGENET_2012_200_MEAN
            std = FLUXOfficialAdapter._IMAGENET_2012_200_STD
        elif latent_stats == "imagenet2012":
            mean = FLUXOfficialAdapter._IMAGENET_2012_MEAN
            std = FLUXOfficialAdapter._IMAGENET_2012_STD
        else:
            raise ValueError(f"Invalid latent stats: {latent_stats}")

        mean = torch.tensor(mean, dtype=dtype, device=device)
        std = torch.tensor(std, dtype=dtype, device=device)
        self._latent_normalizer = latent_norm_type.make_torch(
            mean, std, device)

        self._dtype = dtype

    @property
    def latent_normalizer(self) -> TorchLatentNormalizer:
        return self._latent_normalizer

    def create_preprocessor(self) -> VAEPreprocessor:
        return WANOfficialPreprocessor()

    @torch.inference_mode()
    def encode(self,
               images: torch.Tensor,
               normalize: bool = True,
               sample: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
        shape = images.shape
        if len(shape) == 3:  # (C, H, w)
            images = images.unsqueeze(0)  # (B, C, H, W)
            single = True
        else:  # (B, C, H, W)
            single = False

        with amp.autocast(dtype=self._dtype):
            distribution = self._model.encode(images)
            mean = distribution.mean
            logvar = distribution.logvar

        if sample:
            latents = distribution.sample()
        else:
            latents = distribution.mode()

        if normalize:
            latents = self._latent_normalizer.normalize(latents)

        if single:
            latents = latents.squeeze(0)  # (C, H, W)
            mean = mean.squeeze(0)
            logvar = logvar.squeeze(0)

        info = {
            "raw_distribution": DiagonalGaussianDistribution(mean, logvar)
        }

        return latents, info

    @torch.inference_mode()
    def decode(self,
               latents: torch.Tensor,
               denormalize: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
        shape = latents.shape
        if len(shape) == 3:  # (C, H, w)
            latents = latents.unsqueeze(0)  # (B, C, H, W)
            single = True
        else:  # (B, C, H, W)
            single = False

        if denormalize:
            latents = self._latent_normalizer.denormalize(latents)

        with amp.autocast(dtype=self._dtype):
            images = self._model.decode(latents)

        if single:
            images = images.squeeze(0)  # (C, H, W)
        # todo: check output range
        # images = images.clamp(-1, 1)  # From official code

        info = {}

        return images, info
