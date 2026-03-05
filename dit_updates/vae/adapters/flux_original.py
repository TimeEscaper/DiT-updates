from pathlib import Path
from typing import Any

from dit_updates.utils.files import resolve_path
from dit_updates.vae.adapters.base import VAEAdapter, VAEPreprocessor
from dit_updates.vae.models.flux import flux_vae_f8c16
from dit_updates.vae.models.distributions import DiagonalGaussianDistribution
from dit_updates.vae.models.normalization import LatentNormalizationType, TorchLatentNormalizer
import torch
import torch.cuda.amp as amp


class IdentityPreprocessor(VAEPreprocessor):

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def inverse(self, image: torch.Tensor) -> torch.Tensor:
        return image


class FLUXOfficialAdapter(VAEAdapter):

    # From official code
    _OFFICIAL_MEAN = [
        0.1159
    ]
    _OFFICIAL_STD = [
        0.3611
    ]

    _IMAGENET_2012_200_MEAN = [
        -0.05182335153222084,
        -1.4416066408157349,
        0.49022164940834045,
        -0.10477370023727417,
        -0.2233525812625885,
        0.009221182204782963,
        -1.1337759494781494,
        0.44538432359695435,
        -0.19688397645950317,
        0.22915838658809662,
        0.6006688475608826,
        -0.04875076562166214,
        -0.22705499827861786,
        0.132157102227211,
        -0.7553930878639221,
        -0.35048434138298035
    ]
    _IMAGENET_2012_200_STD = [
        2.015666961669922,
        3.3807647228240967,
        2.042870283126831,
        1.4227722883224487,
        1.4915176630020142,
        1.6953730583190918,
        2.9559166431427,
        2.3253018856048584,
        1.8267412185668945,
        2.650012254714966,
        1.688286542892456,
        1.9062206745147705,
        2.526150703430176,
        2.624682903289795,
        2.193922758102417,
        2.3902266025543213
    ]

    _IMAGENET_2012_MEAN = [
        0.0040035066194832325,
        -1.347395658493042,
        0.4194449782371521,
        -0.06957913935184479,
        -0.21221670508384705,
        0.10140205174684525,
        -1.089916467666626,
        0.4157811403274536,
        -0.16778506338596344,
        0.2511731684207916,
        0.6569024920463562,
        -0.053312670439481735,
        -0.18680237233638763,
        0.08773580938577652,
        -0.8067280054092407,
        -0.36164146661758423
    ]
    _IMAGENET_2012_STD = [
        1.948770523071289,
        3.4378557205200195,
        1.995592713356018,
        1.4158644676208496,
        1.4821546077728271,
        1.692920446395874,
        3.014383554458618,
        2.3437659740448,
        1.7771646976470947,
        2.592388391494751,
        1.6740353107452393,
        1.917815923690796,
        2.569758892059326,
        2.6503844261169434,
        2.189380407333374,
        2.410733938217163
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
        return IdentityPreprocessor()

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
