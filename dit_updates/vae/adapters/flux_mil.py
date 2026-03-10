from dit_updates.vae.adapters.wan_mil import WANYuv2RgbPreprocessor
from sbervae.lib.models.flux.flux_vae import FluxVAEModel
import torch

from typing import Any
from pathlib import Path
from dit_updates.vae.adapters.base import VAEPreprocessor, VAEAdapter
from dit_updates.utils.files import resolve_path
from dit_updates.vae.models.distributions import DiagonalGaussianDistribution
from dit_updates.vae.models.normalization import (LatentNormalizationType,
                                                  TorchLatentNormalizer)


class FLUXYuv2RgbAdapter(VAEAdapter):
    """
    Internal FLUX YUV2RGB adapter implementation.
    """

    _IMAGENET_2012_200_MEAN = [
        -0.001975004794076085,
        0.014901080168783665,
        -0.00634393747895956,
        -0.009670428931713104,
        0.007614071015268564,
        0.022786686196923256,
        0.026518499478697777,
        0.009671240113675594,
        0.0036589442752301693,
        -0.012501779943704605,
        -0.03185223042964935,
        -0.05054309219121933,
        0.01471721287816763,
        0.016083691269159317,
        -0.02177325449883938,
        0.028684722259640694
    ]

    _IMAGENET_2012_200_STD = [
        0.8321698307991028,
        1.0271022319793701,
        0.847089409828186,
        0.21353279054164886,
        0.21509641408920288,
        0.8709070086479187,
        0.8589903116226196,
        0.256315141916275,
        0.9502130746841431,
        0.7277862429618835,
        0.823249101638794,
        1.2654237747192383,
        0.787143349647522,
        0.8983247876167297,
        0.9176579713821411,
        0.9812710881233215
    ]

    _IMAGENET_2012_MEAN = [
        0.003045937977731228,
        -0.00643965695053339,
        -0.0035044148098677397,
        -0.013717574998736382,
        0.006109481677412987,
        0.017682118341326714,
        0.02911437675356865,
        0.013639028184115887,
        0.000154010922415182,
        -0.010936704464256763,
        -0.03706026077270508,
        -0.026477670297026634,
        0.011873207055032253,
        0.01570758782327175,
        -0.019677529111504555,
        0.02383139356970787
    ]

    _IMAGENET_2012_STD = [
        0.8330590128898621,
        1.0145273208618164,
        0.8479723334312439,
        0.21266378462314606,
        0.2141410857439041,
        0.8659865260124207,
        0.8598732948303223,
        0.25278598070144653,
        0.944255530834198,
        0.7272791266441345,
        0.8245658278465271,
        1.2696325778961182,
        0.7889134883880615,
        0.9006738066673279,
        0.9081636071205139,
        0.9726507067680359
    ]

    def __init__(self,
                 name: str = "flux-mil-yuv2rgb",
                 checkpoint: str | Path = "MIL-Flux-YUV2RGB/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANOfficialAdapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan_2.1_official".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "Wan2.1-T2V-14B/Wan2.1_VAE.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "official", or None). Defaults to "imagenet2012".
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        super(FLUXYuv2RgbAdapter, self).__init__(name=name, n_channels=16)
        latent_norm_type = LatentNormalizationType(latent_norm_type)

        checkpoint = resolve_path(checkpoint, "model")
        model = FluxVAEModel(str(checkpoint), output_act="yuv2rgb").to(dtype)

        model = model.eval()
        model = model.to(device)
        self._model = model.model

        if latent_stats is None:
            mean = [0.] * self._model.params.z_channels
            std = [1.] * self._model.params.z_channels
        elif latent_stats == "imagenet2012_200":
            mean = FLUXYuv2RgbAdapter._IMAGENET_2012_200_MEAN
            std = FLUXYuv2RgbAdapter._IMAGENET_2012_200_STD
        elif latent_stats == "imagenet2012":
            mean = FLUXYuv2RgbAdapter._IMAGENET_2012_MEAN
            std = FLUXYuv2RgbAdapter._IMAGENET_2012_STD
        else:
            raise ValueError(f"Invalid latent stats: {latent_stats}")

        mean = torch.tensor(mean, dtype=dtype, device=device)
        std = torch.tensor(std, dtype=dtype, device=device)
        self._latent_normalizer = latent_norm_type.make_torch(
            mean, std, device)

        self._dtype = dtype

    @property
    def latent_normalizer(self) -> TorchLatentNormalizer:
        """
        Return the latent normalizer instance.

        Returns:
            TorchLatentNormalizer: The latent normalizer.
        """
        return self._latent_normalizer

    def create_preprocessor(self) -> VAEPreprocessor:
        """
        Create and return the WANOfficialPreprocessor instance.

        Returns:
            WANOfficialPreprocessor: Preprocessor object.
        """
        return WANYuv2RgbPreprocessor()

    @torch.inference_mode()
    def encode(self,
               images: torch.Tensor,
               normalize: bool = True,
               sample: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Encode images into the latent space of the VAE.

        Args:
            images (torch.Tensor): Images to be encoded, (B, C, H, W) or (C, H, W).
            normalize (bool, optional): Normalize latents using the latent normalizer. Defaults to True.
            sample (bool, optional): Sample from posterior latent distribution. Defaults to True.

        Returns:
            tuple[torch.Tensor, dict[str, Any]]: Encoded latents and info dictionary.
        """
        shape = images.shape
        if len(shape) == 3:  # (C, H, w)
            images = images.unsqueeze(0)  # (B, C, H, W)
            single = True
        else:  # (B, C, H, W)
            single = False

        distribution, _ = self._model.encode(images)
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
        """
        Decode latents into images using the VAE decoder.

        Args:
            latents (torch.Tensor): Latents to decode. Shape can be (B, C, H, W) or (C, H, W).
            denormalize (bool, optional): Denormalize latents before decoding. Defaults to True.

        Returns:
            tuple[torch.Tensor, dict[str, Any]]: Decoded images and info dictionary.
        """
        shape = latents.shape
        if len(shape) == 3:  # (C, H, w)
            latents = latents.unsqueeze(0)  # (B, C, H, W)
            single = True
        else:  # (B, C, H, W)
            single = False

        if denormalize:
            latents = self._latent_normalizer.denormalize(latents)

        images = self._model.decode(latents)

        if single:
            images = images.squeeze(0)  # (C, H, W)

        images = images.clamp(-1, 1)  # From official code

        info = {}

        return images, info
