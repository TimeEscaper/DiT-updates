import torch
import torchvision.transforms as T
import torch.cuda.amp as amp
import torch.nn.functional as F

from typing import Any
from pathlib import Path
from kornia.color.yuv import rgb_to_yuv420, yuv420_to_rgb
from sbervae.lib.models import WanVAEModel
from dit_updates.vae.adapters.base import VAEPreprocessor, VAEAdapter
from dit_updates.utils.files import resolve_path
from dit_updates.vae.models.distributions import DiagonalGaussianDistribution
from dit_updates.vae.models.normalization import (LatentNormalizationType,
                                                  TorchLatentNormalizer)
from dit_updates.vae.adapters.wan_official import WANOfficialPreprocessor


class WANYuv2RgbPreprocessor(VAEPreprocessor):
    """
    Preprocess for internal WAN 2.1 YUV2RGB model.
    """

    def __init__(self):
        """
        Initialize the preprocessor with standard normalization.
        """
        super(WANYuv2RgbPreprocessor, self).__init__()
        
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to the image.

        Args:
            image (torch.Tensor): Input image tensor [0, 1] in RGB space.

        Returns:
            torch.Tensor: Image tensor in YUV space.
        """
        # This transform is aligned with one from SberVAE and current
        # baseline YUV2RGB model, which uses YUV 420d transform.
        batched = image.ndim == 4
        if not batched:
            image = image.unsqueeze(0)
        y, uv = rgb_to_yuv420(image)
        uv = F.interpolate(uv, scale_factor=2, mode='bilinear')
        yuv = torch.cat([y, uv], dim=1)
        if not batched:
            yuv = yuv.squeeze(0)
        return yuv

    def inverse(self, image: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform of the YUV2RGB model.
        It does nothing since output is already in RGB space.

        Args:
            image (torch.Tensor): Image tensor in RGB space.

        Returns:
            torch.Tensor: Same image tensor.
        """
        return image


class WANYuv2YuvPreprocessor(VAEPreprocessor):
    """
    Preprocess for internal WAN 2.1 YUV2YUV model.
    """

    def __init__(self):
        """
        Initialize the preprocessor with standard normalization.
        """
        super(WANYuv2YuvPreprocessor, self).__init__()
        
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to the image.

        Args:
            image (torch.Tensor): Input image tensor [0, 1] in RGB space.

        Returns:
            torch.Tensor: Image tensor in YUV space.
        """
        # This transform is aligned with one from SberVAE and current
        # baseline YUV2RGB model, which uses YUV 420d transform.
        batched = image.ndim == 4
        if not batched:
            image = image.unsqueeze(0)
        y, uv = rgb_to_yuv420(image)
        uv = F.interpolate(uv, scale_factor=2, mode='bilinear')
        yuv = torch.cat([y, uv], dim=1)
        if not batched:
            yuv = yuv.squeeze(0)
        return yuv

    def inverse(self, image: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform of the YUV2RGB model.
        Does inverse YUV to RGB transform.

        Args:
            image (torch.Tensor): Image tensor in RGB space.

        Returns:
            torch.Tensor: Same image tensor.
        """
        batched = image.ndim == 4
        if not batched:
            image = image.unsqueeze(0)
        y = image[:, :1, :, :]
        uv = image[:, 1:, :, :]
        uv = F.interpolate(uv, scale_factor=0.5, mode='bilinear')
        rgb = yuv420_to_rgb(y, uv)
        if not batched:
            rgb = rgb.squeeze(0)
        return rgb


class WANAdapterBase(VAEAdapter):
    """
    Internal WAN 2.1 adapter base class.
    """

    def __init__(self,
                 name: str,
                 checkpoint: str | Path,
                 latent_norm_type: LatentNormalizationType | str,
                 latent_stats_mean: list[float],
                 latent_stats_std: list[float],
                 prerpocessor_cls: type[VAEPreprocessor],
                 wan_kwargs: dict[str, Any],
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
        super(WANAdapterBase, self).__init__(name=name, n_channels=16)
        latent_norm_type = LatentNormalizationType(latent_norm_type)
        self._preprocessor_cls = prerpocessor_cls

        checkpoint = resolve_path(checkpoint, "model")

        model = WanVAEModel(pretrained_path=str(checkpoint),
                            **wan_kwargs)
        model = model.eval()
        model = model.to(device)
        self._model = model

        mean = torch.tensor(latent_stats_mean, dtype=dtype, device=device)
        std = torch.tensor(latent_stats_std, dtype=dtype, device=device)
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
        return self._preprocessor_cls()

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
        
        images = images.unsqueeze(2)  # (B, C, T, H, W) for model

        distribution = self._model.encode(images)
        mean = distribution.mean
        logvar = distribution.logvar
        if sample:
            latents = distribution.sample()
        else:
            latents = distribution.mode()
        latents = latents.squeeze(2)  # (B, C, H, W)
        mean = mean.squeeze(2)
        logvar = logvar.squeeze(2)

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

        latents = latents.unsqueeze(2)  # (B, C, T, H, W) for model

        images = self._model.decode(latents)

        images = images.squeeze(2)  # (B, C, H, W)
        if single:
            images = images.squeeze(0)  # (C, H, W)

        info = {}

        return images, info


class WANYuv2RgbAdapter(WANAdapterBase):
    """
    Internal WAN 2.1 YUV2RGB adapter implementation.
    """

    _IMAGENET_2012_200_MEAN = [
        -7.196443038992584e-05,
        -0.19263234734535217,
        -0.0060315364971756935,
        -0.008871454745531082,
        0.012358635663986206,
        -0.1800689995288849,
        -0.028683781623840332,
        0.09497839212417603,
        0.014750940725207329,
        0.0012209581909701228,
        -0.013892349787056446,
        0.0063743325881659985,
        0.03340122103691101,
        0.11512420326471329,
        0.1839529573917389,
        -0.06042119115591049
    ],

    _IMAGENET_2012_200_STD = [
        0.0006977790035307407,
        0.6388314962387085,
        0.9277921915054321,
        0.7258686423301697,
        0.9773375391960144,
        0.8358289003372192,
        0.7998051643371582,
        1.0100041627883911,
        0.6669394373893738,
        1.0282669067382812,
        0.7665238380432129,
        0.7367441058158875,
        0.8998191356658936,
        0.7038764357566833,
        0.6587797999382019,
        0.6429328322410583
    ]

    _IMAGENET_2012_MEAN = [
        -7.286853360710666e-05,
        -0.18078672885894775,
        -0.008212699554860592,
        -0.007905763573944569,
        0.019507741555571556,
        -0.16465966403484344,
        -0.02668576128780842,
        0.08449669182300568,
        0.008322713896632195,
        0.008041778579354286,
        -0.012736196629703045,
        0.00617537135258317,
        0.03087935410439968,
        0.10986445099115372,
        0.17370596528053284,
        -0.05156862363219261
    ]

    _IMAGENET_2012_STD = [
        0.0006974305724725127,
        0.6393429636955261,
        0.9233031868934631,
        0.7281091213226318,
        0.9659150838851929,
        0.8390931487083435,
        0.7966273427009583,
        1.003061056137085,
        0.6728984117507935,
        1.0245347023010254,
        0.7747780084609985,
        0.7395362854003906,
        0.9027074575424194,
        0.7035373449325562,
        0.6566562056541443,
        0.6456491947174072
    ]

    def __init__(self,
                 name: str = "wan-mil-yuv2rgb",
                 checkpoint: str | Path = "MIL-Wan2.1-YUV2RGB/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANYuv2RgbAdapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-yuv2rgb".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan2.1-YUV2RGB/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        if latent_stats is None:
            mean = [0.] * 16
            std = [1.] * 16
        elif latent_stats == "imagenet2012_200":
            mean = WANYuv2RgbAdapter._IMAGENET_2012_200_MEAN
            std = WANYuv2RgbAdapter._IMAGENET_2012_200_STD
        elif latent_stats == "imagenet2012":
            mean = WANYuv2RgbAdapter._IMAGENET_2012_MEAN
            std = WANYuv2RgbAdapter._IMAGENET_2012_STD
        else:
            raise ValueError(f"Invalid latent stats: {latent_stats}")

        super(WANYuv2RgbAdapter, self).__init__(name=name,
                                                checkpoint=checkpoint,
                                                latent_norm_type=latent_norm_type,
                                                latent_stats_mean=mean,
                                                latent_stats_std=std,
                                                prerpocessor_cls=WANYuv2RgbPreprocessor,
                                                wan_kwargs={"output_act": "yuv2rgb"},
                                                device=device, 
                                                dtype=dtype)



class WANYuv2YuvAdapter(WANAdapterBase):
    """
    Internal WAN 2.1 YUV2YUV adapter implementation.
    """

    def __init__(self,
                 name: str = "wan-mil-yuv2yuv",
                 checkpoint: str | Path = "MIL-Wan2.1-YUV2YUV/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANYuv2YuvAdapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-yuv2yuv".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan2.1-YUV2YUV/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        if latent_stats is None:
            mean = [0.] * 16
            std = [1.] * 16
        elif latent_stats == "imagenet2012_200":
            raise ValueError("Invalid latent stats: {latent_stats}")
        elif latent_stats == "imagenet2012":
            raise ValueError("Invalid latent stats: {latent_stats}")
        else:
            raise ValueError("Invalid latent stats: {latent_stats}")

        super(WANYuv2YuvAdapter, self).__init__(name=name,
                                                checkpoint=checkpoint,
                                                latent_norm_type=latent_norm_type,
                                                latent_stats_mean=mean,
                                                latent_stats_std=std,
                                                prerpocessor_cls=WANYuv2YuvPreprocessor,
                                                wan_kwargs={"output_act": "yuv2yuv"},
                                                device=device, 
                                                dtype=dtype)


class WANRgb2RgbAdapter(WANAdapterBase):
    """
    Internal WAN 2.1 RGB2RGB adapter implementation.
    """

    def __init__(self,
                 name: str = "wan-mil-rgb2rgb",
                 checkpoint: str | Path = "MIL-Wan2.1-RGB2RGB/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANRgb2RgbAdapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-rgb2rgb".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan2.1-RGB2RGB/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        if latent_stats is None:
            mean = [0.] * 16
            std = [1.] * 16
        elif latent_stats == "imagenet2012_200":
            raise ValueError("Invalid latent stats: {latent_stats}")
        elif latent_stats == "imagenet2012":
            raise ValueError("Invalid latent stats: {latent_stats}")
        else:
            raise ValueError("Invalid latent stats: {latent_stats}")

        super(WANRgb2RgbAdapter, self).__init__(name=name,
                                                checkpoint=checkpoint,
                                                latent_norm_type=latent_norm_type,
                                                latent_stats_mean=mean,
                                                latent_stats_std=std,
                                                prerpocessor_cls=WANOfficialPreprocessor,
                                                wan_kwargs={},
                                                device=device, 
                                                dtype=dtype)
