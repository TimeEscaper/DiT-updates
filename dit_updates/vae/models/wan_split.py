import logging
import os

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from kornia.color.yuv import rgb_to_yuv, yuv_to_rgb

from sbervae.lib.models.distributions.distributions import DiagonalGaussianDistribution


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1) if channel_first else ()
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'downsample2d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        return self.resample(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.residual = nn.Sequential(
            RMS_norm(in_dim), nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, 3, padding=1))
        self.shortcut = nn.Conv2d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    Self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b, c, h, w)
        x = self.proj(x)
        return x + identity


class Encoder(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0,
                 in_channels=3):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv1 = nn.Conv2d(in_channels, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                downsamples.append(Resample(out_dim, mode='downsample2d'))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim), nn.SiLU(),
            nn.Conv2d(out_dim, z_dim, 3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.downsamples(x)
        x = self.middle(x)
        x = self.head(x)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0,
                 out_channels=3):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.out_channels = out_channels

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        self.conv1 = nn.Conv2d(z_dim, dims[0], 3, padding=1)

        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                upsamples.append(Resample(out_dim, mode='upsample2d'))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(
            RMS_norm(out_dim), nn.SiLU(),
            nn.Conv2d(out_dim, out_channels, 3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.middle(x)
        x = self.upsamples(x)
        x = self.head(x)
        return x


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion: target attends to source.

    Q is projected from target, K/V from source (projected to target_dim).
    Zero-initialized output projection gives identity at init.
    """

    def __init__(self, source_dim: int, target_dim: int):
        super().__init__()
        self.norm_source = RMS_norm(source_dim)
        self.norm_target = RMS_norm(target_dim)
        self.to_q = nn.Conv2d(target_dim, target_dim, 1)
        self.to_k = nn.Conv2d(source_dim, target_dim, 1)
        self.to_v = nn.Conv2d(source_dim, target_dim, 1)
        self.proj = nn.Conv2d(target_dim, target_dim, 1)

        nn.init.zeros_(self.proj.weight)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        identity = target
        b, c_t, h, w = target.shape

        source = self.norm_source(source)
        target = self.norm_target(target)

        q = self.to_q(target).reshape(b, 1, c_t, -1).permute(0, 1, 3, 2).contiguous()
        k = self.to_k(source).reshape(b, 1, c_t, -1).permute(0, 1, 3, 2).contiguous()
        v = self.to_v(source).reshape(b, 1, c_t, -1).permute(0, 1, 3, 2).contiguous()

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.squeeze(1).permute(0, 2, 1).reshape(b, c_t, h, w)
        out = self.proj(out)
        return out + identity


class FiLMFusion(nn.Module):
    """FiLM fusion: source produces per-channel scale and shift for target.

    output = target * (1 + gamma) + beta, where (gamma, beta) are projected
    from source. Zero-initialized projection gives identity at init.
    """

    def __init__(self, source_dim: int, target_dim: int):
        super().__init__()
        self.norm = RMS_norm(source_dim)
        self.proj = nn.Conv2d(source_dim, target_dim * 2, 1)

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source = self.norm(source)
        gamma, beta = self.proj(source).chunk(2, dim=1)
        return target * (1 + gamma) + beta


class StagedDecoder(nn.Module):
    """Decoder with per-stage access for cross-branch fusion.

    Upsampling stages are stored as nn.ModuleList of nn.Sequential blocks,
    enabling per-stage feature capture. When fusion_role='source', the
    feature at fusion_level is captured and returned alongside the output.
    When fusion_role='target', the fusion module (cross-attention or FiLM)
    is applied at fusion_level using the source feature.
    """

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 out_channels=3,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0,
                 fusion_type: str | None = None,
                 fusion_role: str | None = None,
                 fusion_level: str | None = None,
                 fusion_source_dim: int | None = None):
        if fusion_type is None or fusion_type == "none":
            fusion_type = None
            fusion_role = None
            fusion_level = None

        if fusion_type is not None:
            assert fusion_role in ("source", "target"), f"Invalid fusion_role: {fusion_role}"
            assert fusion_type in ("attention", "film"), f"Invalid fusion_type: {fusion_type}"
            assert (fusion_level in ("conv1", "middle", "head")
                    or (fusion_level is not None and fusion_level.startswith("stage"))), \
                f"Invalid fusion_level: {fusion_level}"

        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.out_channels = out_channels
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales

        self.fusion_role = fusion_role
        self.fusion_level = fusion_level
        self.fusion_type = fusion_type

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        self.bottleneck_dim = dims[0]
        self.conv1 = nn.Conv2d(z_dim, dims[0], 3, padding=1)

        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        self.stages = nn.ModuleList()
        self.stage_out_dims = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            stage_modules = []
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                stage_modules.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    stage_modules.append(AttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                stage_modules.append(Resample(out_dim, mode='upsample2d'))
                self.stage_out_dims.append(out_dim // 2)
                scale *= 2.0
            else:
                self.stage_out_dims.append(out_dim)
            self.stages.append(nn.Sequential(*stage_modules))

        self.head = nn.Sequential(
            RMS_norm(dims[-1]), nn.SiLU(),
            nn.Conv2d(dims[-1], out_channels, 3, padding=1))

        if self.fusion_role == "target":
            assert fusion_source_dim is not None, \
                "fusion_source_dim is required when fusion_role='target'"
            target_dim = self.dim_at_level(fusion_level)
            if self.fusion_type == "attention":
                self.fusion = CrossAttentionFusion(fusion_source_dim, target_dim)
            elif self.fusion_type == "film":
                self.fusion = FiLMFusion(fusion_source_dim, target_dim)
            else:
                raise ValueError(f"Invalid fusion_type: {self.fusion_type}")
        else:
            self.fusion = None

    def dim_at_level(self, level: str) -> int:
        """Return the feature channel dimension at the given fusion level."""
        if level in ("conv1", "middle"):
            return self.bottleneck_dim
        if level.startswith("stage"):
            idx = int(level[5:]) - 1
            return self.stage_out_dims[idx]
        if level == "head":
            return self.out_channels
        raise ValueError(f"Unknown fusion level: {level}")

    def forward(self, x: torch.Tensor, fusion_source: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        y_src = None
        if self.fusion_role == "target":
            assert fusion_source is not None, "fusion_source is required for target fusion"
            y_src = fusion_source

        x = self.conv1(x)
        if self.fusion_level == "conv1":
            x, y_src = self._apply_fusion(x, y_src)

        x = self.middle(x)
        if self.fusion_level == "middle":
            x, y_src = self._apply_fusion(x, y_src)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.fusion_level == f"stage{i+1}":
                x, y_src = self._apply_fusion(x, y_src)

        x = self.head(x)
        if self.fusion_level == "head":
            x, y_src = self._apply_fusion(x, y_src)

        if self.fusion_role == "source":
            assert y_src is not None, \
                f"source feature was not captured — fusion_level '{self.fusion_level}' did not match any decoder level"
            return x, y_src
        return x, None

    def _apply_fusion(self, x: torch.Tensor, fusion_source: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.fusion_role == "source":
            return x, x
        if self.fusion is not None:
            x = self.fusion(fusion_source, x)
        return x, None


class YUVLimiter(nn.Module):

    _Y_LIMITS = (0., 1.)
    _U_LIMITS = (-0.436, 0.436)
    _V_LIMITS = (-0.615, 0.615)

    def __init__(self, mode: str | None = None):
        assert mode in [None, "none", "tanh", "sigmoid", "clamp"], f"Invalid mode: {mode}"
        super(YUVLimiter, self).__init__()
        self._mode = mode

        self.register_buffer("_y_min", torch.tensor(self._Y_LIMITS[0]))
        self.register_buffer("_y_max", torch.tensor(self._Y_LIMITS[1]))
        self.register_buffer("_u_min", torch.tensor(self._U_LIMITS[0]))
        self.register_buffer("_u_max", torch.tensor(self._U_LIMITS[1]))
        self.register_buffer("_v_min", torch.tensor(self._V_LIMITS[0]))
        self.register_buffer("_v_max", torch.tensor(self._V_LIMITS[1]))

    def forward(self, y: torch.Tensor, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._mode is None or self._mode == "none":
            return y, uv

        u = uv[:, 0]
        v = uv[:, 1]

        if self._mode == "tanh":
            y = self._limit_tanh(y, self._y_min, self._y_max)
            u = self._limit_tanh(u, self._u_min, self._u_max)
            v = self._limit_tanh(v, self._v_min, self._v_max)
        elif self._mode == "sigmoid":
            y = self._limit_sigmoid(y, self._y_min, self._y_max)
            u = self._limit_sigmoid(u, self._u_min, self._u_max)
            v = self._limit_sigmoid(v, self._v_min, self._v_max)
        elif self._mode == "clamp":
            y = self._limit_clamp(y, self._y_min, self._y_max)
            u = self._limit_clamp(u, self._u_min, self._u_max)
            v = self._limit_clamp(v, self._v_min, self._v_max)
        else:
            raise ValueError(f"Invalid mode: {self._mode}")
        
        return y, torch.stack([u, v], dim=1)

    def _limit_tanh(self, 
                    x: torch.Tensor, 
                    x_min: torch.Tensor, 
                    x_max: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min
    
    def _limit_sigmoid(self, 
                       x: torch.Tensor, 
                       x_min: torch.Tensor, 
                       x_max: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * (x_max - x_min) + x_min
    
    def _limit_clamp(self, 
                     x: torch.Tensor, 
                     x_min: torch.Tensor, 
                     x_max: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, x_min, x_max)


class WanYUVSplitVAE_(nn.Module):
    """WAN VAE with split Y/UV branches in YUV color space.

    The Y (luma) branch uses a larger base dimension than the UV (chroma)
    branch, reflecting the higher perceptual importance of luminance.
    The combined latent is a channel-wise concatenation of both branches.

    Extension point for cross-branch decoder interaction:
        Override decode() and use StagedDecoder.forward_with_intermediates()
        to access per-stage features at matching spatial resolutions. The
        stage_out_dims attribute on each StagedDecoder lists the channel
        count after each stage for sizing interaction modules.
    """

    def __init__(self,
                 y_dim=80,
                 uv_dim=56,
                 z_dim_y=12,
                 z_dim_uv=4,
                 dim_mult=[1, 2, 4, 4],
                 y_num_res_blocks=2,
                 uv_num_res_blocks=1,
                 attn_scales=[],
                 dropout=0.0,
                 yuv_limit_mode: str | None = None,
                 fusion_type: str | None = None,
                 fusion_level: str | None = None):
        super().__init__()
        self.y_dim = y_dim
        self.uv_dim = uv_dim
        self.z_dim_y = z_dim_y
        self.z_dim_uv = z_dim_uv
        self.z_dim = z_dim_y + z_dim_uv
        self.dim_mult = dim_mult
        self.y_num_res_blocks = y_num_res_blocks
        self.uv_num_res_blocks = uv_num_res_blocks
        self.attn_scales = attn_scales

        # Y (luma) branch — larger capacity, acts as fusion source
        self.y_encoder = Encoder(y_dim, z_dim_y * 2, dim_mult,
                                 y_num_res_blocks, attn_scales, dropout,
                                 in_channels=1)
        self.y_conv1 = nn.Conv2d(z_dim_y * 2, z_dim_y * 2, 1)
        self.y_conv2 = nn.Conv2d(z_dim_y, z_dim_y, 1)
        self.y_decoder = StagedDecoder(dim=y_dim, z_dim=z_dim_y,
                                       out_channels=1,
                                       dim_mult=dim_mult,
                                       num_res_blocks=y_num_res_blocks,
                                       attn_scales=attn_scales,
                                       dropout=dropout,
                                       fusion_type=fusion_type,
                                       fusion_role="source",
                                       fusion_level=fusion_level)

        # Compute source channel dim at fusion level for the target decoder
        fusion_source_dim = (self.y_decoder.dim_at_level(fusion_level)
                             if fusion_type and fusion_level else None)

        # UV (chroma) branch — smaller capacity, acts as fusion target
        self.uv_encoder = Encoder(uv_dim, z_dim_uv * 2, dim_mult,
                                  uv_num_res_blocks, attn_scales, dropout,
                                  in_channels=2)
        self.uv_conv1 = nn.Conv2d(z_dim_uv * 2, z_dim_uv * 2, 1)
        self.uv_conv2 = nn.Conv2d(z_dim_uv, z_dim_uv, 1)
        self.uv_decoder = StagedDecoder(dim=uv_dim, z_dim=z_dim_uv,
                                        out_channels=2,
                                        dim_mult=dim_mult,
                                        num_res_blocks=uv_num_res_blocks,
                                        attn_scales=attn_scales,
                                        dropout=dropout,
                                        fusion_type=fusion_type,
                                        fusion_role="target",
                                        fusion_level=fusion_level,
                                        fusion_source_dim=fusion_source_dim)

        self.yuv_limiter = YUVLimiter(yuv_limit_mode)

    def forward(self, x, sample_posterior=True) -> tuple[torch.Tensor, DiagonalGaussianDistribution]:
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def encode(self, x) -> DiagonalGaussianDistribution:
        yuv = rgb_to_yuv(x)
        y_params = self.y_encoder(yuv[:, :1])
        y_params = self.y_conv1(y_params)
        uv_params = self.uv_encoder(yuv[:, 1:])
        uv_params = self.uv_conv1(uv_params)

        y_mean, y_logvar = y_params.chunk(2, dim=1)
        uv_mean, uv_logvar = uv_params.chunk(2, dim=1)
        combined = torch.cat([y_mean, uv_mean, y_logvar, uv_logvar], dim=1)
        return DiagonalGaussianDistribution(combined)

    def decode(self, z) -> torch.Tensor:
        z_y = self.y_conv2(z[:, :self.z_dim_y])
        z_uv = self.uv_conv2(z[:, self.z_dim_y:])

        y_dec, z_fusion = self.y_decoder(z_y, None)
        uv_dec, _ = self.uv_decoder(z_uv, z_fusion)

        y, uv = self.yuv_limiter(y_dec, uv_dec)

        yuv = torch.cat([y, uv], dim=1)
        rgb = yuv_to_rgb(yuv)

        return rgb


def _image_yuv_split_vae(pretrained_path=None, z_dim_y=None, z_dim_uv=None,
                         yuv_limit_mode: str | None = None,
                         fusion_type: str | None = None,
                         fusion_level: str | None = None,
                         device='cpu', **kwargs):
    cfg = dict(
        y_dim=80,
        uv_dim=56,
        z_dim_y=z_dim_y,
        z_dim_uv=z_dim_uv,
        dim_mult=[1, 2, 4, 4],
        y_num_res_blocks=2,
        uv_num_res_blocks=1,
        attn_scales=[],
        dropout=0.0,
        yuv_limit_mode=yuv_limit_mode,
        fusion_type=fusion_type,
        fusion_level=fusion_level)
    cfg.update(**kwargs)

    model = WanYUVSplitVAE_(**cfg)

    if pretrained_path:
        logging.info(f'loading {pretrained_path}')
        try:
            model.load_state_dict(
                torch.load(pretrained_path, map_location=device), assign=True)
        except RuntimeError:
            logging.warning(f'Failed to load {pretrained_path}! Try to load with `model_state_dict` key.')
            state_dict = torch.load(pretrained_path, map_location=device)['model']

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_key = k.replace('model.', '')
                else:
                    new_key = k
                new_state_dict[new_key] = v

            model.load_state_dict(new_state_dict, assign=True)

    return model


def _get_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype = dtype.lower()
    if hasattr(torch, dtype):
        return getattr(torch, dtype)
    raise ValueError(f"Unsupported dtype: {dtype}")


class WanImageYUVSplitVAE(nn.Module):

    def __init__(self,
                 z_dim_y=12,
                 z_dim_uv=4,
                 pretrained_path=None,
                 dtype=torch.bfloat16,
                 yuv_limit_mode: str | None = None,
                 fusion_type: str | None = None,
                 fusion_level: str | None = None,
                 device: str | None = None):
        super(WanImageYUVSplitVAE, self).__init__()

        dtype = _get_dtype(dtype)
        self.dtype = dtype

        if device is None:
            local_rank = int(os.getenv("LOCAL_RANK", 0))
            self.device = local_rank
        else:
            self.device = device

        if pretrained_path:
            pretrained_path = pretrained_path if pretrained_path.startswith("/") else os.path.join(
                os.environ["PRETRAINED_PATH"], pretrained_path
            )
        self.pretrained_path = pretrained_path
        self.z_dim_y = z_dim_y
        self.z_dim_uv = z_dim_uv
        self.z_dim = z_dim_y + z_dim_uv
        self.yuv_limit_mode = yuv_limit_mode
        self.fusion_type = fusion_type
        self.fusion_level = fusion_level
        
        self.model = _image_yuv_split_vae(
            pretrained_path=pretrained_path,
            z_dim_y=z_dim_y,
            z_dim_uv=z_dim_uv,
            yuv_limit_mode=yuv_limit_mode,
            fusion_type=fusion_type,
            fusion_level=fusion_level,
        ).to(self.device)

    def forward(self, img):
        """Perform forward pass.

        Parameters
        ----------
        img : torch.Tensor
            RGB image tensor with shape [B, 3, H, W].
        """
        with amp.autocast(dtype=self.dtype):
            return self.model(img)

    def get_parameters(self):
        """Return parameters."""
        return {
            "pretrained_path": self.pretrained_path,
            "z_dim_y": self.z_dim_y,
            "z_dim_uv": self.z_dim_uv,
            "yuv_limit_mode": self.yuv_limit_mode,
            "fusion_type": self.fusion_type,
            "fusion_level": self.fusion_level,
        }

    def encode(self, img):
        """Encode RGB image to latent distribution.

        Parameters
        ----------
        img : torch.Tensor
            RGB image with shape [B, 3, H, W].
        """
        return self.model.encode(img)

    def decode(self, zs):
        """Decode latent to RGB image."""
        return self.model.decode(zs)

    def get_last_layer(self):
        """Get last decoder layer (Y branch, the primary branch)."""
        for module in self.model.y_decoder.head.modules():
            if isinstance(module, nn.Conv2d):
                return module.weight
        raise ValueError("No last layer found")

    def get_last_layers(self):
        """Get last decoder layers for both branches.

        Returns dict with 'y' and 'uv' keys mapping to the final
        Conv2d weight of each decoder branch. Useful for per-branch
        adaptive loss weighting.
        """
        y_weight, uv_weight = None, None
        for module in self.model.y_decoder.head.modules():
            if isinstance(module, nn.Conv2d):
                y_weight = module.weight
        for module in self.model.uv_decoder.head.modules():
            if isinstance(module, nn.Conv2d):
                uv_weight = module.weight
        if y_weight is None or uv_weight is None:
            raise ValueError("No last layer found")
        return {'y': y_weight, 'uv': uv_weight}
