import torch
import torch.nn.functional as F


class LogitNormal:
    def __init__(self, sigma: float):
        self.sigma = sigma

    def sample(self, n_samples):
        gauss_rv = torch.randn(n_samples) * self.sigma
        return F.sigmoid(gauss_rv)


def form_flow(x, y, time, mode):
    """Convert model output to flow prediction based on the prediction mode.

    'flow':   model directly predicts the velocity v = x1 - x0
    'target': model predicts x1, converted to velocity via (y - x) / (1 - t)
    """
    if mode == 'flow':
        return y
    elif mode == 'target':
        assert time.ndim == 4, f'time in wrong dimension, expected ndim = 4, found ndim = {time.ndim}'
        scale = (1 / (1 - time)).clip(1, 128)
        return (y - x) * scale
    else:
        raise ValueError(f'Unknown mode {mode}')


class RectifiedFlowMatching:
    """Rectified Flow Matching objective for DiT training.

    Implements straight-path interpolation between noise x0 and data x1:
        x_t = (1 - t) * x0 + t * x1
    with velocity target v = x1 - x0.

    The API (training_losses, p_sample_loop, num_timesteps) is designed to be
    a drop-in replacement for GaussianDiffusion / SpacedDiffusion in the
    existing training loop.
    """
    def __init__(self,
                 time_shift: float,
                 time_emb_scale: float,
                 num_val_steps: int,
                 base_sampler: str,
                 sigma_time_dist: float,
                 pred_mode: str):
        self.time_shift = time_shift
        self.time_emb_scale = time_emb_scale
        self.num_val_steps = num_val_steps
        self.pred_mode = pred_mode

        if base_sampler == 'logit-normal':
            self.base_sampler = LogitNormal(sigma=sigma_time_dist)
        else:
            raise ValueError(f'Unknown base sampler: {base_sampler}')

        # Compatibility shim: the training loop does
        #   t = torch.randint(0, diffusion.num_timesteps, ...)
        # With num_timesteps=1, t is always 0 and ignored by training_losses.
        self.num_timesteps = 1

    def sample_time(self, n_samples: int):
        t = self.base_sampler.sample(n_samples)
        t = self.shift(t)
        return t

    def shift(self, t):
        s = 1 - t
        q = self.time_shift * s / (1 + (self.time_shift - 1) * s)
        return 1 - q

    def training_losses(self, model, data, fake_time, model_kwargs):
        """Compute RFM training loss (MSE between predicted and true flow).

        Signature matches GaussianDiffusion.training_losses so the training
        loop doesn't need changes.
        """
        labels = model_kwargs['y']
        data = data.float()

        t = self.sample_time(data.size(0)).to(data.device).to(data.dtype)
        t = t.view(-1, 1, 1, 1)
        noise = torch.randn_like(data)
        noised = torch.lerp(noise, data, t)

        output = model(noised, t.view(-1) * self.time_emb_scale, labels)
        pred_flow = form_flow(noised, output, t, self.pred_mode)
        flow = data - noise
        return {'loss': F.mse_loss(pred_flow, flow)}

    @torch.no_grad()
    def p_sample_loop(self, model, noise, model_kwargs=None):
        """Deterministic Euler ODE sampling from t=0 (noise) to t=1 (data).

        Args:
            model: callable(x, t, **model_kwargs) -> flow prediction.
                   Can be model.forward or model.forward_with_cfg_flow.
            noise: (B, C, H, W) initial noise tensor.
            model_kwargs: dict passed as **kwargs to model (e.g. y, cfg_scale).
        """
        if model_kwargs is None:
            model_kwargs = {}
        dtype, device = noise.dtype, noise.device

        times = torch.linspace(0, 1, self.num_val_steps + 1, dtype=dtype, device=device)
        times = times[None, :].repeat_interleave(noise.size(0), dim=0)
        times = times.view(noise.size(0), -1, 1, 1)

        x, step = noise, 1 / self.num_val_steps
        for i in range(self.num_val_steps):
            t = times[:, i]
            output = model(x, t.view(-1) * self.time_emb_scale, **model_kwargs)
            flow = form_flow(x, output, t.view(-1, 1, 1, 1), self.pred_mode)
            x = x + flow * step

        return x
