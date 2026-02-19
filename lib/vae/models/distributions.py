import torch
import numpy as np

from abc import ABC, abstractmethod


class AbstractDistribution(ABC):
    """Abstract distribution."""

    @abstractmethod
    def sample(self):
        """Sample from distribution."""
        pass

    @abstractmethod
    def mode(self):
        """Get the distribution mode."""
        pass


class DiagonalGaussianDistribution(AbstractDistribution):
    """Diagonal Gaussian distribution."""

    def __init__(self, 
                 mean: torch.Tensor, 
                 logvar: torch.Tensor, 
                 deterministic=False):
        super(DiagonalGaussianDistribution, self).__init__()
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        """Sample from distribution."""
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        """Compute KL div."""
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1, 2, 3]):
        """Compute nll."""
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        """Get the distribution mode."""
        return self.mean
