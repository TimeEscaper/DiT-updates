import bisect
import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dit_updates.utils.files import resolve_path
from dit_updates.vae.models.normalization import NumPyLatentNormalizer


class LatentsShardDataset(Dataset):
    """
    Dataset for loading VAE latents stored in sharded .npy files. Each sample contains the latent vector
    (optionally sampled with mean/std) and its corresponding label. Supports efficient lazy loading 
    and normalization.
    """

    def __init__(
        self,
        latents_root: str,
        split: str,
        latent_normalizer: NumPyLatentNormalizer,
        sample: bool = True,
        num_samples: int = None
    ):
        """
        Initialize the LatentsShardDataset.

        Args:
            latents_root (str): Path to the root of the latents dataset.
            split (str): Dataset split, e.g., "train" or "val".
            latent_normalizer (NumPyLatentNormalizer): Normalizer instance to apply to latents.
            sample (bool, optional): Whether to sample using mean/std (True) or take only means (False).
            num_samples (int, optional): Maximum number of samples to use. If None, use all samples.
        """
        super(LatentsShardDataset, self).__init__()
        self._split = split
        self._sample = sample
        self._num_samples = num_samples

        latents_root = resolve_path(latents_root, "dataset")

        with open(latents_root / "metadata.json", "r") as f:
            metadata = json.load(f)
        latents_format = metadata["latent_format"]
        if latents_format != "mean_std":
            raise ValueError(f"Invalid latent format: {latents_format}")

        self._latents_dir = Path(latents_root) / split
        chunks = sorted(self._latents_dir.glob("chunk*.npy"))
        self._chunks = chunks
        if len(chunks) == 0:
            raise FileNotFoundError(f"No chunks found in {self._latents_dir}")

        # Build Index Mapping
        # We need to know which global index corresponds to which chunk and local index.
        # This requires reading the header of each file to get the shape (fast).
        self._chunk_sizes = []
        self._cumulative_sizes = []

        current_cum = 0

        for path in chunks:
            # mmap_mode='r' reads metadata without loading data
            # shape is usually (N_samples, N_patches, Dim)
            shape = np.load(path, mmap_mode='r').shape
            count = shape[0]

            self._chunk_sizes.append(count)
            current_cum += count
            self._cumulative_sizes.append(current_cum)

        self._total_size = current_cum

        self._labels = np.load(self._latents_dir / "labels.npy")

        if num_samples is not None:
            self._total_size = min(num_samples, self._total_size)
            self._labels = self._labels[:self._total_size]

        self._latent_normalizer = latent_normalizer

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self._total_size

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (latent, label), where latent is a normalized torch.FloatTensor 
            and label is a torch tensor of type long.

        Raises:
            IndexError: If idx is out of range.
        """
        # Handle standard slicing/indexing
        if idx < 0:
            idx += self._total_size
        if idx >= self._total_size or idx < 0:
            raise IndexError("Index out of range")

        label = torch.tensor(self._labels[idx], dtype=torch.long)

        # Find the correct chunk
        # bisect_right finds the first insertion point strictly greater than idx
        # This gives us the index of the chunk in cumulative_sizes
        chunk_idx = bisect.bisect_right(self._cumulative_sizes, idx)

        # 4. Calculate local index within that chunk
        if chunk_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self._cumulative_sizes[chunk_idx - 1]

        # Load Data
        # We open ONLY the required file in memory-mapped mode.
        # This is fast and memory efficient.
        chunk_path = self._chunks[chunk_idx]

        # Open file (lazy load)
        mmap_data = np.load(chunk_path, mmap_mode='r')

        # Read the specific sample.
        # The copy() ensures we detach from the mmap and return a standard array
        latent_mean_std = mmap_data[local_idx].copy()
        latent = self._sample_latent(latent_mean_std)

        latent = self._latent_normalizer.normalize(latent)

        latent = torch.from_numpy(latent).float()

        return latent, label

    def _sample_latent(self, mean_std: np.ndarray) -> np.ndarray:
        """
        Separate the mean and std from the input array and return either the mean (deterministic)
        or a sampled latent vector (if self._sample=True).

        Args:
            mean_std (np.ndarray): Array containing concatenated means and stds.

        Returns:
            np.ndarray: The sampled latent vector, or the mean if sampling is disabled.
        """
        split_idx = mean_std.shape[0] // 2
        mean = mean_std[:split_idx]
        if not self._sample:
            return mean
        std = mean_std[split_idx:]
        return np.random.randn(*mean.shape) * std + mean
