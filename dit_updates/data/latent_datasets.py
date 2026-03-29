import bisect
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dit_updates.utils.files import resolve_path
from dit_updates.vae.models.normalization import NumPyLatentNormalizer


class DataSource(ABC):
    """
    Interface for accessing raw latents and labels stored in sharded .npy chunk files.

    Subclasses are responsible for all storage and indexing logic.
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the total number of samples.

        Returns:
            int: Total number of samples.
        """
        ...

    @abstractmethod
    def get_latent(self, idx: int) -> np.ndarray:
        """
        Return the raw latent (mean+std) array for the given global index.

        Args:
            idx (int): Global sample index.

        Returns:
            np.ndarray: The raw latent array (mean and std concatenated).
        """
        ...

    @abstractmethod
    def get_label(self, idx: int) -> int:
        """
        Return the label for the given global index.

        Args:
            idx (int): Global sample index.

        Returns:
            int: The integer class label.
        """
        ...


class MemoryMappedDataSource(DataSource):
    """
    DataSource that reads latents lazily from memory-mapped .npy files.
    Uses minimal RAM at the cost of slower random access.
    """

    def __init__(self, chunks: List[Path], labels: np.ndarray):
        """
        Initialize the MemoryMappedDataSource.

        Args:
            chunks (List[Path]): Sorted list of paths to sharded .npy chunk files.
            labels (np.ndarray): Array of integer labels, one per sample.
        """
        self._chunks = chunks
        self._labels = labels

        self._cumulative_sizes: List[int] = []
        current_cum = 0
        for path in chunks:
            current_cum += np.load(path, mmap_mode="r").shape[0]
            self._cumulative_sizes.append(current_cum)

        self._total_size = current_cum

    def __len__(self) -> int:
        """
        Get the total number of samples across all chunks.

        Returns:
            int: Total number of samples.
        """
        return self._total_size

    def get_latent(self, idx: int) -> np.ndarray:
        """
        Return the raw latent (mean+std) array for the given global index.

        Opens the corresponding chunk file in memory-mapped mode and copies
        only the requested row.

        Args:
            idx (int): Global sample index.

        Returns:
            np.ndarray: The raw latent array (mean and std concatenated).
        """
        chunk_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        local_idx = idx if chunk_idx == 0 else idx - self._cumulative_sizes[chunk_idx - 1]
        mmap_data = np.load(self._chunks[chunk_idx], mmap_mode="r")
        return mmap_data[local_idx].copy()

    def get_label(self, idx: int) -> int:
        """
        Return the label for the given global index.

        Args:
            idx (int): Global sample index.

        Returns:
            int: The integer class label.
        """
        return self._labels[idx]


class InMemoryDataSource(DataSource):
    """
    DataSource that loads and concatenates all chunks into a single contiguous
    array in RAM at initialization time for fast random access.
    """

    def __init__(self, chunks: List[Path], labels: np.ndarray):
        """
        Initialize the InMemoryDataSource.

        Loads all chunk files and concatenates them into a single array.

        Args:
            chunks (List[Path]): Sorted list of paths to sharded .npy chunk files.
            labels (np.ndarray): Array of integer labels, one per sample.
        """
        self._labels = labels
        self._data: np.ndarray = np.concatenate(
            [np.load(path) for path in chunks], axis=0
        )

    def __len__(self) -> int:
        """
        Get the total number of samples.

        Returns:
            int: Total number of samples.
        """
        return self._data.shape[0]

    def get_latent(self, idx: int) -> np.ndarray:
        """
        Return the raw latent (mean+std) array for the given global index.

        Args:
            idx (int): Global sample index.

        Returns:
            np.ndarray: The raw latent array (mean and std concatenated).
        """
        return self._data[idx].copy()

    def get_label(self, idx: int) -> int:
        """
        Return the label for the given global index.

        Args:
            idx (int): Global sample index.

        Returns:
            int: The integer class label.
        """
        return self._labels[idx]


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
        num_samples: Optional[int] = None,
        in_memory: bool = False,
    ):
        """
        Initialize the LatentsShardDataset.

        Args:
            latents_root (str): Path to the root of the latents dataset.
            split (str): Dataset split, e.g., "train" or "val".
            latent_normalizer (NumPyLatentNormalizer): Normalizer instance to apply to latents.
            sample (bool, optional): Whether to sample using mean/std (True) or take only means (False).
            num_samples (int, optional): Maximum number of samples to use. If None, use all samples.
            in_memory (bool, optional): If True, load all latents into RAM; otherwise use memory-mapped files.
        """
        super(LatentsShardDataset, self).__init__()
        self._split = split
        self._sample = sample
        self._num_samples = num_samples

        latents_root = resolve_path(latents_root, "dataset")
        self._metadata_path = latents_root / "metadata.json"

        with open(self._metadata_path, "r") as f:
            metadata = json.load(f)
        latents_format = metadata["latent_format"]
        if latents_format != "mean_std":
            raise ValueError(f"Invalid latent format: {latents_format}")

        latents_dir = Path(latents_root) / split
        chunks = sorted(latents_dir.glob("chunk*.npy"))
        if len(chunks) == 0:
            raise FileNotFoundError(f"No chunks found in {latents_dir}")

        labels = np.load(latents_dir / "labels.npy")

        source_cls = InMemoryDataSource if in_memory else MemoryMappedDataSource
        self._source: DataSource = source_cls(chunks, labels)

        self._total_size = len(self._source)
        if num_samples is not None:
            self._total_size = min(num_samples, self._total_size)

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
        if idx < 0:
            idx += self._total_size
        if idx >= self._total_size or idx < 0:
            raise IndexError("Index out of range")

        label = torch.tensor(self._source.get_label(idx), dtype=torch.long)

        latent_mean_std = self._source.get_latent(idx)
        latent = self._sample_latent(latent_mean_std)

        latent = self._latent_normalizer.normalize(latent)

        latent = torch.from_numpy(latent).float()

        return latent, label

    @property
    def metadata_path(self) -> Path:
        """
        Get the path to the metadata file.

        Returns:
            Path: Path to the metadata file.
        """
        return self._metadata_path

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
