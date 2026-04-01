import autoroot
import autorootcwd
import sys
sys.path.append("sbervae")

import os
from pathlib import Path

import json
import fire
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from tqdm import tqdm
from dit_updates.vae.adapters.base import VAEAdapter
from dit_updates.data.imagenet import create_imagenet_dataset
from dit_updates.data.transforms import DiTCenterCrop
from dit_updates.vae.adapters.registry import resolve_adapter


AVAILABLE_DATASETS = [
    "ImageNet2012_200",
    "ImageNet2012"
]


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_distributed() else 1


def _setup_distributed() -> int:
    """Initialize distributed process group if launched via torchrun.

    Returns:
        int: local_rank (0 when not distributed).
    """
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def resolve_model(model: str, device: str) -> VAEAdapter:
    """
    Resolve the model string name to an instantiated VAEAdapter.

    Args:
        model (str): The model name.
        device (str): The device to instantiate the model on.

    Returns:
        VAEAdapter: The corresponding VAE adapter instance.

    Raises:
        ValueError: If an invalid model is supplied.
    """
    return resolve_adapter(model,
                           latent_norm_type="none",
                           latent_stats=None,
                           device=device)


@torch.inference_mode()
def process_split(
    split: str,
    dataset_name: str,
    resolution: int,
    model: VAEAdapter,
    batch_size: int,
    chunk_size: int,
    output_root: str,
    float16: bool,
    n_workers: int,
    device: str
) -> tuple[Path, dict]:
    """
    Process a data split to compute latents and statistics, saving the result as .npy chunks.

    Supports multi-GPU via ``torchrun``: the dataset is sharded across ranks, each
    rank writes its own chunk files (prefixed with ``r{rank:02d}_`` in distributed
    mode), and streaming statistics are all-reduced before the final mean/std is
    computed.  Falls back to single-GPU behaviour when not launched with torchrun.

    Args:
        split (str): The dataset split (e.g., "train", "val").
        dataset_name (str): The name of the dataset (e.g., "ImageNet2012").
        resolution (int): Resolution for input images.
        model (VAEAdapter): VAE adapter/model for encoding images.
        batch_size (int): Batch size for the DataLoader (per GPU).
        chunk_size (int): Number of samples per output chunk file.
        output_root (str): Root directory to store processed latents.
        float16 (bool): Whether to use FP16 precision for output arrays.
        n_workers (int): Number of DataLoader workers.
        device (str): PyTorch device.

    Returns:
        tuple[Path, dict]: The path to the output directory and metadata dictionary.

    Raises:
        FileExistsError: If the output directory already exists.
    """
    rank = _get_rank()
    world_size = _get_world_size()
    distributed = _is_distributed()

    dtype_suffix = "__float16" if float16 else ""
    output_root = Path(output_root) / \
        f"{dataset_name}__{model.name}__resolution_{resolution}{dtype_suffix}" / f"{split}"

    if rank == 0:
        if output_root.is_dir() and any(output_root.iterdir()):
            raise FileExistsError(f"Output dir exists: {output_root}")
        print(f"Creating output directory {output_root}")
        output_root.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
        output_root.mkdir(parents=True, exist_ok=True)

    dataset = create_imagenet_dataset(
        image_dir=f"{dataset_name}/{split}",
        transform=T.Compose([DiTCenterCrop(resolution),
                             T.ToTensor(),
                             model.create_preprocessor()]))

    # Shard the dataset across ranks (contiguous, no duplicates)
    if distributed:
        total_samples = len(dataset)
        per_rank = total_samples // world_size
        start_idx = rank * per_rank
        end_idx = start_idx + per_rank if rank < world_size - 1 else total_samples
        dataset = Subset(dataset, list(range(start_idx, end_idx)))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=False,
    )

    buffer = []
    label_buffer = []
    buffer_size = 0
    chunk_idx = 0
    chunk_prefix = f"r{rank:02d}_" if distributed else ""

    # Streaming per-channel stats accumulators (float64 for numerical stability)
    n_channels = model.n_channels
    channel_sum = np.zeros(n_channels, dtype=np.float64)
    channel_sum_sq = np.zeros(n_channels, dtype=np.float64)
    pixel_count = 0

    show_progress = rank == 0
    desc = f"Processing {dataset_name} {split}"

    for images, labels in tqdm(dataloader, desc=desc, disable=not show_progress):
        images = images.to(device)
        _, info = model.encode(images, normalize=False, sample=False)
        features_mean = info["raw_distribution"].mean.cpu().numpy()
        features_std = info["raw_distribution"].std.cpu().numpy()
        features = np.concatenate([features_mean, features_std], axis=1)

        # Accumulate per-channel stats in float64 before potential float16 cast
        feats_f64 = features_mean.astype(np.float64)  # (B, C, H, W)
        channel_sum += feats_f64.sum(axis=(0, 2, 3))
        channel_sum_sq += (feats_f64 ** 2).sum(axis=(0, 2, 3))
        pixel_count += feats_f64.shape[0] * feats_f64.shape[2] * feats_f64.shape[3]

        if float16:
            features = features.astype(np.float16)

        buffer.append(features)
        label_buffer.append(labels.numpy())
        buffer_size += features.shape[0]

        if buffer_size >= chunk_size:
            chunk_data = np.concatenate(buffer, axis=0)
            save_path = output_root / f"chunk_{chunk_prefix}{chunk_idx:06d}.npy"
            with open(save_path, 'wb') as f:
                np.save(f, chunk_data)

            buffer = []
            buffer_size = 0
            chunk_idx += 1

    if len(buffer) > 0:
        chunk_data = np.concatenate(buffer, axis=0)
        save_path = output_root / f"chunk_{chunk_prefix}{chunk_idx:06d}.npy"
        with open(save_path, 'wb') as f:
            np.save(f, chunk_data)

    # ---- All-reduce streaming statistics across ranks ----
    if distributed:
        channel_sum_t = torch.from_numpy(channel_sum).to(device)
        channel_sum_sq_t = torch.from_numpy(channel_sum_sq).to(device)
        pixel_count_t = torch.tensor(pixel_count, dtype=torch.float64, device=device)
        dist.all_reduce(channel_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(channel_sum_sq_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(pixel_count_t, op=dist.ReduceOp.SUM)
        channel_sum = channel_sum_t.cpu().numpy()
        channel_sum_sq = channel_sum_sq_t.cpu().numpy()
        pixel_count = int(pixel_count_t.item())

    # ---- Save / merge labels ----
    if distributed:
        all_labels = np.concatenate(label_buffer, axis=0)
        tmp_labels_path = output_root / f"_labels_r{rank:02d}.npy"
        with open(tmp_labels_path, 'wb') as f:
            np.save(f, all_labels)
        dist.barrier()
        if rank == 0:
            label_parts = []
            for r in range(world_size):
                part_path = output_root / f"_labels_r{r:02d}.npy"
                label_parts.append(np.load(part_path))
                part_path.unlink()
            merged_labels = np.concatenate(label_parts, axis=0)
            with open(output_root / "labels.npy", 'wb') as f:
                np.save(f, merged_labels)
    else:
        all_labels = np.concatenate(label_buffer, axis=0)
        with open(output_root / "labels.npy", 'wb') as f:
            np.save(f, all_labels)

    # ---- Compute per-channel mean and std from accumulators ----
    channel_mean = channel_sum / pixel_count
    channel_std = np.sqrt(channel_sum_sq / pixel_count - channel_mean ** 2)
    if not float16:
        channel_mean = channel_mean.astype(np.float32)
        channel_std = channel_std.astype(np.float32)
    else:
        channel_mean = channel_mean.astype(np.float16)
        channel_std = channel_std.astype(np.float16)

    metadata = {
        "backend": "numpy",
        "latent_format": "mean_std",
        "dtype": "float16" if float16 else "float32",
        "model": model.name,
        "stats": {
            "mean": channel_mean.tolist(),
            "std": channel_std.tolist(),
        }
    }

    return output_root, metadata


def main(
    dataset: str,
    model: str,
    output_root: str,
    resolution: int = 256,
    batch_size: int = 64,
    n_workers: int = 4,
    chunk_size: int = 10_000,
    float16: bool = False,
    device: str = "cuda",
    local_rank: int = 0,
):
    """
    Main script entry point for precomputing latents from an image dataset.

    Supports multi-GPU via ``torchrun``::

        torchrun --nproc_per_node=NUM_GPUS scripts/precompute_latents.py \\
            --dataset ImageNet2012 --model <model> --output_root <path>

    Single-GPU usage is unchanged (``python scripts/precompute_latents.py ...``).

    Args:
        dataset (str): The input dataset name. Must be in AVAILABLE_DATASETS.
        model (str): The VAE model name. Must be in AVAILABLE_MODELS.
        output_root (str): Output directory for all generated files.
        resolution (int, optional): Image resolution for cropping. Defaults to 256.
        batch_size (int, optional): Per-GPU batch size for encoding. Defaults to 128.
        n_workers (int, optional): Number of DataLoader workers. Defaults to 16.
        chunk_size (int, optional): Samples per chunk file. Defaults to 10_000.
        float16 (bool, optional): Store latents as float16. Defaults to False.
        device (str, optional): Torch device (e.g. "cuda"). Defaults to "cuda".
        local_rank (int, optional): Unused; accepted for torch.distributed.launch
            compatibility.

    Returns:
        None
    """
    torch.multiprocessing.set_sharing_strategy('file_system')

    assert dataset in AVAILABLE_DATASETS, f"Invalid dataset: {dataset}"

    actual_local_rank = _setup_distributed()
    rank = _get_rank()
    world_size = _get_world_size()

    if _is_distributed():
        device = f"cuda:{actual_local_rank}"
        if rank == 0:
            print(f"Running distributed latent precomputation on {world_size} GPUs")

    model = resolve_model(model, device)

    output_dir, metadata = process_split(
        split="train",
        dataset_name=dataset,
        model=model,
        resolution=resolution,
        batch_size=batch_size,
        output_root=output_root,
        n_workers=n_workers,
        device=device,
        chunk_size=chunk_size,
        float16=float16
    )

    _, _ = process_split(
        split="val",
        dataset_name=dataset,
        model=model,
        resolution=resolution,
        batch_size=batch_size,
        output_root=output_root,
        n_workers=n_workers,
        device=device,
        chunk_size=chunk_size,
        float16=float16
    )

    if rank == 0:
        with open(output_dir.parent / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    if _is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
