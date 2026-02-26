import autoroot

from pathlib import Path

import json
import fire
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from dit_updates.vae.adapters.base import VAEAdapter
from dit_updates.vae.adapters.wan_official import WANOfficialAdapter
from dit_updates.data.imagenet import create_imagenet_dataset
from dit_updates.data.transforms import DiTCenterCrop


AVAILABLE_DATASETS = [
    "ImageNet2012_200",
    "ImageNet2012"
]

AVAILABLE_MODELS = [
    "wan-2.1-official"
]


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
    if model == "wan-2.1-official":
        return WANOfficialAdapter(latent_norm_type="none",
                                  latent_stats=None,
                                  device=device)
    raise ValueError(f"Invalid model: {model}")


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

    Args:
        split (str): The dataset split (e.g., "train", "val").
        dataset_name (str): The name of the dataset (e.g., "ImageNet2012").
        resolution (int): Resolution for input images.
        model (VAEAdapter): VAE adapter/model for encoding images.
        batch_size (int): Batch size for the DataLoader.
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
    dtype_suffix = "__float16" if float16 else ""
    output_root = Path(output_root) / f"{dataset_name}__{model.name}__resolution_{resolution}{dtype_suffix}" / f"{split}"
    
    if output_root.is_dir():
        raise FileExistsError(f"Output dir exists: {output_root}")
    else:
        print(f"Creating output directory {output_root}")
        output_root.mkdir(parents=True, exist_ok=True)

    dataset = create_imagenet_dataset(
        image_dir=f"{dataset_name}/{split}", 
        transform=T.Compose([DiTCenterCrop(resolution), 
                             T.ToTensor(),
                             model.create_preprocessor()]))

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=n_workers, 
        pin_memory=True
    )

    buffer = []
    label_buffer = []
    buffer_size = 0
    chunk_idx = 0

    # Streaming per-channel stats accumulators (float64 for numerical stability)
    channel_sum = None      # shape (C,)
    channel_sum_sq = None   # shape (C,)
    pixel_count = 0         # total number of spatial elements per channel

    for images, labels in tqdm(dataloader, desc=f"Processing {dataset_name} {split}"):
        images = images.to(device)
        _, info = model.encode(images, normalize=False, sample=False)
        features_mean = info["raw_distribution"].mean.cpu().numpy()
        features_std = info["raw_distribution"].std.cpu().numpy()
        features = np.concatenate([features_mean, features_std], axis=1)

        # Accumulate per-channel stats in float64 before potential float16 cast
        # Take mean only
        feats_f64 = features_mean.astype(np.float64)  # (B, C, H, W)
        if channel_sum is None:
            n_channels = feats_f64.shape[1]
            channel_sum = np.zeros(n_channels, dtype=np.float64)
            channel_sum_sq = np.zeros(n_channels, dtype=np.float64)
        # Sum over batch and spatial dims, keep channel dim
        channel_sum += feats_f64.sum(axis=(0, 2, 3))      # (C,)
        channel_sum_sq += (feats_f64 ** 2).sum(axis=(0, 2, 3))  # (C,)
        pixel_count += feats_f64.shape[0] * feats_f64.shape[2] * feats_f64.shape[3]

        if float16:
            features = features.astype(np.float16)

        buffer.append(features)
        label_buffer.append(labels.numpy())
        buffer_size += features.shape[0]

        if buffer_size >= chunk_size:
            chunk_data = np.concatenate(buffer, axis=0)
            save_path = output_root / f"chunk_{chunk_idx:06d}.npy"
            np.save(save_path, chunk_data)
            
            buffer = []
            buffer_size = 0
            chunk_idx += 1
        
    if len(buffer) > 0:
        chunk_data = np.concatenate(buffer, axis=0)
        save_path = output_root / f"chunk_{chunk_idx:06d}.npy"
        np.save(save_path, chunk_data)

    # Save all labels as a single array (int64, small footprint)
    all_labels = np.concatenate(label_buffer, axis=0)
    labels_path = output_root / "labels.npy"
    np.save(labels_path, all_labels)

    # Compute per-channel mean and std from accumulators
    channel_mean = channel_sum / pixel_count                              # (C,)
    channel_std = np.sqrt(channel_sum_sq / pixel_count - channel_mean ** 2)  # (C,)
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
    batch_size: int = 8,
    n_workers: int = 8,
    chunk_size: int = 10_000,
    float16: bool = False,
    device: str = "cuda"
):
    """
    Main script entry point for precomputing latents from an image dataset.

    Args:
        dataset (str): The input dataset name. Must be in AVAILABLE_DATASETS.
        model (str): The VAE model name. Must be in AVAILABLE_MODELS.
        output_root (str): Output directory for all generated files.
        resolution (int, optional): Image resolution for cropping. Defaults to 256.
        batch_size (int, optional): Batch size for encoding. Defaults to 8.
        n_workers (int, optional): Number of DataLoader workers. Defaults to 8.
        chunk_size (int, optional): Samples per chunk file. Defaults to 10_000.
        float16 (bool, optional): Store latents as float16. Defaults to False.
        device (str, optional): Torch device (e.g. "cuda"). Defaults to "cuda".

    Returns:
        None
    """
    assert dataset in AVAILABLE_DATASETS, f"Invalid dataset: {dataset}"
    assert model in AVAILABLE_MODELS, f"Invalid model: {model}"

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

    with open(output_dir.parent / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
