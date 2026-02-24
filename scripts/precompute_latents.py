import autoroot

from pathlib import Path

import json
import fire
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from lib.vae.adapters import FLUXOfficialAdapter, VAEAdapter, WANOfficialAdapter
from lib.data.imagenet import create_imagenet_dataset
from lib.data.transforms import DiTCenterCrop


AVAILABLE_DATASETS = [
    "ImageNet2012_200",
    "ImageNet2012"
]

AVAILABLE_MODELS = [
    "wan-2.1-official",
    "flux-official",
]


def resolve_model(model: str, device: str) -> VAEAdapter:
    if model == "wan-2.1-official":
        return WANOfficialAdapter(latent_norm_type="none",
                                  latent_stats=None,
                                  device=device)
    elif model == "flux-official":
        return FLUXOfficialAdapter(latent_norm_type="none",
                                   latent_stats=None,
                                   device=device)
    raise ValueError(f"Invalid model: {model}")


@torch.inference_mode()
def process_split(split: str,
                  dataset_name: str,
                  resolution: int,
                  model: VAEAdapter,
                  batch_size: int,
                  chunk_size: int,
                  output_root: str,
                  float16: bool,
                  n_workers: int,
                  device: str) -> tuple[Path, dict]:
    dtype_suffix = "__float16" if float16 else ""
    output_root = Path(output_root) / \
        f"{dataset_name}__{model.name}__resolution_{resolution}{dtype_suffix}" / f"{split}"

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

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=n_workers,
                            pin_memory=True)

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


def main(dataset: str,
         model: str,
         output_root: str,
         resolution: int = 256,
         batch_size: int = 8,
         n_workers: int = 8,
         chunk_size: int = 10_000,
         float16: bool = False,
         device: str = "cuda"):

    assert dataset in AVAILABLE_DATASETS, f"Invalid dataset: {dataset}"
    assert model in AVAILABLE_MODELS, f"Invalid model: {model}"

    model = resolve_model(model, device)

    output_dir, metadata = process_split(split="train",
                                         dataset_name=dataset,
                                         model=model,
                                         resolution=resolution,
                                         batch_size=batch_size,
                                         output_root=output_root,
                                         n_workers=n_workers,
                                         device=device,
                                         chunk_size=chunk_size,
                                         float16=float16)

    _, _ = process_split(split="val",
                         dataset_name=dataset,
                         model=model,
                         resolution=resolution,
                         batch_size=batch_size,
                         output_root=output_root,
                         n_workers=n_workers,
                         device=device,
                         chunk_size=chunk_size,
                         float16=float16)

    with open(output_dir.parent / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
