# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import autoroot
import autorootcwd
import sys
sys.path.append("sbervae")

import torch
import torch.distributed as dist
from models import DiT_models
from download import find_model
from diffusion import create_diffusion, create_rfm
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from typing import Optional

from dit_updates.vae.adapters.registry import resolve_adapter


def resolve_vae_id(ckpt_path: str, vae_arg: Optional[str]) -> str:
    """Return explicit --vae, or read a single-line VAE id from a file named ``vae`` next to the checkpoint."""
    if vae_arg is not None:
        return vae_arg
    vae_file = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "vae")
    if not os.path.isfile(vae_file):
        raise FileNotFoundError(
            f"--vae was not set and no sidecar file was found at {vae_file}. "
            "Pass --vae or add a one-line VAE id in a file named 'vae' beside the checkpoint."
        )
    with open(vae_file, encoding="utf-8") as f:
        vae_id = f.read().strip()
    if not vae_id:
        raise ValueError(f"VAE sidecar file {vae_file} is empty.")
    return vae_id


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    args.vae = resolve_vae_id(args.ckpt, args.vae)

    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Compute sample folder path and check for existing samples before loading models:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "")
    folder_name = (f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-"
                   f"cfg-{args.cfg_scale}-seed-{args.global_seed}")
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"

    existing_count = 0
    if rank == 0:
        existing_count = len([f for f in os.listdir(sample_folder_dir) if f.endswith('.png')])
    existing_count_tensor = torch.tensor([existing_count], device=device)
    dist.broadcast(existing_count_tensor, src=0)
    existing_count = existing_count_tensor.item()

    if existing_count >= args.num_fid_samples:
        if rank == 0:
            print(f"All {args.num_fid_samples} samples already exist. Skipping model loading.")
            create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
            print("Done.")
        dist.barrier()
        dist.destroy_process_group()
        return

    # Round down to complete global-batch boundary so we never leave index gaps
    complete_samples = (existing_count // global_batch_size) * global_batch_size
    remaining_samples = max(0, total_samples - complete_samples)

    if rank == 0:
        if complete_samples > 0:
            print(f"Found {existing_count} existing samples ({complete_samples} from complete batches).")
            print(f"Resuming generation from sample index {complete_samples}.")
        print(f"Remaining samples to generate: {remaining_samples}")

    # Load VAE:
    vae = resolve_adapter(args.vae, 
                          device=device,
                          latent_norm_type="scale",
                          latent_stats="imagenet2012")
    preprocessor = vae.create_preprocessor()

    # Load model:
    objective = args.objective
    learn_sigma = (objective != "rfm")
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=vae.n_channels,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
    ).to(device)
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    if objective == "rfm":
        diffusion = create_rfm(
            time_shift=args.rfm_time_shift,
            time_emb_scale=args.rfm_time_emb_scale,
            num_val_steps=args.num_sampling_steps,
            base_sampler=args.rfm_base_sampler,
            sigma_time_dist=args.rfm_sigma_time_dist,
            pred_mode=args.rfm_pred_mode,
        )
    else:
        diffusion = create_diffusion(str(args.num_sampling_steps))
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    samples_needed_this_gpu = int(remaining_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = complete_samples
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([args.num_classes] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            if objective == "rfm":
                sample_fn = model.forward_with_cfg_flow
            else:
                sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # Sample images:
        if objective == "rfm":
            samples = diffusion.p_sample_loop(
                sample_fn, z, model_kwargs=model_kwargs
            )
        else:
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples, _ = vae.decode(samples, denormalize=True)
        samples = preprocessor.inverse(samples).clamp(0, 1)
        samples = (samples * 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=str)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=4.)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to a DiT checkpoint (.pt). There is no default; a pretrained model is not auto-loaded.")
    parser.add_argument("--objective", type=str, choices=["ddpm", "rfm"], default="ddpm",
                        help="Training objective: 'ddpm' (epsilon prediction) or 'rfm' (rectified flow matching).")
    parser.add_argument("--rfm-time-shift", type=float, default=3.0)
    parser.add_argument("--rfm-time-emb-scale", type=float, default=6.28)
    parser.add_argument("--rfm-base-sampler", type=str, default="logit-normal")
    parser.add_argument("--rfm-sigma-time-dist", type=float, default=1.0)
    parser.add_argument("--rfm-pred-mode", type=str, choices=["flow", "target"], default="flow")
    parser.add_argument("--local-rank", type=int, default=0) # Dummy arg for Sber server compatibility
    args = parser.parse_args()
    main(args)
