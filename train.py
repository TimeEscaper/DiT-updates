# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import autoroot
import autorootcwd
import sys
sys.path.append("sbervae")

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
from pathlib import Path
import argparse
import logging
import os
import shutil
import yaml

from tqdm import tqdm

from models import DiT_models
from diffusion import create_diffusion, create_rfm
from diffusers.models import AutoencoderKL

from dit_updates.vae.adapters.base import VAEAdapter
from dit_updates.vae.adapters.wan_official import WANOfficialAdapter
from dit_updates.data.latent_datasets import LatentsShardDataset
from dit_updates.data.transforms import DiTCenterCrop
from dit_updates.utils.files import resolve_path
from dit_updates.vae.adapters.registry import resolve_adapter


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


@torch.no_grad()
def generate_samples(ema_model, vae: VAEAdapter, diffusion, latent_size, device,
                     num_classes, cfg_scale=4.0, num_samples=10, seed=0,
                     objective="ddpm"):
    """
    Generate a grid of sample images from the EMA model for TensorBoard logging.
    Uses classifier-free guidance and a fixed seed for consistent comparison
    across epochs.
    """
    rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device)
    torch.manual_seed(seed)

    n = num_samples
    z = torch.randn(n, vae.n_channels, latent_size, latent_size, device=device)
    y = torch.randint(0, num_classes, (n,), device=device)

    z = torch.cat([z, z], 0)
    y_null = torch.tensor([num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    if objective == "rfm":
        samples = diffusion.p_sample_loop(
            ema_model.forward_with_cfg_flow, z, model_kwargs=model_kwargs
        )
    else:
        samples = diffusion.p_sample_loop(
            ema_model.forward_with_cfg, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=False, device=device
        )
    samples, _ = samples.chunk(2, dim=0)

    preprocessor = vae.create_preprocessor()
    samples, _ = vae.decode(samples, denormalize=True)

    torch.random.set_rng_state(rng_state)
    torch.cuda.set_rng_state(cuda_rng_state, device)

    samples = preprocessor.inverse(samples).clamp(0, 1)
    grid = make_grid(samples, nrow=5, padding=2)
    return grid


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        results_dir = str(resolve_path(args.results_dir, "experiment"))
        os.makedirs(results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        shutil.copy2(args.config_path, f"{experiment_dir}/config.yaml")
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    objective = getattr(args, "objective", "ddpm")

    # TODO: Remove hardcoded VAE
    vae = resolve_adapter(args.vae, 
                          device=device,
                          latent_norm_type="scale",
                          latent_stats=Path(args.data_path) / "metadata.json")

    # TODO: Make this factor configurable
    latent_size = args.image_size // 8
    learn_sigma = (objective != "rfm")
    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=vae.n_channels,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])

    if objective == "rfm":
        rfm_cfg = getattr(args, "rfm", {})
        diffusion = create_rfm(**rfm_cfg)
    else:
        diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # ==========================================
    # NEW: CHECKPOINT LOADING LOGIC
    # ==========================================
    train_steps = 0
    if hasattr(args, "pretrained_path") and args.pretrained_path:
        if os.path.isfile(args.pretrained_path):
            logger.info(f"Loading checkpoint from {args.pretrained_path}")
            # Map weights to the correct device for this specific DDP process
            checkpoint = torch.load(args.pretrained_path, map_location=device)
            
            # The saved checkpoint uses model.module.state_dict(), so we load into model.module
            model.module.load_state_dict(checkpoint["model"])
            ema.load_state_dict(checkpoint["ema"])
            opt.load_state_dict(checkpoint["opt"])
            
            # Attempt to parse train_steps from the filename (e.g., '0010000.pt' -> 10000)
            try:
                train_steps = int(Path(args.pretrained_path).stem)
                logger.info(f"Successfully loaded checkpoint. Resuming from step {train_steps}.")
            except ValueError:
                logger.info("Successfully loaded checkpoint, but couldn't parse step count from filename.")
        else:
            logger.warning(f"Pretrained path '{args.pretrained_path}' does not exist! Starting from scratch.")
    # ==========================================

    # Setup TensorBoard and sampling diffusion (rank 0 only):
    writer = None
    if rank == 0:
        tb_dir = f"{experiment_dir}/tensorboard"
        writer = SummaryWriter(log_dir=tb_dir)
        logger.info(f"TensorBoard logs at {tb_dir}")
    if objective == "rfm":
        sample_diffusion = diffusion
    else:
        sample_diffusion = create_diffusion(str(args.num_sampling_steps))

    # Setup data:
    transform = transforms.Compose([
        DiTCenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        vae.create_preprocessor()
    ])

    # dataset = ImageFolder(args.data_path, transform=transform)
    dataset = LatentsShardDataset(args.data_path,
                                  split="train",
                                  latent_normalizer=vae.latent_normalizer.numpy(),
                                  sample=True,
                                  in_memory=args.data_in_memory)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        # pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    # NEW: Only force-sync the EMA if we ARE NOT resuming from a checkpoint
    if not (hasattr(args, "pretrained_path") and args.pretrained_path):
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    # train_steps = 0  <-- Removed, defined above during checkpoint loading
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        batch_iter = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0))
        for x, y in batch_iter:
            x = x.to(device)
            y = y.to(device)
            # with torch.no_grad():
            # Already done by dataset
            # Map input images to latent space + normalize latents:
            # x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if writer is not None:
                    writer.add_scalar("train/loss", avg_loss, train_steps)
                    writer.add_scalar("train/steps_per_sec", steps_per_sec, train_steps)
                    writer.add_scalar("train/epoch", epoch, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        # Generate and log sample images every sample_every epochs:
        if args.sample_every > 0 and (epoch + 1) % args.sample_every == 0:
            if rank == 0:
                logger.info(f"Generating sample images at epoch {epoch}...")
                grid = generate_samples(
                    ema, vae, sample_diffusion, latent_size, device,
                    num_classes=args.num_classes, cfg_scale=args.cfg_scale,
                    num_samples=10, seed=args.global_seed,
                    objective=objective,
                )
                writer.add_image("samples/ema_generations", grid, epoch)
                writer.flush()
                samples_dir = f"{experiment_dir}/samples"
                os.makedirs(samples_dir, exist_ok=True)
                grid_np = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
                Image.fromarray(grid_np).save(f"{samples_dir}/epoch_{epoch:09d}.jpg")
                logger.info(f"Logged sample grid to TensorBoard and saved to disk (epoch {epoch})")
            dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    if writer is not None:
        writer.close()
    logger.info("Done!")
    cleanup()


def load_config(path: str) -> argparse.Namespace:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    required = ["data_path", "vae"]
    for key in required:
        if cfg.get(key) is None:
            raise ValueError(f"Required config field '{key}' is missing or null in {path}")

    assert cfg["model"] in DiT_models, f"Unknown model '{cfg['model']}'. Choose from {list(DiT_models.keys())}"
    assert cfg["image_size"] in (256, 512), f"image_size must be 256 or 512, got {cfg['image_size']}"

    cfg.setdefault("objective", "ddpm")
    cfg.setdefault("data_in_memory", False)
    assert cfg["objective"] in ("ddpm", "rfm"), f"objective must be 'ddpm' or 'rfm', got {cfg['objective']}"
    if cfg["objective"] == "rfm":
        assert "rfm" in cfg, "Config section 'rfm' is required when objective is 'rfm'"

    return argparse.Namespace(**cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--local-rank", type=int, default=0) # Dummy arg for Sber server compatibility
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to checkpoint to resume from (e.g., /path/to/0050000.pt)")
    
    cli = parser.parse_args()
    args = load_config(cli.config)
    
    args.pretrained_path = cli.pretrained_path
    args.config_path = cli.config
    
    main(args)
