import autoroot
import torch

import torchvision.transforms as T
import numpy as np

from PIL import Image

from lib.vae.adapters import WANOfficialAdapter
from lib.data.imagenet import create_imagenet_dataset
from lib.data.transforms import DiTCenterCrop

from torch.utils.data import DataLoader


model = WANOfficialAdapter(name="wan_2.1_official",
                            checkpoint="Wan2.1-T2V-14B/Wan2.1_VAE.pth",
                            latent_norm_type="none",
                            latent_stats=None,
                            device="cuda:5")

preprocessor = model.create_preprocessor()

transforms = T.Compose([
    DiTCenterCrop(256),
    T.ToTensor(),
    preprocessor
])

dataset = create_imagenet_dataset(
    image_dir="ImageNet2012_200/train",
    transform=transforms
)

dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8, pin_memory=True)

for images, _ in dataloader:
    images = images.to("cuda:5")
    latents, _ = model.encode(images, normalize=False, sample=False)
    img, _ = model.decode(latents, denormalize=False)
    img = preprocessor.inverse(img)
    images = preprocessor.inverse(images)
    for i in range(images.shape[0]):
        T.ToPILImage()(images[i]).save(f"orig_{i}.png")
        T.ToPILImage()(img[i]).save(f"rec_{i}.png")
    break


chunk = np.load("/lvmdata/shared/SberVAE/data/latents/ImageNet2012_200__wan_2.1_official__resolution_256/train/chunk_000000.npy")
mean = chunk[:, :16]
std = chunk[:, 16:]

for i in range(mean.shape[0]):
    latent = np.random.randn(*mean[i].shape) * std[i] + mean[i]
    latent = latent[np.newaxis, :, :, :]
    latent = torch.from_numpy(latent).to("cuda:5").to(torch.float32)
    img, _ = model.decode(latent, denormalize=False)
    img = img[0]
    img = preprocessor.inverse(img)
    T.ToPILImage()(img).save(f"latent_{i}.png")
    if i == 1:
        break
