import numpy as np

from PIL import Image


def center_crop_arr(pil_image, image_size):
    """
    Center crop a PIL Image to a square of the desired size.

    Implementation taken from ADM:
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126

    Args:
        pil_image (PIL.Image.Image): The input image to crop.
        image_size (int): The target size for the shortest side.

    Returns:
        PIL.Image.Image: The center-cropped image of size (image_size, image_size).
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class DiTCenterCrop:
    """
    Center crop transformation for images.
    Standard transform for DiT.

    Attributes:
        _image_size (int): The size to which the image will be center cropped.
    """

    def __init__(self, image_size: int):
        """
        Initialize the DiTCenterCrop with the desired image size.

        Args:
            image_size (int): Desired size to center crop the image to (image will become square).
        """
        self._image_size = image_size

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply the center crop transform to the input image.

        Args:
            image (PIL.Image.Image): Input image to transform.

        Returns:
            PIL.Image.Image: Center-cropped image.
        """
        return center_crop_arr(image, self._image_size)
