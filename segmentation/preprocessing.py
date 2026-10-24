import cv2
import numpy as np
import torch
from PIL import Image


def preprocess_for_unet(image):
    """
    Preprocess an MRI image for the trained U-Net.

    Matches the preprocessing used during U-Net training.

    Input:
        PIL Image

    Output:
        Tensor with shape [1, 1, 256, 256]
    """

    # Convert to grayscale
    image = image.convert("L")

    # Convert PIL image to NumPy
    image = np.array(image)

    # Convert to float32
    image = image.astype(np.float32)

    # Min-max normalization
    image = (
        image - image.min()
    ) / (
        image.max() - image.min() + 1e-8
    )

    # Resize exactly like training
    image = cv2.resize(
        image,
        (256, 256),
        interpolation=cv2.INTER_AREA
    )

    # Convert to tensor
    image = torch.from_numpy(image).float()

    # [256, 256] → [1, 256, 256]
    image = image.unsqueeze(0)

    # [1, 256, 256] → [1, 1, 256, 256]
    image = image.unsqueeze(0)

    return image