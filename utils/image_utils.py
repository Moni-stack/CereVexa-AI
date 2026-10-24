import numpy as np
from PIL import Image


def create_segmentation_overlay(image, mask, alpha=0.45):
    """
    Create an overlay showing the U-Net tumor segmentation
    on top of the original MRI.

    Parameters
    ----------
    image : PIL.Image
        Original MRI image.

    mask : numpy.ndarray
        Binary U-Net mask with shape (256, 256).

    alpha : float
        Transparency of the highlighted region.

    Returns
    -------
    PIL.Image
        MRI with segmentation overlay.
    """

    # Convert MRI to RGB
    image = image.convert("RGB")

    # Resize MRI to U-Net output size
    image = image.resize(
        (256, 256),
        Image.Resampling.BILINEAR
    )

    # Convert to NumPy
    image_array = np.array(image).astype(np.float32)

    # Create binary mask
    binary_mask = mask > 0

    # Create overlay
    overlay = image_array.copy()

    # Highlight tumor region
    # Red channel becomes stronger
    overlay[binary_mask, 0] = 255

    # Reduce green and blue
    overlay[binary_mask, 1] *= 0.35
    overlay[binary_mask, 2] *= 0.35

    # Blend original and overlay
    result = (
        image_array * (1 - alpha)
        +
        overlay * alpha
    )

    # Make sure values are valid
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)