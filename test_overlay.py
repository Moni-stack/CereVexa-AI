from PIL import Image
import matplotlib.pyplot as plt

from segmentation.segmenter import BrainTumorSegmenter
from utils.image_utils import create_segmentation_overlay


MODEL_PATH = "models/unet_brain_tumor_best.pth"
IMAGE_PATH = "test_images/test_mri.png"


# Load image
image = Image.open(IMAGE_PATH)


# Load U-Net
segmenter = BrainTumorSegmenter(MODEL_PATH)


# Predict mask
mask = segmenter.predict(image)


# Create overlay
overlay = create_segmentation_overlay(
    image,
    mask
)


# Display
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original MRI")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title("U-Net Segmentation Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()