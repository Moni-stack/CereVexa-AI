from PIL import Image

from segmentation.segmenter import BrainTumorSegmenter


MODEL_PATH = "models/unet_brain_tumor_best.pth"
IMAGE_PATH = "test_images/test_mri.png"


# Load U-Net
segmenter = BrainTumorSegmenter(MODEL_PATH)

print("U-Net loaded successfully!")


# Load MRI
image = Image.open(IMAGE_PATH)


# Predict segmentation
mask = segmenter.predict(image)


print("Mask shape:", mask.shape)

print(
    "Predicted tumor pixels:",
    int(mask.sum())
)

print(
    "Predicted tumor area:",
    f"{mask.mean() * 100:.2f}%"
)