import torch
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp


# -----------------------------
# 1. Settings
# -----------------------------
MODEL_PATH = "models/unet_brain_tumor_best.pth"
IMAGE_PATH = "test_images/test_mri.png"

device = torch.device("cpu")


# -----------------------------
# 2. Create the same U-Net
#    architecture used in training
# -----------------------------
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=1,
    classes=1
)


# -----------------------------
# 3. Load trained weights
# -----------------------------
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("✅ U-Net model loaded successfully!")


# -----------------------------
# 4. Load MRI
# -----------------------------
image = Image.open(IMAGE_PATH).convert("L")

# Convert to NumPy
image = np.array(image)


# -----------------------------
# 5. Preprocessing
#    Same basic preprocessing
#    used during training
# -----------------------------
image = cv2.resize(
    image,
    (256, 256),
    interpolation=cv2.INTER_AREA
)

image = image.astype(np.float32) / 255.0

image_tensor = torch.from_numpy(image)

# [256,256] → [1,256,256]
image_tensor = image_tensor.unsqueeze(0)

# [1,256,256] → [1,1,256,256]
image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)


# -----------------------------
# 6. Prediction
# -----------------------------
with torch.no_grad():

    output = model(image_tensor)

    probability = torch.sigmoid(output)

    mask = (probability > 0.5).float()


# -----------------------------
# 7. Get tumor pixel information
# -----------------------------
mask_np = mask.squeeze().cpu().numpy()

tumor_pixels = int(mask_np.sum())

total_pixels = mask_np.size

tumor_percentage = (
    tumor_pixels / total_pixels
) * 100


# -----------------------------
# 8. Save predicted mask
# -----------------------------
mask_image = (mask_np * 255).astype(np.uint8)

cv2.imwrite(
    "test_images/unet_mask.png",
    mask_image
)


# -----------------------------
# 9. Display result
# -----------------------------
print()
print("================================")
print("         U-NET RESULT")
print("================================")
print("Tumor pixels :", tumor_pixels)
print(f"Tumor area   : {tumor_percentage:.2f}%")
print("Mask saved   : test_images/unet_mask.png")
print("================================")
