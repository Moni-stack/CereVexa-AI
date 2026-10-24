import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from classification.classifier import BrainTumorClassifier
from explainability.gradcam import BrainTumorGradCAM


MODEL_PATH = "models/resnet18_brain_tumor_best.pth"

IMAGE_PATH = "test_images/3027_tumor.png"

# -----------------------------------
# Load classifier
# -----------------------------------
classifier = BrainTumorClassifier(
    MODEL_PATH
)

print("Classifier loaded successfully!")


# -----------------------------------
# Load MRI
# -----------------------------------

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# -----------------------------------
# Get prediction
# -----------------------------------

result = classifier.predict(image)

print("\nPrediction:", result["class"])

print(
    f"Confidence: "
    f"{result['confidence'] * 100:.2f}%"
)


# -----------------------------------
# Get predicted class index
# -----------------------------------

class_names = [
    "Meningioma",
    "Glioma",
    "Pituitary"
]

class_index = class_names.index(
    result["class"]
)


# -----------------------------------
# Generate Grad-CAM
# -----------------------------------

gradcam = BrainTumorGradCAM(
    classifier.model
)

cam_image = gradcam.generate(
    image,
    class_index
)


# -----------------------------------
# Display
# -----------------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original MRI")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cam_image)
plt.title(
    f"Grad-CAM: {result['class']}"
)
plt.axis("off")

plt.tight_layout()
plt.show()