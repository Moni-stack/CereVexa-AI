from PIL import Image

from classification.classifier import BrainTumorClassifier


MODEL_PATH = "models/resnet18_brain_tumor_best.pth"

IMAGE_PATH = "test_images/test_mri.png"


# Load classifier
classifier = BrainTumorClassifier(MODEL_PATH)

# Load MRI
image = Image.open(IMAGE_PATH)


# Predict
result = classifier.predict(image)


print("\nPrediction:", result["class"])
print(
    f"Confidence: "
    f"{result['confidence'] * 100:.2f}%"
)

print("\nClass probabilities:")

for class_name, probability in result["probabilities"].items():

    print(
        f"{class_name}: "
        f"{probability * 100:.2f}%"
    )