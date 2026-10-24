from PIL import Image

from classification.tumor_detector import TumorDetector


MODEL_PATH = "models/tumor_detector_best.pth"
IMAGE_PATH = "test_images/test_mri.png"


detector = TumorDetector(MODEL_PATH)

image = Image.open(IMAGE_PATH)

result = detector.predict(image)

print("Prediction:", result["prediction"])
print(
    f"Tumor probability: "
    f"{result['tumor_probability']:.6f}"
)
print(
    f"Model confidence: "
    f"{result['confidence'] * 100:.2f}%"
)