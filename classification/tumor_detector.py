import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class TumorDetector:

    def __init__(self, model_path, device=None):

        # Select CPU/GPU automatically
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create ResNet18
        self.model = models.resnet18(weights=None)

        # Binary classification:
        # 0 = No Tumor
        # 1 = Tumor
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            1
        )

        # Load trained weights
        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Same preprocessing used during testing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image):

        # Make sure image is RGB
        image = image.convert("RGB")

        # Preprocess
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Prediction
        with torch.no_grad():

            output = self.model(image_tensor)

            # Convert logit → probability
            tumor_probability = torch.sigmoid(output).item()

        # Classification
        if tumor_probability >= 0.5:

            prediction = "Tumor"
            confidence = tumor_probability

        else:

            prediction = "No Tumor"
            confidence = 1 - tumor_probability

        return {
            "prediction": prediction,
            "tumor_probability": tumor_probability,
            "confidence": confidence
        }