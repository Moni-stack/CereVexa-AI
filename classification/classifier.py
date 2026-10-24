import torch
import torch.nn as nn
from torchvision import models

from classification.preprocessing import preprocess_image


class BrainTumorClassifier:

    def __init__(self, model_path, device=None):

        # Select device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create ResNet18
        self.model = models.resnet18(weights=None)

        # Our model has 3 classes
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            3
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

        # Class names
        self.class_names = [
            "Meningioma",
            "Glioma",
            "Pituitary"
        ]

    def predict(self, image):

        # Preprocess image
        image_tensor = preprocess_image(image)
        image_tensor = image_tensor.to(self.device)

        # Prediction
        with torch.no_grad():

            output = self.model(image_tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

        # Get highest probability
        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

        predicted_class = self.class_names[
            predicted_index.item()
        ]


        return {
            "class": predicted_class,
            "confidence": confidence.item(),
            "probabilities": {
                self.class_names[i]: probabilities[0][i].item()
                for i in range(3)
            }
        }

# ============================================================
# BINARY TUMOR DETECTOR
# ============================================================

class TumorDetector:

    def __init__(self, model_path, device=None):

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create ResNet18 with ONE output
        self.model = models.resnet18(
            weights=None
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            1
        )

        # Load binary detector weights
        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image):

        # Use the same preprocessing as the binary detector
        image_tensor = preprocess_image(
            image
        ).to(self.device)

        with torch.no_grad():

            logits = self.model(
                image_tensor
            )

            # Binary probability
            tumor_probability = torch.sigmoid(
                logits
            ).item()

        # Classification
        if tumor_probability >= 0.5:

            predicted_class = "Tumor"
            confidence = tumor_probability

        else:

            predicted_class = "No Tumor"
            confidence = 1 - tumor_probability

        # Debug: inspect the detector's actual output
        print(f"Tumor probability: {tumor_probability:.6f}")
        print(f"Predicted class: {predicted_class}")

        return {
            "class": predicted_class,
            "probability": tumor_probability,
            "confidence": confidence
        }
    