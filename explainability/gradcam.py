import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from classification.preprocessing import preprocess_image


class BrainTumorGradCAM:

    def __init__(self, model, device=None):

        self.model = model

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # ResNet18 final convolutional block
        self.target_layers = [
            self.model.layer4[-1]
        ]

    def generate(self, image, class_index):

        # Prepare image
        image_rgb = image.convert("RGB")

        # Create tensor using the same classifier preprocessing
        input_tensor = preprocess_image(
            image_rgb
        ).to(self.device)

        # Target class for Grad-CAM
        targets = [
            ClassifierOutputTarget(class_index)
        ]

        # Create Grad-CAM
        cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers
        )

        # Generate heatmap
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

        # Prepare image for visualization
        visualization_image = image_rgb.resize(
            (224, 224)
        )

        rgb_image = np.array(
            visualization_image
        ).astype(np.float32) / 255.0

        # Overlay Grad-CAM
        cam_image = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True
        )

        return Image.fromarray(cam_image)