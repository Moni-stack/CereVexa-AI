import torch
import segmentation_models_pytorch as smp

from segmentation.preprocessing import preprocess_for_unet


class BrainTumorSegmenter:

    def __init__(self, model_path, device=None):

        # Select device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create exactly the same U-Net architecture
        # used during training
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=1,
            classes=1
        )

        # Load trained model
        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model = self.model.to(self.device)

        # Evaluation mode
        self.model.eval()

    def predict(self, image):

        # Preprocess image
        image_tensor = preprocess_for_unet(image)

        image_tensor = image_tensor.to(self.device)

        # Run U-Net
        with torch.no_grad():

            output = self.model(image_tensor)

            # Convert logits to probability
            probability = torch.sigmoid(output)

        # Convert probability to binary mask
        mask = (probability > 0.5).float()

        # Remove batch and channel dimensions
        mask = mask.squeeze().cpu().numpy()

        return mask