from PIL import Image
from torchvision import transforms


# Preprocessing used by the ResNet18 classifier
classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image):
    """
    Prepare an MRI image for the ResNet18 classifier.
    """

    # Make sure the image has 3 channels
    image = image.convert("RGB")

    # Apply preprocessing
    image = classifier_transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    return image

