import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Device
device = torch.device("cpu")

# Create the same ResNet18 architecture used during training
model = resnet18(weights=None)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)

# Load trained weights
model.load_state_dict(
    torch.load(
        "models/resnet18_brain_tumor_best.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("✅ ResNet18 model loaded successfully!")
print("Device:", device)
print("Number of classes:", 3)
print("Classes: Meningioma, Glioma, Pituitary")