import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

# -----------------------------
# 1. Settings
# -----------------------------
MODEL_PATH = "models/resnet18_brain_tumor_best.pth"

CLASS_NAMES = [
    "Meningioma",
    "Glioma",
    "Pituitary"
]

device = torch.device("cpu")

# -----------------------------
# 2. Create model
# -----------------------------
model = resnet18(weights=None)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)

# Load trained model
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

# -----------------------------
# 3. Same preprocessing used
#    during ResNet training
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 4. Give your MRI image path
# -----------------------------
image_path = input("Enter MRI image path: ")

image = Image.open(image_path).convert("RGB")

image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)
image_tensor = image_tensor.to(device)

# -----------------------------
# 5. Prediction
# -----------------------------
with torch.no_grad():
    output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, predicted_class = torch.max(
        probabilities, dim=1
    )

# -----------------------------
# 6. Display result
# -----------------------------
predicted_name = CLASS_NAMES[predicted_class.item()]
confidence_percentage = confidence.item() * 100

print()
print("================================")
print("       BRAIN TUMOR RESULT")
print("================================")
print("Predicted tumor :", predicted_name)
print(f"Confidence      : {confidence_percentage:.2f}%")
print("================================")