# utils/features.py
import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(pretrained=True)
model.fc = torch.nn.Identity()
model = model.to(device).eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(cropped_img):
    img = Image.fromarray(cropped_img)
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor)
    return features.squeeze().cpu().numpy()
