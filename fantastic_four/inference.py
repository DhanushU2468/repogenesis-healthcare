import torch
from PIL import Image
from model import OCRModel
from transforms import get_transform
from utils import decode_prediction

image_path = "sample_images/test1.jpg"
MODEL_PATH = "ocr_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = OCRModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = get_transform()

img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    preds = model(img_tensor)

text = decode_prediction(preds)
print("Extracted Text:")
print(text)
