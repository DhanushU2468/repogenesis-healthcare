import streamlit as st
from PIL import Image
import torch
from model import OCRModel
from transforms import get_transform
from utils import decode_prediction

# Correct path
import requests

url = "https://drive.google.com/file/d/1z-j_qUnFK5oajH7bIMh3jVeOYq9aE6IH/view?usp=sharing"
r = requests.get(url)
open("ocr_model.pth", "wb").write(r.content)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = OCRModel()
model.load_state_dict(torch.load("ocr_model.pth", map_location=device))
model.to(device)
model.eval()

transform = get_transform()

st.title("📄 Prescription OCR App")
uploaded_file = st.file_uploader("Upload Prescription Image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)

    text = decode_prediction(preds)

    st.subheader("Extracted Text")
    st.code(text)

    st.download_button("Download Extracted Text", text, "prescription_text.txt")
