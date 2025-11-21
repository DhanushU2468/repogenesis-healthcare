import streamlit as st
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ---------------------------
# Load Pretrained OCR Model
# ---------------------------
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return processor, model

processor, model = load_model()

st.title("📄 Prescription OCR App (TrOCR Powered)")
st.write("Upload a medical prescription image to extract text.")

uploaded_file = st.file_uploader("Upload Prescription Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Prescription", use_column_width=True)

    with st.spinner("Extracting text... Please wait ⏳"):
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    st.subheader("Extracted Text")
    st.code(text)

    st.download_button(
        label="Download Extracted Text",
        data=text,
        file_name="prescription_text.txt",
        mime="text/plain"
    )
