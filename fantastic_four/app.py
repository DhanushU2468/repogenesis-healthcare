import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import easyocr
import numpy as np

# ----------------------------------------
# Load EasyOCR
# ----------------------------------------
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

# ----------------------------------------
# Preprocessing (PIL-only)
# ----------------------------------------
def preprocess_image(pil_img):
    img = pil_img.convert("L")
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return np.array(img)

# ----------------------------------------
# Streamlit UI
# ----------------------------------------
st.title("🩺 Handwritten Prescription OCR (EasyOCR)")
st.write("Upload a handwritten medical prescription to extract text.")

uploaded_file = st.file_uploader("Upload Prescription Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Prescription", use_column_width=True)

    processed = preprocess_image(img)

    with st.spinner("Extracting text… ⏳"):
        result = reader.readtext(processed, detail=0)

    extracted_text = "\n".join(result)

    st.subheader("📌 Extracted Text")
    st.code(extracted_text if extracted_text.strip() else "No readable text found.")

    st.download_button(
        label="Download Extracted Text",
        data=extracted_text,
        file_name="prescription_text.txt",
        mime="text/plain"
    )
