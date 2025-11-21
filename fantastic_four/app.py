import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from paddleocr import PaddleOCR

# ------------------------------------------------------
# Load OCR Model
# ------------------------------------------------------
@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en')

ocr = load_ocr()

# ------------------------------------------------------
# Preprocessing Without OpenCV
# ------------------------------------------------------
def preprocess_image(pil_img):
    # Convert to grayscale
    img = pil_img.convert("L")

    # Increase sharpness
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    # Increase contrast
    img = ImageEnhance.Contrast(img).enhance(2.0)

    # Reduce noise
    img = img.filter(ImageFilter.MedianFilter(size=3))

    return np.array(img)


# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------
st.title("🩺 Handwritten Prescription OCR (PaddleOCR)")
st.write("Upload a handwritten medical prescription to extract text.")

uploaded_file = st.file_uploader("Upload Prescription Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Prescription", use_column_width=True)

    processed = preprocess_image(img)

    with st.spinner("Extracting text... Please wait ⏳"):
        result = ocr.ocr(processed, cls=True)

    extracted_text = ""
    for line in result:
        extracted_text += line[1][0] + "\n"

    st.subheader("📌 Extracted Text")
    st.code(extracted_text if extracted_text.strip() else "No readable text found.")

    st.download_button(
        label="Download Extracted Text",
        data=extracted_text,
        file_name="prescription_text.txt",
        mime="text/plain"
    )
