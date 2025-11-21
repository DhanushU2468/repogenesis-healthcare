import streamlit as st
from PIL import Image
import cv2
import numpy as np
from paddleocr import PaddleOCR

# ------------------------------------------------------
# Load OCR Model (PaddleOCR handles handwriting well)
# ------------------------------------------------------
@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en')

ocr = load_ocr()

# ------------------------------------------------------
# Image Pre-processing (for handwritten prescriptions)
# ------------------------------------------------------
def preprocess_image(pil_img):
    img = np.array(pil_img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=20)

    # Increase contrast
    gray = cv2.equalizeHist(gray)

    return gray


# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------
st.title("🩺 Handwritten Prescription OCR (PaddleOCR Powered)")
st.write("Upload a medical prescription to extract text accurately.")

uploaded_file = st.file_uploader("Upload Prescription Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Prescription", use_column_width=True)

    # Preprocess the image
    processed_img = preprocess_image(img)

    # OCR Extraction
    with st.spinner("Extracting text... Please wait ⏳"):
        result = ocr.ocr(processed_img, cls=True)

    # Extract text lines
    extracted_text = ""
    for line in result:
        extracted_text += line[1][0] + "\n"

    # Display extracted text
    st.subheader("📌 Extracted Text")
    st.code(extracted_text if extracted_text.strip() else "No readable text found.")

    # Download button
    st.download_button(
        label="Download Extracted Text",
        data=extracted_text,
        file_name="prescription_text.txt",
        mime="text/plain"
    )
