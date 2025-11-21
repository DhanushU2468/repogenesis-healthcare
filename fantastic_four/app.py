import streamlit as st
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
import os

# -------------------------
# Config
# -------------------------
HF_MODEL_ID = "naver-clova-ix/donut-base-finetuned-docvqa"
# Example sample path (developer note: you uploaded this file earlier)
EXAMPLE_LOCAL_PATH = "/mnt/data/d98e0ddc-e9f4-492f-a974-443333ed57ad.png"

# -------------------------
# Load model + processor (cached)
# -------------------------
@st.cache_resource
def load_donut(model_id=HF_MODEL_ID):
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    return processor, model

processor, model = load_donut()

# Move to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -------------------------
# Inference helper
# -------------------------
def run_donut(image: Image.Image, task_prompt: str = "what is written?"):
    """
    image: PIL Image
    task_prompt: short question/prompt for docvqa style generation
    returns: string answer
    """
    # Prepare encoder inputs
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Create decoder prompt tokens (Donut style)
    prompt = f"<s_docvqa><s_question>{task_prompt}</s_question><s_answer>"
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    # Generate
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=1024,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_return_sequences=1
    )

    # Decode
    generated = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    # Post-process: strip prompt artifacts if present
    # Donut outputs usually contain the answer text; keep it simple here
    return generated.strip()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Donut Prescription OCR", layout="centered")
st.title("🩺 Donut Prescription OCR (Donut DocVQA)")
st.markdown(
    """
Upload a prescription image and Donut (naver-clova-ix/donut-base-finetuned-docvqa) will extract text.
**Requires GPU** in Spaces — set Space hardware to **GPU** before running.
"""
)

# file uploader
uploaded = st.file_uploader("Upload prescription image (jpg/png)", type=["jpg", "jpeg", "png"])

# quick example button (loads the example image path if present)
if st.button("Use example uploaded file (developer sample)"):
    if os.path.exists(EXAMPLE_LOCAL_PATH):
        uploaded = open(EXAMPLE_LOCAL_PATH, "rb")
        # create a PIL image
        img = Image.open(EXAMPLE_LOCAL_PATH).convert("RGB")
        st.image(img, caption="Example image (local path)", use_column_width=True)
        with st.spinner("Running Donut inference... this can take 10–30s on first run"):
            ans = run_donut(img, task_prompt="what is written?")
        st.subheader("Extracted result")
        st.code(ans)
    else:
        st.error(f"Example file not found at {EXAMPLE_LOCAL_PATH}. Upload an image instead.")

if uploaded:
    # load PIL image
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    # custom prompt (optional)
    task_prompt = st.text_input("Question / prompt for Donut (optional)", value="what is written?")

    if st.button("Extract text with Donut"):
        with st.spinner("Running Donut... this can take 10–30s on first run"):
            answer = run_donut(img, task_prompt=task_prompt)
        st.subheader("Extracted result")
        st.code(answer)

st.markdown("---")
st.markdown(
    """
**Notes:**  
- Donut is a generative doc-qa model. For prescription extraction keep prompt simple like `what is written?` or `please transcribe the prescription`.  
- First model load and first inference are slow (model download + warmup).  
- Use GPU-enabled Space for best speed.
"""
)
