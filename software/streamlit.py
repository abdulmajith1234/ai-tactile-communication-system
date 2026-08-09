

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import torch
import easyocr

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

braille_dict = {
    "a": "⠁", "b": "⠃", "c": "⠉", "d": "⠙", "e": "⠑", "f": "⠋", "g": "⠛", "h": "⠓",
    "i": "⠊", "j": "⠚", "k": "⠅", "l": "⠇", "m": "⠍", "n": "⠝", "o": "⠕", "p": "⠏",
    "q": "⠟", "r": "⠗", "s": "⠎", "t": "⠞", "u": "⠥", "v": "⠧", "w": "⠺", "x": "⠭",
    "y": "⠽", "z": "⠵", " ": " ", ".": ".", ",": ",", "?": "⠹"
}

def text_to_braille(text):
    return ''.join([braille_dict.get(c.lower(), "?") for c in text])

# Streamlit UI
st.title("🧠 Image to Braille Translator")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("⏳ Generating caption...")
    inputs = caption_processor(image, return_tensors="pt").to(device)
    out = caption_model.generate(**inputs)
    caption = caption_processor.decode(out[0], skip_special_tokens=True)
    st.write("🖼️ Caption:", caption)

    st.write("🔍 Extracting text with OCR...")
    result = ocr_reader.readtext(image)
    extracted_text = ' '.join([item[1] for item in result])
    st.write("📜 OCR Text:", extracted_text)

    full_text = caption + ". " + extracted_text

    st.write("📝 Summarizing...")
    summary = summarizer(full_text, max_length=60, min_length=5, do_sample=False)[0]['summary_text']
    st.write("🧾 Summary:", summary)

    st.write("🔡 Converting to Braille...")
    braille_output = text_to_braille(summary)
    st.code(braille_output)
