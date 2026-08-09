
import gradio as gr
gr.close_all()

from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
import pandas as pd
import numpy as np
import torch
import easyocr
import os
from rouge_score import rouge_scorer

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# Dynamic summarization function
def safe_summarize(text, summarizer, default_max=60):
    input_len = len(text.split())
    if input_len < 5:
        return text  # Skip summarization for very short inputs

    max_len = min(default_max, int(input_len * 1.5))
    min_len = max(5, int(input_len * 0.5))

    summary = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )[0]['summary_text']
    return summary

# Load CSV
df = pd.read_csv("test_data.csv")

results = []

for i, row in df.iterrows():
    img_path = row["image_path"]
    expected_caption = row["expected_caption"]
    expected_summary = row["expected_summary"]

    # Load image
    image = Image.open(img_path).convert("RGB")

    # Image Captioning
    inputs = caption_processor(image, return_tensors="pt").to(device)
    output = caption_model.generate(**inputs)
    generated_caption = caption_processor.decode(output[0], skip_special_tokens=True)

    # OCR
    ocr_result = ocr_reader.readtext(np.array(image))
    extracted_text = ' '.join([item[1] for item in ocr_result])

    # Summarization with safe fallback
    full_text = generated_caption + ". " + extracted_text
    summary = safe_summarize(full_text, summarizer)

    # Evaluate summary quality using ROUGE
    scores = scorer.score(expected_summary, summary)

    # Store results
    results.append({
        "image": img_path,
        "expected_caption": expected_caption,
        "generated_caption": generated_caption,
        "expected_summary": expected_summary,
        "generated_summary": summary,
        "ROUGE-1 Score": round(scores["rouge1"].fmeasure, 3),
        "ROUGE-L Score": round(scores["rougeL"].fmeasure, 3)
    })

# Save output
output_df = pd.DataFrame(results)
output_df.to_csv("evaluation_results.csv", index=False)

print("✅ Evaluation complete! Check evaluation_results.csv")
