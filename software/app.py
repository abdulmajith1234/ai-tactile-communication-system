import gradio as gr
from PIL import Image
import numpy as np
import pandas as pd
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import easyocr
from rouge_score import rouge_scorer
import os

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# Braille dictionary
braille_dict = {
    "a": "⠁", "b": "⠃", "c": "⠉", "d": "⠙", "e": "⠑", "f": "⠋", "g": "⠛", "h": "⠓",
    "i": "⠊", "j": "⠚", "k": "⠅", "l": "⠇", "m": "⠍", "n": "⠝", "o": "⠕", "p": "⠏",
    "q": "⠟", "r": "⠗", "s": "⠎", "t": "⠞", "u": "⠥", "v": "⠧", "w": "⠺", "x": "⠭",
    "y": "⠽", "z": "⠵", " ": " ", ".": ".", ",": ",", "?": "⠹"
}

def text_to_braille(text):
    return ''.join([braille_dict.get(c.lower(), "?") for c in text])

def safe_summarize(text):
    input_len = len(text.split())
    if input_len < 5:
        return text
    max_len = min(60, int(input_len * 1.5))
    min_len = max(5, int(input_len * 0.5))
    return summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']

def process_image(image: Image.Image):
    inputs = caption_processor(image.convert("RGB"), return_tensors="pt").to(device)
    out = caption_model.generate(**inputs)
    caption = caption_processor.decode(out[0], skip_special_tokens=True)

    image_np = np.array(image)
    ocr_result = ocr_reader.readtext(image_np)
    extracted_text = ' '.join([item[1] for item in ocr_result])

    full_text = caption + ". " + extracted_text
    summary = safe_summarize(full_text)
    braille = text_to_braille(summary)

    return caption, extracted_text, summary, braille

def run_evaluation():
    if not os.path.exists("test_data.csv"):
        return pd.DataFrame([{"error": "test_data.csv not found"}])

    df = pd.read_csv("test_data.csv")
    results = []

    for _, row in df.iterrows():
        img_path = row["image_path"]
        expected_caption = row["expected_caption"]
        expected_summary = row["expected_summary"]

        if not os.path.exists(img_path):
            results.append({"image": img_path, "error": "File not found"})
            continue

        try:
            # 🔄 Safely open each image fresh
            with open(img_path, "rb") as f:
                image = Image.open(f).convert("RGB")
                image.load()  # Fully load the image

            # Captioning
            inputs = caption_processor(image, return_tensors="pt").to(device)
            out = caption_model.generate(**inputs)
            generated_caption = caption_processor.decode(out[0], skip_special_tokens=True)

            # OCR
            ocr_result = ocr_reader.readtext(np.array(image))
            extracted_text = ' '.join([item[1] for item in ocr_result])

            # Summary
            full_text = generated_caption + ". " + extracted_text
            summary = safe_summarize(full_text)

            # ROUGE scoring
            scores = scorer.score(expected_summary, summary)

            results.append({
                "image": img_path,
                "expected_caption": expected_caption,
                "generated_caption": generated_caption,
                "expected_summary": expected_summary,
                "generated_summary": summary,
                "ROUGE-1": round(scores["rouge1"].fmeasure, 3),
                "ROUGE-L": round(scores["rougeL"].fmeasure, 3)
            })

        except Exception as e:
            results.append({"image": img_path, "error": str(e)})

    result_df = pd.DataFrame(results)
    result_df.to_csv("evaluation_results.csv", index=False)
    return result_df

# ✅ Gradio UI
with gr.Blocks() as demo:
    with gr.Tab("Braille Translator"):
        image_input = gr.Image(type="pil", label="Upload Image")
        caption_output = gr.Textbox(label="Caption")
        ocr_output = gr.Textbox(label="OCR Text")
        summary_output = gr.Textbox(label="Summary")
        braille_output = gr.Textbox(label="Braille Output")

        translate_button = gr.Button("Translate Image to Braille")

        translate_button.click(
            fn=process_image,
            inputs=image_input,
            outputs=[caption_output, ocr_output, summary_output, braille_output]
        )

    with gr.Tab("Evaluate Model"):
        eval_button = gr.Button("Run Evaluation")
        eval_output = gr.Dataframe(label="Evaluation Results")

        eval_button.click(
            fn=run_evaluation,
            inputs=[],
            outputs=eval_output
        )

demo.launch()
