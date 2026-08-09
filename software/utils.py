# utils.py

import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from rouge_score import rouge_scorer

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load summarizer
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# ROUGE scorer
scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rougeL'],
    use_stemmer=True
)

# Braille dictionary
braille_dict = {
    "a": "⠁", "b": "⠃", "c": "⠉", "d": "⠙",
    "e": "⠑", "f": "⠋", "g": "⠛", "h": "⠓",
    "i": "⠊", "j": "⠚", "k": "⠅", "l": "⠇",
    "m": "⠍", "n": "⠝", "o": "⠕", "p": "⠏",
    "q": "⠟", "r": "⠗", "s": "⠎", "t": "⠞",
    "u": "⠥", "v": "⠧", "w": "⠺", "x": "⠭",
    "y": "⠽", "z": "⠵",
    " ": " ",
    ".": ".",
    ",": ",",
    "?": "⠹"
}

def text_to_braille(text):
    """
    Convert normal text into Braille symbols.
    """
    return ''.join([
        braille_dict.get(c.lower(), "?")
        for c in text
    ])

def safe_summarize(text):
    """
    Safely summarize text while handling short inputs.
    """
    input_len = len(text.split())

    if input_len < 5:
        return text

    max_len = min(60, int(input_len * 1.5))
    min_len = max(5, int(input_len * 0.5))

    summary = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )

    return summary[0]['summary_text']

def calculate_rouge(expected, generated):
    """
    Calculate ROUGE scores.
    """
    scores = scorer.score(expected, generated)

    return {
        "ROUGE-1": round(scores["rouge1"].fmeasure, 3),
        "ROUGE-L": round(scores["rougeL"].fmeasure, 3)
    }

def image_to_numpy(image: Image.Image):
    """
    Convert PIL image to NumPy array.
    """
    return np.array(image)
