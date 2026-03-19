# -*- coding: utf-8 -*-
"""
Local fixed version of FakeNewsDetection(DeBERTa-v3 & Transformers).
Removed Colab-specific code and fixed paths for Windows.
"""

import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate
from torch.nn import CrossEntropyLoss
import gradio as gr
from transformers import pipeline
import wikipedia
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import re

# --- WINDOWS PATHS ---
# Replace these with your actual CSV locations if different
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAKE_PATH = os.path.join(BASE_DIR, 'NewsFakeCOVID-19_5.csv')
REAL_PATH = os.path.join(BASE_DIR, 'NewsRealCOVID-19_5.csv')

def setup_data():
    if not (os.path.exists(FAKE_PATH) and os.path.exists(REAL_PATH)):
        print(f"Error: Could not find CSV files at {FAKE_PATH} or {REAL_PATH}")
        return None

    fake_df = pd.read_csv(FAKE_PATH).copy()
    real_df = pd.read_csv(REAL_PATH).copy()

    fake_df['label'] = 0  # 0 = Fake
    real_df['label'] = 1  # 1 = Real

    df = pd.concat([fake_df, real_df], ignore_index=True)
    if 'text' not in df.columns and 'content' in df.columns:
        df = df.rename(columns={'content': 'text'})
    
    df.dropna(subset=['text', 'label'], inplace=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

# ... rest of the training logic can go here if needed ...

# For inference only (using the modular predict logic I already built):
from model.predict import predict_news

def analyze_news_gradio(headline):
    result = predict_news(headline)
    
    md = []
    md.append("### 🔍 Fake News Detection")
    md.append(f"**Prediction**: **{result['label']}**  |  **Confidence**: {result['confidence']:.2f}")
    md.append("")
    md.append("### 🙂 Sentiment Analysis")
    md.append(f"**Sentiment**: **{result.get('sentiment', 'Unknown')}**")
    md.append("")
    md.append("### 📖 Verification Details")
    md.append(f"**Bias Level**: {result.get('bias_level', 'Low')}")
    
    return "\n".join(md)

if __name__ == "__main__":
    iface = gr.Interface(
        fn=analyze_news_gradio,
        inputs=gr.Textbox(lines=3, label="Enter headline"),
        outputs=gr.Markdown(),
        title="Local Fake News Detection (DeBERTa-v3)",
    )
    # Don't use share=True on local unless needed
    iface.launch()
