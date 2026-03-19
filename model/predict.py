import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from services.sentiment import analyze_sentiment
from services.bias_detector import detect_bias

# Model path relative to this script's directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_model")
DEFAULT_MODEL = "microsoft/deberta-v3-small"

# Global classifier for lazy loading
_classifier = None
_tokenizer = None

def get_model():
    """
    Load the classifier and tokenizer from local 'saved_model' or fallback to default.
    """
    global _classifier, _tokenizer
    if _classifier is None:
        try:
            # Check if model exists locally
            if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
                print(f"Loading local model from {MODEL_DIR}...")
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            else:
                print(f"Local model not found. Using pretrained {DEFAULT_MODEL}...")
                _tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
                model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL)
            
            # Using hardware acceleration if available
            device = 0 if torch.cuda.is_available() else -1
            _classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=_tokenizer,
                device=device,
                return_all_scores=True
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
            
    return _classifier

def predict_news(text):
    """
    Analyzes news text for credibility, sentiment, and bias.
    """
    if not text or len(text.strip()) == 0:
        return {"label": "Invalid", "confidence": 0.0, "error": "Empty input"}
        
    clf_pipe = get_model()
    if not clf_pipe or _tokenizer is None:
        return {"label": "Error", "confidence": 0.0, "error": "Model or tokenizer could not be loaded"}
    
    # 1. Prediction with Model
    # Explicitly use the global tokenizer set by get_model
    tokenizer = _tokenizer
    tokens = tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
    
    # Support for GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    with torch.no_grad():
        # Get the model from the pipeline
        model = clf_pipe.model.to(device)
        outputs = model(**tokens)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    # Map predictions to labels
    # 0: Fake, 1: Real
    # Usually: 0: Fake, 1: Real (common convention in HF datasets)
    label_id = np.argmax(probs)
    confidence = float(probs[label_id])
    
    # Map label IDs to readable text
    # Assuming label 0 = Fake, 1 = Real
    # If the user's dataset has different mappings, this should be adjusted
    label_map = {0: "Fake", 1: "Real"}
    label_text = label_map.get(label_id, "Unknown")
    
    # 2. Add Sentiment Bonus
    sentiment_result = analyze_sentiment(text)
    
    # 3. Add Bias Detection Bonus
    bias_result = detect_bias(text)
    
    # 4. Refine Confidence based on Bias level
    # If bias is high and prediction is "Real", but model is unsure, lower confidence
    if bias_result["level"] == "High" and label_text == "Real":
         # Lower confidence in "Real" prediction if bias is high!
         confidence = confidence * 0.95
    
    return {
        "label": label_text,
        "confidence": confidence,
        "sentiment": sentiment_result["label"],
        "sentiment_score": sentiment_result["score"],
        "bias_level": bias_result["level"],
        "bias_score": bias_result["score"],
        "matched_keywords": bias_result["matched_keywords"]
    }

if __name__ == "__main__":
    # Test prediction
    sample_text = "Scientists discover mind-blowing miracle cure for aging in secret underground lab!"
    result = predict_news(sample_text)
    print(f"Text: {sample_text}")
    print(f"Prediction: {result['label']} (Confidence: {result['confidence']:.2f})")
    print(f"Sentiment: {result['sentiment']} | Bias: {result['bias_level']}")
