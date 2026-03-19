import os
import torch  # type: ignore
import numpy as np  # type: ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline  # type: ignore
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
            if os.path.exists(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 0:
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            else:
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
    if not text or len(str(text).strip()) == 0:
        return {"label": "Invalid", "confidence": 0.0, "error": "Empty input"}
        
    clf_pipe = get_model()
    # Explicit check to satisfy static analyzers
    if clf_pipe is None or _tokenizer is None:
        return {"label": "Error", "confidence": 0.0, "error": "Model or tokenizer could not be loaded"}
    
    # Use global tokenizer explicitly
    tokenizer = _tokenizer
    inputs = str(text)
    
    tokens = tokenizer(inputs, truncation=True, max_length=512, return_tensors='pt')
    
    # Support for GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    with torch.no_grad():
        # Get the model from the pipeline
        model_obj = clf_pipe.model.to(device)
        outputs = model_obj(**tokens)
        probs_array = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    # Map predictions to labels (0: Fake, 1: Real)
    label_id = int(np.argmax(probs_array))
    confidence_val = float(probs_array[label_id])
    
    label_map = {0: "Fake", 1: "Real"}
    label_text = str(label_map.get(label_id, "Unknown"))
    
    # 2. Add Sentiment Bonus
    sentiment_result = analyze_sentiment(inputs) or {"label": "Neutral", "score": 0.0}
    
    # 3. Add Bias Detection Bonus
    bias_result = detect_bias(inputs) or {"level": "Low", "score": 0.0, "matched_keywords": []}
    
    # 4. Refine Confidence based on Bias level
    if str(bias_result.get("level")) == "High" and label_text == "Real":
         confidence_val = confidence_val * 0.95
    
    return {
        "label": label_text,
        "confidence": confidence_val,
        "sentiment": str(sentiment_result.get("label", "Neutral")),
        "sentiment_score": float(sentiment_result.get("score", 0.0)),
        "bias_level": str(bias_result.get("level", "Low")),
        "bias_score": float(bias_result.get("score", 0.0)),
        "matched_keywords": list(bias_result.get("matched_keywords", []))
    }

if __name__ == "__main__":
    sample_text = "Scientists discover mind-blowing miracle cure for aging in secret underground lab!"
    result = predict_news(sample_text)
    print(f"Prediction: {result.get('label')} (Confidence: {result.get('confidence'):.2f})")
