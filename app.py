from flask import Flask, request, render_template, jsonify  # type: ignore
import sys
import os

# Load environment variables from .env file (API keys, etc.)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

# Ensure the app can find the local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services import (  # type: ignore
    fetch_related_news,
    check_sources,
    get_video_summary,
    analyze_sentiment,
    detect_bias
)
from model.predict import predict_news  # type: ignore
import wikipedia  # type: ignore

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Form inputs often come as Any, cast to string explicitly
        form_text = request.form.get("news", "")
        text = str(form_text).strip()
        
        if not text:
            return render_template("index.html", error="Please provide news text.")
            
        # 1. Prediction (DeBERTa + sentiment + bias)
        prediction = predict_news(text) or {}
        
        # 2. Fetch Evidence / Related News
        # Break text into search words and limit for linter safety
        words = text.split()
        from itertools import islice
        search_keywords = list(islice(words, 8))
        search_topic = " ".join(search_keywords)
        evidence = fetch_related_news(search_topic) or []
        
        # 3. Check Source Credibility based on evidence
        credibility = check_sources(evidence) or {}
        
        # 4. YouTube Video Summary
        video = get_video_summary(search_topic)
        
        # 5. Final Prediction Refinement (Evidence-based correction)
        # Cast everything clearly to satisfy strict linters
        final_label = str(prediction.get("label", "Unknown"))
        final_confidence = float(prediction.get("confidence", 0.0))
        verified_count = int(credibility.get("verified_count", 0))
        
        if verified_count >= 1 and final_label == "Fake" and final_confidence < 0.80:
            final_label = "Real"
            final_confidence = 0.85 
            
        # Format confidence safely as float using strings
        formatted_conf = float(f"{final_confidence * 100:.1f}")
            
        # Compile result object with safe defaults
        result = {
            "prediction_label": final_label,
            "prediction_confidence": formatted_conf,
            "sentiment": str(prediction.get("sentiment", "Unknown")),
            "bias_level": str(prediction.get("bias_level", "Unknown")),
            "bias_score": float(prediction.get("bias_score", 0.0)),
            "credibility_score": float(credibility.get("score", 0.0)),
            "verified_sources": verified_count,
            "evidence": evidence,
            "video": video,
            "text": text
        }
        
        return render_template("index.html", result=result)
        
    return render_template("index.html", result=None)


if __name__ == "__main__":
    # The application runs on port 5000 by default
    app.run(debug=True, port=5000)
