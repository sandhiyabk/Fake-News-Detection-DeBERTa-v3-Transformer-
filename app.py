from flask import Flask, request, render_template, jsonify
import sys
import os

# Ensure the app can find the local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.predict import predict_news
from services.news_fetcher import fetch_related_news
from services.credibility_checker import check_sources
from services.video_summary import get_video_summary
import wikipedia

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form.get("news", "").strip()
        if not text:
            return render_template("index.html", error="Please provide news text.")
            
        # 1. Prediction (DeBERTa + sentiment + bias)
        prediction = predict_news(text)
        
        # 2. Fetch Evidence / Related News
        # Extract a short keyword representation of the text for search
        search_topic = " ".join(text.split()[:8])
        evidence = fetch_related_news(search_topic)
        
        # 3. Check Source Credibility based on evidence
        credibility = check_sources(evidence)
        
        # 4. YouTube Video Summary
        video = get_video_summary(search_topic)
        
        # Compile result object
        result = {
            "prediction_label": prediction["label"],
            "prediction_confidence": round(prediction["confidence"] * 100, 1),
            "sentiment": prediction.get("sentiment", "Unknown"),
            "bias_level": prediction.get("bias_level", "Unknown"),
            "bias_score": prediction.get("bias_score", 0),
            "credibility_score": credibility["score"],
            "verified_sources": credibility["verified_count"],
            "evidence": evidence,
            "video": video,
            "text": text
        }
        
        return render_template("index.html", result=result)
        
    return render_template("index.html", result=None)


if __name__ == "__main__":
    # The application runs on port 5000 by default
    app.run(debug=True, port=5000)
