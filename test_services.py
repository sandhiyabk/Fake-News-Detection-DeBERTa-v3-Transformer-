import os
import sys

# Add the project root to sys path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.predict import predict_news  # type: ignore
from services import (  # type: ignore
    fetch_related_news,
    analyze_sentiment,
    detect_bias,
    check_sources,
    get_video_summary
)

def run_tests():
    sample_text = "Breaking News: Aliens found living in secret government base in Nevada. It's totally shocking and unbelievable!"
    
    print("Testing Prediction...")
    pred = predict_news(sample_text)
    print(f"Prediction: {pred}")
    
    print("\nTesting Sentiment...")
    sent = analyze_sentiment(sample_text)
    print(f"Sentiment: {sent}")
    
    print("\nTesting Bias Detection...")
    bias = detect_bias(sample_text)
    print(f"Bias: {bias}")
    
    print("\nTesting News Fetcher (Mock)...")
    news = fetch_related_news("Aliens in Nevada")
    print(f"Evidence count: {len(news)}")
    
    print("\nTesting Credibility...")
    cred = check_sources(news)
    print(f"Credibility Score: {cred['score']}")
    
    print("\nTesting Video Summary (Mock)...")
    vid = get_video_summary("Aliens in Nevada")
    print(f"Video: {vid['title'] if vid else 'None'}")
    
    print("\nAll imports and basic flows successful!")

if __name__ == "__main__":
    run_tests()
