import os
import sys

# Add the project root to sys path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.fake_news_model import predict_news
from services.news_fetcher import fetch_related_news
from services.sentiment import analyze_sentiment
from services.bias_detector import detect_bias
from services.credibility_checker import check_sources
from services.video_summary import get_video_summary

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
