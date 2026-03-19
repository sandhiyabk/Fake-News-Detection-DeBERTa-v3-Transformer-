import requests
import os

def fetch_related_news(topic):
    """
    Search for recent news related to the topic using NewsAPI.
    If no API key is set, returns mock evidence data for demonstration.
    """
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        # Mock evidence for demonstration purposes
        return [
            {"title": f"Recent update on: {topic[:30]}...", "source": "BBC News", "url": "#"},
            {"title": "Experts weigh in on the latest controversy", "source": "Reuters", "url": "#"},
            {"title": "Analyzing the recent claims making rounds on social media", "source": "AP News", "url": "#"}
        ]
        
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": topic,
                "language": "en",
                "pageSize": 3,
                "sortBy": "relevancy",
                "apiKey": api_key
            },
            timeout=5
        )
        data = r.json()
        if data.get("status") == "ok":
            articles = data.get("articles", [])
            return [
                {
                    "title": art.get("title"),
                    "source": art.get("source", {}).get("name", "Unknown"),
                    "url": art.get("url")
                }
                for art in articles
            ]
    except Exception as e:
        print("News fetch error:", e)
        
    return []
