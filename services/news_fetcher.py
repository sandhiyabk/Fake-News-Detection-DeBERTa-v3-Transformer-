import requests
import os
from urllib.parse import quote

def fetch_related_news(topic):
    """
    Search for recent news related to the topic using NewsAPI.
    Falls back to curated real search URLs if no API key is set.
    """
    api_key = os.environ.get("NEWSAPI_KEY")

    # URL-encode the topic safely using urllib (more reliable than requests.utils.quote)
    encoded_topic = quote(topic[:80])

    if not api_key:
        # Fallback: real clickable search links on trusted outlets
        return [
            {
                "title": f"BBC: Latest on '{topic[:40]}'",
                "source": "BBC News",
                "url": f"https://www.bbc.co.uk/search?q={encoded_topic}"
            },
            {
                "title": f"Reuters: Expert analysis on '{topic[:35]}'",
                "source": "Reuters",
                "url": f"https://www.reuters.com/search/news?blob={encoded_topic}"
            },
            {
                "title": f"AP News: Fact-check on '{topic[:35]}'",
                "source": "AP News",
                "url": f"https://apnews.com/search?q={encoded_topic}"
            }
        ]

    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": topic[:100],
                "language": "en",
                "pageSize": 5,
                "sortBy": "relevancy",
                "apiKey": api_key
            },
            timeout=8
        )
        data = r.json()

        if data.get("status") == "ok":
            articles = data.get("articles", [])
            results = []
            for art in articles:
                title = art.get("title") or "Untitled"
                source = art.get("source", {}).get("name", "Unknown")
                url = art.get("url") or "#"
                # Skip removed articles
                if "[Removed]" in title:
                    continue
                results.append({"title": title, "source": source, "url": url})
            if results:
                return results[:5]
        else:
            print("NewsAPI error:", data.get("message", "Unknown error"))

    except Exception as e:
        print("News fetch error:", e)

    # Fallback if API call fails
    return [
        {
            "title": f"BBC: Search results for '{topic[:40]}'",
            "source": "BBC News",
            "url": f"https://www.bbc.co.uk/search?q={encoded_topic}"
        }
    ]
