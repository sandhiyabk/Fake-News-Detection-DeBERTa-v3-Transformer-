import os
import requests
from urllib.parse import quote

def get_video_summary(topic):
    """
    Finds a related YouTube video for the news topic using Google API or Search results fallback.
    """
    api_key = os.environ.get("YOUTUBE_API_KEY")

    # Build search query from topic safely
    keywords_list = str(topic).split()[:6]
    keywords_str = " ".join(keywords_list)
    encoded_query = quote(keywords_str + " news fact check")

    if not api_key:
        display_topic = keywords_str[:40] if len(keywords_str) > 40 else keywords_str
        return {
            "title": f"Search: '{display_topic}' – News & Fact Check",
            "url": f"https://www.youtube-nocookie.com/embed?listType=search&list={encoded_query}",
            "channel": "YouTube Search Result",
            "mocked": True
        }

    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": keywords_str + " news",
            "type": "video",
            "maxResults": 1,
            "key": str(api_key)
        }

        response = requests.get(url, params=params, timeout=5)
        data = response.json() or {}
        items = data.get("items", [])

        if items:
            item = items[0]
            snippet = item.get("snippet", {})
            video_id = item.get("id", {}).get("videoId")
            return {
                "title": str(snippet.get("title", "Untitled Video")),
                "url": f"https://www.youtube-nocookie.com/embed/{video_id}",
                "channel": str(snippet.get("channelTitle", "Unknown Channel")),
                "mocked": False
            }
        elif "error" in data:
            print(f"YouTube API Error: {data.get('error', {}).get('message', 'Unknown error')}")

    except Exception as e:
        print("YouTube Search Exception:", e)

    # Final fallback display logic
    display_topic_fallback = keywords_str[:40] if len(keywords_str) > 40 else keywords_str
    return {
        "title": f"Search: '{display_topic_fallback}' – News & Fact Check",
        "url": f"https://www.youtube-nocookie.com/embed?listType=search&list={encoded_query}",
        "channel": "YouTube Search Result",
        "mocked": True
    }
