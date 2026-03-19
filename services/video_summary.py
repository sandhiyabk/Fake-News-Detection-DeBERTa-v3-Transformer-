import os
import requests

def get_video_summary(topic):
    """
    Finds a related YouTube video for the news topic.
    Mocks data if no API key.
    """
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        return {
            "title": f"Summary Video for: {topic[:30]}",
            "url": "https://www.youtube.com/embed/dQw4w9WgXcQ", # Fallback embed
            "channel": "News Channel",
            "mocked": True
        }
        
    try:
        # Extract keywords for better search
        keywords = topic.split()[:5]
        search_query = " ".join(keywords) + " news"
        
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": search_query,
            "type": "video",
            "maxResults": 1,
            "key": api_key
        }
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if "items" in data and len(data["items"]) > 0:
            item = data["items"][0]
            video_id = item["id"]["videoId"]
            return {
                "title": item["snippet"]["title"],
                "url": f"https://www.youtube.com/embed/{video_id}",
                "channel": item["snippet"]["channelTitle"],
                "mocked": False
            }
            
    except Exception as e:
        print("YouTube Search Error:", e)
        
    return None
