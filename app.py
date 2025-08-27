import gradio as gr
from transformers import pipeline
import wikipedia
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import re

# ----------------------------
# 1. NewsAPI Key
# ----------------------------
NEWSAPI_KEY = "dae9f2abd8434e50b6f277863fd81fe1"  # replace with your own key

# ----------------------------
# 2. Load your trained model
# ----------------------------
SAVE_DIR = "/content/final_model"
classifier = pipeline(
    "text-classification",
    model=SAVE_DIR,
    tokenizer=SAVE_DIR,
    return_all_scores=False,
    device=-1  # set to 0 if GPU available
)

# ----------------------------
# 3. Sentiment Analyzer
# ----------------------------
analyzer = SentimentIntensityAnalyzer()

# ----------------------------
# 4. Helper: Extract keywords
# ----------------------------
def extract_keywords(text, top_n=3):
    words = re.findall(r"\w+", text)
    stopwords = {
        "the", "is", "in", "on", "of", "for", "a", "an", "to", "and",
        "with", "show", "study", "results"
    }
    keywords = [w for w in words if w.lower() not in stopwords]
    return " ".join(keywords[:top_n]) if keywords else text

# ----------------------------
# 5. Wikipedia Snippet
# ----------------------------
def get_wiki_snippet(q):
    try:
        try_query = q if len(q.split()) < 6 else " ".join(q.split()[:6])
        return wikipedia.summary(try_query, sentences=3)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous term: refine your query. Options: {', '.join(e.options[:5])} ..."
    except Exception:
        return "No relevant Wikipedia page found."

# ----------------------------
# 6. Live News API Fetch
# ----------------------------
def get_live_api_examples(q):
    if not NEWSAPI_KEY:
        return "⚠️ Live API not configured."
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": q,
                "language": "en",
                "pageSize": 3,
                "sortBy": "publishedAt",
                "apiKey": NEWSAPI_KEY
            },
            timeout=10
        )
        data = r.json()
        if data.get("status") != "ok":
            return f"Live API error: {data.get('message','unknown error')}"
        items = []
        for art in data.get("articles", []):
            items.append(f"- {art.get('title','(no title)')} ({art.get('source',{}).get('name','')})")
        return "\n".join(items) if items else "No recent related articles found."
    except Exception as e:
        return f"Live API request failed: {e}"

# ----------------------------
# 7. Main Analyzer Function
# ----------------------------
def analyze_news(headline):
    headline = (headline or "").strip()
    if not headline:
        return "Please enter a headline."

    # 1) Fake/Real Prediction
    pred = classifier(headline)[0]
    label = pred["label"]
    conf = float(pred["score"])

    # 2) Sentiment
    s = analyzer.polarity_scores(headline)
    sentiment_label = "Neutral"
    if s["compound"] >= 0.05:
        sentiment_label = "Positive"
    elif s["compound"] <= -0.05:
        sentiment_label = "Negative"

    # 3) Wikipedia snippet
    wiki = get_wiki_snippet(headline)

    # 4) Live API (keywords)
    search_query = extract_keywords(headline, top_n=3)
    live_info = get_live_api_examples(search_query)

    # Build Markdown Output
    md = []
    md.append("### 🔍 Fake News Detection")
    md.append(f"**Prediction**: **{label}**  |  **Confidence**: {conf:.2f}")
    md.append("")
    md.append("### 🙂 Sentiment Analysis")
    md.append(f"**Sentiment**: **{sentiment_label}**  |  Scores: {s}")
    md.append("")
    md.append("### 📖 Wikipedia Snippet")
    md.append(wiki)
    md.append("")
    md.append(f"### 📰 Live News (Search: `{search_query}`)")
    md.append(live_info)

    return "\n".join(md)

# ----------------------------
# 8. Gradio UI
# ----------------------------
iface = gr.Interface(
    fn=analyze_news,
    inputs=gr.Textbox(lines=3, label="Enter headline"),
    outputs=gr.Markdown(),
    title="Real-Time Fake News Detection (DeBERTa-v3)",
    description="Classifies news as Fake/Real, gives sentiment, shows Wikipedia snippet, and fetches related live news."
)

iface.launch(share=True, debug=True)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_path = "final_model"   # or path to your uploaded folder
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


