# 🛡️ Fake News Detection System (DeBERTa-v3)

A production-level AI system designed to detect and analyze news credibility using state-of-the-art Transformers, sentiment analysis, and multi-source evidence verification.

## 🚀 Overview
This system uses a fine-tuned **DeBERTa-v3** model to classify news articles as 'Real' or 'Fake'. It goes beyond simple classification by analyzing emotional tone (Sentiment), identifying political or commercial bias, and fetching real-time evidence from trusted news sources.

## ✨ Key Features
- **Deep Learning Classifier**: Powered by `microsoft/deberta-v3-small` for high-accuracy text analysis.
- **Evidence Verification**: Automatically fetches related articles from **NewsAPI**, **BBC**, **Reuters**, and **AP News**.
- **Source Credibility Checker**: Validates sources against a curated whitelist of trusted journalistic outlets.
- **Micro-Analysis**:
  - **Sentiment Analysis**: Detects emotional manipulation.
  - **Bias Detection**: Highlights biased language and high-risk keywords.
- **Contextual Evidence**: Integrated YouTube search to provide video summaries of the topics being analyzed.

## 🛠️ Technology Stack
- **Backend**: Python, Flask
- **AI/ML**: PyTorch, Transformers (Hugging Face), Scikit-learn
- **NLP**: TextBlob, VADER
- **APIs**: NewsAPI, Google API Client (YouTube), Wikipedia
- **Frontend**: HTML5, Vanilla CSS (Modern aesthetic)

## 📦 Installation & Setup

1. **Clone the repository**:
   ```powershell
   git clone https://github.com/sandhiyanayak/Fake-News-Detection-using-DeBERTa-v3-Transformer-.git
   cd Fake-News-Detection-using-DeBERTa-v3-Transformer-
   ```

2. **Create a Virtual Environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the project root and add your API keys:
   ```env
   NEWSAPI_KEY=your_newsapi_key_here
   YOUTUBE_API_KEY=your_youtube_api_key_here
   ```

## 🏃 Running the Application

Start the Flask server:
```powershell
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

## 🧪 Testing the Services
You can run the built-in test suite to verify the model and services:
```powershell
python test_services.py
```

## 📝 License
This project is licensed under the MIT License.
