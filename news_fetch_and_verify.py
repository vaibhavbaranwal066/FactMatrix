import os
import requests
from datetime import datetime
import joblib

# ========================
# API Keys
# ========================
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GNEWS_KEY = os.getenv("GNEWS_KEY")
BING_KEY = os.getenv("BING_KEY")
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")

def assert_keys():
    missing = []
    if not NEWSAPI_KEY: missing.append("NEWSAPI_KEY")
    if not GNEWS_KEY: missing.append("GNEWS_KEY")
    if not BING_KEY: missing.append("BING_KEY")
    if not GOOGLE_FACTCHECK_API_KEY: missing.append("GOOGLE_FACTCHECK_API_KEY")
    if missing:
        raise RuntimeError(f"Missing env variable(s): {', '.join(missing)}")

# ========================
# Model
# ========================
MODEL_PATH = "models/factmatrix_v2_clf.joblib"
pipe = None
def load_model():
    global pipe
    if pipe is None:
        pipe = joblib.load(MODEL_PATH)

# ========================
# Fetch News from APIs
# ========================
def fetch_newsapi(query, from_date=None, to_date=None, language="en", page_size=5):
    url = f"https://newsapi.org/v2/everything?q={query}&language={language}&sortBy=publishedAt&pageSize={page_size}&apiKey={NEWSAPI_KEY}"
    if from_date: url += f"&from={from_date}"
    if to_date: url += f"&to={to_date}"
    resp = requests.get(url)
    if resp.status_code != 200: return []
    return resp.json().get("articles", [])

def fetch_gnews(query, from_date=None, to_date=None, language="en", max_results=5):
    url = f"https://gnews.io/api/v4/search?q={query}&lang={language}&max={max_results}&token={GNEWS_KEY}"
    if from_date: url += f"&from={from_date}"
    if to_date: url += f"&to={to_date}"
    resp = requests.get(url)
    if resp.status_code != 200: return []
    return resp.json().get("articles", [])

def fetch_bing(query, from_date=None, to_date=None, language="en", count=5):
    url = f"https://api.bing.microsoft.com/v7.0/news/search?q={query}&mkt={language}-{language.upper()}&count={count}"
    headers = {"Ocp-Apim-Subscription-Key": BING_KEY}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200: return []
    return resp.json().get("value", [])

# ========================
# Fact Check (Google)
# ========================
def fact_check_claim(claim):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={claim}&key={GOOGLE_FACTCHECK_API_KEY}"
    resp = requests.get(url)
    if resp.status_code != 200: return "Fact-check API error"
    data = resp.json()
    claims = data.get("claims", [])
    if not claims: return "No claims found"
    review = claims[0].get("claimReview", [{}])[0]
    publisher = review.get("publisher", {}).get("name", "Unknown")
    text = review.get("textualRating", "No rating")
    return f"{publisher}: {text}"

# ========================
# Verify
# ========================
def verify_articles(articles, source="Unknown"):
    load_model()
    for art in articles:
        title = art.get("title") or art.get("name", "")
        url = art.get("url", "")
        if not title: continue
        pred = pipe.predict([title])[0]
        fact = fact_check_claim(title)

        print("-" * 80)
        print(f"Source API: {source}")
        print(f"Title: {title}")
        print(f"URL: {url}")
        print(f"Prediction: {pred}")
        print(f"Fact-check: {fact}")

# ========================
# Run Once
# ========================
def run_once(query, from_date=None, to_date=None):
    assert_keys()
    articles_newsapi = fetch_newsapi(query, from_date, to_date)
    articles_gnews = fetch_gnews(query, from_date, to_date)
    articles_bing = fetch_bing(query, from_date, to_date)

    print(f"\n🔹 NewsAPI: {len(articles_newsapi)} results")
    verify_articles(articles_newsapi, "NewsAPI")
    print(f"\n🔹 GNews: {len(articles_gnews)} results")
    verify_articles(articles_gnews, "GNews")
    print(f"\n🔹 Bing: {len(articles_bing)} results")
    verify_articles(articles_bing, "Bing")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--query", type=str, default="climate")
    parser.add_argument("--from_date", type=str)
    parser.add_argument("--to_date", type=str)
    args = parser.parse_args()
    if args.once:
        run_once(query=args.query, from_date=args.from_date, to_date=args.to_date)