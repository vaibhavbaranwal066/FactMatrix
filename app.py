# app.py
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import joblib
import pandas as pd

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="FactMatrix",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Load keys + model
# ---------------------------
load_dotenv()
FACTCHECK_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GNEWS_KEY = os.getenv("GNEWS_KEY")
BING_KEY = os.getenv("BING_KEY")

MODEL_PATH = "models/factmatrix_v2_clf.joblib"
pipe = None
if os.path.exists(MODEL_PATH):
    try:
        pipe = joblib.load(MODEL_PATH)
    except Exception:
        pipe = None

# ---------------------------
# CSS (keep immersive visuals + link-square)
# ---------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f8fc 0%, #ffffff 40%, #fffaf6 100%);
        color: #0f172a;
    }
    .header {
        display: flex;
        gap: 16px;
        align-items: center;
        padding: 18px 22px;
        border-radius: 14px;
        background: linear-gradient(90deg, rgba(255,255,255,0.6), rgba(255,255,255,0.35));
        box-shadow: 0 8px 30px rgba(2,6,23,0.06);
        margin-bottom: 14px;
    }
    .logo { font-size: 32px; padding: 10px; border-radius: 10px; }
    .title { font-weight:700; margin:0; font-size:18px; }
    .subtitle { margin:0; color:#475569; font-size:13px; }
    .card {
        background: rgba(255,255,255,0.95);
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 10px 30px rgba(2,6,23,0.06);
        transition: transform .12s ease, box-shadow .12s ease;
        margin-bottom: 12px;
    }
    .card:hover{ transform: translateY(-6px); box-shadow: 0 20px 60px rgba(2,6,23,0.08); }
    .muted { color:#6b7280; font-size:13px; }
    .small-muted { color:#94a3b8; font-size:12px; }
    .badge { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:600; font-size:13px; color:#fff; }
    .badge-true{ background: linear-gradient(90deg,#10b981,#059669); }
    .badge-false{ background: linear-gradient(90deg,#ef4444,#dc2626); }
    .badge-misleading{ background: linear-gradient(90deg,#f59e0b,#d97706); }
    .badge-unknown{ background: linear-gradient(90deg,#64748b,#475569); }
    .mini { font-size:12px; color:#475569; }

    /* Timeline */
    .timeline { list-style: none; padding-left: 0; margin: 0; }
    .timeline-item { display:flex; gap:12px; align-items:flex-start; margin-bottom:12px; }
    .dot { width:14px; height:14px; border-radius:50%; margin-top:4px; flex:0 0 14px; }
    .dot-google { background:#10b981; }   /* green */
    .dot-newsapi { background:#06b6d4; }  /* cyan */
    .dot-gnews { background:#8b5cf6; }    /* purple */
    .dot-bing { background:#f59e0b; }     /* amber */
    .timeline-content { background: rgba(255,255,255,0.95); padding:8px 12px; border-radius:8px; box-shadow:0 6px 18px rgba(2,6,23,0.04); width:100%; }

    /* article row + link square */
    .article-row { display:flex; gap:12px; align-items:flex-start; }
    .article-main { flex:1 1 auto; }
    .link-square {
      width:56px; height:56px; border-radius:8px; display:flex; align-items:center; justify-content:center;
      background: linear-gradient(180deg,#eef2ff,#e0f2fe); box-shadow:0 6px 20px rgba(2,6,23,0.04);
      text-decoration:none; color:#0f172a; font-weight:700;
    }
    .link-square.disabled {
      background: linear-gradient(180deg,#f3f4f6,#e5e7eb);
      color:#94a3b8;
      pointer-events: none;
      cursor: default;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers: rating mapping + classify
# ---------------------------
def rating_badge_props(rating_text):
    if not rating_text:
        return ("Unknown", "badge-unknown")
    t = str(rating_text).lower()
    if "true" in t or "real" in t:
        return ("True", "badge-true")
    if "false" in t or "fake" in t:
        return ("False", "badge-false")
    if "misleading" in t or "partly" in t:
        return ("Misleading", "badge-misleading")
    return (rating_text, "badge-unknown")

def classify_with_model(text):
    """Return model label or 'Unknown' if model missing or error."""
    if pipe is None:
        return "Unknown"
    try:
        pred = pipe.predict([text])[0]
        return pred
    except Exception:
        return "Unknown"

# ---------------------------
# API wrappers
# ---------------------------
def fact_check_google(query, page_size=10):
    if not FACTCHECK_KEY:
        return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": FACTCHECK_KEY, "languageCode": "en", "pageSize": page_size}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        return r.json().get("claims", [])
    except Exception:
        return []

def fetch_newsapi(query, from_date=None, to_date=None, page_size=5, language="en"):
    if not NEWSAPI_KEY:
        return []
    url = f"https://newsapi.org/v2/everything?q={query}&language={language}&pageSize={page_size}&apiKey={NEWSAPI_KEY}"
    if from_date:
        url += f"&from={from_date}"
    if to_date:
        url += f"&to={to_date}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json().get("articles", [])
    except Exception:
        return []

def fetch_gnews(query, from_date=None, to_date=None, max_results=5, language="en"):
    if not GNEWS_KEY:
        return []
    url = f"https://gnews.io/api/v4/search?q={query}&lang={language}&max={max_results}&token={GNEWS_KEY}"
    if from_date:
        url += f"&from={from_date}"
    if to_date:
        url += f"&to={to_date}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json().get("articles", [])
    except Exception:
        return []

def fetch_bing(query, count=5, language="en"):
    if not BING_KEY:
        return []
    url = f"https://api.bing.microsoft.com/v7.0/news/search?q={query}&count={count}&mkt={language}-{language.upper()}"
    headers = {"Ocp-Apim-Subscription-Key": BING_KEY}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json().get("value", [])
    except Exception:
        return []

# ---------------------------
# Session state init
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []   # each entry: {q, date, counts}
if "saved" not in st.session_state:
    st.session_state.saved = []     # saved articles/claims
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "per_source_counts" not in st.session_state:
    st.session_state.per_source_counts = {"google_factcheck": 0, "newsapi": 0, "gnews": 0, "bing": 0}
if "last_results" not in st.session_state:
    st.session_state.last_results = {"google": [], "newsapi": [], "gnews": [], "bing": []}

# ---------------------------
# Sidebar (date range + options)
# ---------------------------
with st.sidebar:
    st.markdown("### 🔎 Search Options")
    from_date = st.date_input("From date", value=None)
    to_date = st.date_input("To date", value=None)
    max_results = st.slider("Max results per source", 1, 10, 5)
    st.markdown("---")
    st.markdown("### 📊 Session Controls")
    st.write(f"- Queries this session: **{st.session_state.total_queries}**")
    st.write(f"- Saved items: **{len(st.session_state.saved)}**")
    st.markdown("---")
    st.caption("Tip: Provide a short claim/headline for best results")

from_date_str = from_date.strftime("%Y-%m-%d") if from_date else None
to_date_str = to_date.strftime("%Y-%m-%d") if to_date else None

# ---------------------------
# Header
# ---------------------------
st.markdown(
    """
    <div class="header">
        <div class="logo">📊</div>
        <div>
            <p class="title">FactMatrix</p>
            <p class="subtitle">Multi-source verification • Verdicts (ML) • Session insights</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Tabs: Search (first), Overview, History
# ---------------------------
tab_search, tab_overview, tab_history = st.tabs(["Search", "Overview", "History"])

# ---------------------------
# Search tab (main UI preserved) - SEARCH FIRST now
# ---------------------------
with tab_search:
    st.markdown("### Search & Analyze")
    user_input = st.text_area("✍ Enter a short headline, tweet, or claim to analyze:", height=100,
                              placeholder='e.g. "New study shows vaccine microchips"')
    analyze_btn = st.button("🔎 Analyze")

    if analyze_btn:
        q = user_input.strip()
        if not q:
            st.warning("Please enter a statement to analyze.")
        else:
            # update session counts
            st.session_state.total_queries += 1
            today_date = datetime.now().strftime("%Y-%m-%d")
            entry_counts = {"google_factcheck": 0, "newsapi": 0, "gnews": 0, "bing": 0}
            st.session_state.last_results = {"google": [], "newsapi": [], "gnews": [], "bing": []}

            # Google FactCheck
            st.markdown("#### ✅ Verified Claims (Google FactCheck)")
            with st.spinner("Fetching Google FactCheck results..."):
                claims = fact_check_google(q, page_size=max_results)
            if claims:
                entry_counts["google_factcheck"] = len(claims)
                for idx, c in enumerate(claims):
                    text = c.get("text", "N/A")
                    claimant = c.get("claimant", "Unknown")
                    date = c.get("claimDate", "")
                    date_only = date.split("T")[0] if date else "—"
                    review = c.get("claimReview", [{}])[0]
                    site = review.get("publisher", {}).get("name", "Unknown")
                    rating = review.get("textualRating", "No verdict")
                    label, badge_class = rating_badge_props(rating)
                    review_url = review.get("url", "")
                    # show card with right-side link-square if URL present
                    if review_url:
                        link_html = f"<a class='link-square' href='{review_url}' target='_blank'>OPEN</a>"
                    else:
                        link_html = "<div class='link-square disabled'>—</div>"
                    st.markdown(
                        f"<div class='card'><div class='article-row'>"
                        f"<div class='article-main'><strong>🗣 Claim #{idx+1}:</strong> {text}<br/>"
                        f"<span class='muted'>Source: {claimant} • Reviewed by: {site} • 📅 {date_only}</span><br/>"
                        f"<span class='badge {badge_class}'>{label}</span></div>"
                        f"{link_html}"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
                    st.session_state.last_results["google"].append({
                        "text": text, "source": site, "date": date_only, "rating": rating, "url": review_url
                    })
            else:
                st.info("No fact-checked claims found.")
            st.markdown("---")

            # NewsAPI
            st.markdown("#### 📰 News Articles (NewsAPI)")
            with st.spinner("Fetching NewsAPI articles..."):
                newsapi_articles = fetch_newsapi(q, from_date_str, to_date_str, page_size=max_results)
            if newsapi_articles:
                entry_counts["newsapi"] = len(newsapi_articles)
                for art in newsapi_articles:
                    title = art.get("title") or art.get("description") or "Untitled"
                    url = art.get("url")
                    pub = art.get("publishedAt", "")
                    date_only = pub.split("T")[0] if pub else "—"
                    pred = classify_with_model(title)
                    label, badge_class = rating_badge_props(pred)
                    # right-side link square (disabled if no url)
                    if url:
                        link_html = f"<a class='link-square' href='{url}' target='_blank'>OPEN</a>"
                    else:
                        link_html = "<div class='link-square disabled'>—</div>"
                    st.markdown(
                        f"<div class='card'><div class='article-row'>"
                        f"<div class='article-main'><strong>{title}</strong><br/>"
                        f"<span class='muted'>Source: {art.get('source', {}).get('name','Unknown')} • 📅 {date_only}</span><br/>"
                        f"<span class='badge {badge_class}'>{label}</span></div>"
                        f"{link_html}"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
                    # save button
                    if url and st.button("📌 Save article", key=f"save_newsapi_{hash(url)}"):
                        st.session_state.saved.append({
                            "title": title,
                            "source": art.get("source", {}).get("name", "Unknown"),
                            "url": url,
                            "pred": pred,
                            "claim_date": date_only,
                            "saved_on": today_date
                        })
                        st.success("Saved to session")
            else:
                st.caption("No results from NewsAPI.")
            st.markdown("---")

            # GNews
            st.markdown("#### 🌍 Global News (GNews)")
            with st.spinner("Fetching GNews articles..."):
                gnews_articles = fetch_gnews(q, from_date_str, to_date_str, max_results=max_results)
            if gnews_articles:
                entry_counts["gnews"] = len(gnews_articles)
                for art in gnews_articles:
                    title = art.get("title") or art.get("description") or "Untitled"
                    url = art.get("url")
                    pub = art.get("publishedAt", "") or art.get("publishedAt")
                    date_only = pub.split("T")[0] if pub else "—"
                    pred = classify_with_model(title)
                    label, badge_class = rating_badge_props(pred)
                    if url:
                        link_html = f"<a class='link-square' href='{url}' target='_blank'>OPEN</a>"
                    else:
                        link_html = "<div class='link-square disabled'>—</div>"
                    st.markdown(
                        f"<div class='card'><div class='article-row'>"
                        f"<div class='article-main'><strong>{title}</strong><br/>"
                        f"<span class='muted'>Source: {art.get('source','Unknown')} • 📅 {date_only}</span><br/>"
                        f"<span class='badge {badge_class}'>{label}</span></div>"
                        f"{link_html}"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
                    if url and st.button("📌 Save article", key=f"save_gnews_{hash(url)}"):
                        st.session_state.saved.append({
                            "title": title,
                            "source": art.get('source','Unknown') if isinstance(art.get("source"), str) else art.get("source",{}).get("name","Unknown"),
                            "url": url,
                            "pred": pred,
                            "claim_date": date_only,
                            "saved_on": today_date
                        })
                        st.success("Saved to session")
            else:
                st.caption("No results from GNews.")
            st.markdown("---")

            # Bing
            st.markdown("#### 💠 Bing News")
            with st.spinner("Fetching Bing articles..."):
                bing_articles = fetch_bing(q, count=max_results)
            if bing_articles:
                entry_counts["bing"] = len(bing_articles)
                for art in bing_articles:
                    title = art.get("name") or art.get("description") or "Untitled"
                    url = art.get("url")
                    pub = art.get("datePublished", "")
                    date_only = pub.split("T")[0] if pub else "—"
                    pred = classify_with_model(title)
                    label, badge_class = rating_badge_props(pred)
                    if url:
                        link_html = f"<a class='link-square' href='{url}' target='_blank'>OPEN</a>"
                    else:
                        link_html = "<div class='link-square disabled'>—</div>"
                    st.markdown(
                        f"<div class='card'><div class='article-row'>"
                        f"<div class='article-main'><strong>{title}</strong><br/>"
                        f"<span class='muted'>Source: {art.get('provider',[{}])[0].get('name','Unknown')} • 📅 {date_only}</span><br/>"
                        f"<span class='badge {badge_class}'>{label}</span></div>"
                        f"{link_html}"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
                    if url and st.button("📌 Save article", key=f"save_bing_{hash(url)}"):
                        st.session_state.saved.append({
                            "title": title,
                            "source": art.get('provider',[{}])[0].get('name','Unknown'),
                            "url": url,
                            "pred": pred,
                            "claim_date": date_only,
                            "saved_on": today_date
                        })
                        st.success("Saved to session")
            else:
                st.caption("No results from Bing.")

            # update per-source counts
            for k, v in entry_counts.items():
                st.session_state.per_source_counts[k] = st.session_state.per_source_counts.get(k, 0) + v

            # push history entry (only date, no time)
            st.session_state.history.append({
                "q": q,
                "date": today_date,
                "counts": entry_counts
            })

# ---------------------------
# Overview tab
# ---------------------------
with tab_overview:
    st.markdown("### Session overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Queries (session)", st.session_state.total_queries)
    col2.metric("Saved items", len(st.session_state.saved))
    # per-source small metrics:
    col3.metric("FactChecked (Google)", st.session_state.per_source_counts.get("google_factcheck", 0))
    col4.metric("NewsAPI results", st.session_state.per_source_counts.get("newsapi", 0))

    st.markdown("---")
    st.markdown("#### Per-source results (this session)")
    counts = st.session_state.per_source_counts
    df_counts = pd.DataFrame(list(counts.items()), columns=["source", "count"]).set_index("source")
    st.bar_chart(df_counts)

    st.markdown("---")
    st.markdown("Quick actions")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Export saved (CSV)"):
            if st.session_state.saved:
                df = pd.DataFrame(st.session_state.saved)
                st.download_button("Download saved.csv", df.to_csv(index=False).encode("utf-8"), file_name="factmatrix_saved.csv")
            else:
                st.info("No saved items to export.")
    with c2:
        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.total_queries = 0
            st.session_state.per_source_counts = {"google_factcheck": 0, "newsapi": 0, "gnews": 0, "bing": 0}
            st.success("Session cleared.")
    with c3:
        st.write("")  # spacer

# ---------------------------
# History tab
# ---------------------------
with tab_history:
    st.markdown("### Search history (this session)")
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history[-50:])):
            st.markdown(f"- `{h['date']}` — **{h['q']}**  • counts: {h['counts']}")
    else:
        st.info("No searches recorded this session.")

    st.markdown("---")
    st.markdown("### Saved items")
    if st.session_state.saved:
        for s in reversed(st.session_state.saved[-100:]):
            st.markdown(f"- **{s.get('title','—')}** • {s.get('source','—')} • Date: {s.get('claim_date','—')} • [link]({s.get('url')})")
    else:
        st.caption("No saved items yet. Use the 'Save' buttons in Search results.")

# ---------------------------
# End
# ---------------------------
st.markdown("---")
st.markdown("<div class='small-muted'>Built with ❤️ • FactMatrix</div>", unsafe_allow_html=True)