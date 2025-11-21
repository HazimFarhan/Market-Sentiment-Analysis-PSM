# msrt_main_v3.py
# Combined NewsAPI + Google News RSS + Yahoo Finance fetcher integrated into MSRT
# Replace your existing MSRT(main).py with this or copy the fetch_news() and date handling parts.

import os
import requests
import feedparser
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from transformers import pipeline
import torch

# -------------------------
# Configuration (edit as needed)
# -------------------------
MARKET_KEYWORDS = {
    "Stocks": ["stock", "equity", "share", "s&p", "nasdaq", "dow", "index", "bull", "bear", "market"],
    "Forex": ["forex", "currency", "usd", "eur", "gbp", "jpy", "fx", "exchange rate", "dollar", "yen"],
    "Commodities": ["oil", "gold", "silver", "commodity", "crude", "barrel", "ounce", "copper", "futures"],
    "Economy": ["gdp", "inflation", "cpi", "unemployment", "economic", "growth", "recession", "fed", "ecb", "central bank"],
    "Crypto": ["bitcoin", "crypto", "blockchain", "btc", "eth", "digital currency", "token", "defi"],
    "Bonds": ["bond", "yield", "treasury", "debt", "10-year", "notes", "credit rating"]
}
AVAILABLE_MARKETS = list(MARKET_KEYWORDS.keys())

# How many past days to fetch by default
DEFAULT_DAYS_BACK = 5

# NewsAPI endpoint template (supports from= and to=)
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

# Google News RSS search template (we will search for broad queries)
GOOGLE_RSS_TEMPLATE = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

# Optional: list of tickers to query via yfinance for market-specific news (keeps Yahoo footprint small)
YFINANCE_TICKERS = ["^GSPC", "GC=F", "CL=F", "BTC-USD", "AAPL", "MSFT", "NVDA"]  # adjust as needed

# -------------------------
# Helper functions
# -------------------------
def is_financial(text: str) -> bool:
    text_lower = (text or "").lower()
    # accept if any single market has at least one matching term
    for terms in MARKET_KEYWORDS.values():
        if any(term in text_lower for term in terms):
            # exclude obvious non-financial contexts
            if not any(nx in text_lower for nx in ["sports", "entertainment", "celebrity", "movie", "music"]):
                return True
    return False

def parse_rss_date(dt_str):
    """Try a few common RSS date formats, return timezone-aware datetime or None."""
    if not dt_str:
        return None
    # feedparser gives published_parsed as struct_time, but sometimes we get string
    try:
        # feedparser
        import email.utils
        parsed = email.utils.parsedate_to_datetime(dt_str)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        pass
    # fallback
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(dt_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return None

# -------------------------
# Combined news fetcher
# -------------------------
def fetch_news_combined(newsapi_key: str, days_back: int = DEFAULT_DAYS_BACK, query_terms: str = None, max_results: int = 500):
    """
    Fetch news from three sources:
     - NewsAPI (requires API key)
     - Google News RSS (search-based)
     - Yahoo Finance via yfinance (if available)
    Returns a deduplicated list of dicts: {"text": ..., "date": "YYYY-MM-DD", "source": ...}
    """
    now = datetime.now(timezone.utc)
    cutoff_date = now - timedelta(days=int(days_back))  # ensures int delta
    results = []
    seen = set()

    # --- 1) NewsAPI (everything endpoint with from/to) ---
    if newsapi_key:
        try:
            print("MSRT: Fetching NewsAPI articles...")
            # construct params - 'q' optional (query_terms); limit page size and paginate if needed
            params = {
                "pageSize": 100,
                "language": "en",
                "sortBy": "publishedAt",
            }
            if query_terms:
                params["q"] = query_terms
            # from (ISO format)
            params["from"] = cutoff_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            params["to"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            params["apiKey"] = newsapi_key

            page = 1
            fetched = 0
            while True:
                params["page"] = page
                resp = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=15)
                data = resp.json()
                articles = data.get("articles", [])
                if not articles:
                    break
                for a in articles:
                    title = (a.get("title") or "").strip()
                    desc = (a.get("description") or "").strip()
                    text = f"{title}. {desc}".strip()
                    pub = a.get("publishedAt")
                    try:
                        pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00")).astimezone(timezone.utc) if pub else None
                    except Exception:
                        pub_dt = None
                    if pub_dt is None:
                        pub_dt = now
                    if pub_dt < cutoff_date:
                        continue
                    key = (text[:400])
                    if text and key not in seen and is_financial(text):
                        seen.add(key)
                        results.append({"text": text[:1000], "date": pub_dt.date().isoformat(), "source": "NewsAPI"})
                        fetched += 1
                        if fetched >= max_results:
                            break
                if fetched >= max_results:
                    break
                page += 1
        except Exception as e:
            print("⚠️ MSRT: NewsAPI error:", e)

    # --- 2) Google News RSS ---
    try:
        print("MSRT: Fetching Google News RSS...")
        # broad query: if user provided query_terms use it; otherwise search for financial keywords joined by OR
        if not query_terms:
            # create a joined query of important keywords (space joins use OR for Google News)
            joined = " OR ".join([kw for terms in MARKET_KEYWORDS.values() for kw in terms[:3]])  # take a few keywords
            query = joined
        else:
            query = query_terms

        rss_url = GOOGLE_RSS_TEMPLATE.format(query=requests.utils.quote(query))
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            title = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()
            text = f"{title}. {summary}".strip()
            pub_dt = None
            # feedparser exposes published or published_parsed
            if 'published' in entry:
                pub_dt = parse_rss_date(entry.get("published"))
            elif 'updated' in entry:
                pub_dt = parse_rss_date(entry.get("updated"))
            if pub_dt is None:
                pub_dt = now
            if pub_dt < cutoff_date:
                continue
            key = text[:400]
            if text and key not in seen and is_financial(text):
                seen.add(key)
                results.append({"text": text[:1000], "date": pub_dt.date().isoformat(), "source": "GoogleRSS"})
    except Exception as e:
        print("⚠️ MSRT: Google RSS error:", e)

    # --- 3) Yahoo Finance via yfinance (ticker news) ---
    try:
        print("MSRT: Fetching Yahoo Finance news via yfinance tickers...")
        for ticker in YFINANCE_TICKERS:
            try:
                t = yf.Ticker(ticker)
                news_items = t.news  # list of dicts: title, publisher, link, providerPublishTime
                for n in news_items:
                    title = n.get("title", "")
                    summary = n.get("summary") or ""
                    text = f"{title}. {summary}".strip()
                    pub_ts = n.get("providerPublishTime")
                    pub_dt = datetime.fromtimestamp(pub_ts, tz=timezone.utc) if pub_ts else now
                    if pub_dt < cutoff_date:
                        continue
                    key = text[:400]
                    if text and key not in seen and is_financial(text):
                        seen.add(key)
                        results.append({"text": text[:1000], "date": pub_dt.date().isoformat(), "source": f"Yahoo:{ticker}"})
            except Exception as inner:
                # don't fail whole fetch on single ticker
                print(f"  - Warning: yahoo ticker {ticker} fetch error: {inner}")
    except Exception as e:
        print("⚠️ MSRT: yfinance error (skipping Yahoo news):", e)

    # --- Final dedup & sort by date desc ---
    try:
        df = pd.DataFrame(results)
        if df.empty:
            return []
        # keep unique by text
        df = df.drop_duplicates(subset=["text"])
        df = df.sort_values("date", ascending=False)
        # Limit to max_results if too many
        if len(df) > max_results:
            df = df.head(max_results)
        return df.to_dict("records")
    except Exception as e:
        print("⚠️ MSRT: Final aggregation error:", e)
        return results

# -------------------------
# Model loading & signal (use your existing pipelines)
# -------------------------
def load_models(finbert_path=None):
    try:
        device = 0 if torch.cuda.is_available() else -1
        if finbert_path:
            print("MSRT: Loading fine-tuned FinBERT from", finbert_path)
            finbert = pipeline("text-classification", model=finbert_path, device=device)
        else:
            print("MSRT: Loading upstream FinBERT")
            finbert = pipeline("text-classification", model="yiyanghkust/finbert-tone", device=device)

        emotion = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", device=device)
        return finbert, emotion
    except Exception as e:
        print("⚠️ MSRT: Model loading error:", e)
        raise

# Example refined get_signal (same as we previously agreed)
def get_signal(text, finbert_pipeline, emotion_pipeline):
    try:
        finbert_result = finbert_pipeline(text[:512], return_all_scores=True)[0]
        finbert_sorted = sorted(finbert_result, key=lambda x: x['score'], reverse=True)
        sentiment_label = finbert_sorted[0]['label']
        sentiment_conf = finbert_sorted[0]['score']

        emotion_result = emotion_pipeline(text[:512], return_all_scores=True)[0]
        emotion_sorted = sorted(emotion_result, key=lambda x: x['score'], reverse=True)
        emotion_label = emotion_sorted[0]['label']
        emotion_conf = emotion_sorted[0]['score']

        market = next((mkt for mkt, terms in MARKET_KEYWORDS.items()
                       if any(term in text.lower() for term in terms)), "General")

        sentiment_score = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}.get(sentiment_label, 0.0)
        emotion_weight = {
            "joy": 0.4, "optimism": 0.3, "excitement": 0.2,
            "fear": -0.4, "anger": -0.3, "annoyance": -0.1
        }.get(emotion_label.lower(), 0)

        score = (sentiment_score * (0.7 + 0.3 * sentiment_conf)) + (emotion_weight * emotion_conf)
        score *= {"Stocks": 1.2, "Crypto": 1.3, "Commodities": 1.1}.get(market, 1.0)

        if -0.4 < score < 0.4:
            final_signal = "⚪ Neutral"
        elif score >= 1.5:
            final_signal = "🟢 Strong Buy"
        elif score >= 0.7:
            final_signal = "🟢 Buy"
        elif score <= -1.5:
            final_signal = "🔴 Strong Sell"
        elif score <= -0.7:
            final_signal = "🟠 Sell"
        else:
            final_signal = "⚪ Neutral"

        return {
            "Signal": final_signal,
            "FinBERT_Label": sentiment_label,
            "FinBERT_Confidence": round(sentiment_conf, 3),
            "Emotion_Label": emotion_label,
            "Emotion_Confidence": round(emotion_conf, 3),
            "Score": round(score, 3),
        }
    except Exception as e:
        print("⚠️ MSRT: get_signal error:", e)
        return {
            "Signal": "⚪ Neutral",
            "FinBERT_Label": "Error",
            "FinBERT_Confidence": 0.0,
            "Emotion_Label": "Error",
            "Emotion_Confidence": 0.0,
            "Score": 0.0
        }

# -------------------------
# Main analysis wrapper (keeps your interactive flow)
# -------------------------
def analyze_selected_market(target_market, finbert, emotion, newsapi_key=None, days_back=DEFAULT_DAYS_BACK):
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time}] MSRT: Fetching news for last {days_back} days from combined sources...")
    # optional query tailored to the market to focus RSS/NewsAPI searches
    market_query = " OR ".join(MARKET_KEYWORDS.get(target_market, []))
    news_items = fetch_news_combined(newsapi_key, days_back=days_back, query_terms=market_query, max_results=500)

    if not news_items:
        print("⚠️ MSRT: No news items found.")
        return

    keywords = MARKET_KEYWORDS.get(target_market, [])
    filtered = [it for it in news_items if any(kw in it["text"].lower() for kw in keywords)]
    print(f"MSRT: Found {len(filtered)} candidate articles after keyword filtering.")

    results = []
    for item in filtered:
        sig = get_signal(item["text"], finbert, emotion)
        results.append({
            "Market": target_market,
            "Headline": item["text"][:200],
            "Signal": sig["Signal"],
            "Sentiment": sig["FinBERT_Label"],
            "Sent_Conf": sig["FinBERT_Confidence"],
            "Emotion": sig["Emotion_Label"],
            "Emo_Conf": sig["Emotion_Confidence"],
            "Score": sig["Score"],
            "Date": item["date"],
            "Source": item.get("source"),
            "Run_Time": current_time
        })

    if results:
        df = pd.DataFrame(results)
        log_file = f"msrt_{target_market.lower()}_signals.csv"
        write_header = not os.path.exists(log_file)
        df.to_csv(log_file, index=False, mode='a', header=write_header, encoding='utf-8')
        print(f"✅ MSRT: Saved {len(df)} signals to {log_file}")
        print(df.tail())
    else:
        print("⚠️ MSRT: No signals generated.")

# -------------------------
# Interactive entry point
# -------------------------
if __name__ == "__main__":
    load_dotenv()
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
    FINBERT_PATH = os.getenv("FINBERT_MODEL_DIR", None)  # optionally point to your fine-tuned model folder

    finbert, emotion = load_models(FINBERT_PATH)

    while True:
        print("\n" + "="*60)
        print("MSRT Interactive Market Sentiment Analysis")
        print("Available Markets:", ", ".join(AVAILABLE_MARKETS))
        print("Type 'exit' to quit.")
        print("="*60)

        user_input = input("Enter market to analyze (e.g., Stocks, Crypto): ").strip()
        if user_input.lower() == "exit":
            print("MSRT: Goodbye.")
            break
        target = user_input.title()
        if target not in AVAILABLE_MARKETS:
            print("⚠️ Invalid market. Try again.")
            continue

        # Ask how many days back to fetch (press Enter for default)
        days_str = input(f"How many days back to fetch? (default {DEFAULT_DAYS_BACK}): ").strip()
        try:
            days = int(days_str) if days_str else DEFAULT_DAYS_BACK
        except Exception:
            days = DEFAULT_DAYS_BACK

        analyze_selected_market(target, finbert, emotion, newsapi_key=NEWSAPI_KEY, days_back=days)
