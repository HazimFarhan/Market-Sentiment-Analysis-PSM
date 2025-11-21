# MSRT_main_with_targets.py
# Enhanced MSRT: interactive target selection (Gold or Forex pair), combined NewsAPI+GoogleRSS+YahooRSS,
# pick most recent article per day for last 5 days, then analyze & save.

import os
import requests
import feedparser
import pandas as pd
import time
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from transformers import pipeline
from dotenv import load_dotenv
import torch

# ------------ CONFIG -------------
MARKET_KEYWORDS = {
    "Forex": [
        "forex", "currency", "usd", "eur", "gbp", "jpy", "fx", 
        "exchange rate", "dollar", "yen", "pair", "forex market"
    ],
    "Gold": [
        "gold price", "xauusd", "gold market", "spot gold",
        "gold futures", "gold rises", "gold falls",
        "safe haven", "precious metal"
    ],
    "Crypto": ["bitcoin", "crypto", "blockchain", "btc", "eth", "token"],
    "Stocks": ["stock", "equity", "share", "s&p", "nasdaq", "dow"],
    "Commodities": ["oil", "silver", "commodity", "crude", "barrel", "futures"],
    "Economy": ["gdp", "inflation", "recession", "fed", "cpi", "central bank"],
    "Bonds": ["bond", "yield", "treasury", "debt"]
}

AVAILABLE_MARKETS = list(MARKET_KEYWORDS.keys())

# ------------ HELPERS -------------
def is_financial(text):
    text_lower = text.lower()
    # accept if any single market has at least one matching term
    for terms in MARKET_KEYWORDS.values():
        if any(term in text_lower for term in terms):
            if not any(nx in text_lower for nx in ["sports", "entertainment", "celebrity", "movie", "music"]):
                return True
    return False

# ------------ MODEL LOADING -------------
try:
    print("MSRT: Loading AI models (attempt GPU: device=0)...")
    finbert = pipeline("text-classification", model="yiyanghkust/finbert-tone", device=0)
    emotion = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", device=0)
    print("MSRT: AI Models loaded successfully.")
except Exception as e:
    print(f"⚠️ GPU unavailable or load error: {e} — falling back to CPU.")
    finbert = pipeline("text-classification", model="yiyanghkust/finbert-tone", device=-1)
    emotion = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", device=-1)

# ------------ SIGNAL GENERATOR (your existing logic) -------------
def get_signal(text):
    try:
        sentiment = finbert(text[:512])[0]['label']
        emotion_label = emotion(text[:512])[0]['label']
        market = next((mkt for mkt, terms in MARKET_KEYWORDS.items()
                       if any(term in text.lower() for term in terms)), "General")
        score = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}[sentiment]
        score += {"joy": 0.3, "optimism": 0.2, "excitement": 0.1,
                  "fear": -0.3, "anger": -0.2, "annoyance": -0.1}.get(emotion_label.lower(), 0)
        score *= {"Stocks": 1.2, "Crypto": 1.3, "Commodities": 1.1}.get(market, 1.0)
        if score >= 1.5: return "🟢 Strong Buy"
        elif score >= 0.7: return "🟢 Buy"
        elif score <= -1.5: return "🔴 Strong Sell"
        elif score <= -0.7: return "🟠 Sell"
        return "⚪ Neutral"
    except Exception as e:
        print(f"⚠️ MSRT: Analysis error: {e}")
        return "⚪ Neutral"

# ------------ NEWS SOURCES (combined) -------------

def fetch_news_newsapi(api_key, q, page_size=50):
    items = []
    try:
        url = "https://newsapi.org/v2/everything"
        params = {"q": q, "language": "en", "pageSize": page_size, "apiKey": api_key, "sortBy": "publishedAt"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
        for a in j.get("articles", []):
            title = a.get("title", "") or ""
            desc = a.get("description", "") or ""
            text = f"{title}. {desc}".strip()
            published = a.get("publishedAt", None)
            if published:
                try:
                    dt = dateparser.parse(published)
                except:
                    dt = None
            else:
                dt = None
            items.append({"text": text, "date": dt})
    except Exception as e:
        print(f"NewsAPI fetch error: {e}")
    return items

def fetch_news_google_rss(q, max_items=100):
    items = []
    try:
        # google news RSS search
        rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        for e in feed.entries[:max_items]:
            title = e.get("title", "")
            desc = e.get("summary", "")
            text = f"{title}. {desc}".strip()
            # Google provides published time in 'published' or 'updated'
            published = e.get("published", e.get("updated", None))
            dt = None
            if published:
                try:
                    dt = dateparser.parse(published)
                except:
                    dt = None
            items.append({"text": text, "date": dt})
    except Exception as e:
        print(f"Google RSS fetch error: {e}")
    return items

def fetch_news_yahoo_rss(q, max_items=100):
    items = []
    try:
        # Yahoo finance RSS: use the search-style feed for query
        # e.g. https://news.yahoo.com/rss/search?p=gold
        rss_url = f"https://news.yahoo.com/rss/search?p={requests.utils.quote(q)}"
        feed = feedparser.parse(rss_url)
        for e in feed.entries[:max_items]:
            title = e.get("title", "")
            desc = e.get("summary", "")
            text = f"{title}. {desc}".strip()
            published = e.get("published", e.get("updated", None))
            dt = None
            if published:
                try:
                    dt = dateparser.parse(published)
                except:
                    dt = None
            items.append({"text": text, "date": dt})
    except Exception as e:
        print(f"Yahoo RSS fetch error: {e}")
    return items

def fetch_combined_news(newsapi_key, query, limit_days=5):
    """
    Fetch from NewsAPI + Google RSS + Yahoo RSS for the given query.
    Then keep at most one (most recent) article per day, for up to 'limit_days' distinct days.
    Returns list of dicts with 'text' and 'date' (datetime).
    """
    print(f"MSRT: Fetching combined news for query: {query}")
    combined = []
    if newsapi_key:
        combined += fetch_news_newsapi(newsapi_key, query, page_size=50)
    combined += fetch_news_google_rss(query, max_items=80)
    combined += fetch_news_yahoo_rss(query, max_items=80)

    # keep items with dates; if some have no date, set to now (so they'll be recent)
    for it in combined:
        if it["date"] is None:
            it["date"] = datetime.utcnow()

    # Normalize: produce date-only key (YYYY-MM-DD), sort by datetime desc
    combined_sorted = sorted(combined, key=lambda x: x["date"], reverse=True)

    # pick most recent article per day up to limit_days
    selected = []
    seen_days = set()
    for it in combined_sorted:
        day = it["date"].strftime("%Y-%m-%d")
        if day not in seen_days:
            selected.append(it)
            seen_days.add(day)
        if len(seen_days) >= limit_days:
            break

    # If fewer than limit_days available, we still return what we have
    print(f"MSRT: Collected {len(selected)} daily articles (most recent per day, up to {limit_days}).")
    return selected

# ------------ ANALYSIS & CSV LOGGING -------------
def analyze_and_save(target_query, target_market_label, newsapi_key, out_prefix="msrt"):
    items = fetch_combined_news(newsapi_key, target_query, limit_days=5)
    if not items:
        print("MSRT: No news items found for query.")
        return

    results = []
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for it in items:
        txt = it["text"][:1000]
        signal = get_signal(txt)
        results.append({
            "Market": target_market_label,
            "Headline": txt[:200] + ("..." if len(txt) > 200 else ""),
            "Signal": signal,
            "Date": it["date"].strftime("%Y-%m-%d"),
            "Run_Time": run_time
        })

    df = pd.DataFrame(results)
    # filename like msrt_gold_signals.csv or msrt_forex_eurusd_signals.csv
    safe_label = target_market_label.lower().replace(" ", "_")
    filename = f"{out_prefix}_{safe_label}_signals.csv"
    write_header = not os.path.exists(filename)
    df.to_csv(filename, mode='a', index=False, header=write_header, encoding='utf-8')
    print(f"✅ MSRT: Results saved to {filename}")
    print(df)

# ------------ INTERACTIVE FLOW -------------
if __name__ == "__main__":
    load_dotenv()
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")  # optional, works without but better with
    print("MSRT Interactive - Targeted news for MT5 charting.")
    print("Pick a target to display on chart (keyword-based). Options: [gold] or [forex].\n")

    while True:
        choice = input("Target (gold / forex), or 'exit' to quit: ").strip().lower()
        if choice == "exit":
            print("Bye.")
            break
        if choice not in ("gold", "forex"):
            print("Invalid. Enter 'gold' or 'forex'.")
            continue

        if choice == "gold":
            # Accept synonyms
            user_symbol = input("Enter gold keyword (e.g. 'gold' or 'xauusd') (press Enter for 'gold'): ").strip()
            if user_symbol == "":
                user_symbol = "gold"
            # build query - keep it simple & keyword-rich
            query = f"{user_symbol} OR gold OR xauusd OR XAU"
            analyze_and_save(query, "Gold", NEWSAPI_KEY)

        else:  # forex
            pair = input("Enter forex pair (e.g. EURUSD, USDJPY, GBPUSD): ").strip().upper()
            if pair == "":
                print("No pair entered. Try again.")
                continue
            # Create friendly, keyword-rich queries: ticker, symbols, currency names
            pair_clean = pair.replace("/", "").replace("-", "")
            # Expand symbols to names for better search coverage
            pair_map_names = {
                "EURUSD": "EUR USD euro dollar",
                "USDJPY": "USD JPY dollar yen",
                "GBPUSD": "GBP USD pound dollar",
                "AUDUSD": "AUD USD aussie dollar",
                "USDCAD": "USD CAD dollar canadian",
                "USDCHF": "USD CHF dollar swiss"
            }
            name_terms = pair_map_names.get(pair_clean, pair_clean)
            query = f"{pair_clean} OR {name_terms}"
            analyze_and_save(query, f"Forex_{pair_clean}", NEWSAPI_KEY)

        # short sleep to avoid spamming services
        time.sleep(1)
