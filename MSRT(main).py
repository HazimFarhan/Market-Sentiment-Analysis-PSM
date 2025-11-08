import pandas as pd
import requests
import os
import time
from datetime import datetime
from transformers import pipeline
from dotenv import load_dotenv
import torch 

# --- CONFIGURATION (Based on RTV1.ipynb) ---

# [3] Financial Market Detection Config
MARKET_KEYWORDS = {
    "Stocks": ["stock", "equity", "share", "S&P", "NASDAQ", "Dow", "index", "bull", "bear", "market"],
    "Forex": ["forex", "currency", "USD", "EUR", "GBP", "JPY", "FX", "exchange rate", "dollar", "yen"],
    "Commodities": ["oil", "gold", "silver", "commodity", "crude", "barrel", "ounce", "copper", "futures"],
    "Economy": ["GDP", "inflation", "CPI", "unemployment", "economic", "growth", "recession", "Fed", "ECB", "central bank"],
    "Crypto": ["bitcoin", "crypto", "blockchain", "BTC", "ETH", "digital currency", "token", "DeFi"],
    "Bonds": ["bond", "yield", "treasury", "debt", "10-year", "notes", "credit rating"]
}
AVAILABLE_MARKETS = list(MARKET_KEYWORDS.keys()) 

# [4] Relaxed Financial Content Filter
def is_financial(text):
    """More inclusive financial content detection"""
    text_lower = text.lower()
    financial_terms = sum(
        1 for terms in MARKET_KEYWORDS.values() 
        for term in terms if term in text_lower
    )
    non_financial = any(
        term in text_lower 
        for term in ["sports", "entertainment", "celebrity", "movie", "music"]
    )
    return financial_terms >= 2 and not non_financial

# ----------------------------------------------------
# --- GLOBAL MODEL INITIALIZATION ---
try:
    print("MSRT: Loading AI models (using GPU: device=0)...")
    finbert = pipeline("text-classification", model="yiyanghkust/finbert-tone", device=0)
    emotion = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", device=0)
    print("MSRT: AI Models loaded successfully. Ready for user input.")
    
except Exception as e:
    print(f"⚠️ MSRT: GPU initialization failed: {str(e)}. Falling back to CPU.")
    finbert = pipeline("text-classification", model="yiyanghkust/finbert-tone")
    emotion = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
    print("MSRT: AI Models loaded on CPU. Ready for user input.")


# [5] News Fetcher with Better Filtering
def fetch_news(newsapi_key, alphavantage_key):
    """Get news from multiple sources"""
    sources = [
        ("NewsAPI", "https://newsapi.org/v2/top-headlines?category=business&apiKey=" + newsapi_key),
        ("AlphaVantage", "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey=" + alphavantage_key)
    ]
    seen_texts = set()
    all_news = []

    for source_name, url in sources:
        try:
            response = requests.get(url, timeout=10)
            data = response.json()

            if source_name == "NewsAPI":
                for article in data.get('articles', []):
                    title = article.get('title', '')
                    description = article.get('description', '')
                    text = f"{title}. {description}".strip()
                    published = article.get('publishedAt', '')[:10]
                    if is_financial(text) and text not in seen_texts:
                        seen_texts.add(text)
                        all_news.append({"text": text[:1000], "date": published})

            elif source_name == "AlphaVantage":
                for item in data.get('feed', []):
                    title = item.get('title', '')
                    summary = item.get('summary', '')
                    text = f"{title}. {summary}".strip()
                    published = item.get('time_published', '')[:8]
                    published = f"{published[:4]}-{published[4:6]}-{published[6:]}"
                    if is_financial(text) and text not in seen_texts:
                        seen_texts.add(text)
                        all_news.append({"text": text[:1000], "date": published})

        except Exception as e:
            print(f"⚠️ MSRT: {source_name} error: {str(e)}")

    return all_news

# [7] Enhanced Signal Generator (No change)
def get_signal(text):
    """Generate trading signal with market context"""
    try:
        sentiment = finbert(text[:512])[0]['label']
        emotion_label = emotion(text[:512])[0]['label']
        
        market = next((mkt for mkt, terms in MARKET_KEYWORDS.items() 
                       if any(term in text.lower() for term in terms)),
                      "General")
        
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
        print(f"⚠️ MSRT: Analysis error: {str(e)}")
        return "⚪ Neutral"

# [8] Main Analysis Function
def analyze_selected_market(target_market, newsapi_key, alphavantage_key):
    """Triggers news fetch and filtering based on user's choice"""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\n[{current_time}] MSRT: 📞 Fetching all financial news via API...")
    news_items = fetch_news(newsapi_key, alphavantage_key)

    if not news_items:
        print("⚠️ MSRT: No news data found - check APIs.")
        return

    print(f"MSRT: Filtering data for {target_market}...")
    
    results = []
    keywords = MARKET_KEYWORDS.get(target_market)
    
    if keywords:
        market_news = [
            item for item in news_items
            if any(keyword in item["text"].lower() for keyword in keywords)
        ]

        if market_news:
            print(f"MSRT: 📊 Analyzing {target_market} Sector ({len(market_news)} articles)...")
            for item in market_news:
                results.append({
                    "Market": target_market,
                    "Headline": item["text"][:100] + "..." if len(item["text"]) > 100 else item["text"],
                    "Signal": get_signal(item["text"]),
                    "Date": item["date"],
                    "Run_Time": current_time
                })

        results_df = pd.DataFrame(results)

        if not results_df.empty:
            log_file = f"msrt_{target_market.lower()}_signals.csv" 
            write_header = not os.path.exists(log_file)
            results_df.to_csv(log_file, mode='a', index=False, header=write_header)

            print(f"✅ MSRT: Analysis for {target_market} saved to {log_file}.")
            print(f"Latest Signals (Preview):\n{results_df.tail()}")
        else:
            print(f"⚠️ MSRT: No relevant signals generated for {target_market} in this run.")
    else:
        print(f"MSRT: ERROR: Invalid market '{target_market}' passed to analyzer.")


# --- INTERACTIVE LOOPING EXECUTION LOGIC ---
if __name__ == "__main__":
    load_dotenv()
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
    ALPHAVANTAGE_KEY = os.getenv('ALPHAVANTAGE_KEY')
    
    if not NEWSAPI_KEY or not ALPHAVANTAGE_KEY:
        print("MSRT: FATAL ERROR: API keys not found in .env file. System cannot run.")
    else:
        # --- PRIMARY LOOP (Runs indefinitely until user exits) ---
        while True:
            # Display menu and prompt user for input on each loop iteration
            print("\n" + "="*50)
            print("MSRT Interactive Market Sentiment Analysis")
            print("Available Markets: " + ", ".join(AVAILABLE_MARKETS))
            print("Type 'exit' to quit the program.") # New instruction
            print("="*50)
            
            user_input = input("Enter market name to analyze (e.g., Stocks, Crypto): ").strip()

            # --- EXIT CONDITION ---
            if user_input.lower() == 'exit':
                print("\nMSRT: Exiting interactive session. Goodbye.")
                break 

            # --- VALIDATE INPUT AND RUN ANALYSIS ---
            target_market = user_input.title()
            
            if target_market in AVAILABLE_MARKETS:
                print(f"MSRT: Valid market selected. Starting analysis for {target_market}...")
                
                if torch.cuda.is_available():
                     torch.cuda.empty_cache() 
                
                # Execute the analysis (NO BREAK HERE)
                analyze_selected_market(target_market, NEWSAPI_KEY, ALPHAVANTAGE_KEY)
                
            else:
                print(f"⚠️ MSRT: Invalid market name: '{user_input}'. Please try again.")
