import pandas as pd
import sys

# --- INPUT / OUTPUT FILES ---
CRYPTO_FILE = "crypto_sentiment_prediction_dataset.csv"
OUTPUT_FILE = "crypto_training_data.csv"

print(f"Loading crypto dataset: {CRYPTO_FILE}")

try:
    df = pd.read_csv(CRYPTO_FILE, encoding="latin1")
    print(f"Loaded {len(df)} rows.")
except Exception as e:
    print(f"Error: {e}")
    sys.exit()

# --- 1. CHECK REQUIRED COLUMNS ---
required_cols = [
    "cryptocurrency",
    "price_change_24h_percent",
    "market_cap_usd",
    "trading_volume_24h",
    "social_sentiment_score",
    "news_sentiment_score"
]

for col in required_cols:
    if col not in df.columns:
        print(f"FATAL: Required column '{col}' missing.")
        sys.exit()

# --- 2. MAP NUMERIC SENTIMENT TO LABELS ---
def convert_sentiment(score):
    if score >= 0.60:
        return "Positive"
    elif score <= 0.40:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["news_sentiment_score"].apply(convert_sentiment)

# --- 3. GENERATE NATURAL LANGUAGE TEXT ---
def build_text(row):
    return (
        f"{row['cryptocurrency']} recorded a {row['price_change_24h_percent']}% change in the last 24 hours, "
        f"with a trading volume of ${row['trading_volume_24h']} and market cap ${row['market_cap_usd']}. "
        f"News sentiment score is {row['news_sentiment_score']} and social sentiment is {row['social_sentiment_score']}."
    )

df["Text"] = df.apply(build_text, axis=1)

# --- 4. SELECT ONLY NEEDED COLUMNS ---
df_final = df[["Text", "Sentiment"]]

# --- 5. SAVE OUTPUT ---
df_final.to_csv(OUTPUT_FILE, index=False)

print("\n✅ SUCCESS — Crypto-only dataset prepared!")
print(f"Saved: {OUTPUT_FILE}")
print(f"Total usable rows: {len(df_final)}")
