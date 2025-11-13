# prepare_data_v3.py
# Optimized dataset compiler with smart crypto text generation and sentiment balancing

import pandas as pd
import sys
import random

# --- 1. CONFIGURATION ---
BASE_FILE = "overall_dataset.csv"  # main labeled dataset
CRYPTO_FILE = "crypto_sentiment_prediction_dataset.csv"  # numeric crypto dataset
OUTPUT_FILE = "combined_training_data.csv"

CRYPTO_SAMPLE_SIZE = 20000  # increase for better crypto representation
RANDOM_SEED = 42

# --- 2. LOAD BASE DATA ---
print(f"Loading base dataset: {BASE_FILE}")
try:
    df_base = pd.read_csv(BASE_FILE, encoding='latin1')
    df_base = df_base[['Text', 'Sentiment']]
    print(f"✅ Loaded {len(df_base)} rows from base dataset.")
except Exception as e:
    print(f"❌ Error loading base dataset: {e}")
    sys.exit()

# --- 3. LOAD CRYPTO DATA ---
print(f"\nLoading crypto dataset: {CRYPTO_FILE}")
try:
    df_crypto = pd.read_csv(CRYPTO_FILE, encoding='latin1')
    print(f"✅ Loaded {len(df_crypto)} crypto rows.")
except Exception as e:
    print(f"❌ Error loading crypto dataset: {e}")
    sys.exit()

# --- 4. VERIFY REQUIRED COLUMNS ---
required_cols = [
    'cryptocurrency', 'price_change_24h_percent', 
    'news_sentiment_score', 'market_cap_usd'
]
missing_cols = [c for c in required_cols if c not in df_crypto.columns]
if missing_cols:
    print(f"❌ FATAL: Missing required columns in crypto dataset: {missing_cols}")
    sys.exit()

# --- 5. MAP NUMERIC SENTIMENT TO LABELS ---
def map_sentiment(score):
    try:
        if score > 0.6:
            return "Positive"
        elif score < 0.4:
            return "Negative"
        else:
            return "Neutral"
    except:
        return "Neutral"

df_crypto['Sentiment'] = df_crypto['news_sentiment_score'].apply(map_sentiment)

print("\nSentiment distribution in crypto dataset before sampling:")
print(df_crypto['Sentiment'].value_counts())

# --- 6. BUILD SYNTHETIC TEXT FIELD ---
patterns = [
    "{coin} surges {change}% in 24h, sentiment score {score}, market cap ${cap}",
    "{coin} shows {change}% price movement with sentiment {score}, cap ${cap}",
    "Market update: {coin} sentiment {score}, price change {change}%, cap ${cap}",
    "{coin} market reacts with {change}% move, sentiment {score}, cap ${cap}",
    "{coin} trading activity rises, sentiment {score}, price change {change}%"
]

def build_text(row):
    try:
        template = random.choice(patterns)
        return template.format(
            coin=row['cryptocurrency'],
            change=row['price_change_24h_percent'],
            score=row['news_sentiment_score'],
            cap=row['market_cap_usd']
        )
    except Exception:
        return f"{row.get('cryptocurrency', 'Crypto')} market data unavailable."

df_crypto['Text'] = df_crypto.apply(build_text, axis=1)

# --- 7. FILTER & SAMPLE ---
df_crypto = df_crypto[['Text', 'Sentiment']]
df_crypto = df_crypto.dropna(subset=['Text', 'Sentiment'])
df_crypto = df_crypto[df_crypto['Text'].str.strip().astype(bool)]

if len(df_crypto) > CRYPTO_SAMPLE_SIZE:
    print(f"\nSampling {CRYPTO_SAMPLE_SIZE} crypto rows from {len(df_crypto)} total...")
    df_crypto = df_crypto.sample(n=CRYPTO_SAMPLE_SIZE, random_state=RANDOM_SEED)

# --- 8. MERGE DATASETS ---
print("\nCombining base and crypto datasets...")
df_combined = pd.concat([df_base, df_crypto], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# --- 9. FINAL DISTRIBUTION REPORT ---
print("\nSentiment distribution in combined dataset:")
print(df_combined['Sentiment'].value_counts())

# --- 10. SAVE OUTPUT ---
print(f"\nSaving final dataset to {OUTPUT_FILE}...")
df_combined.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

print("\n✅ --- SUCCESS ---")
print(f"Combined dataset '{OUTPUT_FILE}' created successfully.")
print(f"  > {len(df_base)} rows from {BASE_FILE}")
print(f"  > {len(df_crypto)} rows from {CRYPTO_FILE}")
print(f"  → TOTAL: {len(df_combined)} rows ready for fine-tuning.")
