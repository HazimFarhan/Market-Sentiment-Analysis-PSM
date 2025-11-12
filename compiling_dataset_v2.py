# prepare_data_v2.py
# Updated to handle new numeric sentiment dataset for crypto
import pandas as pd
import sys

# --- 1. CONFIGURATION ---
BASE_FILE = "overall_dataset.csv"  # your main dataset
CRYPTO_FILE = "crypto_sentiment_prediction_dataset.csv"  # new crypto dataset with decimal sentiment
OUTPUT_FILE = "combined_training_data.csv"

CRYPTO_SAMPLE_SIZE = 20000  # adjust if your machine can't handle large loads

# --- 2. LOAD BASE DATA ---
print(f"Loading base file: {BASE_FILE}")
try:
    df_base = pd.read_csv(BASE_FILE, encoding='latin1')
    df_base = df_base[['Text', 'Sentiment']]
    print(f"Loaded {len(df_base)} rows from base file.")
except Exception as e:
    print(f"Error loading base file: {e}")
    sys.exit()

# --- 3. LOAD CRYPTO DATA ---
print(f"Loading new crypto dataset: {CRYPTO_FILE}")
try:
    df_crypto = pd.read_csv(CRYPTO_FILE, encoding='latin1')
    print(f"Successfully loaded {len(df_crypto)} crypto rows.")
except Exception as e:
    print(f"Error loading crypto dataset: {e}")
    sys.exit()

# --- 4. TRANSFORM NUMERIC SENTIMENTS ---
# Convert numeric 'news_sentiment_score' to categorical sentiment labels
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

if 'news_sentiment_score' not in df_crypto.columns:
    print("FATAL: Column 'news_sentiment_score' not found in crypto dataset.")
    sys.exit()

df_crypto['Sentiment'] = df_crypto['news_sentiment_score'].apply(map_sentiment)

# --- 5. CREATE TEXT FIELD FOR TRAINING ---
# Build a synthetic text field for the model (headline-style)
df_crypto['Text'] = (
    df_crypto['cryptocurrency'] + " shows " +
    df_crypto['price_change_24h_percent'].astype(str) + "% price change with " +
    "sentiment score " + df_crypto['news_sentiment_score'].astype(str) +
    " and market cap $" + df_crypto['market_cap_usd'].astype(str)
)

# Keep only required columns
df_crypto = df_crypto[['Text', 'Sentiment']]

# --- 6. SAMPLE IF NEEDED ---
if len(df_crypto) > CRYPTO_SAMPLE_SIZE:
    print(f"Sampling {CRYPTO_SAMPLE_SIZE} crypto rows from {len(df_crypto)} total...")
    df_crypto = df_crypto.sample(n=CRYPTO_SAMPLE_SIZE, random_state=42)

# --- 7. COMBINE BOTH DATASETS ---
print("Combining base and crypto datasets...")
df_combined = pd.concat([df_base, df_crypto], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# --- 8. SAVE OUTPUT ---
print(f"Saving final combined dataset to {OUTPUT_FILE}...")
df_combined.to_csv(OUTPUT_FILE, index=False)

print("\n✅ --- SUCCESS ---")
print(f"Combined training dataset created with {len(df_combined)} total rows.")
print(f"  > {len(df_base)} rows from {BASE_FILE}")
print(f"  > {len(df_crypto)} rows from {CRYPTO_FILE}")
