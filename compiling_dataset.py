#
# prepare_data.py
#test
import pandas as pd
import sys

# --- 1. CONFIGURE YOUR FILES ---
BASE_FILE = "overall_dataset.csv"
CRYPTO_FILE = "crypto_dataset.csv" # <-- Make sure this file name is correct
OUTPUT_FILE = "combined_training_data.csv"

# --- 2. CONFIGURE YOUR DATA ---
CRYPTO_SAMPLE_SIZE = 20000 

# --- IMPORTANT: Set your crypto file's column names ---
# --- vvv THIS SECTION IS NOW CORRECT vvv ---
CRYPTO_TEXT_COLUMN = "text"      
CRYPTO_SENTIMENT_COLUMN = "Sentiment" 
# --- ^^^ THIS SECTION IS NOW CORRECT ^^^ ---

# --- 3. LOAD BASE DATA ---
print(f"Loading base file: {BASE_FILE}")
try:
    df_base = pd.read_csv(BASE_FILE, encoding='latin1') 
    df_base = df_base[['Text', 'Sentiment']] 
    print(f"Loaded {len(df_base)} rows from base file.")
except Exception as e:
    print(f"Error loading base file: {e}")
    sys.exit()

# --- 4. LOAD AND SAMPLE CRYPTO DATA ---
print(f"Loading crypto file: {CRYPTO_FILE}...")
print("This may take a long time and use a lot of RAM.")

try:
    df_crypto_full = pd.read_csv(CRYPTO_FILE, encoding='latin1')
    print(f"Successfully loaded {len(df_crypto_full)} crypto rows.")

    # Standardize crypto columns
    df_crypto_full = df_crypto_full[[CRYPTO_TEXT_COLUMN, CRYPTO_SENTIMENT_COLUMN]]
    df_crypto_full = df_crypto_full.rename(columns={
        CRYPTO_TEXT_COLUMN: "Text",
        CRYPTO_SENTIMENT_COLUMN: "Sentiment"
    })

    # Take the random sample
    print(f"Taking a random sample of {CRYPTO_SAMPLE_SIZE} rows...")
    df_crypto_sample = df_crypto_full.sample(n=CRYPTO_SAMPLE_SIZE, random_state=42)

except Exception as e:
    print(f"FATAL ERROR: Could not load or sample the large crypto file. Error: {e}")
    print("This often happens if you do not have enough RAM (e.g., 16GB+) to load the 2.91GB file.")
    sys.exit()


# --- 5. COMBINE AND SAVE ---
print("Combining datasets...")
df_combined = pd.concat([df_base, df_crypto_sample], ignore_index=True)

# Shuffle the combined data
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Saving combined dataset to {OUTPUT_FILE}...")
df_combined.to_csv(OUTPUT_FILE, index=False)

print("\n--- SUCCESS! ---")
print(f"New training file '{OUTPUT_FILE}' created with {len(df_combined)} total rows.")
print(f"  > {len(df_base)} rows from {BASE_FILE}")
print(f"  > {len(df_crypto_sample)} rows from {CRYPTO_FILE}")