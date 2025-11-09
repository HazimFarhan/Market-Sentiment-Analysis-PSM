#
# inspect_crypto.py
#
import pandas as pd

CRYPTO_FILE = "crypto_dataset.csv" # <-- Make sure this file name is correct

try:
    print(f"Inspecting headers for: {CRYPTO_FILE}")
    
    # Read only the first 5 rows to avoid loading the whole file
    df_head = pd.read_csv(CRYPTO_FILE, encoding='latin1', nrows=5)
    
    print("\n--- FILE FOUND ---")
    print("These are the first 5 rows:")
    print(df_head)
    
    print("\n--- COLUMN NAMES ---")
    print("These are the available column names:")
    print(df_head.columns.tolist())
    
except Exception as e:
    print(f"\nError opening file: {e}")