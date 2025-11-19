# msrt_main_v2.py — Enhanced version with Fine-Tuned FinBERT and Emotion Scoring
# -------------------------------------------------------
# Author: Ahmad Hazim Farhan
# Project: Generative AI for Market Sentiment Analysis
# -------------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import datetime
import pandas as pd

# --- 1. INITIAL SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device in use: {device}")

# Load fine-tuned FinBERT (trained by Hazim)
FINBERT_MODEL = "./finbert_crypto_finetuned_v2"  # Adjust if you saved in different folder
finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(device)
finbert = pipeline("text-classification", model=finbert_model, tokenizer=finbert_tokenizer, device=0 if device == "cuda" else -1)

# Load emotion model (lightweight for ensemble refinement)
emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion = pipeline("text-classification", model=emotion_model_name, return_all_scores=False, device=0 if device == "cuda" else -1)

# --- 2. COMBINED SIGNAL FUNCTION ---
def get_signal(text, finbert_pipeline, emotion_pipeline):
    """
    Get market sentiment signal from text using FinBERT + emotion weighting.
    Returns a dict containing sentiment, emotion, confidence, and final score.
    """
    try:
        # Step 1: Run FinBERT
        finbert_result = finbert_pipeline(text, truncation=True, max_length=256)[0]
        sentiment = finbert_result["label"]
        sent_conf = finbert_result["score"]

        # Step 2: Run Emotion Model
        emotion_result = emotion_pipeline(text, truncation=True, max_length=256)[0]
        emotion_label = emotion_result["label"]
        emotion_conf = emotion_result["score"]

        # Step 3: Weighted Scoring Logic
        # Base weights (positive = +1, negative = -1, neutral = 0)
        sent_weight = 1 if sentiment.lower() == "positive" else (-1 if sentiment.lower() == "negative" else 0)

        # Emotion influence
        emo_weight = 0
        if emotion_label.lower() in ["joy", "excitement", "optimism"]:
            emo_weight = +0.3
        elif emotion_label.lower() in ["fear", "anger", "sadness"]:
            emo_weight = -0.3
        else:
            emo_weight = 0

        # Confidence weighting (balances both models)
        final_score = (sent_weight * sent_conf) + (emo_weight * emotion_conf)

        # Step 4: Final decision mapping
        if final_score >= 0.25:
            signal = "🟢 Buy"
        elif final_score <= -0.25:
            signal = "🟠 Sell"
        else:
            signal = "⚪ Neutral"

        return {
            "Signal": signal,
            "Score": round(final_score, 3),
            "FinBERT_Label": sentiment,
            "FinBERT_Confidence": round(sent_conf, 3),
            "Emotion_Label": emotion_label,
            "Emotion_Confidence": round(emotion_conf, 3),
        }

    except Exception as e:
        print(f"Error processing text: {e}")
        return {
            "Signal": "⚪ Neutral",
            "Score": 0.0,
            "FinBERT_Label": "Neutral",
            "FinBERT_Confidence": 0.0,
            "Emotion_Label": "Neutral",
            "Emotion_Confidence": 0.0,
        }


# --- 3. TEST / PIPELINE INTEGRATION EXAMPLE ---
def run_msrt(news_items):
    """
    Takes a list of dictionaries [{'market': ..., 'text': ..., 'date': ...}]
    and returns a DataFrame with signal analysis.
    """
    results = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for item in news_items:
        signal_data = get_signal(item["text"], finbert, emotion)
        results.append({
            "Market": item["market"],
            "Headline": item["text"][:100] + "..." if len(item["text"]) > 100 else item["text"],
            "Signal": signal_data["Signal"],
            "FinBERT_Sentiment": signal_data["FinBERT_Label"],
            "Sent_Conf": signal_data["FinBERT_Confidence"],
            "Emotion": signal_data["Emotion_Label"],
            "Emo_Conf": signal_data["Emotion_Confidence"],
            "Score": signal_data["Score"],
            "Date": item["date"],
            "Run_Time": current_time
        })

    return pd.DataFrame(results)


# --- 4. EXAMPLE USAGE ---
if __name__ == "__main__":
    sample_news = [
        {"market": "Crypto", "text": "Bitcoin surges 12% after BlackRock ETF approval rumors.", "date": "2025-11-13"},
        {"market": "Stocks", "text": "NVIDIA stock plunges as demand for GPUs slows down.", "date": "2025-11-13"},
        {"market": "Economy", "text": "Federal Reserve indicates no rate cuts this quarter.", "date": "2025-11-13"},
        {"market": "Crypto", "text": "Ethereum network upgrade boosts developer confidence.", "date": "2025-11-13"},
        {"market": "Finance", "text": "Major bank warns of increasing recession risk.", "date": "2025-11-13"},
    ]

    df = run_msrt(sample_news)
    print("\n--- Market Sentiment Report ---\n")
    print(df.to_string(index=False))

    df.to_csv("msrt_results.csv", index=False)
    print("\n✅ Results saved to 'msrt_results.csv'")
