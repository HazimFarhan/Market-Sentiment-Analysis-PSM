import pandas as pd
import numpy as np
import evaluate
import transformers
import torch
import re
import random
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# 1. FIX SEED FOR REPRODUCIBILITY
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================
# 2. CONFIG
# ============================================================
MODEL_NAME = "yiyanghkust/finbert-tone"
CRYPTO_FILE = "crypto_training_dataset.csv"   # <-- new dataset
OUTPUT_DIR = "finbert_crypto_finetuned_v2"

label2id = {"Positive": 0, "Negative": 1, "Neutral": 2}
id2label = {0: "Positive", 1: "Negative", 2: "Neutral"}

# ============================================================
# 3. LOAD DATA
# ============================================================
df = pd.read_csv(CRYPTO_FILE, encoding="latin1")

required_cols = [
    "cryptocurrency", "price_change_24h_percent",
    "news_sentiment_score", "market_cap_usd"
]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column in crypto dataset: {c}")

# ============================================================
# 4. MAP NUMERIC SENTIMENT → LABEL
# ============================================================
def map_sentiment(score):
    if score > 0.6:
        return "Positive"
    elif score < 0.4:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["news_sentiment_score"].apply(map_sentiment)

# ============================================================
# 5. CREATE SYNTHETIC TEXT
# ============================================================
df["Text"] = (
    df["cryptocurrency"] + " shows " +
    df["price_change_24h_percent"].astype(str) +
    "% price movement with sentiment score " +
    df["news_sentiment_score"].astype(str) +
    " and market cap $" +
    df["market_cap_usd"].astype(str)
)

df = df[["Text", "Sentiment"]]

# ============================================================
# 6. CLEAN TEXT
# ============================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s.,\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["Text"] = df["Text"].apply(clean_text)

# Convert labels
df["label"] = df["Sentiment"].map(label2id)
df = df.dropna(subset=["Text", "label"])
df = df.rename(columns={"Text": "text"})

# ============================================================
# 7. MAKE DATASET
# ============================================================
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# ============================================================
# 8. TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)

# ============================================================
# 9. MODEL
# ============================================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# ============================================================
# 10. METRICS
# ============================================================
metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    f1 = metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1": f1}

# ============================================================
# 11. TRAINING ARGUMENTS (IMPROVED)
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=6,                    # more epochs (crypto dataset is small)
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.1,                      # ⬅ warmup (10% of training steps)
    lr_scheduler_type="cosine",            # ⬅ smoother decay
    weight_decay=0.05,
    max_grad_norm=1.0,                     # ⬅ gradient clipping
    fp16=False,                            # GTX 1060 does NOT support fp16
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    logging_steps=50
)

# ============================================================
# 12. TRAINER
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

# ============================================================
# 13. TRAIN
# ============================================================
trainer.train()

# ============================================================
# 14. SAVE MODEL
# ============================================================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n🚀 Training complete — improved FinBERT-Crypto saved to:", OUTPUT_DIR)
