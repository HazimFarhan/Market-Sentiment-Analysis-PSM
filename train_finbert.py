import pandas as pd
import numpy as np
import evaluate
import re  # <-- NEW: Import regular expressions
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# --- 1. CONFIGURATION ---

# This MUST match the original finbert-tone model
label2id = {"Positive": 0, "Negative": 1, "Neutral": 2}
id2label = {0: "Positive", 1: "Negative", 2: "Neutral"}
num_labels = 3

MODEL_NAME = "yiyanghkust/finbert-tone"
DATA_FILE = "combined_training_data.csv" # <-- Uses your combined file
OUTPUT_DIR = "finbert_finetuned"

# --- NEW: TEXT CLEANING FUNCTION ---
def clean_text(text):
    """
    Cleans text data by removing emojis, special characters, and extra whitespace.
    """
    if not isinstance(text, str):
        return "" # Return empty string for non-string data (like NaN)
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove emojis
    # This regex matches most common emoji unicodes
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", 
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)
    
    # 3. Remove non-alphanumeric characters (keeps letters, numbers, and basic punctuation)
    # This keeps spaces, periods, commas, apostrophes, and hyphens.
    text = re.sub(r'[^a-zA-Z0-9\s.,\'\-]', '', text)
    
    # 4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- 2. LOAD & PREPARE DATA ---

print(f"Loading data from {DATA_FILE}...")
# Add encoding='latin1' to match the file preparation script
df = pd.read_csv(DATA_FILE, encoding='latin1') 

# --- NEW: Apply the cleaning function ---
print("Cleaning text data (removing emojis, undefined chars)...")
# We apply cleaning to the 'Text' column (from your combined CSV)
df['Text'] = df['Text'].apply(clean_text)
# --- END NEW ---

# Use the correct column names 'Text' and 'Sentiment'
# Drop rows where Sentiment is missing
df = df.dropna(subset=['Sentiment'])

# --- NEW: Also drop rows where cleaning resulted in empty text ---
df = df.dropna(subset=['Text'])
df = df[df['Text'].str.strip().astype(bool)] # Remove rows with only whitespace
# --- END NEW ---

print("Data loaded and cleaned. Mapping labels...")

# Map the text labels to the integer IDs
try:
    df['label'] = df['Sentiment'].str.title().map(label2id)
except Exception as e:
    print(f"Error mapping labels. Make sure 'Sentiment' column has 'positive', 'negative', 'neutral'. Error: {e}")
    df = df.dropna(subset=['label'])

df['label'] = df['label'].astype(int)

# Rename 'Text' column to 'text' for the model
df = df.rename(columns={"Text": "text"})

print("Final data counts after cleaning:")
print(df['label'].value_counts())

# Convert pandas DataFrame to Hugging Face Dataset
ds = Dataset.from_pandas(df)

# train/val split
ds = ds.train_test_split(test_size=0.1, seed=42)

# --- 3. TOKENIZATION ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    # Use the 'text' column for tokenization
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

print("Tokenizing dataset...")
ds = ds.map(tokenize, batched=True)

# --- 4. LOAD MODEL ---
print("Loading pre-trained model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# --- 5. COMPUTE METRICS ---
metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    f1 = metric.compute(predictions=preds, references=labels, average='macro')['f1']
    return {"accuracy": accuracy, "f1": f1}

# --- 6. SET UP TRAINER ---
print("Configuring training arguments...") 
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch size = 4 * 8 = 32
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,   # If it crashes (e.g., no NVIDIA GPU), set to False
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none" 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    compute_metrics=compute_metrics
)

# --- 7. TRAIN & SAVE ---
print("\nStarting training...")
trainer.train()

print("\nTraining complete. Saving best model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to '{OUTPUT_DIR}'")