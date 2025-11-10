import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

BASE_MODEL = "finbert_finetuned"   # your current trained model folder
OUTPUT_DIR = "finbert_finetuned_v2"  # new refined model folder
REFINE_DATA = "crypto_positive_refinement.csv" 

# Load refinement dataset
df = pd.read_csv(REFINE_DATA)
df = df.rename(columns={"Text": "text", "Sentiment": "label"})

label2id = {"Positive": 0, "Negative": 1, "Neutral": 2}
df["label"] = df["label"].map(label2id)

ds = Dataset.from_pandas(df)

# Load tokenizer + trained model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

ds = ds.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    learning_rate=1e-6,   # very small = refinement, not full retraining
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds
)

print("Starting Crypto Refinement Training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Refined model saved to {OUTPUT_DIR}")
