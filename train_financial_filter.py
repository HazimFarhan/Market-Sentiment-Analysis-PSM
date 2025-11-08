import pandas as pd
from glob import glob
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

print("Step 1: Loading datasets...")

files = glob("*.csv")  # adjust if xlsx → glob("*.xlsx")
dfs = []

for f in files:
    df = pd.read_csv(f)
    df['dataset_name'] = f
    df['text'] = df['headlines'].astype(str) + ". " + df['description'].astype(str)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all[['text', 'category']]

print(df_all['category'].value_counts())


# Step 2: Assign Initial Labels
def initial_label(cat):
    if cat == "business":
        return 1
    elif cat in ["education", "entertainment", "sports"]:
        return 0
    elif cat == "technology":
        return None
    return None

df_all["label"] = df_all["category"].apply(initial_label)


# Step 3: Auto-label Technology with FinBERT
print("\nStep 3: Auto-labeling 'technology' category using FinBERT...")
finbert = pipeline("text-classification", model="yiyanghkust/finbert-tone")

df_tech = df_all[df_all["label"].isna()].copy()

def tech_label(text):
    sentiment = finbert(text[:512])[0]['label']
    return 1 if sentiment in ["Positive", "Negative"] else 0

df_tech["label"] = df_tech["text"].apply(tech_label)
df_all.loc[df_all["label"].isna(), "label"] = df_tech["label"]

print(df_all["label"].value_counts())


# Step 4: Train TF-IDF + Logistic Regression Filter
print("\nStep 4: Training Financial Pre-Filter Model...")

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=6000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=300))
])

model.fit(df_all["text"], df_all["label"])

joblib.dump(model, "financial_filter.pkl")

print("\n✅ Training complete! Model saved as: financial_filter.pkl")
