from transformers import pipeline

model = pipeline("text-classification", model="finbert_finetuned", tokenizer="finbert_finetuned")

tests = [
    "Bitcoin surges 12% after BlackRock ETF approval rumors.",
    "NVIDIA stock plunges as demand for GPUs slows down.",
    "Federal Reserve indicates no rate cuts this quarter.",
    "Ethereum network upgrade boosts developer confidence.",
    "Major bank warns of increasing recession risk."
]

for t in tests:
    print(t, "→", model(t)[0]['label'])
