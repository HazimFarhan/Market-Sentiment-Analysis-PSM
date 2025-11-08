import pandas as pd

df = pd.read_csv("business_data.csv")
print(df.head())
print(df.columns)
df['category'].value_counts().head(15)

