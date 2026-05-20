import pandas as pd
from pathlib import Path

path = Path("artifacts/evaluation/full_adss_predictions.csv")
print("Exists:", path.exists())
print("Path:", path.resolve())

df = pd.read_csv(path)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst row:")
print(df.head(1).T)
