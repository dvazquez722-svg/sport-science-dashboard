import pandas as pd

print("\n" + "=" * 80)
print("AUDIT MODEL")
print("=" * 80)

importance = pd.read_csv(
    "data/processed/feature_importance.csv"
)

print(
    importance.head(20)
)