import pandas as pd

FILE = "data/processed/model_dataset_v2.csv"

df = pd.read_csv(FILE)

print("="*80)
print("AUDITORÍA FEATURES V4")
print("="*80)

print("\nNuevas variables")

new_cols = [
    "day_of_week",
    "month",
    "week_of_year",
    "is_official_game",
    "is_friendly_game",
    "is_training"
]

for c in new_cols:
    print(c)

print("\n")
print("="*80)
print("DAY OF WEEK")
print("="*80)

print(df["day_of_week"].value_counts().sort_index())

print("\n")
print("="*80)
print("MONTH")
print("="*80)

print(df["month"].value_counts().sort_index())

print("\n")
print("="*80)
print("SESSION TYPES")
print("="*80)

print("Official games:")
print(df["is_official_game"].sum())

print("\nFriendly games:")
print(df["is_friendly_game"].sum())

print("\nTraining:")
print(df["is_training"].sum())

print("\n")
print("="*80)
print("TARGET BY SESSION TYPE")
print("="*80)

print(
    df.groupby("type_session")[
        "target_distance_next_day"
    ]
    .agg(["count","mean","median"])
    .sort_values("mean", ascending=False)
)

print("\n")
print("="*80)
print("AUDITORÍA FINALIZADA")
print("="*80)