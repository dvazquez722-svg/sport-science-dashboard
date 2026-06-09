from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

FILE_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "Temporada 2022-2023 Las Palmas_CLEAN.csv"
)

df = pd.read_csv(FILE_PATH, low_memory=False)

print("\nPLAYER+DATE")
print(
    df.groupby(
        ["player","date"]
    ).size().describe()
)

print("\nPLAYER+DATE+SESSION")
print(
    df.groupby(
        ["player","date","session"]
    ).size().describe()
)

print("\nPLAYER+DATE+TASK")
print(
    df.groupby(
        ["player","date","task"]
    ).size().describe()
)

print("\nPLAYER+DATE+TASK+REPETITION")
print(
    df.groupby(
        ["player","date","task","repetition"]
    ).size().describe()
)

print(
    df[
        [
            "task",
            "session",
            "type_session",
            "distance_m"
        ]
    ]
    .sort_values("distance_m", ascending=False)
    .head(100)
)

print(
    df["session"]
    .value_counts(dropna=False)
)