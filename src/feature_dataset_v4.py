import pandas as pd
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================

INPUT_FILE = "data/processed/model_dataset.csv"
OUTPUT_FILE = "data/processed/model_dataset_v2.csv"

# =============================================================================
# LOAD
# =============================================================================

print("\n================================================================================")
print("FEATURE ENGINEERING V4")
print("================================================================================")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

# =============================================================================
# DATE FEATURES
# =============================================================================

print("\n[DATE FEATURES]")

df["day_of_week"] = df["date"].dt.dayofweek

df["month"] = df["date"].dt.month

df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

# =============================================================================
# TYPE SESSION FLAGS
# =============================================================================

print("\n[SESSION FEATURES]")

df["is_official_game"] = (
    df["type_session"]
    .fillna("")
    .str.lower()
    .str.contains("official")
    .astype(int)
)

df["is_friendly_game"] = (
    df["type_session"]
    .fillna("")
    .str.lower()
    .str.contains("friendly")
    .astype(int)
)

df["is_training"] = (
    (~df["type_session"]
     .fillna("")
     .str.lower()
     .str.contains("official|friendly"))
).astype(int)

# =============================================================================
# MATCH DAY
# =============================================================================

print("\n[MATCH DAY FEATURES]")

if "week_match_day" in df.columns:

    df["week_match_day"] = (
        pd.to_numeric(
            df["week_match_day"],
            errors="coerce"
        )
    )

# =============================================================================
# SAVE
# =============================================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nArchivo guardado:")
print(OUTPUT_FILE)

print("\nFilas:", len(df))
print("Columnas:", len(df.columns))