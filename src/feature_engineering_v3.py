import pandas as pd
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================

INPUT_FILE = "data/processed/feature_dataset_v2.csv"
OUTPUT_FILE = "data/processed/feature_dataset_v3.csv"

METRICS = [
    "distance_m",
    "abs_hsr_m",
    "player_load_a_u"
]

# =============================================================================
# LOAD
# =============================================================================

print("\n================================================================================")
print("FEATURE ENGINEERING V3")
print("================================================================================")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = (
    df
    .sort_values(["player", "date"])
    .reset_index(drop=True)
)

# =============================================================================
# EWMA FEATURES
# =============================================================================

print("\n[GENERANDO EWMA FEATURES]")

for metric in METRICS:

    print(f"  -> {metric}")

    ewma7_col = f"ewma7_{metric}"
    ewma28_col = f"ewma28_{metric}"
    ratio_col = f"ewma_ratio_{metric}"

    # -------------------------------------------------------------------------
    # EWMA 7
    # -------------------------------------------------------------------------

    df[ewma7_col] = (
        df.groupby("player")[metric]
        .transform(
            lambda x: x.ewm(
                span=7,
                adjust=False
            ).mean()
        )
    )

    # -------------------------------------------------------------------------
    # EWMA 28
    # -------------------------------------------------------------------------

    df[ewma28_col] = (
        df.groupby("player")[metric]
        .transform(
            lambda x: x.ewm(
                span=28,
                adjust=False
            ).mean()
        )
    )

    # -------------------------------------------------------------------------
    # EWMA RATIO
    # -------------------------------------------------------------------------

    df[ratio_col] = np.where(
        df[ewma28_col] > 0,
        df[ewma7_col] / df[ewma28_col],
        np.nan
    )

# =============================================================================
# SAVE
# =============================================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\n================================================================================")
print("FEATURES V3 GENERADAS")
print("================================================================================")

print(f"\nFilas: {len(df):,}")
print(f"Columnas: {len(df.columns)}")

print("\nArchivo guardado:")
print(OUTPUT_FILE)