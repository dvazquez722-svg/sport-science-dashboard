# =============================================================================
# FEATURE ENGINEERING V2
# =============================================================================

import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================

INPUT_FILE = Path("data/processed/feature_dataset.csv")
OUTPUT_FILE = Path("data/processed/feature_dataset_v2.csv")

METRICS = [
    "distance_m",
    "abs_hsr_m",
    "player_load_a_u"
]

# =============================================================================
# LOAD
# =============================================================================

print("=" * 80)
print("FEATURE ENGINEERING V2")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["player", "date"]
).reset_index(drop=True)

print(f"\nFilas cargadas: {len(df):,}")

# =============================================================================
# FEATURES
# =============================================================================

for metric in METRICS:

    print(f"\nGenerando -> {metric}")

    # -------------------------------------------------------------------------
    # LAG 1
    # -------------------------------------------------------------------------

    lag1_col = f"{metric}_lag1"

    df[lag1_col] = (
        df.groupby("player")[metric]
        .shift(1)
    )

    # -------------------------------------------------------------------------
    # LAG 2
    # -------------------------------------------------------------------------

    lag2_col = f"{metric}_lag2"

    df[lag2_col] = (
        df.groupby("player")[metric]
        .shift(2)
    )

    # -------------------------------------------------------------------------
    # ROLLING MEAN 3D
    # -------------------------------------------------------------------------

    avg3_col = f"{metric}_3d_avg"

    df[avg3_col] = (
        df.groupby("player")[metric]
        .transform(
            lambda x:
            x.rolling(
                window=3,
                min_periods=1
            ).mean()
        )
    )

    # -------------------------------------------------------------------------
    # ROLLING MEAN 7D
    # -------------------------------------------------------------------------

    avg7_col = f"{metric}_7d_avg"

    df[avg7_col] = (
        df.groupby("player")[metric]
        .transform(
            lambda x:
            x.rolling(
                window=7,
                min_periods=1
            ).mean()
        )
    )

# =============================================================================
# SAVE
# =============================================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 80)
print("FEATURE ENGINEERING V2 FINALIZADO")
print("=" * 80)

print(f"\nArchivo guardado:")
print(OUTPUT_FILE)

print(f"\nFilas: {len(df):,}")
print(f"Columnas: {df.shape[1]:,}")