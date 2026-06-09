import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(r"E:\sports_science_project")

INPUT_FILE = ROOT / "data" / "processed" / "daily_load.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "feature_dataset.csv"

METRICS = [
    "distance_m",
    "abs_hsr_m",
    "player_load_a_u"
]

# =============================================================================
# LOAD
# =============================================================================

print("=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

print("\nLeyendo dataset:")

print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

# =============================================================================
# PREP
# =============================================================================

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["player", "date"]
).reset_index(drop=True)

# =============================================================================
# FEATURE GENERATION
# =============================================================================

print("\n[GENERANDO FEATURES]")

for metric in METRICS:

    print(f"  -> {metric}")

    acute_col = f"acute_{metric}_7d"
    chronic_col = f"chronic_{metric}_28d"
    acwr_col = f"acwr_{metric}"

    # -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Acute Load
# Últimos 7 días
# -------------------------------------------------------------------------

df[acute_col] = (
    df.groupby("player")[metric]
    .transform(
        lambda x:
        x.rolling(
            window=7,
            min_periods=1
        ).mean()
    )
)

# -------------------------------------------------------------------------
# Chronic Load
# Últimos 28 días
# -------------------------------------------------------------------------

df[chronic_col] = (
    df.groupby("player")[metric]
    .transform(
        lambda x:
        x.rolling(
            window=28,
            min_periods=1
        ).mean()
    )
)

# -------------------------------------------------------------------------
# ACWR
# -------------------------------------------------------------------------

df[acwr_col] = np.where(
    df[chronic_col] > 0,
    df[acute_col] / df[chronic_col],
    np.nan
)
# =============================================================================
# ROUND
# =============================================================================

feature_cols = [
    c for c in df.columns
    if (
        c.startswith("acute_")
        or c.startswith("chronic_")
        or c.startswith("acwr_")
    )
]

df[feature_cols] = df[feature_cols].round(4)

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

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("RESUMEN")
print("=" * 80)

print(f"\nFilas: {len(df):,}")
print(f"Jugadores: {df['player'].nunique():,}")
print(f"Fechas: {df['date'].nunique():,}")

print("\nFeatures generadas:")

for col in feature_cols:
    print(" -", col)

print("\nArchivo generado:")
print(OUTPUT_FILE)

print("\nFINALIZADO")
print("=" * 80)