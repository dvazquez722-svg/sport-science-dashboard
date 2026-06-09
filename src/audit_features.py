import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(r"E:\sports_science_project")

FILE = ROOT / "data" / "processed" / "feature_dataset.csv"

# =============================================================================
# LOAD
# =============================================================================

print("=" * 80)
print("AUDITORÍA FEATURE DATASET")
print("=" * 80)

df = pd.read_csv(FILE)

df["date"] = pd.to_datetime(df["date"])

print("\nFilas:", f"{len(df):,}")
print("Jugadores:", df["player"].nunique())
print("Fechas:", df["date"].nunique())

# =============================================================================
# FEATURES
# =============================================================================

feature_cols = [
    c for c in df.columns
    if (
        c.startswith("acute_")
        or c.startswith("chronic_")
        or c.startswith("acwr_")
    )
]

print("\n")
print("=" * 80)
print("FEATURES GENERADAS")
print("=" * 80)

for col in feature_cols:
    print(col)

# =============================================================================
# NANS
# =============================================================================

print("\n")
print("=" * 80)
print("VALORES NULOS")
print("=" * 80)

for col in feature_cols:

    n_null = df[col].isna().sum()

    print(
        f"{col:<40} {n_null}"
    )

# =============================================================================
# ACWR DISTRIBUTIONS
# =============================================================================

acwr_cols = [
    c for c in df.columns
    if c.startswith("acwr_")
]

print("\n")
print("=" * 80)
print("DISTRIBUCIÓN ACWR")
print("=" * 80)

for col in acwr_cols:

    x = df[col].dropna()

    print(f"\n{col.upper()}")
    print("-" * 40)

    print("Min     :", round(x.min(), 4))
    print("P10     :", round(x.quantile(0.10), 4))
    print("P25     :", round(x.quantile(0.25), 4))
    print("Media   :", round(x.mean(), 4))
    print("Mediana :", round(x.median(), 4))
    print("P75     :", round(x.quantile(0.75), 4))
    print("P90     :", round(x.quantile(0.90), 4))
    print("P95     :", round(x.quantile(0.95), 4))
    print("Max     :", round(x.max(), 4))

# =============================================================================
# ACWR EXTREMOS
# =============================================================================

print("\n")
print("=" * 80)
print("ACWR > 3")
print("=" * 80)

for col in acwr_cols:

    n = (df[col] > 3).sum()

    print(f"{col:<40} {n}")

print("\n")
print("=" * 80)
print("ACWR > 5")
print("=" * 80)

for col in acwr_cols:

    n = (df[col] > 5).sum()

    print(f"{col:<40} {n}")

# =============================================================================
# TOP ACWR
# =============================================================================

for col in acwr_cols:

    print("\n")
    print("=" * 80)
    print(f"TOP 20 {col.upper()}")
    print("=" * 80)

    cols_show = [
        "player",
        "date",
        "distance_m",
        "abs_hsr_m",
        "player_load_a_u",
        col
    ]

    print(
        df[cols_show]
        .sort_values(col, ascending=False)
        .head(20)
        .to_string(index=False)
    )

# =============================================================================
# PLAYER CHECK
# =============================================================================

print("\n")
print("=" * 80)
print("CHECK MANUAL JONATHAN VIERA")
print("=" * 80)

player_df = df[
    df["player"] == "Jonathan Viera"
].copy()

cols = [
    "date",
    "distance_m",
    "acute_distance_m_7d",
    "chronic_distance_m_28d",
    "acwr_distance_m"
]

cols = [c for c in cols if c in player_df.columns]

print(
    player_df[cols]
    .tail(30)
    .to_string(index=False)
)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n")
print("=" * 80)
print("RESUMEN")
print("=" * 80)

for col in acwr_cols:

    x = df[col].dropna()

    print(
        f"{col:<40}"
        f" media={x.mean():.3f}"
        f"  max={x.max():.3f}"
    )

print("\nAUDITORÍA FINALIZADA")
print("=" * 80)