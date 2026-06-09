import pandas as pd
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================

FILE = "data/processed/feature_dataset_v3.csv"

# =============================================================================
# LOAD
# =============================================================================

df = pd.read_csv(FILE)

print("=" * 80)
print("AUDITORÍA FEATURE DATASET V3")
print("=" * 80)

print(f"\nFilas: {len(df):,}")
print(f"Jugadores: {df['player'].nunique():,}")
print(f"Fechas: {df['date'].nunique():,}")

# =============================================================================
# FEATURES V3
# =============================================================================

ewma_cols = [c for c in df.columns if "ewma" in c]

print("\n")
print("=" * 80)
print("FEATURES V3")
print("=" * 80)

for col in ewma_cols:
    print(col)

# =============================================================================
# NULOS
# =============================================================================

print("\n")
print("=" * 80)
print("VALORES NULOS")
print("=" * 80)

print(df[ewma_cols].isna().sum())

# =============================================================================
# DISTRIBUCIONES
# =============================================================================

ratio_cols = [c for c in ewma_cols if "ratio" in c]

print("\n")
print("=" * 80)
print("DISTRIBUCIONES EWMA RATIO")
print("=" * 80)

for col in ratio_cols:

    serie = df[col].dropna()

    print(f"\n{col.upper()}")
    print("-" * 40)

    print(f"Min     : {serie.min():.4f}")
    print(f"P10     : {serie.quantile(0.10):.4f}")
    print(f"P25     : {serie.quantile(0.25):.4f}")
    print(f"Media   : {serie.mean():.4f}")
    print(f"Mediana : {serie.median():.4f}")
    print(f"P75     : {serie.quantile(0.75):.4f}")
    print(f"P90     : {serie.quantile(0.90):.4f}")
    print(f"P95     : {serie.quantile(0.95):.4f}")
    print(f"Máx     : {serie.max():.4f}")

# =============================================================================
# EWMA RATIO ALTOS
# =============================================================================

print("\n")
print("=" * 80)
print("EWMA RATIO > 1.5")
print("=" * 80)

for col in ratio_cols:
    print(f"{col:<40} {(df[col] > 1.5).sum()}")

print("\n")
print("=" * 80)
print("EWMA RATIO > 2")
print("=" * 80)

for col in ratio_cols:
    print(f"{col:<40} {(df[col] > 2).sum()}")

# =============================================================================
# TOP 20 DISTANCE
# =============================================================================

if "ewma_ratio_distance_m" in df.columns:

    print("\n")
    print("=" * 80)
    print("TOP 20 EWMA DISTANCE")
    print("=" * 80)

    cols = [
        "player",
        "date",
        "distance_m",
        "ewma7_distance_m",
        "ewma28_distance_m",
        "ewma_ratio_distance_m"
    ]

    print(
        df[cols]
        .sort_values("ewma_ratio_distance_m", ascending=False)
        .head(20)
        .to_string(index=False)
    )

# =============================================================================
# TOP 20 HSR
# =============================================================================

if "ewma_ratio_abs_hsr_m" in df.columns:

    print("\n")
    print("=" * 80)
    print("TOP 20 EWMA HSR")
    print("=" * 80)

    cols = [
        "player",
        "date",
        "abs_hsr_m",
        "ewma7_abs_hsr_m",
        "ewma28_abs_hsr_m",
        "ewma_ratio_abs_hsr_m"
    ]

    print(
        df[cols]
        .sort_values("ewma_ratio_abs_hsr_m", ascending=False)
        .head(20)
        .to_string(index=False)
    )

# =============================================================================
# TOP 20 PLAYER LOAD
# =============================================================================

if "ewma_ratio_player_load_a_u" in df.columns:

    print("\n")
    print("=" * 80)
    print("TOP 20 EWMA PLAYER LOAD")
    print("=" * 80)

    cols = [
        "player",
        "date",
        "player_load_a_u",
        "ewma7_player_load_a_u",
        "ewma28_player_load_a_u",
        "ewma_ratio_player_load_a_u"
    ]

    print(
        df[cols]
        .sort_values("ewma_ratio_player_load_a_u", ascending=False)
        .head(20)
        .to_string(index=False)
    )

print("\n")
print("=" * 80)
print("AUDITORÍA FINALIZADA")
print("=" * 80)