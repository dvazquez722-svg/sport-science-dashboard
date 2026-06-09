# =============================================================================
# AUDIT FEATURES V2
# =============================================================================

import pandas as pd

# =============================================================================
# LOAD
# =============================================================================

df = pd.read_csv(
    "data/processed/feature_dataset_v2.csv"
)

df["date"] = pd.to_datetime(df["date"])

print("=" * 80)
print("AUDITORÍA FEATURE DATASET V2")
print("=" * 80)

print(f"\nFilas: {len(df):,}")
print(f"Jugadores: {df['player'].nunique():,}")
print(f"Fechas: {df['date'].nunique():,}")

# =============================================================================
# FEATURES V2
# =============================================================================

features_v2 = [

    "distance_m_lag1",
    "distance_m_lag2",
    "distance_m_3d_avg",
    "distance_m_7d_avg",

    "abs_hsr_m_lag1",
    "abs_hsr_m_lag2",
    "abs_hsr_m_3d_avg",
    "abs_hsr_m_7d_avg",

    "player_load_a_u_lag1",
    "player_load_a_u_lag2",
    "player_load_a_u_3d_avg",
    "player_load_a_u_7d_avg"
]

print("\n")
print("=" * 80)
print("FEATURES V2")
print("=" * 80)

for col in features_v2:
    if col in df.columns:
        print(col)

# =============================================================================
# NULOS
# =============================================================================

print("\n")
print("=" * 80)
print("VALORES NULOS")
print("=" * 80)

for col in features_v2:
    print(f"{col:<35} {df[col].isna().sum()}")

# =============================================================================
# DISTRIBUCIONES
# =============================================================================

metrics = [
    "distance_m_3d_avg",
    "distance_m_7d_avg",
    "abs_hsr_m_3d_avg",
    "abs_hsr_m_7d_avg",
    "player_load_a_u_3d_avg",
    "player_load_a_u_7d_avg"
]

print("\n")
print("=" * 80)
print("DISTRIBUCIONES")
print("=" * 80)

for col in metrics:

    print(f"\n{col.upper()}")
    print("-" * 40)

    print("Min     :", round(df[col].min(), 4))
    print("P10     :", round(df[col].quantile(0.10), 4))
    print("P25     :", round(df[col].quantile(0.25), 4))
    print("Media   :", round(df[col].mean(), 4))
    print("Mediana :", round(df[col].median(), 4))
    print("P75     :", round(df[col].quantile(0.75), 4))
    print("P90     :", round(df[col].quantile(0.90), 4))
    print("Máx     :", round(df[col].max(), 4))

# =============================================================================
# CHECK MANUAL JUGADOR
# =============================================================================

print("\n")
print("=" * 80)
print("CHECK MANUAL JONATHAN VIERA")
print("=" * 80)

sample = (
    df[df["player"] == "Jonathan Viera"]
    .sort_values("date")
    .tail(20)
)

cols = [

    "date",

    "distance_m",
    "distance_m_lag1",
    "distance_m_lag2",
    "distance_m_3d_avg",
    "distance_m_7d_avg",

    "player_load_a_u",
    "player_load_a_u_lag1",
    "player_load_a_u_lag2",
    "player_load_a_u_3d_avg",
    "player_load_a_u_7d_avg"
]

print(
    sample[cols]
    .to_string(index=False)
)

# =============================================================================
# RESUMEN
# =============================================================================

print("\n")
print("=" * 80)
print("RESUMEN")
print("=" * 80)

print("\nNulos lag1 esperados:")
print(df["distance_m_lag1"].isna().sum())

print("\nNulos lag2 esperados:")
print(df["distance_m_lag2"].isna().sum())

print("\nAuditoría finalizada.")
print("=" * 80)