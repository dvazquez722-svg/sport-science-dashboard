# src/daily_dataset_builder.py

from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(r"E:\sports_science_project")

RAW_FILE = (
    PROJECT_ROOT /
    "data" /
    "raw" /
    "Temporada 2022-2023 Las Palmas_CLEAN.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "daily_load.csv"
)

LOG_FILE = (
    PROJECT_ROOT /
    "reports" /
    "audits" /
    "daily_dataset_log.txt"
)

# =============================================================================
# LOAD
# =============================================================================

print("=" * 80)
print("DAILY DATASET BUILDER")
print("=" * 80)

print("\nLeyendo archivo:")
print(RAW_FILE)

df = pd.read_csv(
    RAW_FILE,
    low_memory=False
)

initial_rows = len(df)

# =============================================================================
# CLEAN
# =============================================================================

print("\n[FILTRADO SESSION]")

df["task"] = df["task"].astype(str).str.strip()

df = df[
    df["task"] == "Session"
].copy()

print(f"Filas Session: {len(df):,}")

# =============================================================================
# REMOVE TEAM
# =============================================================================

print("\n[ELIMINANDO TEAM]")

df["position"] = df["position"].astype(str).str.strip()

before_team = len(df)

df = df[
    df["position"] != "TEAM"
].copy()

print(f"Filas eliminadas TEAM: {before_team - len(df):,}")

# =============================================================================
# DATE
# =============================================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.dropna(subset=["date"])

# =============================================================================
# NUMERIC CONVERSION
# =============================================================================

metrics = [
    "distance_m",
    "abs_hsr_m",
    "sprints_abs_count",
    "accelerations_count",
    "player_load_a_u",
    "max_speed_km_h",
    "energy_expenditure_kcal",
    "hmld_m",
]

for col in metrics:

    if col not in df.columns:
        df[col] = np.nan

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

print("\n[AGREGANDO PLAYER + DATE]")

agg_dict = {
    "distance_m": "sum",
    "abs_hsr_m": "sum",
    "sprints_abs_count": "sum",
    "accelerations_count": "sum",
    "player_load_a_u": "sum",
    "energy_expenditure_kcal": "sum",
    "hmld_m": "sum",
    "max_speed_km_h": "max",
}

daily = (
    df
    .groupby(
        ["player", "date"],
        as_index=False
    )
    .agg(agg_dict)
)

# =============================================================================
# POSITION
# =============================================================================

position_map = (
    df
    .groupby("player")["position"]
    .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan)
)

daily["position"] = (
    daily["player"]
    .map(position_map)
)

# =============================================================================
# SESSION TYPE
# =============================================================================

session_type = (
    df
    .groupby(["player", "date"])["type_session"]
    .agg(
        lambda x:
        x.mode().iloc[0]
        if len(x.mode())
        else np.nan
    )
)

daily["type_session"] = (
    daily
    .set_index(["player", "date"])
    .index
    .map(session_type)
)

# =============================================================================
# DERIVED VARIABLES
# =============================================================================

print("\n[VARIABLES DERIVADAS]")

daily["hsr_pct_distance"] = np.where(
    daily["distance_m"] > 0,
    daily["abs_hsr_m"] / daily["distance_m"] * 100,
    np.nan
)

daily["sprint_density"] = np.where(
    daily["distance_m"] > 0,
    daily["sprints_abs_count"] /
    (daily["distance_m"] / 1000),
    np.nan
)

daily["acc_density"] = np.where(
    daily["distance_m"] > 0,
    daily["accelerations_count"] /
    (daily["distance_m"] / 1000),
    np.nan
)

daily["player_load_per_km"] = np.where(
    daily["distance_m"] > 0,
    daily["player_load_a_u"] /
    (daily["distance_m"] / 1000),
    np.nan
)

# =============================================================================
# VALIDATION
# =============================================================================

print("\n[VALIDACIONES]")

duplicates = daily.duplicated(
    subset=["player", "date"]
).sum()

print(
    f"Duplicados player+date: {duplicates}"
)

negative_values = (
    daily[
        [
            "distance_m",
            "abs_hsr_m",
            "player_load_a_u"
        ]
    ] < 0
).sum().sum()

print(
    f"Valores negativos: {negative_values}"
)

# =============================================================================
# SORT
# =============================================================================

daily = daily.sort_values(
    ["player", "date"]
).reset_index(drop=True)

# =============================================================================
# SAVE
# =============================================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

daily.to_csv(
    OUTPUT_FILE,
    index=False
)

# =============================================================================
# LOG
# =============================================================================

with open(
    LOG_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write("DAILY DATASET LOG\n")
    f.write("=" * 50 + "\n\n")

    f.write(
        f"Raw rows: {initial_rows:,}\n"
    )

    f.write(
        f"Daily rows: {len(daily):,}\n"
    )

    f.write(
        f"Players: {daily['player'].nunique():,}\n"
    )

    f.write(
        f"Dates: {daily['date'].nunique():,}\n"
    )

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("RESUMEN")
print("=" * 80)

print(
    f"Filas finales: {len(daily):,}"
)

print(
    f"Jugadores: {daily['player'].nunique():,}"
)

print(
    f"Fechas: {daily['date'].nunique():,}"
)

print("\nArchivo generado:")
print(OUTPUT_FILE)

print("\nFINALIZADO")