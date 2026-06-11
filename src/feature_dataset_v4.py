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
# SESSION PHASE (TEAM MICROCYCLE)
# =============================================================================

print("\n[SESSION PHASE FEATURES]")

df["session_phase"] = ""

official_games = sorted(

    df[
        df["type_session"]
        .fillna("")
        .str.lower()
        .str.contains("official")
    ]["date"]

    .drop_duplicates()

)

# Etiquetar partidos

df.loc[
    df["date"].isin(official_games),
    "session_phase"
] = "MD"

# Construcción de microciclos

for i in range(len(official_games) - 1):

    current_game = official_games[i]
    next_game = official_games[i + 1]

    block = df[
        (df["date"] > current_game)
        &
        (df["date"] < next_game)
    ].copy()

    session_dates = (
        block["date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    n_sessions = len(session_dates)

    if n_sessions == 0:
        continue

    # Primera sesión = MD+1

    phase_map = {}

    phase_map[
        session_dates[0]
    ] = "MD+1"

    start_negative = 1

    # Segunda sesión = MD+2
    # solo si existen suficientes sesiones

    if n_sessions >= 6:

        phase_map[
            session_dates[1]
        ] = "MD+2"

        start_negative = 2

    remaining = session_dates[
        start_negative:
    ]

    negative = len(remaining)

    for d in remaining:

        phase_map[d] = f"MD-{negative}"

        negative -= 1

    for d, label in phase_map.items():

        df.loc[
            df["date"] == d,
            "session_phase"
        ] = label

print(
    df["session_phase"]
    .value_counts()
    .sort_index()
)

print(
    [c for c in df.columns
     if "hsr" in c.lower()
     or "speed" in c.lower()
     or "sprint" in c.lower()]
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