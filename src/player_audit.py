from pathlib import Path
import pandas as pd

# =====================================================
# CONFIGURACIÓN
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FILE_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "Temporada 2022-2023 Las Palmas_CLEAN.csv"
)

# =====================================================
# CARGA DE DATOS
# =====================================================

print("\n" + "=" * 80)
print("PLAYER AUDIT")
print("=" * 80)

df = pd.read_csv(FILE_PATH, low_memory=False)

if len(df.columns) == 1:
    df = pd.read_csv(FILE_PATH, sep=";", low_memory=False)

# =====================================================
# LIMPIEZA BÁSICA
# =====================================================

df["player"] = df["player"].astype(str).str.strip()
df["position"] = df["position"].astype(str).str.strip()

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

# =====================================================
# RESUMEN GENERAL
# =====================================================

print("\nRESUMEN GENERAL")
print("-" * 80)

print(f"Jugadores únicos : {df['player'].nunique()}")
print(f"Posiciones únicas: {df['position'].nunique()}")
print(f"Registros totales: {len(df):,}")

# =====================================================
# FRECUENCIA DE JUGADORES
# =====================================================

print("\nJUGADORES POR NÚMERO DE REGISTROS")
print("-" * 80)

player_counts = (
    df["player"]
    .value_counts()
)

print(player_counts)

# =====================================================
# RESUMEN DETALLADO POR JUGADOR
# =====================================================

summary = (
    df.groupby("player")
    .agg(
        records=("player", "size"),
        first_date=("date", "min"),
        last_date=("date", "max")
    )
)

position_mode = (
    df.groupby("player")["position"]
    .agg(
        lambda x: (
            x.mode().iloc[0]
            if not x.mode().empty
            else "UNKNOWN"
        )
    )
)

summary["position"] = position_mode

summary = (
    summary[
        [
            "position",
            "records",
            "first_date",
            "last_date"
        ]
    ]
    .sort_values(
        by="records",
        ascending=False
    )
)

print("\nRESUMEN DETALLADO")
print("-" * 80)

print(summary)

# =====================================================
# TOP 20 JUGADORES
# =====================================================

print("\nTOP 20 JUGADORES")
print("-" * 80)

print(summary.head(20))

# =====================================================
# POSIBLES CANTERANOS / ERRORES
# =====================================================

print("\nJUGADORES CON MENOS DE 100 REGISTROS")
print("-" * 80)

low_records = summary[
    summary["records"] < 100
]

print(low_records)

# =====================================================
# POSICIONES
# =====================================================

print("\nDISTRIBUCIÓN POR POSICIÓN")
print("-" * 80)

print(
    df["position"]
    .value_counts()
)

# =====================================================
# TEAM
# =====================================================

print("\nREGISTROS TEAM")
print("-" * 80)

team_rows = df[
    df["position"] == "TEAM"
]

print(
    f"Registros TEAM: {len(team_rows):,}"
)

if len(team_rows) > 0:

    print("\nJugadores etiquetados como TEAM:")

    print(
        team_rows["player"]
        .value_counts()
    )

# =====================================================
# JUGADORES ACTIVOS TODA LA TEMPORADA
# =====================================================

print("\nJUGADORES CON MÁS DE 200 REGISTROS")
print("-" * 80)

core_players = summary[
    summary["records"] >= 200
]

print(core_players)

# =====================================================
# EXPORTACIÓN CSV
# =====================================================

OUTPUT_DIR = BASE_DIR / "reports" / "audits"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

summary.to_csv(
    OUTPUT_DIR / "player_audit.csv"
)

print("\nArchivo generado:")

print(
    OUTPUT_DIR / "player_audit.csv"
)

print("\nAUDITORÍA FINALIZADA")
print("=" * 80)