from pathlib import Path
import pandas as pd

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FILE_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "daily_load.csv"
)

# ==========================================================
# CARGA
# ==========================================================

print("=" * 80)
print("AUDITORÍA DAILY LOAD")
print("=" * 80)

df = pd.read_csv(FILE_PATH)

print(f"\nFilas: {len(df):,}")
print(f"Jugadores: {df['player'].nunique()}")
print(f"Fechas: {df['date'].nunique()}")

# ==========================================================
# VARIABLES CLAVE
# ==========================================================

KEY_METRICS = [
    "distance_m",
    "abs_hsr_m",
    "sprints_abs_count",
    "accelerations_count",
    "player_load_a_u",
    "max_speed_km_h"
]

# ==========================================================
# ESTADÍSTICAS
# ==========================================================

print("\n" + "=" * 80)
print("ESTADÍSTICAS GENERALES")
print("=" * 80)

for metric in KEY_METRICS:

    if metric not in df.columns:
        continue

    series = pd.to_numeric(
        df[metric],
        errors="coerce"
    )

    print(f"\n{metric}")
    print("-" * 40)

    print(f"Min     : {series.min():.2f}")
    print(f"P10     : {series.quantile(0.10):.2f}")
    print(f"P25     : {series.quantile(0.25):.2f}")
    print(f"Media   : {series.mean():.2f}")
    print(f"Mediana : {series.median():.2f}")
    print(f"P75     : {series.quantile(0.75):.2f}")
    print(f"P90     : {series.quantile(0.90):.2f}")
    print(f"P95     : {series.quantile(0.95):.2f}")
    print(f"Máx     : {series.max():.2f}")

# ==========================================================
# TOP CARGAS DIARIAS
# ==========================================================

print("\n" + "=" * 80)
print("TOP 20 DISTANCIAS")
print("=" * 80)

cols = [
    "player",
    "date",
    "type_session",
    "distance_m"
]

print(
    df[cols]
    .sort_values(
        "distance_m",
        ascending=False
    )
    .head(20)
)

# ==========================================================
# TOP HSR
# ==========================================================

print("\n" + "=" * 80)
print("TOP 20 HSR")
print("=" * 80)

cols = [
    "player",
    "date",
    "type_session",
    "abs_hsr_m"
]

print(
    df[cols]
    .sort_values(
        "abs_hsr_m",
        ascending=False
    )
    .head(20)
)

# ==========================================================
# TOP PLAYER LOAD
# ==========================================================

print("\n" + "=" * 80)
print("TOP 20 PLAYER LOAD")
print("=" * 80)

cols = [
    "player",
    "date",
    "type_session",
    "player_load_a_u"
]

print(
    df[cols]
    .sort_values(
        "player_load_a_u",
        ascending=False
    )
    .head(20)
)

# ==========================================================
# TOP SPEED
# ==========================================================

print("\n" + "=" * 80)
print("TOP 20 MAX SPEED")
print("=" * 80)

cols = [
    "player",
    "date",
    "type_session",
    "max_speed_km_h"
]

print(
    df[cols]
    .sort_values(
        "max_speed_km_h",
        ascending=False
    )
    .head(20)
)

# ==========================================================
# DETECCIÓN DE POSIBLES ERRORES
# ==========================================================

print("\n" + "=" * 80)
print("POTENCIALES ANOMALÍAS")
print("=" * 80)

distance_anomaly = df[df["distance_m"] > 20000]

print(
    f"\nDistancias > 20.000 m: "
    f"{len(distance_anomaly)}"
)

hsr_anomaly = df[df["abs_hsr_m"] > 3000]

print(
    f"HSR > 3000 m: "
    f"{len(hsr_anomaly)}"
)

speed_anomaly = df[df["max_speed_km_h"] > 40]

print(
    f"Velocidad > 40 km/h: "
    f"{len(speed_anomaly)}"
)

load_anomaly = df[df["player_load_a_u"] > 2000]

print(
    f"Player Load > 2000: "
    f"{len(load_anomaly)}"
)

print("\nAUDITORÍA FINALIZADA")
print("=" * 80)