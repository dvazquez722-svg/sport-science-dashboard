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
print("AUDITORÍA DE BASE DE DATOS - SPORT SCIENCE PROJECT")
print("=" * 80)

print(f"\nArchivo: {FILE_PATH.name}")

# Intento automático de separador
try:
    df = pd.read_csv(FILE_PATH)
    
    if len(df.columns) == 1:
        print("\n[INFO] Detectado posible separador ';'")
        df = pd.read_csv(FILE_PATH, sep=";")

except Exception as e:
    print(f"\nERROR AL LEER EL ARCHIVO:\n{e}")
    raise

# =====================================================
# ESTRUCTURA GENERAL
# =====================================================

print("\n" + "=" * 80)
print("ESTRUCTURA GENERAL")
print("=" * 80)

print(f"Registros : {len(df):,}")
print(f"Columnas  : {len(df.columns)}")

# =====================================================
# PRIMERAS COLUMNAS
# =====================================================

print("\n" + "=" * 80)
print("PRIMERAS 30 COLUMNAS")
print("=" * 80)

for i, col in enumerate(df.columns[:30], start=1):
    print(f"{i:02d}. {col}")

# =====================================================
# LISTADO COMPLETO DE COLUMNAS
# =====================================================

print("\n" + "=" * 80)
print("LISTADO COMPLETO DE COLUMNAS")
print("=" * 80)

for i, col in enumerate(df.columns, start=1):
    print(f"{i:03d}. {col}")

# =====================================================
# JUGADORES
# =====================================================

print("\n" + "=" * 80)
print("JUGADORES")
print("=" * 80)

if "Player" in df.columns:

    players = sorted(df["Player"].dropna().unique())

    print(f"Jugadores únicos: {len(players)}\n")

    for player in players:
        print(player)

else:
    print("No existe columna 'Player'")

# =====================================================
# POSICIONES
# =====================================================

print("\n" + "=" * 80)
print("POSICIONES")
print("=" * 80)

if "Position" in df.columns:

    positions = (
        df["Position"]
        .fillna("VACÍO")
        .value_counts()
    )

    print(positions)

else:
    print("No existe columna 'Position'")

# =====================================================
# FECHAS
# =====================================================

print("\n" + "=" * 80)
print("FECHAS")
print("=" * 80)

if "Date" in df.columns:

    dates = pd.to_datetime(df["Date"], errors="coerce")

    print(f"Primer registro : {dates.min()}")
    print(f"Último registro : {dates.max()}")
    print(f"Fechas únicas   : {dates.nunique()}")

else:
    print("No existe columna 'Date'")

# =====================================================
# SEMANAS
# =====================================================

print("\n" + "=" * 80)
print("SEMANAS")
print("=" * 80)

for col in ["Week Calendar", "Week Team"]:

    if col in df.columns:

        print(f"\n{col}")

        print(
            df[col]
            .fillna("VACÍO")
            .value_counts()
            .head(20)
        )

# =====================================================
# SESIONES
# =====================================================

print("\n" + "=" * 80)
print("SESIONES")
print("=" * 80)

if "Session" in df.columns:

    print(f"Sesiones únicas: {df['Session'].nunique()}")

if "Type Session" in df.columns:

    print("\nTipos de sesión:\n")

    session_types = (
        df["Type Session"]
        .fillna("VACÍO")
        .value_counts()
    )

    print(session_types)

# =====================================================
# MATCH DAY
# =====================================================

print("\n" + "=" * 80)
print("MATCH DAYS")
print("=" * 80)

for col in ["Match Day", "Week Match Day"]:

    if col in df.columns:

        print(f"\n{col}")

        print(
            df[col]
            .fillna("VACÍO")
            .value_counts()
            .head(30)
        )

# =====================================================
# TASKS
# =====================================================

print("\n" + "=" * 80)
print("TASKS")
print("=" * 80)

if "Task" in df.columns:

    print(f"Tareas únicas: {df['Task'].nunique()}")

# =====================================================
# REPETICIONES
# =====================================================

print("\n" + "=" * 80)
print("REPETITIONS")
print("=" * 80)

if "Repetition" in df.columns:

    print(
        df["Repetition"]
        .fillna("VACÍO")
        .value_counts()
        .head(20)
    )

# =====================================================
# CALIDAD DE DATOS
# =====================================================

print("\n" + "=" * 80)
print("VALORES NULOS")
print("=" * 80)

nulls = pd.DataFrame({
    "Column": df.columns,
    "Nulls": df.isnull().sum(),
    "Null_%": round(df.isnull().mean() * 100, 2)
})

nulls = nulls.sort_values(
    by="Null_%",
    ascending=False
)

print(nulls.to_string(index=False))

# =====================================================
# COLUMNAS CON MÁS DEL 50% DE NULOS
# =====================================================

print("\n" + "=" * 80)
print("COLUMNAS > 50% NULOS")
print("=" * 80)

high_nulls = nulls[nulls["Null_%"] > 50]

if len(high_nulls) == 0:
    print("Ninguna")
else:
    print(high_nulls.to_string(index=False))

# =====================================================
# DUPLICADOS
# =====================================================

print("\n" + "=" * 80)
print("DUPLICADOS")
print("=" * 80)

duplicates = df.duplicated().sum()

print(f"Filas duplicadas: {duplicates}")

# =====================================================
# TIPOS DE DATOS
# =====================================================

print("\n" + "=" * 80)
print("TIPOS DE DATOS")
print("=" * 80)

print(df.dtypes)

# =====================================================
# GRANULARIDAD
# =====================================================

print("\n" + "=" * 80)
print("ANÁLISIS DE GRANULARIDAD")
print("=" * 80)

granularity_cols = [
    col for col in [
        "Player",
        "Date",
        "Session",
        "Task",
        "Repetition"
    ]
    if col in df.columns
]

if granularity_cols:

    unique_rows = (
        df[granularity_cols]
        .drop_duplicates()
        .shape[0]
    )

    print("\nColumnas utilizadas:")
    print(granularity_cols)

    print(f"\nCombinaciones únicas: {unique_rows:,}")

# =====================================================
# VARIABLES FÍSICAS DETECTADAS
# =====================================================

print("\n" + "=" * 80)
print("VARIABLES FÍSICAS CANDIDATAS")
print("=" * 80)

keywords = [
    "distance",
    "explosive",
    "hsr",
    "sprint",
    "speed",
    "velocity",
    "acc",
    "dec",
    "load",
    "heart",
    "power"
]

physical_columns = []

for col in df.columns:

    col_lower = col.lower()

    if any(word in col_lower for word in keywords):
        physical_columns.append(col)

for col in sorted(physical_columns):
    print(col)

print(f"\nTotal variables físicas detectadas: {len(physical_columns)}")

# =====================================================
# RESUMEN FINAL
# =====================================================

print("\n" + "=" * 80)
print("RESUMEN EJECUTIVO")
print("=" * 80)

print(f"Registros totales : {len(df):,}")
print(f"Columnas totales  : {len(df.columns)}")

if "Player" in df.columns:
    print(f"Jugadores únicos  : {df['Player'].nunique()}")

if "Session" in df.columns:
    print(f"Sesiones únicas   : {df['Session'].nunique()}")

if "Date" in df.columns:
    print(
        f"Fechas únicas     : "
        f"{pd.to_datetime(df['Date'], errors='coerce').nunique()}"
    )

print(f"Duplicados        : {duplicates}")

print("\nAUDITORÍA FINALIZADA")
print("=" * 80)

print("\n" + "=" * 80)
print("AUDITORÍA REAL")
print("=" * 80)

print(f"\nJugadores únicos: {df['player'].nunique()}")

print(f"Posiciones únicas: {df['position'].nunique()}")

print(f"Fechas únicas: {df['date'].nunique()}")

print(f"Sesiones únicas: {df['session'].nunique()}")

print(f"Tareas únicas: {df['task'].nunique()}")

print(f"Repeticiones únicas: {df['repetition'].nunique()}")

print("\nPOSICIONES")

print(df['position'].value_counts())

print("\nTIPOS DE SESIÓN")

print(df['type_session'].value_counts())

print("\nTOP 20 SESIONES")

print(df['session'].value_counts().head(20))

print("\n" + "=" * 80)
print("GRANULARIDAD REAL")
print("=" * 80)

granularity = (
    df[
        [
            'player',
            'date',
            'session',
            'task',
            'repetition'
        ]
    ]
    .drop_duplicates()
)

print(
    f"Combinaciones únicas: "
    f"{len(granularity):,}"
)

print(
    f"Filas originales: "
    f"{len(df):,}"
)