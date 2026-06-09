import pandas as pd

PATH = r"E:\sports_science_project\data\processed\daily_load.csv"

df = pd.read_csv(PATH)

print("=" * 80)
print("VALORES REPETIDOS SOSPECHOSOS")
print("=" * 80)

variables = [
    "abs_hsr_m",
    "player_load_a_u",
    "max_speed_km_h",
    "accelerations_count"
]

for col in variables:

    print(f"\n\n{col.upper()}")
    print("-" * 50)

    vc = df[col].value_counts(dropna=False)

    print("\nTOP 20 VALORES MÁS REPETIDOS")
    print(vc.head(20))

    print("\nVALORES REPETIDOS > 20 VECES")
    print(vc[vc > 20].head(50))

    print("\nNº VALORES ÚNICOS")
    print(df[col].nunique())


print("\n")
print("=" * 80)
print("ACCELERATIONS_COUNT")
print("=" * 80)

print(df["accelerations_count"].describe())

print("\nTOP 30 VALORES")
print(
    df["accelerations_count"]
    .sort_values(ascending=False)
    .head(30)
)

print("\nTOP 30 FILAS ACCELERATIONS")
print(
    df[
        [
            "player",
            "date",
            "type_session",
            "accelerations_count",
            "distance_m",
            "player_load_a_u"
        ]
    ]
    .sort_values("accelerations_count", ascending=False)
    .head(30)
)


print("\n")
print("=" * 80)
print("MAX SPEED REPETIDA")
print("=" * 80)

print(
    df["max_speed_km_h"]
    .value_counts()
    .head(30)
)

print("\nFILAS CON MAX SPEED = 33.224489")

print(
    df.loc[
        df["max_speed_km_h"].round(6) == 33.224489,
        [
            "player",
            "date",
            "type_session",
            "max_speed_km_h"
        ]
    ]
    .head(100)
)


print("\n")
print("=" * 80)
print("PLAYER LOAD REPETIDO")
print("=" * 80)

top_pl = df["player_load_a_u"].value_counts().head(20)

print(top_pl)

valor = top_pl.index[0]

print("\nEJEMPLOS DEL VALOR MÁS REPETIDO")

print(
    df.loc[
        df["player_load_a_u"] == valor,
        [
            "player",
            "date",
            "type_session",
            "player_load_a_u",
            "distance_m"
        ]
    ]
    .head(100)
)


print("\n")
print("=" * 80)
print("HSR REPETIDO")
print("=" * 80)

top_hsr = df["abs_hsr_m"].value_counts().head(20)

print(top_hsr)

valor = top_hsr.index[0]

print("\nEJEMPLOS DEL VALOR MÁS REPETIDO")

print(
    df.loc[
        df["abs_hsr_m"] == valor,
        [
            "player",
            "date",
            "type_session",
            "abs_hsr_m",
            "distance_m"
        ]
    ]
    .head(100)
)

print("\nFIN")