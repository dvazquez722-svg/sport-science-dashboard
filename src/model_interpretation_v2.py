import pandas as pd

print("\n" + "="*80)
print("MODEL INTERPRETATION")
print("="*80)

df = pd.read_csv(
    "data/processed/xgb_feature_importance.csv"
)

df = df.sort_values(
    "importance",
    ascending=False
)

print("\nTOP 15 FEATURES\n")
print(df.head(15).to_string(index=False))

print("\n")
print("="*80)
print("FOOTBALL INTERPRETATION")
print("="*80)

for _, row in df.head(15).iterrows():

    feature = row["feature"]
    importance = row["importance"]

    print("\n----------------------------------------")
    print(feature)
    print(f"Importance: {importance:.4f}")

    if "chronic" in feature:
        print(
            "Representa carga acumulada del jugador. "
            "Indica el estado físico construido durante semanas."
        )

    elif "acute" in feature:
        print(
            "Representa carga reciente. "
            "Detecta picos de entrenamiento."
        )

    elif "acwr" in feature:
        print(
            "Relaciona carga reciente con carga histórica."
        )

    elif "ewma" in feature:
        print(
            "Carga ponderada dando más importancia "
            "a los días recientes."
        )

    elif "lag1" in feature:
        print(
            "Información procedente de la sesión anterior."
        )

    elif "lag2" in feature:
        print(
            "Información procedente de dos sesiones atrás."
        )

    elif "7d_avg" in feature:
        print(
            "Promedio de la última semana."
        )

    elif "3d_avg" in feature:
        print(
            "Promedio de los últimos tres días."
        )

    else:
        print(
            "Variable de carga externa o interna."
        )

print("\n")
print("="*80)
print("TOP 10 FEATURES")
print("="*80)

top10 = df.head(10)

for i, row in enumerate(top10.itertuples(), start=1):

    print(
        f"{i}. {row.feature} "
        f"({row.importance:.4f})"
    )

print("\n")
print("="*80)
print("CONCLUSION")
print("="*80)

print("""
El modelo depende principalmente de variables
relacionadas con carga acumulada, carga reciente
y tendencias de entrenamiento.

Las métricas históricas tienen mayor peso que las
métricas instantáneas de una única sesión.

Esto sugiere que la carga futura de un jugador
está más relacionada con su historial reciente
que con el contenido aislado de una sesión.
""")