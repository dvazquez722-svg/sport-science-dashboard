import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

FILE = "data/processed/model_dataset.csv"

# =============================================================================
# LOAD
# =============================================================================

df = pd.read_csv(FILE)

print("=" * 80)
print("AUDITORÍA TARGET")
print("=" * 80)

print(f"\nFilas: {len(df):,}")

# =============================================================================
# TARGET
# =============================================================================

target = "target_distance_next_day"

print("\n")
print("=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)

print(df[target].describe())

print("\nPercentiles")

for p in [0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]:
    print(
        f"P{int(p*100):02d}: "
        f"{df[target].quantile(p):.2f}"
    )

# =============================================================================
# TOP TARGETS
# =============================================================================

print("\n")
print("=" * 80)
print("TOP 20 TARGETS")
print("=" * 80)

cols = [
    "player",
    "date",
    "distance_m",
    "target_distance_next_day"
]

print(
    df[cols]
    .sort_values(
        "target_distance_next_day",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

# =============================================================================
# CORRELATION
# =============================================================================

print("\n")
print("=" * 80)
print("CORRELACIÓN CON DISTANCIA ACTUAL")
print("=" * 80)

corr = df[
    ["distance_m", "target_distance_next_day"]
].corr().iloc[0,1]

print(round(corr,4))

print("\n")
print("=" * 80)
print("AUDITORÍA FINALIZADA")
print("=" * 80)