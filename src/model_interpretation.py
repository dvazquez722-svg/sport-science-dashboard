import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

# =============================================================================
# LOAD
# =============================================================================

print("\n" + "=" * 80)
print("MODEL INTERPRETATION")
print("=" * 80)

df = pd.read_csv(
    "data/processed/model_dataset.csv"
)

df["date"] = pd.to_datetime(df["date"])

# =============================================================================
# ENCODE POSITION
# =============================================================================

if "position" in df.columns:

    df["position"] = (
        pd.Categorical(
            df["position"]
        ).codes
    )

# =============================================================================
# FEATURES
# =============================================================================

excluded = [
    "player",
    "date",
    "target_distance_next_day",
    "type_session"
]

feature_cols = [
    c for c in df.columns
    if c not in excluded
]

X = (
    df[feature_cols]
    .fillna(0)
)

y = df["target_distance_next_day"]

# =============================================================================
# TRAIN FULL MODEL
# =============================================================================

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 80)
print("TOP 30 FEATURES")
print("=" * 80)

print(
    importance.head(30)
    .to_string(index=False)
)

# =============================================================================
# CUMULATIVE IMPORTANCE
# =============================================================================

importance["cum_importance"] = (
    importance["importance"]
    .cumsum()
)

print("\n" + "=" * 80)
print("TOP 10 FEATURES")
print("=" * 80)

print(
    importance.head(10)
    .to_string(index=False)
)

# =============================================================================
# FEATURES NECESARIAS PARA 80%
# =============================================================================

top80 = importance[
    importance["cum_importance"] <= 0.80
]

print("\n" + "=" * 80)
print("FEATURES PARA EXPLICAR 80% DEL MODELO")
print("=" * 80)

print(
    top80[[
        "feature",
        "importance",
        "cum_importance"
    ]]
    .to_string(index=False)
)

# =============================================================================
# GROUP ANALYSIS
# =============================================================================

groups = {
    "current_load": [
        "distance_m",
        "abs_hsr_m",
        "player_load_a_u"
    ],

    "lags": [
        c for c in feature_cols
        if "_lag" in c
    ],

    "rolling": [
        c for c in feature_cols
        if "_avg" in c
    ],

    "ewma": [
        c for c in feature_cols
        if "ewma" in c
    ],

    "acwr": [
        c for c in feature_cols
        if "acwr" in c
    ]
}

print("\n" + "=" * 80)
print("GROUP IMPORTANCE")
print("=" * 80)

for group_name, vars_group in groups.items():

    score = importance.loc[
        importance["feature"].isin(vars_group),
        "importance"
    ].sum()

    print(
        f"{group_name:<20} {score:.4f}"
    )

# =============================================================================
# SAVE
# =============================================================================

importance.to_csv(
    "data/processed/feature_importance.csv",
    index=False
)

print(
    "\nFeature importance guardada en:"
)

print(
    "data/processed/feature_importance.csv"
)

print("\nFIN")