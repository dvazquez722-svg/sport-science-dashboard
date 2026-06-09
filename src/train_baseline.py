import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n" + "=" * 80)
print("TRAIN BASELINE MODEL")
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
# SORT
# =============================================================================

df = df.sort_values(
    ["date", "player"]
).reset_index(drop=True)

# =============================================================================
# CHECK NON NUMERIC
# =============================================================================

print("\nCOLUMNAS NO NUMÉRICAS:")

print(
    df.select_dtypes(
        exclude=["number"]
    ).columns.tolist()
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

print("\nFEATURES UTILIZADAS")

for c in feature_cols:
    print(c)

# =============================================================================
# X / y
# =============================================================================

X = df[feature_cols].copy()

print("\nNULOS POR FEATURE")

nulls = X.isna().sum()

print(
    nulls[nulls > 0]
)

# rellenar nulos para baseline

X = X.fillna(0)

print(
    "\nNaNs restantes:",
    X.isna().sum().sum()
)

y = df["target_distance_next_day"]

# =============================================================================
# TEMPORAL SPLIT
# =============================================================================

unique_dates = sorted(
    df["date"].unique()
)

split_idx = int(
    len(unique_dates) * 0.80
)

train_dates = unique_dates[:split_idx]
test_dates = unique_dates[split_idx:]

train_mask = df["date"].isin(train_dates)
test_mask = df["date"].isin(test_dates)

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("\n" + "=" * 80)
print("DATA SPLIT")
print("=" * 80)

print("Train rows:", len(X_train))
print("Test rows :", len(X_test))

print(
    "Train dates:",
    min(train_dates),
    "->",
    max(train_dates)
)

print(
    "Test dates:",
    min(test_dates),
    "->",
    max(test_dates)
)

# =============================================================================
# LINEAR REGRESSION
# =============================================================================

print("\n" + "=" * 80)
print("LINEAR REGRESSION")
print("=" * 80)

lr = LinearRegression()

lr.fit(
    X_train,
    y_train
)

pred_lr = lr.predict(
    X_test
)

lr_mae = mean_absolute_error(
    y_test,
    pred_lr
)

lr_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred_lr
    )
)

lr_r2 = r2_score(
    y_test,
    pred_lr
)

print(f"MAE  : {lr_mae:.2f}")
print(f"RMSE : {lr_rmse:.2f}")
print(f"R²   : {lr_r2:.4f}")

# =============================================================================
# RANDOM FOREST
# =============================================================================

print("\n" + "=" * 80)
print("RANDOM FOREST")
print("=" * 80)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

pred_rf = rf.predict(
    X_test
)

rf_mae = mean_absolute_error(
    y_test,
    pred_rf
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred_rf
    )
)

rf_r2 = r2_score(
    y_test,
    pred_rf
)

print(f"MAE  : {rf_mae:.2f}")
print(f"RMSE : {rf_rmse:.2f}")
print(f"R²   : {rf_r2:.4f}")

# =============================================================================
# COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [lr_mae, rf_mae],
    "RMSE": [lr_rmse, rf_rmse],
    "R2": [lr_r2, rf_r2]
})

print(
    comparison.to_string(index=False)
)

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

print("\n" + "=" * 80)
print("TOP 20 FEATURE IMPORTANCE")
print("=" * 80)

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance
    .head(20)
    .to_string(index=False)
)

print("\nFIN")