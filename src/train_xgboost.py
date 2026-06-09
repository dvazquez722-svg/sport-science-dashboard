import os

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

import joblib

print("\n" + "="*80)
print("TRAIN XGBOOST")
print("="*80)

# =============================================================================
# LOAD
# =============================================================================

df = pd.read_csv(
    "data/processed/model_dataset.csv"
)

df["date"] = pd.to_datetime(df["date"])

# =============================================================================
# DROP NON NUMERIC
# =============================================================================

drop_cols = [
    "player",
    "date",
    "type_session",
    "target_distance_next_day"
]

X = df.drop(columns=drop_cols)

X = pd.get_dummies(
    X,
    columns=["position"],
    drop_first=True
)

X = X.fillna(0)

y = df["target_distance_next_day"]

# =============================================================================
# TIME SPLIT
# =============================================================================

split_date = df["date"].quantile(0.8)

train_mask = df["date"] <= split_date

X_train = X[train_mask]
X_test = X[~train_mask]

y_train = y[train_mask]
y_test = y[~train_mask]

print("\nTRAIN:", len(X_train))
print("TEST :", len(X_test))

# =============================================================================
# MODEL
# =============================================================================

model = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

# =============================================================================
# PREDICT
# =============================================================================

pred = model.predict(X_test)

# =============================================================================
# METRICS
# =============================================================================

mae = mean_absolute_error(
    y_test,
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)

r2 = r2_score(
    y_test,
    pred
)

print("\n" + "="*80)
print("XGBOOST RESULTS")
print("="*80)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
)

print("\n" + "="*80)
print("TOP 20 FEATURES")
print("="*80)

print(
    importance.head(20)
    .to_string(index=False)
)

importance.to_csv(
    "data/processed/xgb_feature_importance.csv",
    index=False
)

print("\nArchivo guardado:")
print("data/processed/xgb_feature_importance.csv")

# =============================================================================
# SAVE MODEL
# =============================================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/xgb_distance_predictor.pkl"
)

print("\nModelo guardado:")
print("models/xgb_distance_predictor.pkl")

print("\nFIN")