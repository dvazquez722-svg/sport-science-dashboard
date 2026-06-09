import pandas as pd
import joblib

MODEL_PATH = "models/xgb_distance_predictor.pkl"


def predict_next_session(player_df):

    model = joblib.load(MODEL_PATH)

    row = player_df.iloc[[-1]].copy()

    drop_cols = [
        "player",
        "date",
        "type_session",
        "target_distance_next_day"
    ]

    X = row.drop(
        columns=drop_cols,
        errors="ignore"
    )

    X = pd.get_dummies(
        X,
        columns=["position"],
        drop_first=True
    )

    model_features = model.feature_names_in_

    X = X.reindex(
        columns=model_features,
        fill_value=0
    )

    prediction = model.predict(X)[0]

    return prediction