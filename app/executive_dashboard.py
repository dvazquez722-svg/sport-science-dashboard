import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="⚽",
    layout="wide"
)

df = pd.read_csv(
    "data/processed/model_dataset.csv"
)

df["date"] = pd.to_datetime(
    df["date"]
)

latest_rows = (
    df
    .sort_values("date")
    .groupby("player")
    .tail(1)
    .copy()
)

def calculate_risk_score(row):

    score = 0

    # EWMA
    score += min(
        row["ewma_ratio_player_load_a_u"] * 25,
        35
    )

    # ACWR
    score += min(
        row["acwr_player_load_a_u"] * 15,
        20
    )

    # HSR
    score += min(
        row["abs_hsr_m"] / 25,
        15
    )

    # Accelerations
    score += min(
        row["accelerations_count"] / 8,
        15
    )

    # Player Load
    score += min(
        row["player_load_a_u"] / 4,
        15
    )

    return round(score, 1)

latest_rows["Risk Score"] = (
    latest_rows
    .apply(
        calculate_risk_score,
        axis=1
    )
)

st.title(
    "⚽ Executive Dashboard"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Players",
    len(latest_rows)
)

c2.metric(
    "High Risk",
    len(
        latest_rows[
            latest_rows["Risk Score"] >= 80
        ]
    )
)

c3.metric(
    "Medium Risk",
    len(
        latest_rows[
            (
                latest_rows["Risk Score"] >= 60
            )
            &
            (
                latest_rows["Risk Score"] < 80
            )
        ]
    )
)

c4.metric(
    "Low Risk",
    len(
        latest_rows[
            latest_rows["Risk Score"] < 60
        ]
    )
)

st.subheader(
    "Top Risk Players"
)

top_risk = (
    latest_rows[
        [
            "player",
            "Risk Score",
            "ewma_ratio_player_load_a_u",
            "acwr_player_load_a_u"
        ]
    ]
    .sort_values(
        "Risk Score",
        ascending=False
    )
)

st.dataframe(
    top_risk.head(10),
    use_container_width=True
)

st.subheader(
    "Highest Workload Players"
)

workload = (
    latest_rows[
        [
            "player",
            "distance_m",
            "abs_hsr_m",
            "player_load_a_u"
        ]
    ]
    .sort_values(
        "player_load_a_u",
        ascending=False
    )
)

st.dataframe(
    workload.head(10),
    use_container_width=True
)

squad_health = (
    100
    -
    latest_rows["Risk Score"].mean()
)

st.metric(
    "Squad Health Score",
    f"{squad_health:.1f}/100"
)