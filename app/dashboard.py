import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Football Load Monitoring",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "data/processed/model_dataset.csv"
)

df["date"] = pd.to_datetime(df["date"])

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Filters")

player = st.sidebar.selectbox(
    "Player",
    sorted(df["player"].unique())
)

# =====================================================
# PLAYER DATA
# =====================================================

player_df = (
    df[df["player"] == player]
    .sort_values("date")
)

# =====================================================
# HEADER
# =====================================================

st.title("Football Load Monitoring Dashboard")

st.subheader(player)

# =====================================================
# KPI
# =====================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Sessions",
    len(player_df)
)

col2.metric(
    "Average Distance",
    round(player_df["distance_m"].mean())
)

col3.metric(
    "Average Player Load",
    round(player_df["player_load_a_u"].mean(),1)
)

# =====================================================
# DISTANCE
# =====================================================

st.markdown("## Distance")

st.line_chart(
    player_df.set_index("date")["distance_m"]
)

# =====================================================
# HSR
# =====================================================

st.markdown("## High Speed Running")

st.line_chart(
    player_df.set_index("date")["abs_hsr_m"]
)

# =====================================================
# PLAYER LOAD
# =====================================================

st.markdown("## Player Load")

st.line_chart(
    player_df.set_index("date")["player_load_a_u"]
)

# =====================================================
# EWMA
# =====================================================

if "ewma_ratio_player_load_a_u" in player_df.columns:

    st.markdown("## EWMA Ratio")

    st.line_chart(
        player_df.set_index("date")[
            "ewma_ratio_player_load_a_u"
        ]
    )

# =====================================================
# TABLE
# =====================================================

st.markdown("## Last Sessions")

cols = [
    "date",
    "type_session",
    "distance_m",
    "abs_hsr_m",
    "player_load_a_u"
]

st.dataframe(
    player_df[cols]
    .sort_values("date", ascending=False)
    .head(20)
)