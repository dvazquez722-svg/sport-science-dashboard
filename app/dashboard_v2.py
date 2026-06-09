import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.predictor import predict_next_session

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Football Load Monitoring",
    page_icon="⚽",
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
# SQUAD OVERVIEW DATASET
# =====================================================

latest_rows = (
    df
    .sort_values("date")
    .groupby("player")
    .tail(1)
    .copy()
)

# =====================================================
# RISK CLASSIFICATION
# =====================================================

def classify_risk(ewma):

    if ewma < 0.8:
        return "🔵 LOW"

    elif ewma < 1.3:
        return "🟢 NORMAL"

    elif ewma < 1.5:
        return "🟡 MEDIUM"

    else:
        return "🔴 HIGH"


latest_rows["risk"] = (
    latest_rows["ewma_ratio_player_load_a_u"]
    .apply(classify_risk)
)

# =====================================================
# RISK FUNCTIONS
# =====================================================

def classify_risk(ewma):

    if ewma < 0.8:
        return "🔵 LOW"

    elif ewma < 1.3:
        return "🟢 NORMAL"

    elif ewma < 1.5:
        return "🟡 MEDIUM"

    else:
        return "🔴 HIGH"


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

# =====================================================
# SQUAD OVERVIEW DATASET
# =====================================================

latest_rows = (
    df
    .sort_values("date")
    .groupby("player")
    .tail(1)
    .copy()
)

latest_rows["risk"] = (
    latest_rows["ewma_ratio_player_load_a_u"]
    .apply(classify_risk)
)

latest_rows["risk_score"] = (
    latest_rows
    .apply(
        calculate_risk_score,
        axis=1
    )
)

# =====================================================
# SQUAD OVERVIEW
# =====================================================

st.subheader("Squad Overview")

high_risk = len(
    latest_rows[
        latest_rows["ewma_ratio_player_load_a_u"] >= 1.5
    ]
)

medium_risk = len(
    latest_rows[
        (
            latest_rows["ewma_ratio_player_load_a_u"] >= 1.3
        )
        &
        (
            latest_rows["ewma_ratio_player_load_a_u"] < 1.5
        )
    ]
)

normal_risk = len(
    latest_rows[
        latest_rows["ewma_ratio_player_load_a_u"] < 1.3
    ]
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "🔴 High Risk",
    high_risk
)

c2.metric(
    "🟡 Medium Risk",
    medium_risk
)

c3.metric(
    "🟢 Normal / Low",
    normal_risk
)

overview_cols = [
    "player",
    "risk",
    "risk_score",
    "distance_m",
    "abs_hsr_m",
    "player_load_a_u",
    "ewma_ratio_player_load_a_u"
]

overview = latest_rows[
    overview_cols
].copy()

overview.columns = [
    "Player",
    "Risk",
    "Risk Score",
    "Distance",
    "HSR",
    "Player Load",
    "EWMA"
]

st.dataframe(
    overview.sort_values(
        "Risk Score",
        ascending=False
    ),
    use_container_width=True
)

st.divider()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚽ Football Load Monitoring")

player = st.sidebar.selectbox(
    "Player",
    sorted(df["player"].unique())
)

player_df_full = (
    df[df["player"] == player]
    .sort_values("date")
    .copy()
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=player_df_full["date"].min()
)

end_date = st.sidebar.date_input(
    "End Date",
    value=player_df_full["date"].max()
)

player_df = player_df_full[
    (
        player_df_full["date"]
        >= pd.to_datetime(start_date)
    )
    &
    (
        player_df_full["date"]
        <= pd.to_datetime(end_date)
    )
].copy()


last_session = player_df.iloc[-1]

# =====================================================
# AI PREDICTION
# =====================================================

pred_distance = predict_next_session(
    player_df
)

latest_ewma = player_df[
    "ewma_ratio_player_load_a_u"
].iloc[-1]

if latest_ewma < 0.8:

    risk = "LOW"
    color = "🔵"

elif latest_ewma < 1.3:

    risk = "NORMAL"
    color = "🟢"

elif latest_ewma < 1.5:

    risk = "MEDIUM"
    color = "🟡"

else:

    risk = "HIGH"
    color = "🔴"

# =====================================================
# HEADER
# =====================================================

st.title("⚽ Football Load Monitoring Dashboard")

# =====================================================
# EXECUTIVE SUMMARY
# =====================================================

st.markdown("---")

st.subheader("Executive Summary")

current_risk_score = calculate_risk_score(
    last_session
)

e1, e2, e3, e4 = st.columns(4)

e1.metric(
    "Predicted Distance",
    f"{pred_distance:,.0f} m"
)

e2.metric(
    "EWMA Ratio",
    f"{latest_ewma:.2f}"
)

e3.metric(
    "Risk Status",
    f"{color} {risk}"
)

e4.metric(
    "Injury Risk Score",
    f"{current_risk_score:.0f}/100"
)

if current_risk_score < 40:

    st.success(
        "Low injury risk"
    )

elif current_risk_score < 60:

    st.info(
        "Normal injury risk"
    )

elif current_risk_score < 80:

    st.warning(
        "Elevated injury risk"
    )

else:

    st.error(
        "High injury risk"
    )

st.markdown("---")

# =====================================================
# TOP RISK PLAYERS
# =====================================================

st.subheader("Top Risk Players")

top_risk = latest_rows.copy()

top_risk["Risk Score"] = (
    top_risk
    .apply(
        calculate_risk_score,
        axis=1
    )
)

top_risk = (
    top_risk[
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
    .head(10)
)

top_risk.columns = [
    "Player",
    "Risk Score",
    "EWMA",
    "ACWR"
]

st.dataframe(
    top_risk,
    use_container_width=True
)

st.markdown("---")

st.markdown(
    f"### {player}"
)

# =====================================================
# SQUAD BENCHMARKING
# =====================================================

st.subheader("Squad Benchmarking")


squad_avg_distance = df["distance_m"].mean()
squad_avg_hsr = df["abs_hsr_m"].mean()
squad_avg_acc = df["accelerations_count"].mean()
squad_avg_sprints = df["sprints_abs_count"].mean()
squad_avg_load = df["player_load_a_u"].mean()

player_distance = player_df["distance_m"].mean()
player_hsr = player_df["abs_hsr_m"].mean()
player_acc = player_df["accelerations_count"].mean()
player_sprints = player_df["sprints_abs_count"].mean()
player_load = player_df["player_load_a_u"].mean()

benchmark = pd.DataFrame({

    "Metric": [
        "Distance",
        "HSR",
        "Sprints",
        "Accelerations",
        "Player Load"
    ],

    "Player": [
        round(player_distance, 1),
        round(player_hsr, 1),
        round(player_sprints, 1),
        round(player_acc, 1),
        round(player_load, 1)
    ],

    "Squad Average": [
        round(squad_avg_distance, 1),
        round(squad_avg_hsr, 1),
        round(squad_avg_sprints, 1),
        round(squad_avg_acc, 1),
        round(squad_avg_load, 1)
    ]
})

benchmark["Difference %"] = round(
    (
        benchmark["Player"]
        /
        benchmark["Squad Average"]
        - 1
    ) * 100,
    1
)

def classify_benchmark(x):

    if x >= 20:
        return f"🔴 +{x}%"

    elif x >= 10:
        return f"🟡 +{x}%"

    elif x <= -20:
        return f"🔵 {x}%"

    elif x <= -10:
        return f"🟢 {x}%"

    else:
        return f"⚪ {x}%"

benchmark["Difference %"] = (
    benchmark["Difference %"]
    .apply(classify_benchmark)
)

st.dataframe(
    benchmark,
    use_container_width=True
)

st.caption(
    """
    🔴 Much higher than squad average (>20%)

    🟡 Higher than squad average (+10% to +20%)

    ⚪ Similar to squad average

    🟢 Lower than squad average (-10% to -20%)

    🔵 Much lower than squad average (<-20%)
    """
)

st.divider()


st.divider()

# =====================================================
# POSITIONAL BENCHMARKING
# =====================================================

st.subheader("Positional Benchmarking")

player_position = (
    player_df_full["position"]
    .iloc[-1]
)

position_df = df[
    df["position"] == player_position
]

pos_avg_distance = position_df["distance_m"].mean()
pos_avg_hsr = position_df["abs_hsr_m"].mean()
pos_avg_sprints = position_df["sprints_abs_count"].mean()
pos_avg_acc = position_df["accelerations_count"].mean()
pos_avg_load = position_df["player_load_a_u"].mean()

position_benchmark = pd.DataFrame({

    "Metric": [
        "Distance",
        "HSR",
        "Sprints",
        "Accelerations",
        "Player Load"
    ],

    "Player": [
        round(player_distance,1),
        round(player_hsr,1),
        round(player_sprints,1),
        round(player_acc,1),
        round(player_load,1)
    ],

    "Position Average": [
        round(pos_avg_distance,1),
        round(pos_avg_hsr,1),
        round(pos_avg_sprints,1),
        round(pos_avg_acc,1),
        round(pos_avg_load,1)
    ]
})

position_benchmark["Difference %"] = round(
    (
        position_benchmark["Player"]
        /
        position_benchmark["Position Average"]
        - 1
    ) * 100,
    1
)

position_benchmark["Difference %"] = (
    position_benchmark["Difference %"]
    .apply(classify_benchmark)
)

st.write(
    f"Position: {player_position}"
)

st.dataframe(
    position_benchmark,
    use_container_width=True
)

st.divider()

# =====================================================
# KPI BLOCK
# =====================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Sessions",
    len(player_df)
)

col2.metric(
    "Avg Distance",
    f"{player_df['distance_m'].mean():.0f} m"
)

col3.metric(
    "Avg HSR",
    f"{player_df['abs_hsr_m'].mean():.0f} m"
)

col4.metric(
    "Avg Player Load",
    f"{player_df['player_load_a_u'].mean():.1f}"
)

st.divider()

st.subheader("AI Prediction")

p1, p2 = st.columns(2)

p1.metric(
    "Predicted Next Distance",
    f"{pred_distance:,.0f} m"
)

p2.metric(
    "Player Average Distance",
    f"{player_df['distance_m'].mean():,.0f} m"
)
# =====================================================
# LAST SESSION
# =====================================================

st.subheader("Latest Session")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Date",
    last_session["date"].strftime("%d/%m/%Y")
)

c2.metric(
    "Session",
    last_session["type_session"]
)

c3.metric(
    "Distance",
    f"{last_session['distance_m']:.0f} m"
)

c4.metric(
    "HSR",
    f"{last_session['abs_hsr_m']:.0f} m"
)

c5.metric(
    "Player Load",
    f"{last_session['player_load_a_u']:.1f}"
)

st.divider()

# =====================================================
# DISTANCE
# =====================================================

st.subheader("Distance Evolution")

fig = px.line(
    player_df,
    x="date",
    y="distance_m"
)

fig.update_layout(
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# HSR
# =====================================================

st.subheader("High Speed Running")

fig = px.line(
    player_df,
    x="date",
    y="abs_hsr_m"
)

fig.update_layout(
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# PLAYER LOAD
# =====================================================

st.subheader("Player Load")

fig = px.line(
    player_df,
    x="date",
    y="player_load_a_u"
)

fig.update_layout(
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# ACUTE VS CHRONIC
# =====================================================

if (
    "acute_player_load_a_u_7d" in player_df.columns
    and
    "chronic_player_load_a_u_28d" in player_df.columns
):

    st.subheader(
        "Acute vs Chronic Load"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=player_df["date"],
            y=player_df["acute_player_load_a_u_7d"],
            name="Acute"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=player_df["date"],
            y=player_df["chronic_player_load_a_u_28d"],
            name="Chronic"
        )
    )

    fig.update_layout(
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# EWMA
# =====================================================

if "ewma_ratio_player_load_a_u" in player_df.columns:

    st.subheader("EWMA Ratio")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=player_df["date"],
            y=player_df["ewma_ratio_player_load_a_u"],
            name="EWMA Ratio"
        )
    )

    fig.add_hline(
        y=0.8,
        line_dash="dash"
    )

    fig.add_hline(
        y=1.3,
        line_dash="dash"
    )

    fig.add_hline(
        y=1.5,
        line_dash="dash"
    )

    fig.update_layout(
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# LOAD TREND
# =====================================================

trend = (
    player_df["ewma_ratio_player_load_a_u"]
    .tail(7)
    .mean()
)

st.subheader("Load Trend")

st.metric(
    "7-Day EWMA Trend",
    f"{trend:.2f}"
)

# =====================================================
# WEEKLY MICROCYCLE ANALYSIS
# =====================================================

st.subheader("Weekly Microcycle Analysis")

player_df["week_of_year"] = (
    player_df["date"]
    .dt.isocalendar()
    .week
)

weekly = (
    player_df
    .groupby("week_of_year")
    .agg({
        "distance_m": "sum",
        "abs_hsr_m": "sum",
        "sprints_abs_count": "sum",
        "accelerations_count": "sum",
        "player_load_a_u": "sum"
    })
    .reset_index()
)

weekly.columns = [
    "Week",
    "Distance",
    "HSR",
    "Sprints",
    "Accelerations",
    "Player Load"
]

st.dataframe(
    weekly.sort_values(
        "Week",
        ascending=False
    ),
    use_container_width=True
)

st.divider()

# =====================================================
# WEEKLY HSR
# =====================================================

st.subheader("Weekly High Speed Running")

fig = px.bar(
    weekly,
    x="Week",
    y="HSR"
)

fig.update_layout(
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# WEEKLY ACCELERATIONS
# =====================================================

st.subheader("Weekly Accelerations")

fig = px.bar(
    weekly,
    x="Week",
    y="Accelerations"
)

fig.update_layout(
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# LAST 20 SESSIONS
# =====================================================

st.subheader("Last Sessions")

cols = [
    "date",
    "type_session",
    "distance_m",
    "abs_hsr_m",
    "player_load_a_u"
]

st.dataframe(
    player_df[cols]
    .sort_values(
        "date",
        ascending=False
    )
    .head(20),
    use_container_width=True
)

# =====================================================
# AI INSIGHTS
# =====================================================

st.subheader("AI Insights")

if pred_distance > player_df["distance_m"].mean():

    st.info(
        "Predicted workload is above player's average."
    )

else:

    st.info(
        "Predicted workload is below player's average."
    )

if latest_ewma > 1.5:

    st.error(
        "Workload spike detected."
    )

elif latest_ewma > 1.3:

    st.warning(
        "Workload increasing."
    )

else:

    st.success(
        "Workload stable."
    )

# =====================================================
# SESSION PLANNER
# =====================================================

st.subheader("Session Planner")

planner1, planner2 = st.columns(2)

planned_distance = planner1.number_input(
    "Planned Distance (m)",
    value=5000
)

planned_hsr = planner1.number_input(
    "Planned HSR (m)",
    value=250
)

planned_acc = planner2.number_input(
    "Planned Accelerations",
    value=60
)

planned_load = planner2.number_input(
    "Planned Player Load",
    value=50
)

future_ewma = (
    latest_ewma * 0.8
    +
    (
        planned_load
        /
        max(
            player_df_full["player_load_a_u"].mean(),
            1
        )
    )
    * 0.2
)

if future_ewma < 0.8:

    future_risk = "🔵 LOW"

elif future_ewma < 1.3:

    future_risk = "🟢 NORMAL"

elif future_ewma < 1.5:

    future_risk = "🟡 MEDIUM"

else:

    future_risk = "🔴 HIGH"


st.markdown("---")

p1, p2 = st.columns(2)

p1.metric(
    "Projected EWMA",
    f"{future_ewma:.2f}"
)

p2.metric(
    "Projected Risk",
    future_risk
)


# =====================================================
# SUMMARY
# =====================================================

st.subheader("Player Summary")

st.write(
    f"""
    This player completed {len(player_df)} sessions.

    Average distance:
    {player_df['distance_m'].mean():.0f} m

    Average HSR:
    {player_df['abs_hsr_m'].mean():.0f} m

    Average Player Load:
    {player_df['player_load_a_u'].mean():.1f}
    """
)