import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.predictor import predict_next_session

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Rendimiento y Control de Carga",
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
        return "🔵 BAJO"

    elif ewma < 1.3:
        return "🟢 NORMAL"

    elif ewma < 1.5:
        return "🟡 MEDIO"

    else:
        return "🔴 ALTO"


latest_rows["risk"] = (
    latest_rows["ewma_ratio_player_load_a_u"]
    .apply(classify_risk)
)

# =====================================================
# RISK FUNCTIONS
# =====================================================

def classify_risk(ewma):

    if ewma < 0.8:
        return "🔵 BAJO"

    elif ewma < 1.3:
        return "🟢 NORMAL"

    elif ewma < 1.5:
        return "🟡 MEDIO"

    else:
        return "🔴 ALTO"


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
# HEADER
# =====================================================

st.title("⚽ Monitorización de la carga")

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
# SIDEBAR
# =====================================================

st.sidebar.title("⚽ Control de Carga")

player = st.sidebar.selectbox(
    "Jugador",
    sorted(df["player"].unique())
)

player_df_full = (
    df[df["player"] == player]
    .sort_values("date")
    .copy()
)

start_date = st.sidebar.date_input(
    "Fecha inicio",
    value=player_df_full["date"].min()
)

end_date = st.sidebar.date_input(
    "Fecha fin",
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
# TABS
# =====================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Plantilla",
    "👤 Jugador",
    "⚖️ Benchmarking",
    "🚨 Riesgo",
    "🏃 Sesiones"
])

with tab1:
    
    st.subheader("Resumen de Plantilla")

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
    "🔴 Riesgo Alto",
    high_risk
)

c2.metric(
    "🟡 Riesgo Medio",
    medium_risk
)

c3.metric(
    "🟢 Normal / Bajo",
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
    "Jugador",
    "Riesgo",
    "Índice de Riesgo",
    "Distancia Total",
    "HSR",
    "Carga del jugador",
    "EWMA"
]

st.dataframe(
    overview.sort_values(
        "Índice de Riesgo",
        ascending=False
    ),
    use_container_width=True
)

st.divider()

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
# EXECUTIVE SUMMARY
# =====================================================

st.markdown("---")

st.subheader("Resumen Ejecutivo")

current_risk_score = calculate_risk_score(
    last_session
)

e1, e2, e3, e4, e5 = st.columns(5)

e1.metric(
    "Distancia Prevista",
    f"{pred_distance:,.0f} m"
)

e2.metric(
    "Ratio EWMA",
    f"{latest_ewma:.2f}"
)

e3.metric(
    "Estado de Riesgo",
    f"{color} {risk}"
)

e4.metric(
    "HSR Última Sesión",
    f"{last_session['abs_hsr_m']:.0f} m"
)

e5.metric(
    "Aceleraciones",
    f"{last_session['accelerations_count']:.0f}"
)

st.markdown("---")

# =====================================================
# INJURY RISK GAUGE
# =====================================================

fig = go.Figure(
    go.Indicator(

        mode="gauge+number",

        value=current_risk_score,

        title={
            "text": "Índice de Riesgo de Lesión"
        },

        gauge={

            "axis": {
                "range": [0, 100]
            },

            "bar": {
                "thickness": 0.3
            },

            "steps": [

                {
                    "range": [0, 40],
                    "color": "lightgreen"
                },

                {
                    "range": [40, 70],
                    "color": "gold"
                },

                {
                    "range": [70, 100],
                    "color": "lightcoral"
                }

            ]

        }

    )
)

fig.update_layout(
    height=300
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# RISK BREAKDOWN
# =====================================================

st.subheader(
    "Desglose del Riesgo"
)

ewma_points = min(
    last_session["ewma_ratio_player_load_a_u"] * 25,
    35
)

acwr_points = min(
    last_session["acwr_player_load_a_u"] * 15,
    20
)

hsr_points = min(
    last_session["abs_hsr_m"] / 25,
    15
)

acc_points = min(
    last_session["accelerations_count"] / 8,
    15
)

load_points = min(
    last_session["player_load_a_u"] / 4,
    15
)

risk_breakdown = pd.DataFrame({

    "Factor": [
        "EWMA",
        "ACWR",
        "HSR",
        "Aceleraciones",
        "Player Load"
    ],

    "Puntos": [
        ewma_points,
        acwr_points,
        hsr_points,
        acc_points,
        load_points
    ]

})

risk_breakdown = (
    risk_breakdown
    .sort_values(
        "Puntos",
        ascending=False
    )
)

fig = px.bar(

    risk_breakdown,

    x="Puntos",

    y="Factor",

    orientation="h"

)

fig.update_layout(
    height=350
)

st.plotly_chart(
    fig,
    use_container_width=True
)

top_factor = risk_breakdown.iloc[0]

st.info(
    f"Principal factor de riesgo: "
    f"{top_factor['Factor']} "
    f"({top_factor['Puntos']:.1f} puntos)"
)

# =====================================================
# RISK INSIGHTS
# =====================================================

st.subheader(
    "Interpretación Automática"
)

drivers = []

if (
    last_session["ewma_ratio_player_load_a_u"]
    >= 1.3
):

    drivers.append(
        "EWMA elevada"
    )

if (
    last_session["acwr_player_load_a_u"]
    >= 1.2
):

    drivers.append(
        "Incremento reciente de carga"
    )

if (
    last_session["abs_hsr_m"]
    >
    player_df_full["abs_hsr_m"].mean()
):

    drivers.append(
        "Exposición alta a HSR"
    )

if (
    last_session["accelerations_count"]
    >
    player_df_full["accelerations_count"].mean()
):

    drivers.append(
        "Número elevado de aceleraciones"
    )

if (
    last_session["player_load_a_u"]
    >
    player_df_full["player_load_a_u"].mean()
):

    drivers.append(
        "Carga externa elevada"
    )

if len(drivers) == 0:

    st.success(
        "No se detectan factores de riesgo relevantes."
    )

else:

    insight = (
        "El riesgo actual está impulsado por: "
        + ", ".join(drivers)
        + "."
    )

    st.info(
        insight
    )

# =====================================================
# OPERATIONAL RECOMMENDATION
# =====================================================

st.subheader(
    "Recomendación Operativa"
)

if current_risk_score < 40:

    st.success(
        """
        Riesgo bajo.

        El jugador puede tolerar una carga normal o incluso
        un incremento moderado de la exposición.
        """
    )

elif current_risk_score < 60:

    st.info(
        """
        Riesgo controlado.

        Mantener la planificación prevista y continuar
        monitorizando la evolución de la carga.
        """
    )

elif current_risk_score < 80:

    st.warning(
        """
        Riesgo elevado.

        Considerar limitar la exposición a HSR,
        sprints y aceleraciones en las próximas sesiones.
        """
    )

else:

    st.error(
        """
        Riesgo alto.

        Se recomienda reducir la carga de entrenamiento,
        especialmente las acciones de alta intensidad,
        y realizar un seguimiento individualizado.
        """
    )
 
# =====================================================
# TOP RISK PLAYERS
# =====================================================

st.subheader("Jugadores con más riesgo")

top_risk = latest_rows.copy()

top_risk["Índice de Riesgo"] = (
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
            "Índice de Riesgo",
            "ewma_ratio_player_load_a_u",
            "acwr_player_load_a_u"
        ]
    ]
    .sort_values(
        "Índice de Riesgo",
        ascending=False
    )
    .head(10)
)

top_risk.columns = [
    "Jugador",
    "Índice de Riesgo",
    "EWMA",
    "ACWR"
]
fig = px.bar(

    top_risk,

    x="Índice de Riesgo",

    y="Jugador",

    orientation="h",

    color="Índice de Riesgo"

)

fig.update_layout(

    height=450,

    yaxis={
        "categoryorder": "total ascending"
    }

)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("---")

st.markdown(
    f"### {player}"
)


# =====================================================
# SQUAD BENCHMARKING
# =====================================================

st.subheader("Comparativa con la Plantilla")


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

    "Métrica": [
        "Distancia Total",
        "HSR",
        "Sprints",
        "Aceleraciones",
        "Player Load"
    ],

    "Jugador": [
        round(player_distance, 1),
        round(player_hsr, 1),
        round(player_sprints, 1),
        round(player_acc, 1),
        round(player_load, 1)
    ],

    "Media de la plantilla": [
        round(squad_avg_distance, 1),
        round(squad_avg_hsr, 1),
        round(squad_avg_sprints, 1),
        round(squad_avg_acc, 1),
        round(squad_avg_load, 1)
    ]
})

benchmark["Diferencia %"] = round(
    (
        benchmark["Jugador"]
        /
        benchmark["Media de la plantilla"]
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

benchmark["Diferencia %"] = (
    benchmark["Diferencia %"]
    .apply(classify_benchmark)
)

st.dataframe(
    benchmark,
    use_container_width=True
)

st.caption(
    """
    🔴 Mucho más alto que la media de la plantilla (>20%)

    🟡 Más alto que la media de la plantilla (+10% to +20%)

    ⚪ Similar a la media de la plantilla

    🟢 Más bajo que la media de la plantilla (-10% to -20%)

    🔵 Mucho más bajo que la media de la plantilla (<-20%)
    """
)

st.divider()


st.divider()

# =====================================================
# POSITIONAL BENCHMARKING
# =====================================================

st.subheader("Comparativa por Posición")

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

    "Métrica": [
        "Distancia",
        "HSR",
        "Sprints",
        "Aceleraciones",
        "Player Load"
    ],

    "Jugador": [
        round(player_distance,1),
        round(player_hsr,1),
        round(player_sprints,1),
        round(player_acc,1),
        round(player_load,1)
    ],

    "Media en su posición": [
        round(pos_avg_distance,1),
        round(pos_avg_hsr,1),
        round(pos_avg_sprints,1),
        round(pos_avg_acc,1),
        round(pos_avg_load,1)
    ]
})

position_benchmark["Diferencia %"] = round(
    (
        position_benchmark["Jugador"]
        /
        position_benchmark["Media en su posición"]
        - 1
    ) * 100,
    1
)

position_benchmark["Diferencia %"] = (
    position_benchmark["Diferencia %"]
    .apply(classify_benchmark)
)

st.write(
    f"Posición: {player_position}"
)

st.dataframe(
    position_benchmark,
    use_container_width=True
)

# =====================================================
# PERFIL AUTOMÁTICO
# =====================================================

distance_diff = (
    (
        player_distance
        /
        pos_avg_distance
    ) - 1
) * 100

hsr_diff = (
    (
        player_hsr
        /
        pos_avg_hsr
    ) - 1
) * 100

acc_diff = (
    (
        player_acc
        /
        pos_avg_acc
    ) - 1
) * 100


def classify_profile(diff):

    if diff > 20:
        return "muy superior"

    elif diff > 10:
        return "superior"

    elif diff < -20:
        return "muy inferior"

    elif diff < -10:
        return "inferior"

    else:
        return "similar"


profile_text = f"""
Distancia total: {classify_profile(distance_diff)} a la media de su posición.

HSR: {classify_profile(hsr_diff)} respecto a jugadores de la misma posición.

Aceleraciones: {classify_profile(acc_diff)} respecto a jugadores de la misma posición.
"""

st.subheader(
    "Perfil Automático"
)

st.info(
    profile_text
)

st.divider()

# =====================================================
# KPI BLOCK
# =====================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Sesiones",
    len(player_df)
)

col2.metric(
    "Distancia Media",
    f"{player_df['distance_m'].mean():.0f} m"
)

col3.metric(
    "Media HSR",
    f"{player_df['abs_hsr_m'].mean():.0f} m"
)

col4.metric(
    "Media Player Load",
    f"{player_df['player_load_a_u'].mean():.1f}"
)

st.divider()

st.subheader("Predicción del Modelo")

p1, p2 = st.columns(2)

p1.metric(
    "Distancia próxima sesión",
    f"{pred_distance:,.0f} m"
)

p2.metric(
    "Media distancia jugador",
    f"{player_df['distance_m'].mean():,.0f} m"
)
# =====================================================
# LAST SESSION
# =====================================================

st.subheader("Última Sesión")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Fecha",
    last_session["date"].strftime("%d/%m/%Y")
)

c2.metric(
    "Sesión",
    last_session["type_session"]
)

c3.metric(
    "Distancia",
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
# SESSION TYPE ANALYSIS
# =====================================================

st.subheader("Análisis por Tipo de Sesión")

session_profile = (
    player_df_full
    .groupby("type_session")
    [
        [
            "distance_m",
            "abs_hsr_m",
            "sprints_abs_count",
            "accelerations_count",
            "player_load_a_u"
        ]
    ]
    .mean()
    .reset_index()
)

st.dataframe(
    session_profile,
    use_container_width=True
)

st.subheader("HSR por Tipo de Sesión")

fig = px.bar(
    session_profile,
    x="type_session",
    y="abs_hsr_m"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("Aceleraciones por Tipo de Sesión")

fig = px.bar(
    session_profile,
    x="type_session",
    y="accelerations_count"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# POSITIONAL SESSION BENCHMARKING
# =====================================================

st.subheader(
    "Comparativa de la Sesión"
)

selected_date = st.selectbox(
    "Seleccionar sesión",
    player_df["date"]
    .sort_values(
        ascending=False
    )
    .dt.strftime("%d/%m/%Y")
)

selected_row = player_df[
    player_df["date"]
    ==
    pd.to_datetime(
        selected_date,
        dayfirst=True
    )
].iloc[0]

session_type = selected_row[
    "type_session"
]

player_position = selected_row[
    "position"
]

reference_df = df[
    (
        df["position"]
        ==
        player_position
    )
    &
    (
        df["type_session"]
        ==
        session_type
    )
]

st.info(
    f"""
    Fecha: {selected_date}

    Tipo de sesión: {session_type}

    Posición: {player_position}
    """
)

comparison = pd.DataFrame({

    "Métrica": [
        "Distancia",
        "HSR",
        "Sprints",
        "Aceleraciones",
        "Player Load"
    ],

    "Sesión": [

        selected_row["distance_m"],
        selected_row["abs_hsr_m"],
        selected_row["sprints_abs_count"],
        selected_row["accelerations_count"],
        selected_row["player_load_a_u"]

    ],

    "Media Posición": [

        reference_df["distance_m"].mean(),
        reference_df["abs_hsr_m"].mean(),
        reference_df["sprints_abs_count"].mean(),
        reference_df["accelerations_count"].mean(),
        reference_df["player_load_a_u"].mean()

    ]

})

comparison["Dif %"] = round(

    (
        comparison["Sesión"]
        /
        comparison["Media Posición"]
        - 1
    ) * 100,

    1
)

st.dataframe(
    comparison,
    use_container_width=True
)

fig = px.bar(

    comparison,

    x="Métrica",

    y=["Sesión", "Media Posición"],

    barmode="group"

)

fig.update_layout(
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# EXIGENCIA DE LA SESIÓN
# =====================================================

comparison["Ratio"] = (
    comparison["Sesión"]
    /
    comparison["Media Posición"]
)

session_score = round(

    comparison["Ratio"]
    .mean()
    * 50,

    0
)

session_score = max(
    0,
    min(
        session_score,
        100
    )
)

st.metric(
    "Exigencia Sesión",
    f"{session_score:.0f}/100"
)

if session_score < 40:

    st.success(
        "Sesión ligera respecto a jugadores de la misma posición."
    )

elif session_score < 70:

    st.info(
        "Sesión dentro de parámetros normales."
    )

elif session_score < 85:

    st.warning(
        "Sesión exigente respecto a jugadores de la misma posición."
    )

else:

    st.error(
        "Sesión muy exigente respecto a jugadores de la misma posición."
    )



# =====================================================
# SESSION PROFILE RADAR
# =====================================================

st.subheader(
    "Perfil de la Sesión"
)

radar_df = comparison.copy()

radar_df["Sesión Norm"] = (
    radar_df["Sesión"]
    /
    radar_df["Media Posición"]
) * 100

radar_df["Posición Norm"] = 100

fig = go.Figure()

fig.add_trace(
    go.Scatterpolar(
        r=radar_df["Sesión Norm"],
        theta=radar_df["Métrica"],
        fill="toself",
        name="Sesión Seleccionada"
    )
)

fig.add_trace(
    go.Scatterpolar(
        r=radar_df["Posición Norm"],
        theta=radar_df["Métrica"],
        fill="toself",
        name="Media Posición"
    )
)

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True
        )
    ),
    height=600
)

st.plotly_chart(
    fig,
    use_container_width=True
)

top_metric = comparison.loc[
    comparison["Dif %"].idxmax(),
    "Métrica"
]

top_diff = comparison["Dif %"].max()

st.info(
    f"La mayor diferencia respecto a su posición se observa en "
    f"{top_metric} "
    f"({top_diff:+.1f}%)."
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
# SPRINTS
# =====================================================

st.subheader("Sprints")

fig = px.line(
    player_df,
    x="date",
    y="sprints_abs_count"
)

fig.update_layout(
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# ACCELERATIONS
# =====================================================

st.subheader("Aceleraciones")

fig = px.line(
    player_df,
    x="date",
    y="accelerations_count"
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
# DISTANCE
# =====================================================

st.subheader("Evolución Distancia")

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
# ACUTE VS CHRONIC
# =====================================================

if (
    "acute_player_load_a_u_7d" in player_df.columns
    and
    "chronic_player_load_a_u_28d" in player_df.columns
):

    st.subheader(
        "Carga Aguda vs Crónica"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=player_df["date"],
            y=player_df["acute_player_load_a_u_7d"],
            name="Aguda"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=player_df["date"],
            y=player_df["chronic_player_load_a_u_28d"],
            name="Crónica"
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

st.subheader("Tendencia de Carga")

st.metric(
    "Tendencia semanal EWMA",
    f"{trend:.2f}"
)

# =====================================================
# WEEKLY MICROCYCLE ANALYSIS
# =====================================================

st.subheader("Análisis Microciclo Semanal")

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
    "Semana",
    "Distancia",
    "HSR",
    "Sprints",
    "Aceleraciones",
    "Player Load"
]

st.dataframe(
    weekly.sort_values(
        "Semana",
        ascending=False
    ),
    use_container_width=True
)

st.divider()

# =====================================================
# WEEKLY HSR
# =====================================================

st.subheader("HSR Semanal")

fig = px.bar(
    weekly,
    x="Semana",
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

st.subheader("Aceleraciones Semanal")

fig = px.bar(
    weekly,
    x="Semana",
    y="Aceleraciones"
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

st.subheader("Últimas Sesiones")

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

st.subheader("Observaciones del modelo")

if pred_distance > player_df["distance_m"].mean():

    st.info(
        "La carga predecida está por encima de la media del jugador."
    )

else:

    st.info(
        "La carga predecida está por encima de la media del jugador."
    )

if latest_ewma > 1.5:

    st.error(
        "Pico de carga de trabajo detectado."
    )

elif latest_ewma > 1.3:

    st.warning(
        "Carga de trabajo aumentando."
    )

else:

    st.success(
        "Carga de trabajo estable."
    )

# =====================================================
# SESSION PLANNER
# =====================================================

st.subheader("Planificador de Sesión")

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

    future_risk = "🔵 BAJO"

elif future_ewma < 1.3:

    future_risk = "🟢 NORMAL"

elif future_ewma < 1.5:

    future_risk = "🟡 MEDIO"

else:

    future_risk = "🔴 ALTO"


st.markdown("---")

p1, p2 = st.columns(2)

p1.metric(
    "EWMA Estimada",
    f"{future_ewma:.2f}"
)

p2.metric(
    "Riesgo Estimado",
    future_risk
)


# =====================================================
# TOP SESIONES MÁS EXIGENTES
# =====================================================

st.subheader(
    "Top Sesiones Más Exigentes"
)

player_sessions = player_df_full.copy()

player_sessions["Session Score"] = (

    player_sessions["distance_m"]
    /
    player_sessions["distance_m"].mean()

) * 100

top_sessions = (

    player_sessions[
        [
            "date",
            "type_session",
            "Session Score",
            "distance_m",
            "abs_hsr_m"
        ]
    ]

    .sort_values(
        "Session Score",
        ascending=False
    )

    .head(10)

)

top_sessions.columns = [

    "Fecha",
    "Tipo de Sesión",
    "Índice de Exigencia",
    "Distancia",
    "HSR"

]

fig = px.bar(

    top_sessions.sort_values(
        "Índice de Exigencia"
    ),

    x="Índice de Exigencia",

    y="Fecha",

    orientation="h"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# SUMMARY
# =====================================================

st.subheader("Resumen del jugador")

st.write(
    f"""
    Este jugador ha completado {len(player_df)} sesiones.

    Distancia Media:
    {player_df['distance_m'].mean():.0f} m

   HSR Medio :
    {player_df['abs_hsr_m'].mean():.0f} m

    Player Load Medio:
    {player_df['player_load_a_u'].mean():.1f}
    """
)
