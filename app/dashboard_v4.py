import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.predictor import predict_next_session
from scipy.stats import percentileofscore


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
    "data/processed/model_dataset_v2.csv"
)

df["date"] = pd.to_datetime(df["date"])

# =====================================================
# HEADER
# =====================================================

st.title("⚽ Monitorización de la carga")

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

latest_rows["risk_score"] = (
    latest_rows
    .apply(
        calculate_risk_score,
        axis=1
    )
)

def classify_risk_score(score):

    if score < 40:
        return "🟢 BAJO"

    elif score < 60:
        return "🟡 MEDIO"

    elif score < 80:
        return "🟠 ALTO"

    else:
        return "🔴 MUY ALTO"

latest_rows["risk"] = (
    latest_rows["risk_score"]
    .apply(classify_risk_score)
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
# PLAYER VARIABLES
# =====================================================

last_session = player_df.iloc[-1]

pred_distance = predict_next_session(
    player_df
)

latest_ewma = (
    player_df["ewma_ratio_player_load_a_u"]
    .iloc[-1]
)

current_risk_score = calculate_risk_score(
    last_session
)

if latest_ewma < 0.8:

    risk = "BAJO"
    color = "🔵"

elif latest_ewma < 1.3:

    risk = "NORMAL"
    color = "🟢"

elif latest_ewma < 1.5:

    risk = "MEDIO"
    color = "🟡"

else:

    risk = "ALTO"
    color = "🔴"

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Plantilla",
    "👤 Jugador",
    "⚖️ Comparativas",
    "🚨 Riesgo",
    "🏃 Sesiones"
])

with tab1:
    
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

    st.subheader("Resumen de Plantilla")

    very_high_risk = len(
    latest_rows[
        latest_rows["risk_score"] >= 80
    ]
)

    high_risk = len(
    latest_rows[
        (
            latest_rows["risk_score"] >= 60
        )
        &
        (
            latest_rows["risk_score"] < 80
        )
    ]
)

    medium_risk = len(
    latest_rows[
        (
            latest_rows["risk_score"] >= 40
        )
        &
        (
            latest_rows["risk_score"] < 60
        )
    ]
)

    low_risk = len(
    latest_rows[
        latest_rows["risk_score"] < 40
    ]
)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
    "🔴 Muy Alto",
    very_high_risk
)

    c2.metric(
    "🟠 Alto",
    high_risk
)

    c3.metric(
    "🟡 Medio",
    medium_risk
)

    c4.metric(
    "🟢 Bajo",
    low_risk
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

    # =====================================================
# MICROCYCLE DISTRIBUTION
# =====================================================

    st.subheader("Distribución del Microciclo")

    phase_order = [
    
        "MD+1",
        "MD+2",
        "MD-4",
        "MD-3",
        "MD-2",
        "MD-1",
        "MD"
    ]


    phase_counts = (
    df["session_phase"]
    .value_counts()
    .reset_index()
)

    phase_counts.columns = [
    "Fase",
    "Sesiones"
]

    phase_counts = phase_counts[
    phase_counts["Fase"].isin(phase_order)
]

    phase_counts["Fase"] = pd.Categorical(
    phase_counts["Fase"],
    categories=phase_order,
    ordered=True
)

    phase_counts = (
    phase_counts
    .sort_values("Fase")
)

    fig = px.bar(
    phase_counts,
    x="Fase",
    y="Sesiones",
    text="Sesiones"
)

    fig.update_layout(
    height=450,
    showlegend=False
)

    st.plotly_chart(
    fig,
    use_container_width=True
)
        # =====================================================
# MICROCYCLE PROFILE
# =====================================================

    st.subheader("Perfil Medio del Microciclo")

    microcycle_profile = (
    df
    .groupby("session_phase")
    [
        [
            "distance_m",
            "abs_hsr_m",
            "player_load_a_u"
        ]
    ]
    .mean()
    .reset_index()
)

    microcycle_profile = microcycle_profile[
    microcycle_profile["session_phase"].isin(
        phase_order
    )
]

    microcycle_profile["session_phase"] = pd.Categorical(
    microcycle_profile["session_phase"],
    categories=phase_order,
    ordered=True
)

    microcycle_profile = (
    microcycle_profile
    .sort_values("session_phase")
)
    
    c1, c2, c3 = st.columns(3)

    with c1:

        fig = px.line(
        microcycle_profile,
        x="session_phase",
        y="distance_m",
        markers=True,
        title="Distancia"
    )

        fig.update_layout(
        height=350
    )

        st.plotly_chart(
        fig,
        use_container_width=True
    )

    with c2:

        fig = px.line(
        microcycle_profile,
        x="session_phase",
        y="abs_hsr_m",
        markers=True,
        title="HSR"
    )

        fig.update_layout(
        height=350
    )

        st.plotly_chart(
        fig,
        use_container_width=True
    )

    with c3:

        fig = px.line(
        microcycle_profile,
        x="session_phase",
        y="player_load_a_u",
        markers=True,
        title="Player Load"
    )

        fig.update_layout(
        height=350
    )

        st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()


with tab2:

    st.markdown("---")

    st.subheader("Resumen Ejecutivo")

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
    # MICROCYCLE DEVIATION
    # =====================================================

    st.subheader("Comparativa con el Patrón del Microciclo")

    last_phase = (
    player_df["session_phase"]
    .iloc[-1]
)

    last_distance = (
    player_df["distance_m"]
    .iloc[-1]
)

    last_hsr = (
    player_df["abs_hsr_m"]
    .iloc[-1]
)

    last_load = (
    player_df["player_load_a_u"]
    .iloc[-1]
)

    phase_reference = (
    df[
        df["session_phase"] == last_phase
    ]
    [
        [
            "distance_m",
            "abs_hsr_m",
            "player_load_a_u"
        ]
    ]
    .mean()
)

    distance_diff = (
    (
        last_distance
        /
        phase_reference["distance_m"]
    )   - 1
)       * 100

    hsr_diff = (
    (
        last_hsr
        /
        phase_reference["abs_hsr_m"]
    ) - 1
)       * 100

    load_diff = (
    (
        last_load
        /
        phase_reference["player_load_a_u"]
    ) - 1
)       * 100
    
    st.markdown(
    f"#### Última sesión: {last_phase}"
)

    c1, c2, c3 = st.columns(3)

    c1.metric(
    "Distancia",
    f"{last_distance:,.0f} m",
    f"{distance_diff:+.1f}%"
)

    c2.metric(
    "HSR",
    f"{last_hsr:,.0f} m",
    f"{hsr_diff:+.1f}%"
)

    c3.metric(
    "Player Load",
    f"{last_load:.1f}",
    f"{load_diff:+.1f}%"
)
    
    alerts = []

    if distance_diff > 20:
        alerts.append(
        "Distancia significativamente superior al patrón esperado."
    )

    if hsr_diff > 20:
        alerts.append(
        "Exposición HSR superior a la habitual para esta fase."
    )

    if load_diff > 20:
        alerts.append(
        "Player Load elevado respecto al microciclo."
    )

    if len(alerts) > 0:

        st.warning(
        " ".join(alerts)
    )

    else:

        st.success(
        "La sesión se encuentra dentro de los valores esperados para esta fase del microciclo."
    )


    # =====================================================
# MICROCYCLE COMPLIANCE
# =====================================================

    st.subheader(
        "Índice de Cumplimiento del Microciclo"
)

    phase_order =[
        "MD+1",
        "MD+2",
        "MD-4",
        "MD-3",
        "MD-2",
        "MD-1",
        "MD"
    ]
    
    player_phase = (
        player_df
        .groupby("session_phase")
    [
        [
            "distance_m",
            "abs_hsr_m",
            "player_load_a_u"
        ]
    ]
    .mean()
)

    team_phase = (
        df
        .groupby("session_phase")
    [
        [
            "distance_m",
            "abs_hsr_m",
            "player_load_a_u"
        ]
    ]
    .mean()
)
    compliance = []

    for phase in phase_order:

            if (
        phase in player_phase.index
        and
        phase in team_phase.index
    ):

                distance_ratio = (
                player_phase.loc[
                phase,
                "distance_m"
            ]
            /
            team_phase.loc[
                phase,
                "distance_m"
            ]
        )

            hsr_ratio = (
            player_phase.loc[
                phase,
                "abs_hsr_m"
            ]
            /
            team_phase.loc[
                phase,
                "abs_hsr_m"
            ]
        )

            compliance_score = (
            distance_ratio
            +
            hsr_ratio
        ) / 2 * 100

    compliance.append(
            {
                "Fase": phase,
                "Cumplimiento": round(
                    compliance_score,
                    1
                )
            }
        )

    compliance = pd.DataFrame(
            compliance
)
        
    global_compliance = (
        compliance["Cumplimiento"]
        .mean()
        )
    st.metric(
        "Índice Global",
        f"{global_compliance:.0f}%")
        

    fig = px.bar(
        compliance,
        x="Fase",
        y="Cumplimiento",
        text="Cumplimiento"
)

    fig.add_hline(
        y=100,
        line_dash="dash"
)

    fig.update_layout(
        height=400
)

    st.plotly_chart(
        fig,
        use_container_width=True
)


    if global_compliance > 110:

            st.warning(
        "El jugador trabaja sistemáticamente por encima del patrón habitual del equipo."
    )

    elif global_compliance < 90:

            st.info(
        "El jugador trabaja sistemáticamente por debajo del patrón habitual del equipo."
    )

    else:

            st.success(
        "El jugador sigue de forma consistente el patrón de carga del microciclo."
    )


    phase_order = [
    
        "MD+1",
        "MD+2",
        "MD-4",
        "MD-3",
        "MD-2",
        "MD-1",
        "MD"
    ]
    
# =====================================================
# PERFIL DEL MICROCICLO
# =====================================================

    st.subheader(
"Perfil del Microciclo"
)

    phase_order = [
    
        "MD+1",
        "MD+2",
        "MD-4",
        "MD-3",
        "MD-2",
        "MD-1",
        "MD"
    ]
    

    metric = st.selectbox(
"Métrica",
[
"distance_m",
"abs_hsr_m",
"sprints_abs_count"
]
)

    player_microcycle = (
    player_df
    .groupby("session_phase")[metric]
    .mean()
    .reindex(phase_order)
)

    team_microcycle = (
    df
    .groupby("session_phase")[metric]
    .mean()
    .reindex(phase_order)
)

    plot_df = pd.DataFrame({
    "Fase": phase_order,
    "Jugador": player_microcycle.values,
    "Equipo": team_microcycle.values
})

    fig = go.Figure()

    fig.add_trace(
go.Scatter(
x=plot_df["Fase"],
y=plot_df["Jugador"],
mode="lines+markers",
name="Jugador"
)
)

    fig.add_trace(
go.Scatter(
x=plot_df["Fase"],
y=plot_df["Equipo"],
mode="lines+markers",
name="Equipo"
)
)

    metric_names = {
"distance_m": "Distancia",
"abs_hsr_m": "HSR",
"sprints_abs_count": "Sprints"
}

    fig.update_layout(
height=500,
title=f"{metric_names[metric]} por Fase del Microciclo",
xaxis_title="Fase",
yaxis_title=metric_names[metric]
)

    st.plotly_chart(
fig,
use_container_width=True
)

# =====================================================
# INTERPRETACIÓN AUTOMÁTICA
# =====================================================

    comparison = (
    (
    plot_df["Jugador"]
    /
    plot_df["Equipo"]
    ) - 1
    ) * 100

    max_diff_idx = comparison.abs().idxmax()

    phase = plot_df.loc[
    max_diff_idx,
    "Fase"
    ]

    diff = comparison.iloc[
    max_diff_idx
    ]

    if diff > 0:
        message = (
        f"La mayor diferencia respecto al patrón del equipo "
        f"se observa en {phase} (+{diff:.1f}%)."
    )
    else:
        message = (
        f"La mayor diferencia respecto al patrón del equipo "
        f"se observa en {phase} ({diff:.1f}%)."
    )

    st.info(message)

    
    
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
        # ACWR ALERT
            # =====================================================

    if "acwr_player_load_a_u" in player_df.columns:

            current_acwr = (
            player_df["acwr_player_load_a_u"]
            .iloc[-1]
    )

    st.subheader("Interpretación ACWR")

    if current_acwr < 0.8:

        st.info(
            f"ACWR actual: {current_acwr:.2f}. "
            "La carga reciente está por debajo de la carga crónica."
        )

    elif current_acwr < 1.3:

        st.success(
            f"ACWR actual: {current_acwr:.2f}. "
            "La relación carga aguda/crónica se encuentra en rango normal."
        )

    elif current_acwr < 1.5:

        st.warning(
            f"ACWR actual: {current_acwr:.2f}. "
            "Se observa un incremento relevante de carga reciente."
        )

    else:

        st.error(
            f"ACWR actual: {current_acwr:.2f}. "
            "Pico de carga detectado. Riesgo aumentado."
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
        # EWMA ALERT
        # =====================================================

    latest_ewma = (
            player_df["ewma_ratio_player_load_a_u"]
            .iloc[-1]
)

    st.subheader("Interpretación EWMA")

    if latest_ewma < 0.8:

        st.info(
        f"EWMA actual: {latest_ewma:.2f}. "
        "La carga reciente está claramente por debajo de la tendencia habitual."
    )

    elif latest_ewma < 1.3:

        st.success(
        f"EWMA actual: {latest_ewma:.2f}. "
        "La carga se encuentra dentro de parámetros normales."
    )

    elif latest_ewma < 1.5:

        st.warning(
        f"EWMA actual: {latest_ewma:.2f}. "
        "La carga está aumentando respecto al patrón habitual."
    )

    else:

        st.error(
        f"EWMA actual: {latest_ewma:.2f}. "
        "Pico de carga detectado. Conviene monitorizar próximas sesiones."
    )

        # =====================================================
        # LOAD STATUS
        # =====================================================

    st.subheader("Estado Actual de la Carga")

    if (
            latest_ewma >= 1.5
            and
            current_acwr >= 1.5
):

            st.error(
        """
        EWMA y ACWR elevados.

        Existe evidencia de un incremento importante de carga
        reciente respecto al historial del jugador.
        """
    )

    elif (
        latest_ewma >= 1.3
        or
        current_acwr >= 1.3
):

            st.warning(
        """
        La carga muestra una tendencia ascendente.

        Se recomienda monitorizar la evolución durante las
        próximas sesiones.
        """
    )

    else:

            st.success(
        """
        La carga actual se encuentra dentro de parámetros
        estables para el jugador.
        """
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

    # =====================================================
# INTENSITY EVOLUTION
# =====================================================

    st.subheader(
    "Evolución de Intensidad"
)

    intensity_df = player_df.copy()

    intensity_df["HSR"] = (
    intensity_df["abs_hsr_m"]
    /
    intensity_df["abs_hsr_m"].max()
) * 100

    intensity_df["Sprints"] = (
    intensity_df["sprints_abs_count"]
    /
    intensity_df["sprints_abs_count"].max()
) * 100

    intensity_df["Accelerations"] = (
    intensity_df["accelerations_count"]
    /
    intensity_df["accelerations_count"].max()
) * 100

    fig = go.Figure()

    fig.add_trace(
    go.Scatter(
        x=intensity_df["date"],
        y=intensity_df["HSR"],
        name="HSR"
    )
)

    fig.add_trace(
    go.Scatter(
        x=intensity_df["date"],
        y=intensity_df["Sprints"],
        name="Sprints"
    )
)

    fig.add_trace(
    go.Scatter(
        x=intensity_df["date"],
        y=intensity_df["Accelerations"],
        name="Accelerations"
    )
)

    fig.update_layout(
    height=500,
    yaxis_title="Índice (0-100)"
)

    st.plotly_chart(
        fig,
    use_container_width=True
)
    st.divider()

with tab3:

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

    squad_distance_diff = (
    (player_distance / squad_avg_distance) - 1
    ) * 100

    squad_hsr_diff = (
    (player_hsr / squad_avg_hsr) - 1
) * 100

    squad_sprint_diff = (
    (player_sprints / squad_avg_sprints) - 1
) * 100

    squad_acc_diff = (
    (player_acc / squad_avg_acc) - 1
) * 100

    squad_load_diff = (
    (player_load / squad_avg_load) - 1
) * 100

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
    "Distancia",
    f"{squad_distance_diff:+.1f}%"
)

    c2.metric(
    "HSR",
    f"{squad_hsr_diff:+.1f}%"
)

    c3.metric(
    "Sprints",
    f"{squad_sprint_diff:+.1f}%"
)

    c4.metric(
    "Aceleraciones",
    f"{squad_acc_diff:+.1f}%"
)

    c5.metric(
    "Player Load",
    f"{squad_load_diff:+.1f}%"
)
    benchmark_chart = pd.DataFrame({

    "Métrica": [
        "Distancia",
        "HSR",
        "Sprints",
        "Aceleraciones",
        "Player Load"
    ],

    "Dif %": [
        squad_distance_diff,
        squad_hsr_diff,
        squad_sprint_diff,
        squad_acc_diff,
        squad_load_diff
    ],

    })

    fig = px.bar(

    benchmark_chart,

    x="Dif %",

    y="Métrica",

    orientation="h",

    color="Dif %",

    color_continuous_scale="RdYlGn"

    )

    fig.update_layout(
    height=350,
    coloraxis_showscale=False
)

    st.plotly_chart(
    fig,
    use_container_width=True
)

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

    distance_diff = (
    (player_distance / pos_avg_distance) - 1
    ) * 100

    hsr_diff = (
    (player_hsr / pos_avg_hsr) - 1
) * 100

    sprint_diff = (
    (player_sprints / pos_avg_sprints) - 1
) * 100

    acc_diff = (
    (player_acc / pos_avg_acc) - 1
) * 100

    load_diff = (
    (player_load / pos_avg_load) - 1
) * 100

    st.write(
    f"Posición: {player_position}"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
    "Distancia",
    f"{distance_diff:+.1f}%"
    )

    c2.metric(
    "HSR",
    f"{hsr_diff:+.1f}%"
    )

    c3.metric(
    "Sprints",
    f"{sprint_diff:+.1f}%"
    )

    c4.metric(
    "Aceleraciones",
    f"{acc_diff:+.1f}%"
    )

    c5.metric(
    "Player Load",
    f"{load_diff:+.1f}%"
    )

    position_chart = pd.DataFrame({

    "Métrica": [
        "Distancia",
        "HSR",
        "Sprints",
        "Aceleraciones",
        "Player Load"
    ],

    "Dif %": [
        distance_diff,
        hsr_diff,
        sprint_diff,
        acc_diff,
        load_diff
    ]

    })

    fig = px.bar(

    position_chart,

    x="Dif %",

    y="Métrica",

    orientation="h",

    color="Dif %",

    color_continuous_scale="RdYlGn"

    )

    fig.add_vline(
    x=0,
    line_dash="dash"
    )

    fig.update_layout(
    height=350,
    coloraxis_showscale=False
    )

    st.plotly_chart(
    fig,
    use_container_width=True
    )

    st.subheader("Percentiles por Posición")

    distance_pct = percentileofscore(
    position_df["distance_m"],
    player_distance
)

    hsr_pct = percentileofscore(
    position_df["abs_hsr_m"],
    player_hsr
)

    sprint_pct = percentileofscore(
    position_df["sprints_abs_count"],
    player_sprints
)

    acc_pct = percentileofscore(
    position_df["accelerations_count"],
    player_acc
)

    load_pct = percentileofscore(
    position_df["player_load_a_u"],
    player_load
)
    
    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric("Distancia", f"P{distance_pct:.0f}")
    c2.metric("HSR", f"P{hsr_pct:.0f}")
    c3.metric("Sprints", f"P{sprint_pct:.0f}")
    c4.metric("Aceleraciones", f"P{acc_pct:.0f}")
    c5.metric("Player Load", f"P{load_pct:.0f}")
    
    st.subheader("Ranking en su Posición")
    num_players = position_df["player"].nunique()
    rank_distance = (
    position_df
    .groupby("player")["distance_m"]
    .mean()
    .rank(ascending=False)
)

    rank_hsr = (
    position_df
    .groupby("player")["abs_hsr_m"]
    .mean()
    .rank(ascending=False)
)

    rank_sprints = (
    position_df
    .groupby("player")["sprints_abs_count"]
    .mean()
    .rank(ascending=False)
)

    rank_acc = (
    position_df
    .groupby("player")["accelerations_count"]
    .mean()
    .rank(ascending=False)
)

    rank_load = (
    position_df
    .groupby("player")["player_load_a_u"]
    .mean()
    .rank(ascending=False)
)
    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric(
    "Distancia",
    f"{int(rank_distance[player])}/{num_players}"
)

    c2.metric(
    "HSR",
    f"{int(rank_hsr[player])}/{num_players}"
)

    c3.metric(
    "Sprints",
    f"{int(rank_sprints[player])}/{num_players}"
    )

    c4.metric(
    "Aceleraciones",
    f"{int(rank_acc[player])}/{num_players}"
)

    c5.metric(
    "Player Load",
    f"{int(rank_load[player])}/{num_players}"
)
    
    st.subheader("Radar Posicional")

    radar_position = pd.DataFrame({

    "Métrica":[
        "Distancia",
        "HSR",
        "Sprints",
        "Aceleraciones",
        "Player Load"
    ],

    "Jugador":[
        player_distance,
        player_hsr,
        player_sprints,
        player_acc,
        player_load
    ],

    "Posición":[
        pos_avg_distance,
        pos_avg_hsr,
        pos_avg_sprints,
        pos_avg_acc,
        pos_avg_load
    ]})

    radar_position["Jugador Norm"] = (
    radar_position["Jugador"]
    /
    radar_position["Posición"]
    ) * 100

    radar_position["Posición Norm"] = 100
    
    fig = go.Figure()

    fig.add_trace(
    go.Scatterpolar(
        r=radar_position["Jugador Norm"],
        theta=radar_position["Métrica"],
        fill="toself",
        name=player
    )
)

    fig.add_trace(
    go.Scatterpolar(
        r=radar_position["Posición Norm"],
        theta=radar_position["Métrica"],
        fill="toself",
        name="Media Posición"
    )
)

    fig.update_layout(
    height=600
)

    st.plotly_chart(
    fig,
    use_container_width=True
)

    st.divider()



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

with tab4:

    # =====================================================
    # RISK GAUGE
    # =====================================================

    st.subheader("Índice Global de Riesgo")

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

    st.divider()

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

    st.divider()

    # =====================================================
    # INTERPRETACIÓN AUTOMÁTICA
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

        st.info(
            "El riesgo actual está impulsado por: "
            + ", ".join(drivers)
            + "."
        )

    st.divider()

    # =====================================================
    # RECOMENDACIÓN OPERATIVA
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

with tab5:

    # =====================================================
    # SESSION BENCHMARKING
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

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # SESSION SCORE
    # =====================================================

    comparison["Ratio"] = (
        comparison["Sesión"]
        /
        comparison["Media Posición"]
    )

    session_score = round(
        comparison["Ratio"].mean() * 50,
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
        "Índice de Exigencia",
        f"{session_score:.0f}/100"
    )

    if session_score < 40:

        st.success(
            "Sesión ligera respecto a su posición."
        )

    elif session_score < 70:

        st.info(
            "Sesión dentro de parámetros normales."
        )

    elif session_score < 85:

        st.warning(
            "Sesión exigente respecto a su posición."
        )

    else:

        st.error(
            "Sesión muy exigente respecto a su posición."
        )

    st.divider()

    # =====================================================
    # SESSION RADAR
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
            name="Sesión"
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
        f"La principal diferencia respecto a su posición se observa en "
        f"{top_metric} ({top_diff:+.1f}%)."
    )

    st.divider()

    # =====================================================
    # TOP SESIONES
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
    # SESSION SUMMARY
    # =====================================================

    st.subheader(
        "Resumen de la Sesión"
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    summary_col1.metric(
    "Distancia",
    f"{selected_row['distance_m']:.0f} m"
    )

    summary_col2.metric(
    "HSR",
    f"{selected_row['abs_hsr_m']:.0f} m"
    )   

    summary_col3.metric(
    "Player Load",
    f"{selected_row['player_load_a_u']:.1f}"
    )

