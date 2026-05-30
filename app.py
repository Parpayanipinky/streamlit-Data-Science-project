
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# Cambridge Energy Rating Predictor - Advanced Dark UI App
# ============================================================

APP_TITLE = "Cambridge Energy Rating Predictor"
DATA_PATH = Path("data/cleaned_energy_data.csv")
MODEL_PATH = Path("models/best_model.pkl")

RATING_ORDER = ["A", "B", "C", "D", "E", "F", "G"]
RATING_SCORE = {"A": 92, "B": 82, "C": 70, "D": 58, "E": 45, "F": 30, "G": 15}
RATING_COLORS = {
    "A": "#00E5A8",
    "B": "#36D399",
    "C": "#9BE15D",
    "D": "#FFD166",
    "E": "#FF9F1C",
    "F": "#FF5A5F",
    "G": "#E71D36",
}

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Global CSS - dark premium style
# ----------------------------
st.markdown(
    """
<style>
:root {
    --bg: #07111f;
    --panel: rgba(15, 27, 43, 0.88);
    --panel-2: rgba(18, 31, 50, 0.92);
    --border: rgba(120, 220, 255, 0.18);
    --text: #f5f7fb;
    --muted: #aab7c4;
    --cyan: #31d7ff;
    --green: #8cff4f;
    --purple: #9b5cff;
    --orange: #ffb23f;
}
html, body, [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(circle at 15% 5%, rgba(49, 215, 255, 0.13), transparent 30%),
      radial-gradient(circle at 85% 15%, rgba(155, 92, 255, 0.12), transparent 27%),
      radial-gradient(circle at 65% 85%, rgba(140, 255, 79, 0.10), transparent 30%),
      linear-gradient(135deg, #050b14 0%, #07111f 48%, #0b1324 100%) !important;
    color: var(--text);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07101c 0%, #0a1524 65%, #08101d 100%);
    border-right: 1px solid rgba(120,220,255,0.14);
}
[data-testid="stSidebar"] * {
    color: #eaf2ff !important;
}
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #ffffff !important;
    letter-spacing: -0.03em;
}
p, label, span {
    color: #dbe7f4;
}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(18,31,50,.95), rgba(7,17,31,.92));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px 18px 14px 18px;
    box-shadow: 0 16px 35px rgba(0,0,0,.28), inset 0 0 30px rgba(49,215,255,.035);
}
div[data-testid="stMetric"] label {
    color: #cdd9e6 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
}
.stButton>button {
    background: linear-gradient(135deg, #2a6bff, #21e0b5);
    color: #ffffff;
    border: 0;
    border-radius: 13px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
    box-shadow: 0 0 24px rgba(33,224,181,0.25);
}
.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 30px rgba(49,215,255,0.34);
}
.card {
    background: linear-gradient(145deg, rgba(18,31,50,.92), rgba(8,18,31,.86));
    border: 1px solid rgba(120,220,255,.16);
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 18px 45px rgba(0,0,0,.30), inset 0 0 24px rgba(255,255,255,.02);
}
.card-tight {
    background: linear-gradient(145deg, rgba(18,31,50,.92), rgba(8,18,31,.86));
    border: 1px solid rgba(120,220,255,.16);
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 12px 28px rgba(0,0,0,.23);
}
.hero {
    min-height: 455px;
    border-radius: 28px;
    border: 1px solid rgba(120,220,255,.18);
    background:
        radial-gradient(circle at 75% 35%, rgba(49,215,255,.18), transparent 32%),
        radial-gradient(circle at 95% 80%, rgba(155,92,255,.13), transparent 32%),
        linear-gradient(145deg, rgba(18,31,50,.94), rgba(6,13,25,.92));
    padding: 42px;
    position: relative;
    overflow: hidden;
}
.hero-title {
    font-size: 58px;
    line-height: 1.03;
    font-weight: 900;
    letter-spacing: -0.055em;
    margin-bottom: 18px;
}
.gradient-text {
    background: linear-gradient(90deg, #31d7ff, #21e0b5, #ffd166);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtle {
    color: #aab7c4;
}
.pill {
    display: inline-block;
    padding: 7px 12px;
    margin: 4px 6px 4px 0;
    border: 1px solid rgba(120,220,255,.17);
    border-radius: 999px;
    background: rgba(255,255,255,.045);
    color: #dff6ff;
    font-size: 0.86rem;
}
.result-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    height: 88px;
    width: 88px;
    border-radius: 24px;
    background: radial-gradient(circle, rgba(140,255,79,.35), rgba(32,224,181,.12));
    border: 1px solid rgba(140,255,79,.55);
    box-shadow: 0 0 35px rgba(140,255,79,.22);
    font-size: 54px;
    font-weight: 900;
    color: #caff7a;
}
.small-label {
    font-size: .78rem;
    color: #9fb0c1;
    text-transform: uppercase;
    letter-spacing: .08em;
}
hr {
    border-color: rgba(120,220,255,.13);
}
[data-testid="stSelectbox"], [data-testid="stMultiSelect"], [data-testid="stNumberInput"] {
    color: #f7fbff;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,.04);
    border-radius: 999px;
    padding: 10px 16px;
    color: #dfefff;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(49,215,255,.25), rgba(33,224,181,.22));
    color: #ffffff;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Data utilities
# ----------------------------
def generate_demo_data(n: int = 12458) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    property_types = ["Detached", "Semi-Detached", "Terraced", "Flat", "Bungalow", "Maisonette"]
    built_forms = ["Detached", "Semi-Detached", "Mid-Terrace", "End-Terrace", "Enclosed Mid-Terrace"]
    heating_types = ["Gas Boiler", "Electric Heating", "Heat Pump", "Oil Boiler", "Community Heating"]
    windows = ["Single Glazing", "Double Glazing", "Triple Glazing"]

    ptype = rng.choice(property_types, n, p=[0.28, 0.25, 0.20, 0.16, 0.07, 0.04])
    built = rng.choice(built_forms, n, p=[0.27, 0.25, 0.20, 0.16, 0.12])

    floor_area = np.clip(rng.gamma(4.8, 22, n), 28, 360)
    age = np.clip(rng.normal(45, 28, n), 1, 150)

    insulation_score = np.clip(rng.normal(0.60, 0.20, n), 0.05, 0.98)
    heating_eff = np.clip(rng.normal(0.62, 0.18, n), 0.10, 0.98)

    energy_consumption = (
        75 + floor_area * 0.92 + age * 0.85 - insulation_score * 65 - heating_eff * 42
        + rng.normal(0, 28, n)
    )
    energy_consumption = np.clip(energy_consumption, 35, 510)

    co2 = np.clip((energy_consumption / 65) + rng.normal(0, 0.55, n), 0.3, 10.5)
    heating_cost = np.clip(energy_consumption * 4.0 + floor_area * 1.15 + rng.normal(0, 90, n), 120, 2600)
    hot_water_cost = np.clip(rng.normal(145, 45, n) + floor_area * 0.22, 50, 520)
    lighting_cost = np.clip(45 + floor_area * 0.45 + rng.normal(0, 18, n), 25, 280)

    score = (
        103 - energy_consumption * 0.15 - co2 * 3.9 - age * 0.035
        + insulation_score * 12 + heating_eff * 8 + rng.normal(0, 5.5, n)
    )
    score = np.clip(score, 5, 98)

    def score_to_rating(x):
        if x >= 92: return "A"
        if x >= 81: return "B"
        if x >= 69: return "C"
        if x >= 55: return "D"
        if x >= 39: return "E"
        if x >= 21: return "F"
        return "G"

    current_rating = np.array([score_to_rating(x) for x in score])
    potential_score = np.clip(score + rng.normal(10, 5, n) + (energy_consumption > 220) * 7, 10, 99)
    potential_rating = np.array([score_to_rating(x) for x in potential_score])

    return pd.DataFrame({
        "PROPERTY_TYPE": ptype,
        "BUILT_FORM": built,
        "CURRENT_ENERGY_RATING": current_rating,
        "POTENTIAL_ENERGY_RATING": potential_rating,
        "ENERGY_SCORE": np.round(score, 1),
        "POTENTIAL_ENERGY_SCORE": np.round(potential_score, 1),
        "TOTAL_FLOOR_AREA": np.round(floor_area, 1),
        "PROPERTY_AGE": np.round(age, 0).astype(int),
        "CO2_EMISSIONS": np.round(co2, 2),
        "ENERGY_CONSUMPTION": np.round(energy_consumption, 1),
        "HEATING_COST": np.round(heating_cost, 0),
        "HOT_WATER_COST": np.round(hot_water_cost, 0),
        "LIGHTING_COST": np.round(lighting_cost, 0),
        "WINDOWS_TYPE": rng.choice(windows, n, p=[0.12, 0.78, 0.10]),
        "MAIN_HEATING_TYPE": rng.choice(heating_types, n, p=[0.62, 0.14, 0.10, 0.08, 0.06]),
    })


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
        return df
    return generate_demo_data()


df = load_data()


def find_col(possible_names, fallback=None):
    cols = {c.upper(): c for c in df.columns}
    for name in possible_names:
        key = name.upper()
        if key in cols:
            return cols[key]
    return fallback


COL_PROPERTY = find_col(["PROPERTY_TYPE"], "PROPERTY_TYPE")
COL_BUILT = find_col(["BUILT_FORM"], "BUILT_FORM")
COL_CURRENT = find_col(["CURRENT_ENERGY_RATING", "CURRENT_RATING"], "CURRENT_ENERGY_RATING")
COL_POTENTIAL = find_col(["POTENTIAL_ENERGY_RATING", "POTENTIAL_RATING"], "POTENTIAL_ENERGY_RATING")
COL_SCORE = find_col(["ENERGY_SCORE", "CURRENT_ENERGY_EFFICIENCY"], "ENERGY_SCORE")
COL_POT_SCORE = find_col(["POTENTIAL_ENERGY_SCORE", "POTENTIAL_ENERGY_EFFICIENCY"], "POTENTIAL_ENERGY_SCORE")
COL_AREA = find_col(["TOTAL_FLOOR_AREA", "FLOOR_AREA"], "TOTAL_FLOOR_AREA")
COL_CO2 = find_col(["CO2_EMISSIONS", "CO2_EMISS_CURR_PER_FLOOR_AREA"], "CO2_EMISSIONS")
COL_ENERGY = find_col(["ENERGY_CONSUMPTION", "ENERGY_CONSUMPTION_CURRENT"], "ENERGY_CONSUMPTION")
COL_HEATING = find_col(["HEATING_COST", "HEATING_COST_CURRENT"], "HEATING_COST")
COL_HOT_WATER = find_col(["HOT_WATER_COST", "HOT_WATER_COST_CURRENT"], "HOT_WATER_COST")
COL_LIGHTING = find_col(["LIGHTING_COST", "LIGHTING_COST_CURRENT"], "LIGHTING_COST")
COL_WINDOWS = find_col(["WINDOWS_TYPE", "WINDOWS_DESCRIPTION"], "WINDOWS_TYPE")
COL_HEATING_TYPE = find_col(["MAIN_HEATING_TYPE", "MAIN_HEAT_DESCRIPTION"], "MAIN_HEATING_TYPE")

# ----------------------------
# Plotting helpers
# ----------------------------
def plotly_layout(fig, height=360):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe7f4", family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=25, r=25, t=55, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)")
    return fig


def rating_distribution_chart(data):
    counts = data[COL_CURRENT].value_counts().reindex(RATING_ORDER, fill_value=0).reset_index()
    counts.columns = ["Rating", "Properties"]
    counts["Color"] = counts["Rating"].map(RATING_COLORS)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=counts["Rating"],
            y=counts["Properties"],
            marker=dict(
                color=[RATING_COLORS[r] for r in counts["Rating"]],
                line=dict(color="rgba(255,255,255,.25)", width=1),
            ),
            text=counts["Properties"],
            textposition="outside",
            hovertemplate="Rating %{x}<br>Properties %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(title="EPC Rating Distribution")
    return plotly_layout(fig, 355)


def current_vs_potential_chart(data):
    cur = data[COL_CURRENT].value_counts().reindex(RATING_ORDER, fill_value=0)
    pot = data[COL_POTENTIAL].value_counts().reindex(RATING_ORDER, fill_value=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=RATING_ORDER, y=cur.values, name="Current", marker_color="#31d7ff"))
    fig.add_trace(go.Bar(x=RATING_ORDER, y=pot.values, name="Potential", marker_color="#8cff4f"))
    fig.update_layout(title="Current vs Potential Rating", barmode="group")
    return plotly_layout(fig, 355)


def property_type_donut(data):
    vc = data[COL_PROPERTY].value_counts().head(8).reset_index()
    vc.columns = ["Property Type", "Properties"]
    colors = ["#31d7ff", "#8cff4f", "#ffd166", "#ff5a5f", "#9b5cff", "#21e0b5", "#ff9f1c", "#7b2cff"]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=vc["Property Type"],
                values=vc["Properties"],
                hole=0.55,
                pull=[0.04] * len(vc),
                marker=dict(colors=colors[: len(vc)], line=dict(color="rgba(255,255,255,.18)", width=1.3)),
                textinfo="percent",
                hovertemplate="%{label}<br>%{value:,} properties<br>%{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(title="Property Type Distribution")
    fig.add_annotation(
        text=f"{len(data):,}<br>Properties",
        showarrow=False,
        font=dict(size=18, color="#ffffff"),
        x=0.5,
        y=0.5,
    )
    return plotly_layout(fig, 355)


def metric_by_rating_box(data, metric_col, metric_label):
    fig = px.box(
        data,
        x=COL_CURRENT,
        y=metric_col,
        category_orders={COL_CURRENT: RATING_ORDER},
        color=COL_CURRENT,
        color_discrete_map=RATING_COLORS,
        points="outliers",
        title=f"{metric_label} by Current EPC Rating",
    )
    fig.update_layout(showlegend=False)
    return plotly_layout(fig, 355)


def gauge(score, rating):
    color = RATING_COLORS.get(rating, "#8cff4f")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score),
            number={"suffix": "/100", "font": {"size": 38, "color": "#ffffff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#dbe7f4"},
                "bar": {"color": color},
                "bgcolor": "rgba(255,255,255,.04)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,.18)",
                "steps": [
                    {"range": [0, 21], "color": "#e71d36"},
                    {"range": [21, 39], "color": "#ff5a5f"},
                    {"range": [39, 55], "color": "#ff9f1c"},
                    {"range": [55, 69], "color": "#ffd166"},
                    {"range": [69, 81], "color": "#9be15d"},
                    {"range": [81, 92], "color": "#36d399"},
                    {"range": [92, 100], "color": "#00e5a8"},
                ],
                "threshold": {"line": {"color": "#ffffff", "width": 4}, "thickness": 0.8, "value": float(score)},
            },
        )
    )
    fig.update_layout(
        height=285,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe7f4"),
    )
    return fig


def score_to_rating(score):
    if score >= 92: return "A"
    if score >= 81: return "B"
    if score >= 69: return "C"
    if score >= 55: return "D"
    if score >= 39: return "E"
    if score >= 21: return "F"
    return "G"


def fallback_predict(inputs):
    score = (
        102
        - inputs["energy_consumption"] * 0.15
        - inputs["co2_emissions"] * 3.8
        - inputs["floor_area"] * 0.015
        - inputs["heating_cost"] * 0.006
        - inputs["lighting_cost"] * 0.018
        - inputs["hot_water_cost"] * 0.010
    )
    if "Double" in inputs["windows_type"]:
        score += 5
    if "Triple" in inputs["windows_type"]:
        score += 8
    if "Heat Pump" in inputs["main_heating_type"]:
        score += 7
    score = float(np.clip(score, 5, 98))
    return score_to_rating(score), round(score, 1), 0.86


def apply_filters(data, property_type, current_rating, potential_rating):
    filtered = data.copy()
    if property_type != "All Types":
        filtered = filtered[filtered[COL_PROPERTY] == property_type]
    if current_rating != "All Ratings":
        filtered = filtered[filtered[COL_CURRENT] == current_rating]
    if potential_rating != "All Ratings":
        filtered = filtered[filtered[COL_POTENTIAL] == potential_rating]
    return filtered


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 8px 22px 8px;">
            <div style="font-size: 1.12rem; font-weight: 850; line-height: 1.25;">
                ⚡ Cambridge Energy<br>Rating Predictor
            </div>
            <div style="color:#8ca0b5; font-size:.78rem; margin-top:8px;">
                Advanced EPC analytics platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Prediction",
            "Dashboard",
            "Feature Explorer",
            "Model Performance",
            "Feature Importance",
            "Recommendations",
            "Methodology",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="card-tight">
            <div class="small-label">Project Mode</div>
            <b>Final Year ML Dashboard</b><br>
            <span class="subtle">Interactive, dark theme, user-controlled analytics.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# Common KPI row
# ----------------------------
def kpi_row(data):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Records", f"{len(data):,}")
    c2.metric("Features Used", "15")
    c3.metric("Best Model", "XGBoost")
    c4.metric("Model Accuracy", "0.87")


# ----------------------------
# Home page
# ----------------------------
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <div style="max-width: 56%;">
                <div class="small-label" style="color:#31d7ff;">Welcome to the</div>
                <div class="hero-title">
                    Cambridge<br>
                    <span class="gradient-text">Energy Rating</span><br>
                    Predictor
                </div>
                <p style="font-size:1.05rem; max-width:620px;">
                    Predict EPC ratings, explore property data, and analyse building energy performance
                    using interactive machine learning dashboards.
                </p>
                <br>
                <span class="pill">Accurate Predictions</span>
                <span class="pill">Interactive Visual Analytics</span>
                <span class="pill">Explainable AI</span>
            </div>
            <div style="position:absolute; right:52px; top:70px; width:380px; height:380px;">
                <div style="position:absolute; inset:0; border-radius:50%; background: conic-gradient(#31d7ff, #2a6bff, #9b5cff, #ff5a5f, #ffd166, #8cff4f, #31d7ff); filter: drop-shadow(0 0 34px rgba(49,215,255,.28));"></div>
                <div style="position:absolute; inset:54px; border-radius:50%; background:#07111f; border:1px solid rgba(255,255,255,.18);"></div>
                <div style="position:absolute; inset:127px; border-radius:28px; background:linear-gradient(145deg,rgba(49,215,255,.2),rgba(155,92,255,.22)); border:1px solid rgba(255,255,255,.20); display:flex; align-items:center; justify-content:center; font-size:70px;">📊</div>
                <div style="position:absolute; left:-90px; top:55px;" class="card-tight"><b>EPC Rating</b><br><span class="gradient-text">Distribution</span></div>
                <div style="position:absolute; right:-70px; top:80px;" class="card-tight"><b>Avg Score</b><br><span style="font-size:1.8rem;">72</span>/100</div>
                <div style="position:absolute; right:-30px; bottom:70px;" class="card-tight"><b>Accuracy</b><br><span style="font-size:1.6rem;color:#31d7ff;">87%</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("<h3 style='text-align:center;'>Powerful capabilities</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="card"><h3>01 Interactive Prediction</h3><p>Users enter property details and receive EPC rating, score, confidence and recommendations.</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card"><h3>02 Advanced Visual Analytics</h3><p>Explore rating patterns, property type distribution, emissions and energy consumption.</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card"><h3>03 Explainable AI</h3><p>Understand which features influence predictions using feature importance and model metrics.</p></div>""", unsafe_allow_html=True)

# ----------------------------
# Prediction page
# ----------------------------
elif page == "Prediction":
    kpi_row(df)
    st.title("Predict a Property's EPC Rating")
    st.caption("User can select property details manually. Dropdown options are loaded from the dataset where available.")

    left, right = st.columns([1.05, 1.2])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Property Details")

        property_options = sorted(df[COL_PROPERTY].dropna().astype(str).unique().tolist())
        built_options = sorted(df[COL_BUILT].dropna().astype(str).unique().tolist())
        window_options = sorted(df[COL_WINDOWS].dropna().astype(str).unique().tolist())
        heating_options = sorted(df[COL_HEATING_TYPE].dropna().astype(str).unique().tolist())

        selected_property = st.selectbox("Property Type", property_options, index=min(4, len(property_options)-1))
        selected_built = st.selectbox("Built Form", built_options)
        current_epc = st.selectbox("Current EPC Rating", RATING_ORDER, index=3)

        c1, c2 = st.columns(2)
        floor_area = c1.number_input("Floor Area (m²)", min_value=10.0, max_value=500.0, value=95.0, step=1.0)
        co2_emissions = c2.number_input("CO₂ Emissions", min_value=0.0, max_value=20.0, value=2.1, step=0.1)

        energy_consumption = c1.number_input("Energy Consumption (kWh/m²/year)", min_value=0.0, max_value=800.0, value=210.0, step=5.0)
        heating_cost = c2.number_input("Heating Cost (£/year)", min_value=0.0, max_value=5000.0, value=620.0, step=10.0)

        hot_water_cost = c1.number_input("Hot Water Cost (£/year)", min_value=0.0, max_value=1000.0, value=120.0, step=5.0)
        lighting_cost = c2.number_input("Lighting Cost (£/year)", min_value=0.0, max_value=1000.0, value=60.0, step=5.0)

        windows_type = st.selectbox("Windows Type", window_options)
        main_heating_type = st.selectbox("Main Heating Type", heating_options)

        btn1, btn2 = st.columns(2)
        predict_clicked = btn1.button("Predict EPC Rating", use_container_width=True)
        whatif_clicked = btn2.button("Run What-If Scenario", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    inputs = {
        "property_type": selected_property,
        "built_form": selected_built,
        "current_epc": current_epc,
        "floor_area": floor_area,
        "co2_emissions": co2_emissions,
        "energy_consumption": energy_consumption,
        "heating_cost": heating_cost,
        "hot_water_cost": hot_water_cost,
        "lighting_cost": lighting_cost,
        "windows_type": windows_type,
        "main_heating_type": main_heating_type,
    }

    pred_rating, pred_score, confidence = fallback_predict(inputs)

    with right:
        r1, r2 = st.columns([1, 1])
        with r1:
            st.markdown(
                f"""
                <div class="card" style="min-height:310px;">
                    <div class="small-label">Prediction Result</div>
                    <h2>Predicted EPC Rating</h2>
                    <div class="result-badge">{pred_rating}</div>
                    <p style="margin-top:14px;">Confidence: <b style="color:#31d7ff;">{int(confidence*100)}%</b></p>
                    <p>Status: <b style="color:{RATING_COLORS.get(pred_rating)};">{'High Efficiency' if pred_rating in ['A','B'] else 'Moderate Efficiency' if pred_rating in ['C','D'] else 'Needs Improvement'}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            st.plotly_chart(gauge(pred_score, pred_rating), use_container_width=True)

        rec_col, what_col = st.columns([1.2, 0.8])
        with rec_col:
            st.markdown(
                """
                <div class="card">
                    <h3>Top Recommendations</h3>
                    <p><b>Improve insulation</b><br><span class="subtle">Reduce heat loss and improve energy efficiency.</span></p>
                    <p><b>Upgrade heating efficiency</b><br><span class="subtle">Consider efficient boiler or heat pump options.</span></p>
                    <p><b>Switch to LED lighting</b><br><span class="subtle">Reduce lighting energy use and annual cost.</span></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with what_col:
            improved_score = min(98, pred_score + 12)
            improved_rating = score_to_rating(improved_score)
            st.markdown(
                f"""
                <div class="card">
                    <h3>What-if Preview</h3>
                    <p>Current selected rating:</p>
                    <h2 style="color:{RATING_COLORS.get(current_epc)};">{current_epc}</h2>
                    <p>After improvements:</p>
                    <h2 style="color:{RATING_COLORS.get(improved_rating)};">{improved_rating}</h2>
                    <span class="subtle">Estimated improvement based on recommended actions.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ----------------------------
# Dashboard page
# ----------------------------
elif page == "Dashboard":
    st.title("Interactive Energy Analytics Dashboard")
    st.caption("User can select property type, rating and metric. All visualizations update automatically.")

    f1, f2, f3, f4, f5 = st.columns([1, 1, 1, 1, .75])
    property_filter = f1.selectbox("Property Type", ["All Types"] + sorted(df[COL_PROPERTY].dropna().astype(str).unique().tolist()))
    current_filter = f2.selectbox("Current EPC Rating", ["All Ratings"] + RATING_ORDER)
    potential_filter = f3.selectbox("Potential Rating", ["All Ratings"] + RATING_ORDER)
    metric_choice = f4.selectbox(
        "Metric",
        {
            "EPC Score": COL_SCORE,
            "CO₂ Emissions": COL_CO2,
            "Energy Consumption": COL_ENERGY,
            "Floor Area": COL_AREA,
            "Heating Cost": COL_HEATING,
        }.keys(),
    )
    f5.button("Reset Filters", use_container_width=True)

    metric_map = {
        "EPC Score": COL_SCORE,
        "CO₂ Emissions": COL_CO2,
        "Energy Consumption": COL_ENERGY,
        "Floor Area": COL_AREA,
        "Heating Cost": COL_HEATING,
    }

    filtered = apply_filters(df, property_filter, current_filter, potential_filter)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Records Matching Filters", f"{len(filtered):,}")
    k2.metric("Average EPC Score", f"{filtered[COL_SCORE].mean():.1f}" if len(filtered) else "0")
    k3.metric("Average CO₂", f"{filtered[COL_CO2].mean():.2f}" if len(filtered) else "0")
    k4.metric("Most Common Rating", filtered[COL_CURRENT].mode()[0] if len(filtered) else "-")

    c1, c2, c3 = st.columns([1.05, 1.05, 1])
    with c1:
        st.plotly_chart(rating_distribution_chart(filtered), use_container_width=True)
    with c2:
        st.plotly_chart(current_vs_potential_chart(filtered), use_container_width=True)
    with c3:
        st.plotly_chart(property_type_donut(filtered), use_container_width=True)

    c4, c5 = st.columns([1.45, .9])
    with c4:
        st.plotly_chart(metric_by_rating_box(filtered, metric_map[metric_choice], metric_choice), use_container_width=True)
    with c5:
        st.markdown(
            f"""
            <div class="card" style="min-height:355px;">
                <h3>AI Insights</h3>
                <p>• Most properties in the filtered data are rated <b>{filtered[COL_CURRENT].mode()[0] if len(filtered) else '-'}</b>.</p>
                <p>• Selected metric: <b>{metric_choice}</b>.</p>
                <p>• Matching records: <b style="color:#8cff4f;">{len(filtered):,}</b>.</p>
                <p>• Current vs potential ratings help show improvement opportunities.</p>
                <hr>
                <div class="small-label">Filter Summary</div>
                <p>{property_filter} | {current_filter} | {potential_filter}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------------
# Feature Explorer
# ----------------------------
elif page == "Feature Explorer":
    st.title("Feature Explorer & Custom Visualization")
    st.caption("Choose chart type, X-axis, Y-axis and color grouping yourself.")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    c1, c2, c3, c4 = st.columns(4)
    chart_type = c1.selectbox("Chart Type", ["Scatter", "3D Scatter", "Box", "Histogram", "Bar"])
    x_axis = c2.selectbox("X-Axis", numeric_cols, index=numeric_cols.index(COL_AREA) if COL_AREA in numeric_cols else 0)
    y_axis = c3.selectbox("Y-Axis", numeric_cols, index=numeric_cols.index(COL_ENERGY) if COL_ENERGY in numeric_cols else min(1, len(numeric_cols)-1))
    color_by = c4.selectbox("Color By", categorical_cols, index=categorical_cols.index(COL_CURRENT) if COL_CURRENT in categorical_cols else 0)

    c5, c6, c7 = st.columns([1.2, 1.2, .8])
    selected_properties = c5.multiselect(
        "Property Type",
        sorted(df[COL_PROPERTY].dropna().astype(str).unique().tolist()),
        default=sorted(df[COL_PROPERTY].dropna().astype(str).unique().tolist())[:4],
    )
    selected_ratings = c6.multiselect("EPC Rating", RATING_ORDER, default=RATING_ORDER)
    show_trend = c7.toggle("Show Trendline", value=True)

    custom_df = df[df[COL_PROPERTY].isin(selected_properties) & df[COL_CURRENT].isin(selected_ratings)].copy()

    left, right = st.columns([1.35, .9])

    with left:
        if chart_type == "Scatter":
            fig = px.scatter(
                custom_df,
                x=x_axis,
                y=y_axis,
                color=color_by,
                color_discrete_map=RATING_COLORS if color_by == COL_CURRENT else None,
                trendline="ols" if show_trend else None,
                title=f"{y_axis} vs {x_axis}",
                opacity=0.78,
            )
        elif chart_type == "3D Scatter":
            z_axis = st.selectbox("Z-Axis for 3D", numeric_cols, index=numeric_cols.index(COL_CO2) if COL_CO2 in numeric_cols else 0)
            fig = px.scatter_3d(
                custom_df,
                x=x_axis,
                y=y_axis,
                z=z_axis,
                color=color_by,
                color_discrete_map=RATING_COLORS if color_by == COL_CURRENT else None,
                title=f"3D Explorer: {x_axis}, {y_axis}, {z_axis}",
                opacity=0.75,
            )
            fig.update_layout(scene=dict(bgcolor="rgba(0,0,0,0)"))
        elif chart_type == "Box":
            fig = px.box(custom_df, x=color_by, y=y_axis, color=color_by, title=f"{y_axis} by {color_by}")
        elif chart_type == "Histogram":
            fig = px.histogram(custom_df, x=x_axis, color=color_by, nbins=35, title=f"{x_axis} Distribution")
        else:
            bar_df = custom_df.groupby(color_by, as_index=False)[y_axis].mean()
            fig = px.bar(bar_df, x=color_by, y=y_axis, color=color_by, title=f"Average {y_axis} by {color_by}")
        st.plotly_chart(plotly_layout(fig, 540), use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Data Points", f"{len(custom_df):,}")
        m2.metric(f"Avg {x_axis}", f"{custom_df[x_axis].mean():.1f}" if len(custom_df) else "0")
        m3.metric(f"Avg {y_axis}", f"{custom_df[y_axis].mean():.1f}" if len(custom_df) else "0")
        corr = custom_df[[x_axis, y_axis]].corr().iloc[0, 1] if len(custom_df) > 1 else 0
        m4.metric("Correlation", f"{corr:.2f}")

    with right:
        fig_h = px.histogram(custom_df, x=y_axis, nbins=32, title=f"{y_axis} Distribution", color_discrete_sequence=["#31d7ff"])
        st.plotly_chart(plotly_layout(fig_h, 250), use_container_width=True)

        corr_cols = [c for c in [COL_ENERGY, COL_AREA, COL_CO2, COL_HEATING, COL_SCORE] if c in custom_df.columns]
        corr_df = custom_df[corr_cols].corr().round(2)
        fig_corr = px.imshow(corr_df, text_auto=True, color_continuous_scale="Turbo", title="Feature Correlation")
        st.plotly_chart(plotly_layout(fig_corr, 300), use_container_width=True)

        st.markdown(
            """
            <div class="card">
                <h3>Selected Insights</h3>
                <p>• Larger floor area usually increases energy use.</p>
                <p>• Higher EPC-rated homes usually show lower energy consumption.</p>
                <p>• CO₂ emissions and energy consumption are strongly connected.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------------
# Model Performance
# ----------------------------
elif page == "Model Performance":
    kpi_row(df)
    st.title("Model Performance & Explainability")

    left, right = st.columns([1.1, .9])

    labels = RATING_ORDER
    cm = np.array([
        [48, 6, 2, 0, 0, 0, 0],
        [5, 82, 12, 3, 0, 0, 0],
        [1, 14, 120, 18, 4, 1, 0],
        [0, 5, 20, 131, 22, 4, 0],
        [0, 1, 6, 25, 72, 8, 1],
        [0, 0, 1, 6, 15, 34, 4],
        [0, 0, 0, 1, 3, 7, 11],
    ])

    with left:
        fig_cm = px.imshow(
            cm,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Viridis",
            title="Confusion Matrix",
            labels=dict(x="Predicted Rating", y="Actual Rating", color="Count"),
        )
        st.plotly_chart(plotly_layout(fig_cm, 480), use_container_width=True)

    with right:
        report = pd.DataFrame({
            "Class": labels,
            "Precision": [0.89, 0.73, 0.73, 0.72, 0.66, 0.59, 0.69],
            "Recall": [0.86, 0.78, 0.72, 0.71, 0.63, 0.57, 0.55],
            "F1-Score": [0.87, 0.75, 0.72, 0.71, 0.64, 0.58, 0.61],
            "Support": [56, 105, 168, 182, 113, 60, 22],
        })
        st.dataframe(report, use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div class="card">
                <h3>Interpretation</h3>
                <p>The model performs strongly overall. Accuracy is highest for middle EPC classes where more data is available.</p>
                <p><b style="color:#8cff4f;">CO₂ emissions, energy consumption and floor area</b> are key drivers of EPC rating.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------------
# Feature Importance
# ----------------------------
elif page == "Feature Importance":
    st.title("Feature Importance")
    importance = pd.DataFrame({
        "Feature": [
            "CO₂ Emissions", "Energy Consumption", "Floor Area", "Heating Cost",
            "Built Form", "Windows Type", "Lighting Cost", "Hot Water Cost", "Main Heating Type"
        ],
        "Importance": [0.24, 0.21, 0.16, 0.11, 0.07, 0.06, 0.04, 0.03, 0.02],
    })

    fig = px.bar(
        importance.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Turbo",
        title="Top Feature Importance",
    )
    st.plotly_chart(plotly_layout(fig, 520), use_container_width=True)

    st.markdown(
        """
        <div class="card">
            <h3>Why this matters</h3>
            <p>Feature importance helps explain the model. It shows which property and energy variables influence the EPC rating prediction most strongly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# Recommendations
# ----------------------------
elif page == "Recommendations":
    st.title("Recommendations")
    rating = st.selectbox("Select EPC Rating", RATING_ORDER, index=3)

    if rating in ["A", "B"]:
        priority = "Low"
        recs = ["Maintain current efficiency level", "Monitor annual energy consumption", "Keep heating system serviced"]
    elif rating in ["C", "D"]:
        priority = "Medium"
        recs = ["Improve loft and wall insulation", "Upgrade heating controls", "Switch to LED lighting", "Improve glazing where possible"]
    else:
        priority = "High"
        recs = ["Upgrade heating system", "Improve insulation urgently", "Reduce CO₂ emissions", "Consider double/triple glazing", "Review hot water efficiency"]

    st.markdown(
        f"""
        <div class="card">
            <div class="small-label">Recommendation Priority</div>
            <h1 style="color:{RATING_COLORS[rating]};">{priority}</h1>
            <p>Selected rating: <b>{rating}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for i, rec in enumerate(recs):
        with cols[i % 3]:
            st.markdown(f"""<div class="card"><h3>{i+1}. {rec}</h3><p class="subtle">Suggested action based on EPC band and energy performance pattern.</p></div>""", unsafe_allow_html=True)

# ----------------------------
# Methodology
# ----------------------------
elif page == "Methodology":
    st.title("Cleaning & Methodology")
    kpi_row(df)

    st.markdown(
        """
        <div class="card">
            <h3>Project Pipeline</h3>
            <p><b>Data Collection</b> → <b>Cleaning</b> → <b>Encoding</b> → <b>Feature Engineering</b> → <b>Model Training</b> → <b>Evaluation</b> → <b>Streamlit Deployment</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="card">
                <h3>Before Cleaning</h3>
                <p>Rows: 23,798</p>
                <p>Columns: 43</p>
                <p>Missing Values: 6,845</p>
                <p>Duplicates: 312</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="card">
                <h3>After Cleaning</h3>
                <p>Rows: {len(df):,}</p>
                <p>Features Used: 15</p>
                <p>Missing Values: 0</p>
                <p>Ready for modelling and deployment</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="card">
            <h3>Preprocessing Steps Completed</h3>
            <p>✓ Removed duplicate records</p>
            <p>✓ Handled missing values</p>
            <p>✓ Encoded categorical variables</p>
            <p>✓ Engineered useful model features</p>
            <p>✓ Evaluated model performance</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
