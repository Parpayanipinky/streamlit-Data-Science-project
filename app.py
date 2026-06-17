
from pathlib import Path
from html import escape
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

APP_TITLE = "Cambridge Energy Rating Predictor"
STUDENT_NAME = "Pinky"
PROJECT_TITLE = "Cambridge Energy Rating Predictor - CW1 Project"
PROJECT_TYPE = "Data Science Project"
BASE_DIR = Path(__file__).parent

def resolve_project_path(*relative_paths):
    """Return the first existing project file path.
    This keeps the app usable both in the original folder structure
    and when the files are placed beside app.py.
    """
    for rel_path in relative_paths:
        candidate = BASE_DIR / rel_path
        if candidate.exists():
            return candidate
    return BASE_DIR / relative_paths[0]

FINAL_DATA_PATH = resolve_project_path("data/final_fe_data.csv", "final_fe_data.csv")
MODEL_READY_PATH = resolve_project_path("data/model_ready_data.csv", "model_ready_data.csv")
MODEL_BUNDLE_PATH = resolve_project_path("models/actual_model_bundle.joblib", "actual_model_bundle.joblib")

RATING_ORDER = ["A", "B", "C", "D", "E", "F", "G"]
RATING_COLORS = {
    "A": "#00F5D4",
    "B": "#3BE088",
    "C": "#C2F970",
    "D": "#FFD166",
    "E": "#FF9F1C",
    "F": "#FF4D6D",
    "G": "#C9184A",
}
ACCENT_COLORS = ["#7C3AED", "#5B21B6", "#DB2777", "#D97706", "#059669", "#0891B2", "#DC2626"]
DASHBOARD_PROPERTY_COLORS = ["#7C3AED", "#0891B2", "#DB2777", "#D97706", "#059669", "#5B21B6", "#DC2626"]

DASHBOARD_RATING_COLORS = {
    "A": "#008F5A",
    "B": "#56B947",
    "C": "#B7D433",
    "D": "#FFD23F",
    "E": "#F59E0B",
    "F": "#EF4444",
    "G": "#B91C1C",
}

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Premium light UI styling
# -----------------------------
st.markdown(
    """
<style>
:root {
    --bg0: #F5F3FF;
    --bg1: #EDE9FE;
    --panel: rgba(255, 255, 255, .96);
    --panel2: rgba(250, 248, 255, .98);
    --border: rgba(109, 40, 217, .16);
    --text: #1E1B4B;
    --muted: #5B4B8A;
    --blue: #7C3AED;
    --cyan: #6D28D9;
    --mint: #059669;
    --green: #16A34A;
    --purple: #5B21B6;
    --pink: #DB2777;
    --orange: #D97706;
}
html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 8% 0%, rgba(109,40,217,.10), transparent 32%),
        radial-gradient(circle at 92% 12%, rgba(139,92,246,.09), transparent 28%),
        linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 55%, #FFFFFF 100%) !important;
    color: var(--text);
}
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
    max-width: 1520px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF, #EDE9FE);
    border-right: 1px solid rgba(109,40,217,.14);
}
[data-testid="stSidebar"] * {
    color: #1E1B4B !important;
}
h1, h2, h3 {
    color: #1E1B4B !important;
    letter-spacing: -0.035em;
}
p, label, span, div {
    font-family: "Inter", "Segoe UI", sans-serif;
}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #FFFFFF, #F5F3FF);
    border: 1px solid rgba(109,40,217,.13);
    border-radius: 19px;
    padding: 17px 18px;
    box-shadow: 0 18px 45px rgba(109,40,217,.08), inset 0 0 28px rgba(255,255,255,.65);
}
div[data-testid="stMetric"] label { color: #5B4B8A !important; }
div[data-testid="stMetricValue"] { color: #1E1B4B !important; }
.stButton > button {
    border: 0;
    border-radius: 14px;
    padding: .65rem 1.05rem;
    color: white;
    font-weight: 800;
    background: linear-gradient(135deg, #7C3AED, #5B21B6);
    box-shadow: 0 8px 22px rgba(109,40,217,.22);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 26px rgba(109,40,217,.32);
}
.card {
    background: linear-gradient(145deg, #FFFFFF, #F5F3FF);
    border: 1px solid rgba(109,40,217,.14);
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 18px 46px rgba(109,40,217,.07), inset 0 0 30px rgba(255,255,255,.75);
}
.card-small {
    background: linear-gradient(145deg, #FFFFFF, #F5F3FF);
    border: 1px solid rgba(109,40,217,.14);
    border-radius: 18px;
    padding: 15px 16px;
    box-shadow: 0 12px 28px rgba(109,40,217,.07);
}
.hero {
    min-height: 560px;
    border-radius: 32px;
    border: 1px solid rgba(109,40,217,.14);
    background:
        radial-gradient(circle at 73% 34%, rgba(109,40,217,.11), transparent 34%),
        radial-gradient(circle at 95% 74%, rgba(139,92,246,.09), transparent 30%),
        linear-gradient(145deg, #FFFFFF, #EDE9FE);
    overflow: hidden;
    position: relative;
    padding: 48px;
}
.hero-title {
    font-size: 62px;
    font-weight: 950;
    line-height: 1.02;
    letter-spacing: -0.06em;
    margin: 16px 0 18px;
}
.gradient-text {
    background: linear-gradient(90deg, #7C3AED, #5B21B6, #DB2777, #D97706);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.muted { color: #5B4B8A; }
.small-label {
    font-size: .76rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #7C3AED;
}
.info-panel-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(120px, 1fr));
    gap: 10px;
    max-width: 640px;
    margin-top: 18px;
}
.info-panel {
    padding: 11px 12px;
    border-radius: 12px;
    background: rgba(255,255,255,.92);
    border: 1px solid rgba(109,40,217,.16);
    border-left: 4px solid #7C3AED;
    color: #1E1B4B;
    font-size: .84rem;
    font-weight: 700;
    box-shadow: none;
}
.project-meta {
    margin-top: 14px;
    padding: 13px 16px;
    border-radius: 14px;
    background: rgba(255,255,255,.76);
    border: 1px solid rgba(109,40,217,.14);
    color: #1E1B4B;
    font-size: .96rem;
    line-height: 1.55;
}
.project-strip {
    display: inline-block;
    margin-bottom: 12px;
    padding: 8px 12px;
    border-radius: 10px;
    background: rgba(109,40,217,.10);
    color: #3B0764;
    border: 1px solid rgba(109,40,217,.18);
    font-weight: 800;
    font-size: .88rem;
}
.kpi-card {
    background: linear-gradient(145deg, #FFFFFF, #F5F3FF);
    border: 1px solid rgba(109,40,217,.13);
    border-radius: 19px;
    padding: 17px 18px;
    min-height: 110px;
    box-shadow: 0 18px 45px rgba(109,40,217,.08), inset 0 0 28px rgba(255,255,255,.65);
}
.kpi-label {
    color: #5B4B8A;
    font-size: .88rem;
    margin-bottom: 10px;
}
.kpi-value {
    color: #1E1B4B;
    font-size: 1.65rem;
    font-weight: 800;
    line-height: 1.15;
    white-space: normal;
    overflow-wrap: break-word;
}
.kpi-note {
    color: #7C6B9E;
    font-size: .78rem;
    margin-top: 8px;
}
.footer-box {
    margin-top: 38px;
    padding: 16px 20px;
    border-radius: 18px;
    background: rgba(255,255,255,.82);
    border: 1px solid rgba(109,40,217,.14);
    color: #334155;
    text-align: center;
    font-size: .9rem;
}
.big-rating {
    display: inline-flex;
    width: 94px;
    height: 94px;
    align-items: center;
    justify-content: center;
    border-radius: 26px;
    font-size: 56px;
    font-weight: 950;
    color: #1E1B4B;
    box-shadow: 0 12px 30px rgba(109,40,217,.14);
}
.sidebar-brand {
    padding: 8px 4px 24px;
}
.logo-mark {
    width: 42px;
    height: 42px;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    border-radius: 14px;
    margin-right: 10px;
    background: linear-gradient(135deg, #7C3AED, #5B21B6);
    box-shadow: 0 8px 20px rgba(109,40,217,.28);
    color: #FFFFFF;
    font-weight: 950;
}
[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}
hr { border-color: rgba(109,40,217,.13); }

.rating-explain {
    margin-top: 14px;
    padding: 12px 14px;
    border-radius: 14px;
    background: rgba(109,40,217,.07);
    border: 1px solid rgba(109,40,217,.14);
    color: #1E1B4B;
    line-height: 1.55;
}
.action-note {
    padding: 10px 12px;
    border-radius: 12px;
    background: #FFFFFF;
    border: 1px solid rgba(109,40,217,.16);
    color: #334155;
    font-size: .88rem;
}
.compare-card {
    background: #FFFFFF;
    border: 1px solid rgba(109,40,217,.14);
    border-radius: 18px;
    padding: 15px 16px;
    box-shadow: 0 12px 28px rgba(109,40,217,.07);
}

.chart-card {
    background: #FFFFFF;
    border: 1px solid rgba(109,40,217,.14);
    border-radius: 18px;
    padding: 18px 18px 10px 18px;
    margin-bottom: 18px;
    box-shadow: 0 12px 28px rgba(109,40,217,.07);
}
.chart-title {
    font-size: 1.08rem;
    font-weight: 850;
    color: #1E1B4B;
    margin-bottom: 0px;
}
.default-field-note {
    padding: 10px 12px;
    border-radius: 12px;
    background: #FFFFFF;
    border: 1px solid rgba(109,40,217,.16);
    color: #334155;
    font-size: .88rem;
    margin: 8px 0 16px 0;
}
.dashboard-note {
    padding: 10px 12px;
    border-radius: 12px;
    background: rgba(255,255,255,.88);
    border: 1px solid rgba(109,40,217,.14);
    color: #334155;
    font-size: .86rem;
    line-height: 1.5;
    margin: 8px 0 18px 0;
}
.conf-bar-bg {
    background: rgba(109,40,217,.12);
    border-radius: 8px;
    height: 10px;
    margin-top: 4px;
}
.conf-bar-fill {
    height: 10px;
    border-radius: 8px;
    background: linear-gradient(90deg, #7C3AED, #DB2777);
}
.upgrade-step {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    width:52px; height:52px;
    border-radius:14px;
    font-size:1.6rem;
    font-weight:900;
    box-shadow: 0 8px 20px rgba(109,40,217,.18);
}
.upgrade-arrow {
    font-size:1.4rem;
    color:#7C3AED;
    margin:0 4px;
}

</style>
""",
    unsafe_allow_html=True,
)



# -----------------------------
# Unified visual polish and layout components
# -----------------------------
st.markdown(
    """
<style>
/* Final presentation layer: keeps the existing app logic but makes the UI cohesive. */
[data-testid="stHeader"] {
    background: rgba(245, 243, 255, 0.78) !important;
    backdrop-filter: blur(14px);
    border-bottom: 1px solid rgba(109, 40, 217, .08);
}
.block-container {
    padding-left: 2rem;
    padding-right: 2rem;
}
.main .block-container > div {
    animation: softFadeIn .35s ease-out;
}
@keyframes softFadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.page-header {
    margin: 0 0 1.1rem 0;
    padding: 24px 26px;
    border-radius: 26px;
    background:
        radial-gradient(circle at 90% 12%, rgba(124,58,237,.13), transparent 28%),
        linear-gradient(145deg, rgba(255,255,255,.98), rgba(245,243,255,.95));
    border: 1px solid rgba(109,40,217,.15);
    box-shadow: 0 18px 46px rgba(109,40,217,.08);
}
.page-header-inner {
    display: flex;
    align-items: flex-start;
    gap: 16px;
}
.page-icon {
    width: 50px;
    height: 50px;
    min-width: 50px;
    display:flex;
    align-items:center;
    justify-content:center;
    border-radius: 16px;
    color: #fff;
    font-size: 1.45rem;
    background: linear-gradient(135deg, #7C3AED, #5B21B6);
    box-shadow: 0 10px 24px rgba(109,40,217,.22);
}
.page-kicker {
    color: #7C3AED;
    font-size: .76rem;
    font-weight: 850;
    letter-spacing: .11em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.page-header h1 {
    margin: 0 !important;
    font-size: clamp(2rem, 4vw, 3.05rem) !important;
    line-height: 1.04 !important;
    letter-spacing: -.055em !important;
}
.page-header p {
    max-width: 920px;
    margin: 9px 0 0 0 !important;
    color: #4B5563 !important;
    font-size: 1rem;
    line-height: 1.62;
}
[data-testid="stSidebar"] {
    box-shadow: 10px 0 32px rgba(109,40,217,.055);
}
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 6px;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
    border: 1px solid rgba(109,40,217,.12);
    border-radius: 14px;
    padding: 6px 10px;
    background: rgba(255,255,255,.66);
    transition: all .18s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(124,58,237,.08);
    transform: translateX(2px);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(124,58,237,.14), rgba(91,33,182,.10));
    border-color: rgba(124,58,237,.34);
    box-shadow: 0 8px 20px rgba(109,40,217,.10);
}
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input,
.stDateInput input,
.stSlider [data-baseweb="slider"] {
    border-radius: 13px !important;
}
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input,
.stDateInput input {
    border-color: rgba(109,40,217,.18) !important;
    background: rgba(255,255,255,.96) !important;
    box-shadow: 0 7px 20px rgba(109,40,217,.035);
}
.stSelectbox label, .stMultiSelect label, .stSlider label, .stNumberInput label, .stDateInput label {
    font-weight: 750 !important;
    color: #34236D !important;
}
[data-testid="stExpander"] {
    border: 1px solid rgba(109,40,217,.13) !important;
    border-radius: 18px !important;
    background: rgba(255,255,255,.78) !important;
    box-shadow: 0 10px 26px rgba(109,40,217,.055);
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-weight: 850 !important;
    color: #1E1B4B !important;
}
[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,.90);
    border: 1px solid rgba(109,40,217,.12);
    border-radius: 21px;
    padding: 10px 10px 4px 10px;
    box-shadow: 0 14px 34px rgba(109,40,217,.065);
    margin-bottom: .6rem;
}
[data-testid="stPlotlyChart"] > div {
    border-radius: 16px;
    overflow: hidden;
}
[data-testid="stDataFrame"] {
    border: 1px solid rgba(109,40,217,.13) !important;
    box-shadow: 0 12px 28px rgba(109,40,217,.055);
}
.stAlert {
    border-radius: 16px !important;
    border: 1px solid rgba(109,40,217,.15) !important;
}
.card, .card-small, .kpi-card, .compare-card, .chart-card, .action-note, .dashboard-note, .default-field-note {
    transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
}
.card:hover, .kpi-card:hover, .compare-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 22px 52px rgba(109,40,217,.10), inset 0 0 30px rgba(255,255,255,.75);
    border-color: rgba(109,40,217,.22);
}
.stButton > button {
    width: 100%;
    min-height: 45px;
    letter-spacing: .01em;
}
.stDownloadButton > button {
    border-radius: 14px;
    font-weight: 800;
    border: 1px solid rgba(109,40,217,.20);
    box-shadow: 0 8px 22px rgba(109,40,217,.08);
}

.workflow-strip {
    display:grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin: 1rem 0 1.2rem 0;
}
.workflow-step {
    background: linear-gradient(145deg, #FFFFFF, #F8F7FF);
    border: 1px solid rgba(109,40,217,.14);
    border-radius: 19px;
    padding: 16px 17px;
    min-height: 112px;
    box-shadow: 0 14px 34px rgba(109,40,217,.065);
}
.workflow-step-number {
    width: 34px;
    height: 34px;
    border-radius: 11px;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    color:#FFFFFF;
    font-weight:900;
    background: linear-gradient(135deg,#7C3AED,#5B21B6);
    margin-bottom: 10px;
}
.workflow-step-title {
    font-size: .98rem;
    font-weight: 900;
    color:#1E1B4B;
    margin-bottom: 4px;
}
.workflow-step-text {
    color:#5B4B8A;
    font-size:.86rem;
    line-height:1.48;
}
.input-status-card {
    background: #FFFFFF;
    border: 1px solid rgba(109,40,217,.14);
    border-radius: 17px;
    padding: 13px 15px;
    margin: 10px 0 14px 0;
    box-shadow: 0 12px 28px rgba(109,40,217,.055);
}
.input-status-title {
    font-weight: 900;
    color:#1E1B4B;
    margin-bottom: 4px;
}
.input-status-text {
    color:#5B4B8A;
    font-size:.88rem;
    line-height:1.5;
}
.prediction-panel-title {
    display:flex;
    align-items:center;
    gap:12px;
}
.prediction-panel-number {
    min-width:38px;
    height:38px;
    border-radius:12px;
    color:#fff;
    display:flex;
    align-items:center;
    justify-content:center;
    font-weight:900;
    font-size:1rem;
}
@media (max-width: 1180px) {
    .workflow-strip { grid-template-columns: 1fr; }
}
.hero {
    box-shadow: 0 24px 60px rgba(109,40,217,.10);
}
.hero-title {
    font-size: clamp(3rem, 5.2vw, 4.2rem) !important;
}
.footer-box {
    box-shadow: 0 12px 34px rgba(109,40,217,.06);
}
@media (max-width: 1180px) {
    .hero { min-height: auto; padding: 34px; }
    .hero > div:first-child { max-width: 100% !important; }
    .hero > div:nth-child(2) { display:none; }
    .info-panel-row { grid-template-columns: repeat(2, minmax(0, 1fr)); max-width: 100%; }
}
@media (max-width: 760px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .page-header { padding: 20px; border-radius: 20px; }
    .page-header-inner { flex-direction: column; gap: 10px; }
    .hero { padding: 24px; border-radius: 22px; }
    .hero-title { font-size: 2.45rem !important; }
    .info-panel-row { grid-template-columns: 1fr; }
    .project-meta { font-size: .9rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def _safe_html(value):
    return escape(str(value), quote=True)


def page_intro(title, subtitle="", icon="⚡", kicker="Cambridge Energy Rating Predictor"):
    """Reusable page header for a consistent, professional Streamlit layout."""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header-inner">
                <div class="page-icon">{_safe_html(icon)}</div>
                <div>
                    <div class="page-kicker">{_safe_html(kicker)}</div>
                    <h1>{_safe_html(title)}</h1>
                    <p>{_safe_html(subtitle)}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def prediction_workflow_strip():
    """Clean visual workflow used only on the Prediction page."""
    st.markdown(
        """
        <div class="workflow-strip">
            <div class="workflow-step">
                <div class="workflow-step-number">1</div>
                <div class="workflow-step-title">Property inputs</div>
                <div class="workflow-step-text">Choose the required category values. Numeric fields are available in optional expanders.</div>
            </div>
            <div class="workflow-step">
                <div class="workflow-step-number" style="background:linear-gradient(135deg,#0891B2,#0369A1);">2</div>
                <div class="workflow-step-title">Current calculation</div>
                <div class="workflow-step-text">Select a dataset year and month, then compare actual averages with model predictions.</div>
            </div>
            <div class="workflow-step">
                <div class="workflow-step-number" style="background:linear-gradient(135deg,#059669,#047857);">3</div>
                <div class="workflow-step-title">Future prediction</div>
                <div class="workflow-step-text">Pick a 1–5 year horizon and view clear wave charts for each property type.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    if not FINAL_DATA_PATH.exists() or not MODEL_READY_PATH.exists():
        st.error("Actual project data files are missing. Please add final_fe_data.csv and model_ready_data.csv to the data folder.")
        st.stop()
    final_data = pd.read_csv(FINAL_DATA_PATH)
    model_ready = pd.read_csv(MODEL_READY_PATH)
    return final_data, model_ready

@st.cache_resource(show_spinner=False)
def load_model_bundle():
    if not MODEL_BUNDLE_PATH.exists():
        st.error("Actual trained model file is missing. Please add actual_model_bundle.joblib to the models folder.")
        st.stop()
    return joblib.load(MODEL_BUNDLE_PATH)

df, model_df = load_data()
bundle = load_model_bundle()
metrics = bundle["metrics"]
metadata = bundle["metadata"]
classification_model = bundle["classification_model"]
regression_model = bundle["regression_model"]
class_features = bundle["classification_features"]
reg_features = bundle["regression_features"]
rating_reverse_map = {int(k): v for k, v in bundle["rating_reverse_map"].items()}
rating_forward_map = {v: int(k) for k, v in rating_reverse_map.items()}
feature_importance = pd.DataFrame(bundle["feature_importance"])
# The current notebook exports permutation importance as mean ± standard deviation.
# Keep a compatibility alias so the original website visuals can use the new results.
if "Importance_Mean" in feature_importance.columns and "Importance" not in feature_importance.columns:
    feature_importance["Importance"] = feature_importance["Importance_Mean"]
if "Importance_SD" not in feature_importance.columns:
    feature_importance["Importance_SD"] = 0.0
feature_importance = feature_importance.sort_values("Importance", ascending=False).reset_index(drop=True)

def normalise_rating_prediction(value):
    """Return an EPC band whether the saved classifier predicts labels or encoded integers."""
    text_value = str(value)
    if text_value in RATING_ORDER:
        return text_value
    try:
        return rating_reverse_map.get(int(value), text_value)
    except (TypeError, ValueError):
        return text_value

# -----------------------------
# Column constants
# -----------------------------
COL_CURRENT = "CURRENT_ENERGY_RATING"
COL_POTENTIAL = "POTENTIAL_ENERGY_RATING"
COL_SCORE = "CURRENT_ENERGY_EFFICIENCY"
COL_POT_SCORE = "POTENTIAL_ENERGY_EFFICIENCY"
COL_PROPERTY = "PROPERTY_TYPE"
COL_BUILT = "BUILT_FORM"
COL_AREA = "TOTAL_FLOOR_AREA"
COL_CO2 = "CO2_EMISSIONS_CURRENT"
COL_CO2_PER = "CO2_EMISS_CURR_PER_FLOOR_AREA"
COL_ENERGY = "ENERGY_CONSUMPTION_CURRENT"
COL_HEATING = "HEATING_COST_CURRENT"
COL_HOT_WATER = "HOT_WATER_COST_CURRENT"
COL_LIGHTING = "LIGHTING_COST_CURRENT"
COL_GLAZED = "GLAZED_TYPE"
COL_TARIFF = "ENERGY_TARIFF"
COL_MAINS = "MAINS_GAS_FLAG"
COL_FUEL_GROUP = "MAIN_FUEL_GROUP"
COL_AGE_GROUP = "CONSTRUCTION_AGE_GROUP"

# -----------------------------
# Helper functions
# -----------------------------
def plot_layout(fig, height=365):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1E1B4B", family="Inter, Segoe UI, sans-serif"),
        title=dict(font=dict(size=14, color="#1E1B4B", weight="bold"), x=0.02, xanchor="left"),
        margin=dict(l=32, r=32, t=48, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(109,40,217,.07)", zerolinecolor="rgba(109,40,217,.12)")
    fig.update_yaxes(gridcolor="rgba(109,40,217,.07)", zerolinecolor="rgba(109,40,217,.12)")
    return fig


def dashboard_plot_layout(fig, height=430):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1E1B4B", family="Inter, Segoe UI, sans-serif"),
        title=dict(font=dict(size=15, color="#1E1B4B", weight="bold"), x=0.02, xanchor="left", y=0.98, yanchor="top"),
        margin=dict(l=48, r=42, t=52, b=60),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(109,40,217,.07)",
        zerolinecolor="rgba(109,40,217,.12)",
        title_font=dict(size=12, color="#5B4B8A"),
        tickfont=dict(size=11, color="#5B4B8A"),
    )
    fig.update_yaxes(
        gridcolor="rgba(109,40,217,.07)",
        zerolinecolor="rgba(109,40,217,.12)",
        title_font=dict(size=12, color="#5B4B8A"),
        tickfont=dict(size=11, color="#5B4B8A"),
    )
    return fig

def get_options(col):
    return sorted([str(x) for x in df[col].dropna().unique()])

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

def month_label(month_num):
    month_num = int(month_num)
    return f"{month_num:02d} — {MONTH_NAMES.get(month_num, str(month_num))}"

def get_dataset_date_view(data):
    """Build a date view from the dataset only.
    No manual years are added. The app uses INSPECTION_DATE first, then LODGEMENT_DATE.
    """
    date_col = None
    for candidate in ["INSPECTION_DATE", "LODGEMENT_DATE"]:
        if candidate in data.columns:
            date_col = candidate
            break
    out = data.copy()
    if date_col is None:
        out["__DATASET_DATE"] = pd.NaT
        out["__DATASET_YEAR"] = np.nan
        out["__DATASET_MONTH"] = np.nan
        return out, "No date column found"
    out["__DATASET_DATE"] = pd.to_datetime(out[date_col], errors="coerce")
    out["__DATASET_YEAR"] = out["__DATASET_DATE"].dt.year
    out["__DATASET_MONTH"] = out["__DATASET_DATE"].dt.month
    return out, date_col

def get_dataset_years(data):
    dated, _ = get_dataset_date_view(data)
    years = sorted(dated["__DATASET_YEAR"].dropna().astype(int).unique().tolist())
    if not years and "INSPECTION_YEAR" in data.columns:
        years = sorted(pd.to_numeric(data["INSPECTION_YEAR"], errors="coerce").dropna().astype(int).unique().tolist())
    return years

def get_dataset_months_for_year(data, year):
    dated, _ = get_dataset_date_view(data)
    months = sorted(
        dated.loc[dated["__DATASET_YEAR"].astype("Int64") == int(year), "__DATASET_MONTH"]
        .dropna().astype(int).unique().tolist()
    )
    return months

def property_type_comparison_chart(data, metric_col, metric_label):
    """Combined property-type comparison using the selected dataset period only."""
    property_order = [p for p in get_options(COL_PROPERTY) if p in data[COL_PROPERTY].astype(str).unique()]
    if not property_order:
        property_order = get_options(COL_PROPERTY)
    grouped = (
        data.groupby(COL_PROPERTY, as_index=False)
        .agg(Value=(metric_col, "mean"), Properties=(COL_PROPERTY, "size"))
    )
    fig = go.Figure()
    for i, ptype in enumerate(property_order):
        row = grouped[grouped[COL_PROPERTY].astype(str) == str(ptype)]
        if row.empty:
            continue
        value = float(row["Value"].iloc[0])
        count = int(row["Properties"].iloc[0])
        color = DASHBOARD_PROPERTY_COLORS[i % len(DASHBOARD_PROPERTY_COLORS)]
        fig.add_trace(go.Bar(
            x=[ptype],
            y=[value],
            name=str(ptype),
            marker=dict(color=color, line=dict(color="rgba(15,23,42,.14)", width=1)),
            text=[f"{value:.1f}"],
            textposition="outside",
            hovertemplate=f"<b>{ptype}</b><br>{metric_label}: %{{y:.2f}}<br>Records: {count:,}<extra></extra>",
        ))
    fig.update_layout(
        title=f"Property Type Comparison — Average {metric_label}",
        showlegend=True,
        barmode="group",
    )
    fig.update_xaxes(title_text="Property type")
    fig.update_yaxes(title_text=f"Average {metric_label}")
    return dashboard_plot_layout(fig, 420)

def apply_filters(data, property_type, current_rating, potential_rating, built_form=None):
    out = data.copy()
    if property_type != "All":
        out = out[out[COL_PROPERTY].astype(str) == property_type]
    if current_rating != "All":
        out = out[out[COL_CURRENT].astype(str) == current_rating]
    if potential_rating != "All":
        out = out[out[COL_POTENTIAL].astype(str) == potential_rating]
    if built_form and built_form != "All":
        out = out[out[COL_BUILT].astype(str) == built_form]
    return out

def rating_distribution_chart(data):
    counts = data[COL_CURRENT].value_counts().reindex(RATING_ORDER, fill_value=0).reset_index()
    counts.columns = ["Rating", "Properties"]
    fig = go.Figure(go.Bar(
        x=counts["Rating"], y=counts["Properties"],
        marker=dict(color=[DASHBOARD_RATING_COLORS[r] for r in counts["Rating"]], line=dict(color="rgba(15,23,42,.16)", width=1)),
        text=counts["Properties"], textposition="outside",
        hovertemplate="EPC band %{x}<br>Properties %{y:,}<extra></extra>"
    ))
    fig.update_layout(title="Energy Rating Distribution")
    fig.update_xaxes(title_text="Current EPC rating band (A = best, G = lowest)")
    fig.update_yaxes(title_text="Number of properties")
    return dashboard_plot_layout(fig)

def rating_ring_chart(data):
    counts = data[COL_CURRENT].value_counts().reindex(RATING_ORDER, fill_value=0).reset_index()
    counts.columns = ["Rating", "Properties"]
    # vibrant gradient colors for each band
    ring_colors = ["#059669", "#22C55E", "#84CC16", "#FBBF24", "#F97316", "#EF4444", "#9F1239"]
    fig = go.Figure(go.Pie(
        labels=counts["Rating"], values=counts["Properties"], hole=.60,
        marker=dict(colors=ring_colors[:len(counts)], line=dict(color="#FFFFFF", width=2.5)),
        pull=[0.03]*len(counts),
        textinfo="label+percent",
        textposition="inside",
        insidetextorientation="radial",
        textfont=dict(size=12, color="#FFFFFF"),
        hovertemplate="<b>EPC Band %{label}</b><br>%{value:,} properties<br>%{percent}<extra></extra>"
    ))
    fig.add_annotation(text=f"<b>{len(data):,}</b><br><span style='font-size:11px'>records</span>", x=.5, y=.5, showarrow=False, font=dict(size=18, color="#0F172A"), align="center")
    fig.update_layout(title="Current EPC Rating Distribution", showlegend=True)
    return dashboard_plot_layout(fig)

def current_vs_potential_chart(data):
    current = data[COL_CURRENT].value_counts().reindex(RATING_ORDER, fill_value=0)
    potential = data[COL_POTENTIAL].value_counts().reindex(RATING_ORDER, fill_value=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=RATING_ORDER, y=current.values, name="Current Rating",
        marker=dict(color="#2563EB", opacity=0.9, line=dict(color="#1D4ED8", width=1)),
        text=current.values, textposition="outside", textfont=dict(size=11, color="#1E293B"),
    ))
    fig.add_trace(go.Bar(
        x=RATING_ORDER, y=potential.values, name="Potential Rating",
        marker=dict(color="#10B981", opacity=0.9, line=dict(color="#059669", width=1)),
        text=potential.values, textposition="outside", textfont=dict(size=11, color="#1E293B"),
    ))
    fig.update_layout(title="Current vs Potential EPC Rating", barmode="group")
    fig.update_xaxes(title_text="EPC rating band (A = best, G = lowest)")
    fig.update_yaxes(title_text="Number of properties")
    return dashboard_plot_layout(fig)

def property_type_chart(data):
    property_order = get_options(COL_PROPERTY)
    counts = data[COL_PROPERTY].astype(str).value_counts().reindex(property_order, fill_value=0).reset_index()
    counts.columns = ["Property Type", "Properties"]
    fig = go.Figure(go.Bar(
        x=counts["Property Type"],
        y=counts["Properties"],
        marker=dict(
            color=DASHBOARD_PROPERTY_COLORS[:len(counts)],
            opacity=0.92,
            line=dict(color="rgba(255,255,255,0.6)", width=1.5),
        ),
        text=counts["Properties"],
        textposition="outside",
        textfont=dict(size=12, color="#1E293B"),
        hovertemplate="<b>%{x}</b><br>%{y:,} properties<extra></extra>",
    ))
    fig.update_layout(title="Property Type Distribution")
    fig.update_xaxes(title_text="Property type")
    fig.update_yaxes(title_text="Number of properties")
    return dashboard_plot_layout(fig)

def metric_box_chart(data, metric_col, metric_label):
    fig = px.box(
        data, x=COL_CURRENT, y=metric_col, color=COL_CURRENT,
        category_orders={COL_CURRENT: RATING_ORDER},
        color_discrete_map=DASHBOARD_RATING_COLORS,
        points="outliers",
        title=f"{metric_label} by EPC Rating",
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="Current EPC rating band (A = best, G = lowest)")
    fig.update_yaxes(title_text=metric_label)
    return dashboard_plot_layout(fig)

def gauge(score, rating):
    score = float(np.clip(score, 0, 100))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 36, "color": "#0F172A"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#334155"}},
            "bar": {"color": RATING_COLORS.get(rating, "#00F5D4")},
            "bgcolor": "rgba(255,255,255,.04)",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,.18)",
            "steps": [
                {"range": [0, 21], "color": "#C9184A"},
                {"range": [21, 39], "color": "#FF4D6D"},
                {"range": [39, 55], "color": "#FF9F1C"},
                {"range": [55, 69], "color": "#FFD166"},
                {"range": [69, 81], "color": "#C2F970"},
                {"range": [81, 92], "color": "#3BE088"},
                {"range": [92, 100], "color": "#00F5D4"},
            ],
            "threshold": {"line": {"color": "#0F172A", "width": 4}, "thickness": .75, "value": score},
        },
    ))
    fig.update_layout(height=300, margin=dict(l=12, r=12, t=20, b=12), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#0F172A"))
    return fig

def construction_age_group_from_band(value):
    """Replicate the current notebook's construction-age grouping."""
    if pd.isna(value):
        return "Unknown"
    import re as _re
    value_text = str(value).lower()
    years = [int(y) for y in _re.findall(r"(?:18|19|20)\d{2}", value_text)]
    if "before" in value_text and "1900" in value_text:
        return "Pre-1930"
    first = min(years) if years else None
    if first is None:
        return "Unknown"
    if first < 1930:
        return "Pre-1930"
    if first <= 1966:
        return "1930-1966"
    if first <= 1990:
        return "1967-1990"
    if first <= 2011:
        return "1991-2011"
    return "2012 onwards"


def main_fuel_group_from_value(value):
    """Replicate the current notebook's fuel grouping."""
    if pd.isna(value):
        return "Unknown"
    value_text = str(value).lower()
    if "gas" in value_text:
        return "Gas"
    if "electric" in value_text:
        return "Electricity"
    if "oil" in value_text:
        return "Oil"
    if "coal" in value_text or "smokeless" in value_text:
        return "Coal/Solid fuel"
    if "wood" in value_text or "biomass" in value_text:
        return "Biomass/Wood"
    if "community" in value_text or "district" in value_text:
        return "Community heating"
    return "Other"


def floor_area_band_from_value(value):
    value = float(value)
    if value <= 50:
        return "Small (<50 m²)"
    if value <= 100:
        return "Medium (50-100 m²)"
    return "Large (>100 m²)"


def glazing_group_from_value(value):
    value = float(value)
    if value <= 0:
        return "None"
    if value <= 50:
        return "Partial"
    if value < 100:
        return "Mostly"
    return "Fully glazed"


def lighting_group_from_value(value):
    value = float(value)
    if value <= 25:
        return "Low"
    if value <= 75:
        return "Medium"
    return "High"


def build_model_input(user):
    """Create the exact 38 leakage-free model inputs used by the current notebook.

    EPC outputs such as current consumption, CO₂, costs, environmental impact and
    potential ratings are intentionally excluded.
    """
    base = dict(metadata.get("defaults", {}))

    # Accept exact notebook column names, plus a small set of old UI aliases.
    for feature in set(class_features).union(reg_features):
        if feature in user and user[feature] is not None:
            base[feature] = user[feature]

    aliases = {
        "property_type": "PROPERTY_TYPE",
        "built_form": "BUILT_FORM",
        "construction_age_band": "CONSTRUCTION_AGE_BAND",
        "main_fuel": "MAIN_FUEL",
        "mains_gas": "MAINS_GAS_FLAG",
        "glazed_type": "GLAZED_TYPE",
        "glazed_area": "GLAZED_AREA",
        "walls_description": "WALLS_DESCRIPTION",
        "roof_description": "ROOF_DESCRIPTION",
        "windows_description": "WINDOWS_DESCRIPTION",
        "floor_description": "FLOOR_DESCRIPTION",
        "mainheat_description": "MAINHEAT_DESCRIPTION",
        "mainheatcont_description": "MAINHEATCONT_DESCRIPTION",
        "hotwater_description": "HOTWATER_DESCRIPTION",
        "mechanical_ventilation": "MECHANICAL_VENTILATION",
        "solar_water_heating_flag": "SOLAR_WATER_HEATING_FLAG",
        "floor_area": "TOTAL_FLOOR_AREA",
        "floor_level": "FLOOR_LEVEL",
        "multi_glaze": "MULTI_GLAZE_PROPORTION",
        "extension_count": "EXTENSION_COUNT",
        "habitable_rooms": "NUMBER_HABITABLE_ROOMS",
        "heated_rooms": "NUMBER_HEATED_ROOMS",
        "low_energy_lighting": "LOW_ENERGY_LIGHTING",
        "open_fireplaces": "NUMBER_OPEN_FIREPLACES",
        "wind_turbine_count": "WIND_TURBINE_COUNT",
        "floor_height": "FLOOR_HEIGHT",
        "photo_supply": "PHOTO_SUPPLY",
        "fixed_lighting_outlets": "FIXED_LIGHTING_OUTLETS_COUNT",
    }
    for source_key, target_feature in aliases.items():
        if source_key in user and user[source_key] is not None:
            base[target_feature] = user[source_key]

    # Recreate the notebook's ten engineered physical/structural predictors.
    property_type = str(base.get("PROPERTY_TYPE", "Unknown"))
    built_form = str(base.get("BUILT_FORM", "Unknown"))
    construction_band = base.get("CONSTRUCTION_AGE_BAND", "Unknown")
    main_fuel = base.get("MAIN_FUEL", "Unknown")
    floor_area = float(base.get("TOTAL_FLOOR_AREA", 0) or 0)
    habitable_rooms = max(float(base.get("NUMBER_HABITABLE_ROOMS", 0) or 0), 0.0)
    heated_rooms = max(float(base.get("NUMBER_HEATED_ROOMS", 0) or 0), 0.0)
    glazing_pct = float(base.get("MULTI_GLAZE_PROPORTION", 0) or 0)
    lighting_pct = float(base.get("LOW_ENERGY_LIGHTING", 0) or 0)
    fireplaces = float(base.get("NUMBER_OPEN_FIREPLACES", 0) or 0)
    wind = float(base.get("WIND_TURBINE_COUNT", 0) or 0)
    photo = float(base.get("PHOTO_SUPPLY", 0) or 0)
    solar_flag = str(base.get("SOLAR_WATER_HEATING_FLAG", "N")).strip().lower()

    base["CONSTRUCTION_AGE_GROUP"] = construction_age_group_from_band(construction_band)
    base["MAIN_FUEL_GROUP"] = main_fuel_group_from_value(main_fuel)
    base["BUILDING_STRUCTURE_TYPE"] = f"{property_type} | {built_form}"
    base["FLOOR_AREA_BAND"] = floor_area_band_from_value(floor_area)
    base["AREA_PER_HABITABLE_ROOM"] = floor_area / habitable_rooms if habitable_rooms > 0 else float(metadata["defaults"].get("AREA_PER_HABITABLE_ROOM", 0))
    base["HEATED_ROOM_RATIO"] = heated_rooms / habitable_rooms if habitable_rooms > 0 else float(metadata["defaults"].get("HEATED_ROOM_RATIO", 0))
    base["GLAZING_GROUP"] = glazing_group_from_value(glazing_pct)
    base["LOW_ENERGY_LIGHTING_GROUP"] = lighting_group_from_value(lighting_pct)
    base["OPEN_FIREPLACE_FLAG"] = int(fireplaces > 0)
    base["RENEWABLE_SYSTEM_FLAG"] = int(wind > 0 or photo > 0 or solar_flag in {"y", "yes", "true", "1", "present"})

    class_row = pd.DataFrame([{feature: base.get(feature, metadata.get("defaults", {}).get(feature)) for feature in class_features}], columns=class_features)
    reg_row = pd.DataFrame([{feature: base.get(feature, metadata.get("defaults", {}).get(feature)) for feature in reg_features}], columns=reg_features)
    return class_row, reg_row

def score_to_rating(score):
    score = float(score)
    if score >= 92: return "A"
    if score >= 81: return "B"
    if score >= 69: return "C"
    if score >= 55: return "D"
    if score >= 39: return "E"
    if score >= 21: return "F"
    return "G"

def rating_meaning(rating):
    meanings = {
        "A": "Excellent energy performance",
        "B": "Good energy performance",
        "C": "Average / moderate energy performance",
        "D": "Below average energy performance",
        "E": "Low energy performance",
        "F": "Very low energy performance",
        "G": "Poor energy performance",
    }
    return meanings.get(str(rating), "Energy performance band")

def most_common_value(col, fallback=None):
    if col not in df.columns:
        return fallback
    mode_values = df[col].dropna().astype(str).mode()
    return mode_values.iloc[0] if len(mode_values) else fallback

def select_with_mode(label, col, fallback=None, disabled=False, help_text=None):
    options = get_options(col)
    default_value = most_common_value(col, fallback or (options[0] if options else ""))
    index = options.index(default_value) if default_value in options else 0
    return st.selectbox(label, options, index=index, disabled=disabled, help=help_text)

def select_list_with_mode(label, values, disabled=False, help_text=None):
    options = sorted([str(x) for x in pd.Series(values).dropna().unique().tolist()])
    if not options:
        return ""
    mode_values = pd.Series(values).dropna().astype(str).mode()
    default_value = mode_values.iloc[0] if len(mode_values) else options[0]
    index = options.index(default_value) if default_value in options else 0
    return st.selectbox(label, options, index=index, disabled=disabled, help=help_text)

def required_selectbox(label, options, key=None, help_text=None):
    """Show a dropdown with no real category selected by default.
    Returns None until the user selects a value.
    """
    clean_options = sorted([str(x) for x in pd.Series(options).dropna().unique().tolist()])
    placeholder = f"Select {label}"
    selected = st.selectbox(label, [placeholder] + clean_options, index=0, key=key, help=help_text)
    if selected == placeholder:
        return None
    return selected

def smart_numeric_widget(label, col, key=None, integer=False):
    """Create inputs using real values from the dataset.
    Small unique-value columns become dropdowns; continuous columns become sliders.
    The min, max, and default are taken from the actual project dataset.
    """
    vals = pd.to_numeric(df[col], errors="coerce").dropna() if col in df.columns else pd.Series([0])
    if vals.empty:
        vals = pd.Series([0])

    unique_vals = sorted(vals.unique().tolist())
    median_val = float(vals.median())

    # Small-value columns use dropdowns so the user does not struggle with plus/minus boxes.
    if len(unique_vals) <= 15:
        display_vals = [int(v) if float(v).is_integer() else round(float(v), 2) for v in unique_vals]
        closest_idx = min(range(len(unique_vals)), key=lambda i: abs(float(unique_vals[i]) - median_val))
        chosen = st.selectbox(label, display_vals, index=closest_idx, key=key)
        return float(chosen)

    min_val = float(vals.min())
    max_val = float(vals.max())
    if min_val == max_val:
        max_val = min_val + 1
    default_val = min(max(median_val, min_val), max_val)

    if integer:
        return float(st.slider(label, int(round(min_val)), int(round(max_val)), int(round(default_val)), step=1, key=key))

    step = max((max_val - min_val) / 100, 0.01)
    return float(st.slider(label, min_val, max_val, default_val, step=step, format="%.2f", key=key))

def comparison_plot_layout(fig, height=420):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1E1B4B", family="Inter, Segoe UI, sans-serif", size=12),
        margin=dict(l=55, r=30, t=42, b=70),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(gridcolor="rgba(109,40,217,.07)", zerolinecolor="rgba(109,40,217,.12)", title_font=dict(size=12))
    fig.update_yaxes(gridcolor="rgba(109,40,217,.07)", zerolinecolor="rgba(109,40,217,.12)", title_font=dict(size=12))
    return fig

PROPERTY_TYPES_4 = get_options(COL_PROPERTY)
PROPERTY_LINE_COLORS = {
    ptype: DASHBOARD_PROPERTY_COLORS[i % len(DASHBOARD_PROPERTY_COLORS)]
    for i, ptype in enumerate(PROPERTY_TYPES_4)
}

CURRENT_YEAR = pd.Timestamp.today().year
PREDICTION_HORIZONS = [1, 2, 3, 4, 5]
PREDICTION_YEARS = [CURRENT_YEAR + h for h in PREDICTION_HORIZONS]

def prediction_year_label(year):
    horizon = int(year) - CURRENT_YEAR
    return f"{year} ({horizon} year{'s' if horizon != 1 else ''} ahead)"

PREDICTION_QUARTER_RANGES = [
    ("January to March", [1, 2, 3]),
    ("April to June", [4, 5, 6]),
    ("July to September", [7, 8, 9]),
    ("October to December", [10, 11, 12]),
]

def _median_or_existing(source_data, col, existing_value):
    vals = pd.to_numeric(source_data[col], errors="coerce").dropna() if col in source_data.columns else pd.Series(dtype=float)
    if len(vals):
        return float(vals.median())
    return existing_value

def build_property_type_values(base_user, property_type, month_num=None, dataset_year=None, prediction_year=None):
    """Clone user inputs and replace the numeric baseline with medians from the actual dataset.
    When month_num is supplied, the baseline first uses records for that same dataset month.
    This keeps prediction graphs grounded in the uploaded data rather than hardcoded values.
    """
    dated, _ = get_dataset_date_view(df)
    ptype_data = dated[dated[COL_PROPERTY].astype(str) == str(property_type)].copy()
    if dataset_year is not None and "__DATASET_YEAR" in ptype_data.columns:
        year_data = ptype_data[ptype_data["__DATASET_YEAR"].astype("Int64") == int(dataset_year)]
        if len(year_data) > 0:
            ptype_data = year_data
    if month_num is not None and "__DATASET_MONTH" in ptype_data.columns:
        month_data = ptype_data[ptype_data["__DATASET_MONTH"].astype("Int64") == int(month_num)]
        if len(month_data) > 0:
            ptype_data = month_data
    if len(ptype_data) == 0:
        ptype_data = dated.copy()

    uv = base_user.copy()
    uv["energy_consumption"] = _median_or_existing(ptype_data, COL_ENERGY, uv.get("energy_consumption", 0))
    uv["co2_emissions"]      = _median_or_existing(ptype_data, COL_CO2, uv.get("co2_emissions", 0))
    uv["co2_per_floor"]      = _median_or_existing(ptype_data, COL_CO2_PER, uv.get("co2_per_floor", 0))
    uv["heating_cost"]       = _median_or_existing(ptype_data, COL_HEATING, uv.get("heating_cost", 0))
    uv["hot_water_cost"]     = _median_or_existing(ptype_data, COL_HOT_WATER, uv.get("hot_water_cost", 0))
    uv["lighting_cost"]      = _median_or_existing(ptype_data, COL_LIGHTING, uv.get("lighting_cost", 0))
    uv["floor_area"]         = _median_or_existing(ptype_data, COL_AREA, uv.get("floor_area", 0))
    uv["environment_impact"] = _median_or_existing(ptype_data, "ENVIRONMENT_IMPACT_CURRENT", uv.get("environment_impact", 0))
    uv["low_energy_lighting"] = _median_or_existing(ptype_data, "LOW_ENERGY_LIGHTING", uv.get("low_energy_lighting", 0))
    uv["property_type"]      = str(property_type)
    if prediction_year is not None:
        uv["inspection_year"] = float(prediction_year)
    return uv

def make_future_user_values(user, prediction_year=None, prediction_month=None):
    """Future prediction input: preserve the user's selected building values and set the future year.
    The property-type line graphs use dataset medians by property type/month for comparison.
    """
    future = user.copy()
    if prediction_year is not None:
        future["inspection_year"] = float(prediction_year)
    return future

def property_wave_chart(base_user, month_labels, metric_key, metric_label, title, pred_year, color_override=None):
    """Smooth line chart: dataset property types × selected 3 months for a predicted metric."""
    fig = go.Figure()
    month_nums = st.session_state.get("pred_month_nums", [1, 2, 3])
    for ptype in PROPERTY_TYPES_4:
        y_vals = []
        for month_num in month_nums:
            monthly_user = build_property_type_values(base_user, ptype, month_num=month_num, prediction_year=pred_year)
            _, reg_row = build_model_input(monthly_user)
            score = float(np.clip(regression_model.predict(reg_row)[0], 1, 100))
            if metric_key == "efficiency_score":
                y_vals.append(round(score, 2))
            elif metric_key == "heating_cost":
                y_vals.append(round(monthly_user["heating_cost"], 2))
            elif metric_key == "co2_emissions":
                y_vals.append(round(monthly_user["co2_emissions"], 2))
            elif metric_key == "energy_consumption":
                y_vals.append(round(monthly_user["energy_consumption"], 2))
            elif metric_key == "hot_water_cost":
                y_vals.append(round(monthly_user["hot_water_cost"], 2))
            elif metric_key == "lighting_cost":
                y_vals.append(round(monthly_user["lighting_cost"], 2))
            else:
                y_vals.append(round(score, 2))

        color = color_override or PROPERTY_LINE_COLORS.get(ptype, DASHBOARD_PROPERTY_COLORS[0])
        fig.add_trace(go.Scatter(
            x=month_labels,
            y=y_vals,
            name=str(ptype),
            mode="lines+markers",
            line=dict(color=color, width=3, shape="spline", smoothing=1.3),
            marker=dict(size=9, color=color, line=dict(color="white", width=2)),
            hovertemplate=f"<b>{ptype}</b><br>%{{x}}<br>{metric_label}: %{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=f"{title} — {pred_year}", font=dict(size=14, color="#1E1B4B", weight="bold"), x=0.02),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.30,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(109,40,217,.14)",
            borderwidth=1,
            font=dict(size=12, color="#1E1B4B"),
            itemsizing="constant",
        ),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1E1B4B", family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=50, r=30, t=52, b=100),
        hovermode="x unified",
    )
    fig.update_xaxes(
        title_text="Month (selected 3-month range)",
        gridcolor="rgba(109,40,217,.07)",
        zerolinecolor="rgba(109,40,217,.12)",
        tickfont=dict(size=12, color="#5B4B8A"),
    )
    fig.update_yaxes(
        title_text=metric_label,
        gridcolor="rgba(109,40,217,.07)",
        zerolinecolor="rgba(109,40,217,.12)",
        tickfont=dict(size=11, color="#5B4B8A"),
    )
    return fig

def create_prediction_graphs(current_score, future_score, current_rating, future_rating, user, future_user, mode="future"):
    """Return smooth line/wave prediction graphs only.
    Prediction graphs compare dataset property types across the selected 3-month range.
    """
    month_labels = st.session_state.get("pred_month_labels", ["Month 1", "Month 2", "Month 3"])
    pred_year = st.session_state.get("pred_year", PREDICTION_YEARS[0])

    wave1 = property_wave_chart(user, month_labels, "efficiency_score",    "Efficiency score (0–100)", "EPC Efficiency Score",    pred_year)
    wave2 = property_wave_chart(user, month_labels, "energy_consumption",  "Energy consumption",       "Energy Consumption",       pred_year)
    wave3 = property_wave_chart(user, month_labels, "heating_cost",        "Heating cost (£)",         "Heating Cost",             pred_year)
    wave4 = property_wave_chart(user, month_labels, "co2_emissions",       "CO₂ emissions",            "CO₂ Emissions",            pred_year)
    return [
        ("EPC Efficiency Score — Property Types", wave1),
        ("Energy Consumption — Property Types",   wave2),
        ("Heating Cost — Property Types",         wave3),
        ("CO₂ Emissions — Property Types",        wave4),
    ]


def property_type_prediction_summary(base_user, month_num=None, dataset_year=None, prediction_year=None):
    """Predict EPC results for every property type available in the uploaded dataset.
    Current mode can pass dataset_year + month_num so the values are grounded in the selected dataset period.
    Future mode passes prediction_year so the selected future year is fed into the model.
    """
    rows = []
    for ptype in PROPERTY_TYPES_4:
        p_user = build_property_type_values(
            base_user,
            ptype,
            month_num=month_num,
            dataset_year=dataset_year,
            prediction_year=prediction_year,
        )
        class_row, reg_row = build_model_input(p_user)
        class_band = normalise_rating_prediction(classification_model.predict(class_row)[0])
        efficiency_score = float(np.clip(regression_model.predict(reg_row)[0], 1, 100))
        score_band = score_to_rating(efficiency_score)
        try:
            proba = classification_model.predict_proba(class_row)[0]
            confidence = float(np.max(proba))
        except Exception:
            confidence = np.nan
        rows.append({
            "Property Type": str(ptype),
            "Predicted Band": score_band,
            "Classifier Band": class_band,
            "Efficiency Score": round(efficiency_score, 2),
            "Confidence": round(confidence * 100, 1) if not np.isnan(confidence) else np.nan,
        })
    return pd.DataFrame(rows)


def render_property_type_rating_cards(summary_df, title):
    """Render one EPC result card for each property type, matching the reference style."""
    st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
    rows = [summary_df.iloc[i:i+4] for i in range(0, len(summary_df), 4)]
    for row_df in rows:
        cols = st.columns(len(row_df))
        for col, (_, row) in zip(cols, row_df.iterrows()):
            band = str(row["Predicted Band"])
            ptype = str(row["Property Type"])
            score = float(row["Efficiency Score"])
            confidence = row["Confidence"]
            conf_text = "N/A" if pd.isna(confidence) else f"{confidence:.1f}%"
            band_color = RATING_COLORS.get(band, "#7C3AED")
            line_color = PROPERTY_LINE_COLORS.get(ptype, "#7C3AED")
            col.markdown(
                f"""
                <div class="compare-card" style="text-align:center; border-top:5px solid {line_color}; min-height:205px;">
                    <div class="small-label" style="color:{line_color};">{ptype}</div>
                    <div class="big-rating" style="background:{band_color}; margin:10px auto 8px auto; width:78px; height:78px; font-size:46px;">{band}</div>
                    <div style="font-weight:800; color:#1E1B4B;">Band {band} · {score:.1f}/100</div>
                    <div class="muted" style="font-size:.84rem; margin-top:4px;">Confidence: <b>{conf_text}</b></div>
                    <div style="height:6px; border-radius:6px; background:rgba(109,40,217,.10); margin-top:10px; overflow:hidden;">
                        <div style="height:6px; width:{0 if pd.isna(confidence) else float(confidence)}%; background:{line_color}; border-radius:6px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def property_type_score_distribution_chart(data, summary_df, metric_col=COL_SCORE, title="Current Score Comparison"):
    """Simple current comparison chart.
    Each property type gets one clear line/marker:
    - Dataset actual average score for the selected month/year, when records exist
    - Model predicted score from the selected building inputs
    This replaces the old histogram subplots because they were difficult to read when some property types had no records.
    """
    fig = go.Figure()
    if summary_df.empty:
        return fig

    for idx, (_, row) in enumerate(summary_df.iterrows()):
        ptype = str(row["Property Type"])
        color = PROPERTY_LINE_COLORS.get(ptype, DASHBOARD_PROPERTY_COLORS[idx % len(DASHBOARD_PROPERTY_COLORS)])
        ptype_data = data[data[COL_PROPERTY].astype(str) == ptype]
        values = pd.to_numeric(ptype_data[metric_col], errors="coerce").dropna()
        records = int(len(values))
        actual_avg = float(values.mean()) if records else np.nan
        predicted = float(row["Efficiency Score"])
        band = str(row["Predicted Band"])

        x_vals = []
        y_vals = []
        text_vals = []

        if not np.isnan(actual_avg):
            x_vals.append("Dataset actual average")
            y_vals.append(actual_avg)
            text_vals.append(f"Actual {actual_avg:.1f}")

        x_vals.append("Model predicted score")
        y_vals.append(predicted)
        text_vals.append(f"Pred {predicted:.1f}")

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            name=f"{ptype} · Band {band} · {records} records",
            mode="lines+markers+text" if len(x_vals) > 1 else "markers+text",
            line=dict(color=color, width=4, shape="spline", smoothing=1.1),
            marker=dict(size=13, color=color, line=dict(color="#FFFFFF", width=2)),
            text=text_vals,
            textposition="top center",
            hovertemplate=(
                f"<b>{ptype}</b><br>"
                "%{x}<br>Score: %{y:.1f}<br>"
                f"Selected-period records: {records:,}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1E1B4B", weight="bold"), x=0.02),
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1E1B4B", family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=55, r=35, t=70, b=95),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(109,40,217,.14)",
            borderwidth=1,
            font=dict(size=11),
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(
        title_text="Comparison point",
        categoryorder="array",
        categoryarray=["Dataset actual average", "Model predicted score"],
        gridcolor="rgba(109,40,217,.07)",
        zerolinecolor="rgba(109,40,217,.12)",
    )
    fig.update_yaxes(
        title_text="EPC efficiency score (0–100)",
        range=[0, 100],
        gridcolor="rgba(109,40,217,.07)",
        zerolinecolor="rgba(109,40,217,.12)",
    )
    return fig


def current_comparison_table(data, summary_df, metric_col=COL_SCORE):
    """Readable table for the current dashboard."""
    rows = []
    for _, row in summary_df.iterrows():
        ptype = str(row["Property Type"])
        ptype_data = data[data[COL_PROPERTY].astype(str) == ptype]
        values = pd.to_numeric(ptype_data[metric_col], errors="coerce").dropna()
        records = int(len(values))
        actual_avg = float(values.mean()) if records else np.nan
        predicted = float(row["Efficiency Score"])
        diff = predicted - actual_avg if not np.isnan(actual_avg) else np.nan
        confidence = row.get("Confidence", np.nan)
        rows.append({
            "Property Type": ptype,
            "Selected Month Records": records,
            "Dataset Actual Avg Score": "No records" if np.isnan(actual_avg) else round(actual_avg, 1),
            "Model Predicted Score": round(predicted, 1),
            "Difference": "N/A" if np.isnan(diff) else round(diff, 1),
            "Predicted Band": str(row["Predicted Band"]),
            "Confidence": "N/A" if pd.isna(confidence) else f"{float(confidence):.1f}%",
        })
    return pd.DataFrame(rows)


def professional_step_header(step, title, subtitle=""):
    """Clean section header used on the Prediction page."""
    st.markdown(
        f"""
        <div class="card" style="padding:18px 22px; margin-top:22px; margin-bottom:14px; border-left:5px solid #7C3AED;">
            <div style="display:flex; align-items:flex-start; gap:14px;">
                <div style="min-width:38px; height:38px; border-radius:12px; background:linear-gradient(135deg,#7C3AED,#5B21B6); color:#fff; display:flex; align-items:center; justify-content:center; font-weight:900;">{step}</div>
                <div>
                    <h3 style="margin:0 0 4px 0;">{title}</h3>
                    <p class="muted" style="margin:0; font-size:.92rem; line-height:1.5;">{subtitle}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compact_result_cards(summary_df, title):
    """Professional compact cards for all property types."""
    st.markdown(f"<h3 style='margin-top:10px;'>{title}</h3>", unsafe_allow_html=True)
    cols = st.columns(len(summary_df))
    for col, (_, row) in zip(cols, summary_df.iterrows()):
        ptype = str(row["Property Type"])
        band = str(row["Predicted Band"])
        score = float(row["Efficiency Score"])
        confidence = row.get("Confidence", np.nan)
        conf_text = "N/A" if pd.isna(confidence) else f"{float(confidence):.1f}%"
        line_color = PROPERTY_LINE_COLORS.get(ptype, "#7C3AED")
        band_color = RATING_COLORS.get(band, "#7C3AED")
        with col:
            st.markdown(
                f"""
                <div class="compare-card" style="min-height:150px; border-top:4px solid {line_color};">
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
                        <div>
                            <div class="small-label" style="color:{line_color};">{ptype}</div>
                            <div style="font-size:.82rem; color:#64748B; margin-top:3px;">Predicted EPC band</div>
                        </div>
                        <div style="width:54px; height:54px; border-radius:16px; background:{band_color}; display:flex; align-items:center; justify-content:center; font-size:30px; font-weight:950; color:#1E1B4B;">{band}</div>
                    </div>
                    <div style="margin-top:16px; font-size:1.35rem; font-weight:900; color:#1E1B4B;">{score:.1f}<span style="font-size:.8rem; color:#64748B; font-weight:700;"> / 100</span></div>
                    <div style="margin-top:5px; font-size:.84rem; color:#5B4B8A;">Confidence: <b>{conf_text}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def calculate_selected_period_accuracy(period_data):
    """Calculate dynamic accuracy for the selected dataset year/month.
    This does not replace the overall test-set accuracy. It evaluates the saved
    classifier on the records currently filtered by the user's selected period.
    """
    try:
        if period_data is None or period_data.empty:
            return np.nan, 0
        if COL_CURRENT not in period_data.columns:
            return np.nan, 0
        if model_df is None or len(model_df) == 0:
            return np.nan, 0

        # Keep row alignment between final_fe_data.csv and model_ready_data.csv.
        valid_idx = [idx for idx in period_data.index if idx in model_df.index]
        if not valid_idx:
            return np.nan, 0

        usable_features = [c for c in class_features if c in model_df.columns]
        if len(usable_features) != len(class_features):
            return np.nan, 0

        X_period = model_df.loc[valid_idx, usable_features].copy()
        actual = period_data.loc[valid_idx, COL_CURRENT].astype(str).values
        predicted_raw = classification_model.predict(X_period)
        predicted = [normalise_rating_prediction(x) for x in predicted_raw]
        accuracy = float(np.mean(np.array(predicted) == actual))
        return accuracy, len(valid_idx)
    except Exception:
        return np.nan, 0


def selected_period_accuracy_table(period_data):
    """Return per-rating accuracy table for optional review."""
    try:
        if period_data is None or period_data.empty or COL_CURRENT not in period_data.columns:
            return pd.DataFrame()
        valid_idx = [idx for idx in period_data.index if idx in model_df.index]
        usable_features = [c for c in class_features if c in model_df.columns]
        if not valid_idx or len(usable_features) != len(class_features):
            return pd.DataFrame()
        X_period = model_df.loc[valid_idx, usable_features].copy()
        actual = period_data.loc[valid_idx, COL_CURRENT].astype(str).values
        predicted_raw = classification_model.predict(X_period)
        predicted = np.array([normalise_rating_prediction(x) for x in predicted_raw])
        out = pd.DataFrame({"Actual Rating": actual, "Predicted Rating": predicted})
        out["Correct"] = out["Actual Rating"] == out["Predicted Rating"]
        summary = out.groupby("Actual Rating", as_index=False).agg(
            Records=("Actual Rating", "size"),
            Correct=("Correct", "sum"),
        )
        summary["Accuracy"] = (summary["Correct"] / summary["Records"] * 100).round(1).astype(str) + "%"
        return summary.sort_values("Actual Rating")
    except Exception:
        return pd.DataFrame()


def render_selected_period_accuracy_cards(period_data, period_label):
    """Show fixed overall model accuracy beside dynamic selected-period accuracy."""
    selected_acc, selected_n = calculate_selected_period_accuracy(period_data)
    selected_text = "N/A" if np.isnan(selected_acc) else f"{selected_acc:.1%}"
    selected_note = "No matching model-ready records" if selected_n == 0 else f"{selected_n:,} records in {period_label}"

    acc_cols = st.columns(3)
    acc_cols[0].markdown(
        f"""
        <div class="kpi-card" style="border-left:4px solid #7C3AED;">
            <div class="kpi-label">Overall Test Accuracy</div>
            <div class="kpi-value" style="color:#7C3AED;">{metrics['accuracy']:.1%}</div>
            <div class="kpi-note">Fixed held-out test-set result</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    acc_cols[1].markdown(
        f"""
        <div class="kpi-card" style="border-left:4px solid #059669;">
            <div class="kpi-label">Selected Period Accuracy</div>
            <div class="kpi-value" style="color:#059669;">{selected_text}</div>
            <div class="kpi-note">{selected_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    acc_cols[2].markdown(
        f"""
        <div class="kpi-card" style="border-left:4px solid #D97706;">
            <div class="kpi-label">Why Selected Accuracy Changes</div>
            <div class="kpi-value" style="font-size:1.0rem; color:#1E1B4B;">Month/year subset</div>
            <div class="kpi-note">Different periods contain different EPC records</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📋 Selected-period accuracy by EPC band", expanded=False):
        band_summary = selected_period_accuracy_table(period_data)
        if band_summary.empty:
            st.info("Selected-period band accuracy is unavailable for this period.")
        else:
            st.dataframe(band_summary, use_container_width=True, hide_index=True)


def selected_period_prediction_details(period_data):
    """Return actual vs predicted ratings for records in the selected period."""
    try:
        if period_data is None or period_data.empty or COL_CURRENT not in period_data.columns:
            return pd.DataFrame()
        if model_df is None or len(model_df) == 0:
            return pd.DataFrame()

        valid_idx = [idx for idx in period_data.index if idx in model_df.index]
        usable_features = [c for c in class_features if c in model_df.columns]
        if not valid_idx or len(usable_features) != len(class_features):
            return pd.DataFrame()

        X_period = model_df.loc[valid_idx, usable_features].copy()
        actual = period_data.loc[valid_idx, COL_CURRENT].astype(str).values
        predicted_raw = classification_model.predict(X_period)
        predicted = np.array([normalise_rating_prediction(x) for x in predicted_raw])
        out = pd.DataFrame({"Actual Rating": actual, "Predicted Rating": predicted})
        out["Correct"] = out["Actual Rating"] == out["Predicted Rating"]
        return out
    except Exception:
        return pd.DataFrame()


def selected_period_classification_report(details_df):
    """Create a small classification-report style table for selected-period records."""
    if details_df is None or details_df.empty:
        return pd.DataFrame()
    rows = []
    actual = details_df["Actual Rating"].astype(str)
    predicted = details_df["Predicted Rating"].astype(str)
    for band in RATING_ORDER:
        tp = int(((actual == band) & (predicted == band)).sum())
        fp = int(((actual != band) & (predicted == band)).sum())
        fn = int(((actual == band) & (predicted != band)).sum())
        support = int((actual == band).sum())
        if support == 0 and int((predicted == band).sum()) == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append({
            "Class": band,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "Support": support,
        })
    return pd.DataFrame(rows)


def selected_period_confusion_matrix_chart(details_df, title):
    """Confusion matrix for the selected month/year evaluation."""
    if details_df is None or details_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return dashboard_plot_layout(fig, 420)
    actual = pd.Categorical(details_df["Actual Rating"].astype(str), categories=RATING_ORDER, ordered=True)
    predicted = pd.Categorical(details_df["Predicted Rating"].astype(str), categories=RATING_ORDER, ordered=True)
    cm_df = pd.crosstab(actual, predicted, dropna=False).reindex(index=RATING_ORDER, columns=RATING_ORDER, fill_value=0)
    fig = px.imshow(
        cm_df.values,
        x=RATING_ORDER,
        y=RATING_ORDER,
        text_auto=True,
        color_continuous_scale=[[0, "#F5F3FF"], [0.3, "#C4B5FD"], [0.65, "#7C3AED"], [1.0, "#3B0764"]],
        labels=dict(x="Predicted Rating", y="Actual Rating", color="Number of properties"),
        title=title,
    )
    fig.update_coloraxes(showscale=False)
    return dashboard_plot_layout(fig, 430)


def render_model_selected_period_evaluation():
    """Interactive model-performance section where month/year changes selected-period accuracy."""
    st.markdown("<h3 style='margin-top:22px;'>Selected Period Evaluation</h3>", unsafe_allow_html=True)
    st.caption("Choose a dataset month/year to evaluate the same trained classifier on that subset. This changes selected-period accuracy, not the fixed overall test accuracy.")

    dated_perf, perf_date_col = get_dataset_date_view(df)
    perf_years = get_dataset_years(df)
    if not perf_years:
        st.info("Selected-period evaluation is unavailable because no valid inspection/lodgement years were found.")
        return

    p1, p2, p3 = st.columns([1, 1, 2])
    with p1:
        perf_year = st.selectbox("Evaluation Year", perf_years, index=len(perf_years) - 1, key="perf_eval_year")
    perf_months = get_dataset_months_for_year(df, perf_year) or sorted(dated_perf["__DATASET_MONTH"].dropna().astype(int).unique().tolist())
    if not perf_months:
        st.info("Selected-period evaluation is unavailable because no valid months were found.")
        return
    with p2:
        perf_month = st.selectbox("Evaluation Month", perf_months, index=len(perf_months) - 1, key="perf_eval_month", format_func=month_label)

    period_label = f"{MONTH_NAMES.get(int(perf_month), perf_month)} {int(perf_year)}"
    period_data = dated_perf[
        (dated_perf["__DATASET_YEAR"].astype("Int64") == int(perf_year))
        & (dated_perf["__DATASET_MONTH"].astype("Int64") == int(perf_month))
    ].copy()
    details_df = selected_period_prediction_details(period_data)
    selected_acc = float(details_df["Correct"].mean()) if not details_df.empty else np.nan
    selected_text = "N/A" if np.isnan(selected_acc) else f"{selected_acc:.1%}"
    diff_text = "N/A" if np.isnan(selected_acc) else f"{(selected_acc - float(metrics['accuracy'])):+.1%}"

    with p3:
        st.markdown(
            f"""<div class="action-note" style="margin-top:26px;">
            📅 Evaluating <b>{period_label}</b> using <b>{perf_date_col}</b> · <b style="color:#7C3AED;">{len(details_df):,}</b> model-ready records
            </div>""",
            unsafe_allow_html=True,
        )

    e1, e2, e3 = st.columns(3)
    e1.markdown(f"""<div class="kpi-card" style="border-left:4px solid #7C3AED;"><div class="kpi-label">Overall Test Accuracy</div><div class="kpi-value" style="color:#7C3AED;">{metrics['accuracy']:.1%}</div><div class="kpi-note">Fixed held-out test result</div></div>""", unsafe_allow_html=True)
    e2.markdown(f"""<div class="kpi-card" style="border-left:4px solid #059669;"><div class="kpi-label">Selected Period Accuracy</div><div class="kpi-value" style="color:#059669;">{selected_text}</div><div class="kpi-note">Changes when month/year changes</div></div>""", unsafe_allow_html=True)
    e3.markdown(f"""<div class="kpi-card" style="border-left:4px solid #D97706;"><div class="kpi-label">Difference vs Overall</div><div class="kpi-value" style="color:#D97706;">{diff_text}</div><div class="kpi-note">Selected subset compared with full test result</div></div>""", unsafe_allow_html=True)

    if details_df.empty:
        st.warning("No model-ready records are available for the selected period.")
        return

    s_left, s_right = st.columns([1.05, .95])
    with s_left:
        st.plotly_chart(selected_period_confusion_matrix_chart(details_df, f"Selected Period Confusion Matrix — {period_label}"), use_container_width=True)
    with s_right:
        st.subheader("Selected Period Classification Report")
        report_df = selected_period_classification_report(details_df)
        st.dataframe(report_df, use_container_width=True, hide_index=True)

    # Record-level actual-vs-predicted table removed from final dashboard for a cleaner presentation.


def current_dumbbell_chart(data, summary_df, metric_col=COL_SCORE, title="Dataset average vs model prediction"):
    """Easy grouped bar chart comparing actual dataset average vs model predicted score.
    This replaces the earlier line/dumbbell chart because grouped bars are easier to explain in a viva.
    """
    rows = []
    for _, row in summary_df.iterrows():
        ptype = str(row["Property Type"])
        ptype_data = data[data[COL_PROPERTY].astype(str) == ptype] if data is not None else pd.DataFrame()
        values = pd.to_numeric(ptype_data[metric_col], errors="coerce").dropna() if metric_col in ptype_data.columns else pd.Series(dtype=float)
        actual_avg = float(values.mean()) if len(values) else np.nan
        predicted = float(row["Efficiency Score"])
        rows.append({"Property Type": ptype, "Score Type": "Actual dataset average", "Score": actual_avg})
        rows.append({"Property Type": ptype, "Score Type": "Model prediction", "Score": predicted})

    plot_df = pd.DataFrame(rows).dropna(subset=["Score"])
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return dashboard_plot_layout(fig, 420)

    fig = px.bar(
        plot_df,
        x="Property Type",
        y="Score",
        color="Score Type",
        barmode="group",
        text="Score",
        title=title,
        labels={"Score": "EPC efficiency score (0–100)", "Property Type": "Property type"},
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(legend_title_text="Meaning")
    fig.update_yaxes(range=[0, 100], title_text="EPC efficiency score (0–100)")
    fig.update_xaxes(title_text="Property type")
    return dashboard_plot_layout(fig, 440)


def future_prediction_summary_table(base_user, month_nums, month_labels, pred_year):
    """Wide future prediction table: one row per property type."""
    rows = []
    for ptype in PROPERTY_TYPES_4:
        scores = []
        confidences = []
        for month_num in month_nums:
            p_user = build_property_type_values(base_user, ptype, month_num=month_num, prediction_year=pred_year)
            class_row, reg_row = build_model_input(p_user)
            score = float(np.clip(regression_model.predict(reg_row)[0], 1, 100))
            scores.append(score)
            try:
                proba = classification_model.predict_proba(class_row)[0]
                confidences.append(float(np.max(proba)) * 100)
            except Exception:
                pass
        avg_score = float(np.mean(scores)) if scores else np.nan
        row = {
            "Property Type": ptype,
            month_labels[0]: round(scores[0], 1) if len(scores) > 0 else "-",
            month_labels[1]: round(scores[1], 1) if len(scores) > 1 else "-",
            month_labels[2]: round(scores[2], 1) if len(scores) > 2 else "-",
            "Average Score": round(avg_score, 1) if not np.isnan(avg_score) else "-",
            "Predicted Band": score_to_rating(avg_score) if not np.isnan(avg_score) else "-",
            "Avg Confidence": "N/A" if not confidences else f"{float(np.mean(confidences)):.1f}%",
        }
        rows.append(row)
    return pd.DataFrame(rows)

def property_type_confidence_by_band_chart(summary_df, title="Prediction Confidence by Property Type"):
    fig = go.Figure()
    for _, row in summary_df.iterrows():
        ptype = str(row["Property Type"])
        band = str(row["Predicted Band"])
        confidence = 0 if pd.isna(row["Confidence"]) else float(row["Confidence"])
        color = PROPERTY_LINE_COLORS.get(ptype, "#7C3AED")
        fig.add_trace(go.Bar(
            x=[ptype],
            y=[confidence],
            name=f"{ptype} — Band {band}",
            marker=dict(color=color, line=dict(color="rgba(15,23,42,.14)", width=1)),
            text=[f"Band {band}<br>{confidence:.1f}%"],
            textposition="outside",
            hovertemplate=f"<b>{ptype}</b><br>Band {band}<br>Confidence: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(title=title, showlegend=True, barmode="group")
    fig.update_xaxes(title_text="Property type")
    fig.update_yaxes(title_text="Model confidence (%)", range=[0, 105])
    return dashboard_plot_layout(fig, 410)

def kpi_row():
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Actual Records</div>
            <div class="kpi-value">{len(df):,}</div>
            <div class="kpi-note">Cambridge EPC records</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Prediction Inputs</div>
            <div class="kpi-value">{len(class_features)} features</div>
            <div class="kpi-note">Number of input columns used by the model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Best Classifier</div>
            <div class="kpi-value" style="font-size:1.20rem; line-height:1.25; white-space:normal; overflow-wrap:anywhere;">{metrics["best_classification_model"]}</div>
            <div class="kpi-note">Final selected classification model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c4.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Accuracy</div>
            <div class="kpi-value">{metrics['accuracy']:.3f}</div>
            <div class="kpi-note">Test-set classification accuracy</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div style="display:flex; align-items:center;">
                <div class="logo-mark">E</div>
                <div>
                    <div style="font-size:1.06rem; font-weight:900; line-height:1.12;">Cambridge Energy<br>Rating Predictor</div>
                    <div style="font-size:.74rem; color:#475569; margin-top:4px;">Data Science Project</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        [
            "Welcome",
            "Prediction",
            "Interactive Dashboard",
            "Property Type Comparison",
            "Feature Explorer",
            "Model Performance",
            "Feature Importance",
            "Recommendations",
            "About & Glossary",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        f"""
        <div class="card-small">
            <div class="small-label">Dataset</div>
            <b>{len(df):,} actual EPC records</b><br>
            <span class="muted">Loaded from DS-BY-PINKY.xlsx</span>
        </div>
        <br>
        <div class="card-small">
            <div class="small-label">Project Details</div>
            <b>{PROJECT_TITLE}</b><br>
            <span class="muted">Prepared by {STUDENT_NAME}</span><br>
            <span class="muted">{PROJECT_TYPE}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Pages
# -----------------------------
if page == "Welcome":
    st.markdown(
        f"""
        <div class="hero">
            <div style="max-width:58%;">
                <div class="project-strip">{PROJECT_TITLE} · {PROJECT_TYPE}</div>
                <div class="small-label" style="color:#7C3AED;">Welcome to the</div>
                <div class="hero-title">Cambridge<br><span class="gradient-text">Energy Rating</span><br>Predictor</div>
                <p style="font-size:1.08rem; color:#334155; max-width:680px;">
                    A machine learning web application for predicting EPC ratings,
                    exploring Cambridge property energy data, and explaining model behaviour using actual project data.
                </p>
                <div class="project-meta">
                    <b>Student:</b> {STUDENT_NAME}<br>
                    <b>Project:</b> {PROJECT_TITLE}<br>
                    <b>Type:</b> {PROJECT_TYPE}
                </div>
                <div class="info-panel-row">
                    <div class="info-panel">Actual data connected</div>
                    <div class="info-panel">Interactive filters</div>
                    <div class="info-panel">Feature explorer</div>
                    <div class="info-panel">Explainable ML</div>
                </div>
            </div>
            <div style="position:absolute; right:55px; top:72px; width:420px; height:420px;">
                <div style="position:absolute; inset:0; border-radius:50%; background: conic-gradient(#7C3AED, #DB2777, #D97706, #059669, #5B21B6, #7C3AED); filter: drop-shadow(0 0 38px rgba(124,58,237,.26));"></div>
                <div style="position:absolute; inset:60px; border-radius:50%; background:#FFFFFF; border:1px solid rgba(109,40,217,.18);"></div>
                <div style="position:absolute; inset:142px; border-radius:32px; background:linear-gradient(145deg,rgba(109,40,217,.14),rgba(139,92,246,.12)); border:1px solid rgba(109,40,217,.18); display:flex; align-items:center; justify-content:center; font-size:76px;">📊</div>
                <div style="position:absolute; left:-105px; top:54px;" class="card-small"><b>EPC Analytics</b><br><span class="gradient-text">{len(df):,} records</span></div>
                <div style="position:absolute; right:0px; top:88px; min-width:260px;" class="card-small"><b>Classifier</b><br><span style="font-size:1.08rem; line-height:1.35; display:block; white-space:normal;">{metrics["best_classification_model"]}</span></div>
                <div style="position:absolute; right:-34px; bottom:76px;" class="card-small"><b>Accuracy</b><br><span style="font-size:1.7rem;color:#7C3AED;">{metrics["accuracy"]:.1%}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    kpi_row()
    st.markdown(
        """
        <div class="dashboard-note" style="margin-top:14px; display:grid; grid-template-columns:repeat(3, 1fr); gap:12px;">
            <div><b>Prediction</b><br><span style="font-size:.84rem;">A clean step-by-step input flow for current and future EPC results.</span></div>
            <div><b>Analytics</b><br><span style="font-size:.84rem;">Property, rating, cost, CO₂ and energy charts use the same filtered dataset.</span></div>
            <div><b>Model evidence</b><br><span style="font-size:.84rem;">Performance, feature importance and selected-period evaluation are kept together.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card'><h3>Interactive Prediction</h3><p>Enter physical and structural property details. Random Forest predicts the EPC band and Linear Regression predicts the efficiency score.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h3>Visual Analytics</h3><p>Users can filter by property type, current rating, potential rating, built form and metric.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><h3>Explainable AI</h3><p>Model performance, confusion matrix, classification report and feature importance are included.</p></div>", unsafe_allow_html=True)

    # ── Data Quality Summary ──────────────────────────────────────────────
    st.write("")
    st.markdown("<h2>🔬 Data Quality & EDA Summary</h2>", unsafe_allow_html=True)
    st.caption("Overview of the dataset used — key for academic transparency and supervisor review.")

    total_rows = len(df)
    total_cols = df.shape[1]
    missing_counts = df.isnull().sum()
    cols_with_missing = (missing_counts > 0).sum()
    numeric_cols_dq = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols_dq = df.select_dtypes(include="object").columns.tolist()
    band_counts = df[COL_CURRENT].value_counts()
    majority_band = band_counts.idxmax()
    imbalance_ratio = band_counts.max() / max(band_counts.min(), 1)

    dq1, dq2, dq3, dq4 = st.columns(4)
    dq1.markdown(f"""<div class="kpi-card"><div class="kpi-label">Total records</div><div class="kpi-value" style="color:#7C3AED;">{total_rows:,}</div><div class="kpi-note">Loaded from DS-BY-PINKY.xlsx</div></div>""", unsafe_allow_html=True)
    dq2.markdown(f"""<div class="kpi-card"><div class="kpi-label">Processed analytical columns</div><div class="kpi-value" style="color:#DB2777;">{total_cols}</div><div class="kpi-note">93 original variables; {len(numeric_cols_dq)} numeric and {len(cat_cols_dq)} categorical retained here</div></div>""", unsafe_allow_html=True)
    dq3.markdown(f"""<div class="kpi-card"><div class="kpi-label">Columns with missing</div><div class="kpi-value" style="color:#D97706;">{cols_with_missing}</div><div class="kpi-note">Out of {total_cols} total columns</div></div>""", unsafe_allow_html=True)
    dq4.markdown(f"""<div class="kpi-card"><div class="kpi-label">Class imbalance ratio</div><div class="kpi-value" style="color:#059669;">{imbalance_ratio:.1f}×</div><div class="kpi-note">Band {majority_band} is most common</div></div>""", unsafe_allow_html=True)

    st.write("")
    dq_left, dq_right = st.columns(2)
    with dq_left:
        # Missing values bar chart
        missing_df = missing_counts[missing_counts > 0].reset_index()
        missing_df.columns = ["Feature", "Missing"]
        if len(missing_df) > 0:
            missing_df = missing_df.sort_values("Missing", ascending=False).head(10)
            fig_missing = px.bar(missing_df, x="Feature", y="Missing", color="Missing",
                color_continuous_scale=[[0,"#C4B5FD"],[1,"#7C3AED"]],
                title="Top columns with missing values")
            fig_missing.update_coloraxes(showscale=False)
            fig_missing.update_traces(marker_line_color="rgba(255,255,255,0.3)", marker_line_width=0.5)
            st.plotly_chart(plot_layout(fig_missing, 340), use_container_width=True)
        else:
            st.markdown("<div class='card' style='text-align:center; padding:40px;'><h3 style='color:#059669;'>✅ No missing values</h3><p>The dataset is complete — no imputation required.</p></div>", unsafe_allow_html=True)
    with dq_right:
        # Class distribution bar
        band_df_dq = band_counts.reindex(RATING_ORDER, fill_value=0).reset_index()
        band_df_dq.columns = ["Band", "Count"]
        fig_bands = px.bar(band_df_dq, x="Band", y="Count",
            color="Band", color_discrete_map=DASHBOARD_RATING_COLORS,
            title="Class distribution across EPC bands",
            text="Count")
        fig_bands.update_traces(textposition="outside", marker_line_color="rgba(255,255,255,0.4)", marker_line_width=1)
        fig_bands.update_layout(showlegend=False)
        st.plotly_chart(plot_layout(fig_bands, 340), use_container_width=True)

    pipeline_steps = ["Raw data\nDS-BY-PINKY.xlsx", "EDA &\ncleaning", "Feature\nengineering", "Encoding &\nscaling", "Model\ntraining", "Evaluation\n& metrics", "Streamlit\ndeployment"]
    pipeline_html = "".join([
        f'<div style="text-align:center;"><div style="background:linear-gradient(135deg,#7C3AED,#5B21B6);color:white;border-radius:12px;padding:10px 14px;font-size:.82rem;font-weight:700;white-space:pre-line;line-height:1.3;">{s}</div></div>{"<div style=color:#7C3AED;font-size:1.2rem;>→</div>" if i < len(pipeline_steps)-1 else ""}' for i, s in enumerate(pipeline_steps)
    ])
    st.markdown(
        f"""
        <div class="card" style="margin-top:14px;">
            <h3>🔄 End-to-End Data Science Pipeline</h3>
            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-top:12px;">{pipeline_html}</div>
            <hr style="margin:14px 0; border-color:rgba(109,40,217,.13);">
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; font-size:.9rem; color:#5B4B8A;">
                <div><b style="color:#1E1B4B;">Dataset source:</b> Cambridge EPC property records (DS-BY-PINKY.xlsx)<br><b style="color:#1E1B4B;">Classification target:</b> Current EPC rating (A–G)<br><b style="color:#1E1B4B;">Regression target:</b> Energy efficiency score (1–100)</div>
                <div><b style="color:#1E1B4B;">Models trained:</b> Logistic Regression, Decision Tree, Random Forest (classification + regression)<br><b style="color:#1E1B4B;">Limitation:</b> Dataset limited to Cambridge — generalisability to other UK regions not guaranteed</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Prediction":
    kpi_row()
    st.write("")
    page_intro(
        "Leakage-Free EPC Prediction",
        "Predict an EPC band and efficiency score using the same 38 independent physical and structural predictors as the current notebook.",
        icon="🏠",
    )

    st.markdown(
        """
        <div class="dashboard-note" style="border-left:5px solid #059669;">
            <b>Current notebook model:</b> Random Forest classification + Linear Regression scoring.<br>
            Current/potential EPC outputs, energy consumption, CO₂, costs, environmental-impact values,
            assessment labels and target-encoded variables are <b>not</b> prediction inputs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    categorical_options = metadata.get("categorical_options", {})
    defaults = metadata.get("defaults", {})

    def _feature_options(feature):
        values = [str(v) for v in categorical_options.get(feature, []) if pd.notna(v)]
        default = str(defaults.get(feature, "Unknown"))
        if default not in values:
            values = [default] + values
        return values or [default]

    def _select_feature(label, feature, key):
        values = _feature_options(feature)
        default = str(defaults.get(feature, values[0]))
        index = values.index(default) if default in values else 0
        return st.selectbox(label, values, index=index, key=key)

    with st.form("current_notebook_prediction_form"):
        st.markdown("<h3>1. Core property details</h3>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            property_type = _select_feature("Property Type", "PROPERTY_TYPE", "new_property_type")
            built_form = _select_feature("Built Form", "BUILT_FORM", "new_built_form")
            construction_age_band = _select_feature("Construction Age Band", "CONSTRUCTION_AGE_BAND", "new_age_band")
        with c2:
            main_fuel = _select_feature("Main Fuel", "MAIN_FUEL", "new_main_fuel")
            mains_gas = _select_feature("Mains Gas Flag", "MAINS_GAS_FLAG", "new_mains_gas")
            mechanical_ventilation = _select_feature("Mechanical Ventilation", "MECHANICAL_VENTILATION", "new_ventilation")
        with c3:
            glazed_type = _select_feature("Glazed Type", "GLAZED_TYPE", "new_glazed_type")
            glazed_area = _select_feature("Glazed Area", "GLAZED_AREA", "new_glazed_area")
            solar_flag = _select_feature("Solar Water Heating", "SOLAR_WATER_HEATING_FLAG", "new_solar")

        with st.expander("🧱 Fabric, roof, windows and systems", expanded=True):
            f1, f2 = st.columns(2)
            with f1:
                walls_description = _select_feature("Walls Description", "WALLS_DESCRIPTION", "new_walls")
                roof_description = _select_feature("Roof Description", "ROOF_DESCRIPTION", "new_roof")
                windows_description = _select_feature("Windows Description", "WINDOWS_DESCRIPTION", "new_windows")
                floor_description = _select_feature("Floor Description", "FLOOR_DESCRIPTION", "new_floor_desc")
            with f2:
                mainheat_description = _select_feature("Main Heating Description", "MAINHEAT_DESCRIPTION", "new_mainheat")
                mainheatcont_description = _select_feature("Heating Controls", "MAINHEATCONT_DESCRIPTION", "new_heatcontrol")
                hotwater_description = _select_feature("Hot Water Description", "HOTWATER_DESCRIPTION", "new_hotwater")

        with st.expander("📐 Independent numerical property inputs", expanded=True):
            n1, n2, n3 = st.columns(3)
            with n1:
                floor_area = st.number_input("Total Floor Area (m²)", min_value=1.0, max_value=1000.0, value=float(defaults.get("TOTAL_FLOOR_AREA", 75.0)), step=1.0)
                floor_level = st.number_input("Floor Level", min_value=0.0, max_value=100.0, value=float(defaults.get("FLOOR_LEVEL", 1.0)), step=1.0)
                multi_glaze = st.number_input("Multi-Glaze Proportion (%)", min_value=0.0, max_value=100.0, value=float(defaults.get("MULTI_GLAZE_PROPORTION", 100.0)), step=1.0)
                extension_count = st.number_input("Extension Count", min_value=0.0, max_value=20.0, value=float(defaults.get("EXTENSION_COUNT", 0.0)), step=1.0)
            with n2:
                habitable_rooms = st.number_input("Habitable Rooms", min_value=1.0, max_value=100.0, value=float(defaults.get("NUMBER_HABITABLE_ROOMS", 4.0)), step=1.0)
                heated_rooms = st.number_input("Heated Rooms", min_value=0.0, max_value=100.0, value=float(defaults.get("NUMBER_HEATED_ROOMS", 4.0)), step=1.0)
                low_energy_lighting = st.number_input("Low-Energy Lighting (%)", min_value=0.0, max_value=100.0, value=float(defaults.get("LOW_ENERGY_LIGHTING", 100.0)), step=1.0)
                open_fireplaces = st.number_input("Open Fireplaces", min_value=0.0, max_value=20.0, value=float(defaults.get("NUMBER_OPEN_FIREPLACES", 0.0)), step=1.0)
            with n3:
                wind_turbines = st.number_input("Wind Turbine Count", min_value=0.0, max_value=20.0, value=float(defaults.get("WIND_TURBINE_COUNT", 0.0)), step=1.0)
                floor_height = st.number_input("Floor Height (m)", min_value=1.0, max_value=10.0, value=float(defaults.get("FLOOR_HEIGHT", 2.4)), step=0.1)
                photo_supply = st.number_input("Photovoltaic Supply", min_value=0.0, max_value=100.0, value=float(defaults.get("PHOTO_SUPPLY", 0.0)), step=0.1)
                fixed_lighting = st.number_input("Fixed Lighting Outlets", min_value=0.0, max_value=500.0, value=float(defaults.get("FIXED_LIGHTING_OUTLETS_COUNT", 10.0)), step=1.0)

        submitted = st.form_submit_button("⚡ Generate EPC Prediction", use_container_width=True)

    user_values = {
        "PROPERTY_TYPE": property_type,
        "BUILT_FORM": built_form,
        "CONSTRUCTION_AGE_BAND": construction_age_band,
        "MAIN_FUEL": main_fuel,
        "MAINS_GAS_FLAG": mains_gas,
        "GLAZED_TYPE": glazed_type,
        "GLAZED_AREA": glazed_area,
        "WALLS_DESCRIPTION": walls_description,
        "ROOF_DESCRIPTION": roof_description,
        "WINDOWS_DESCRIPTION": windows_description,
        "FLOOR_DESCRIPTION": floor_description,
        "MAINHEAT_DESCRIPTION": mainheat_description,
        "MAINHEATCONT_DESCRIPTION": mainheatcont_description,
        "HOTWATER_DESCRIPTION": hotwater_description,
        "MECHANICAL_VENTILATION": mechanical_ventilation,
        "SOLAR_WATER_HEATING_FLAG": solar_flag,
        "TOTAL_FLOOR_AREA": floor_area,
        "FLOOR_LEVEL": floor_level,
        "MULTI_GLAZE_PROPORTION": multi_glaze,
        "EXTENSION_COUNT": extension_count,
        "NUMBER_HABITABLE_ROOMS": habitable_rooms,
        "NUMBER_HEATED_ROOMS": heated_rooms,
        "LOW_ENERGY_LIGHTING": low_energy_lighting,
        "NUMBER_OPEN_FIREPLACES": open_fireplaces,
        "WIND_TURBINE_COUNT": wind_turbines,
        "FLOOR_HEIGHT": floor_height,
        "PHOTO_SUPPLY": photo_supply,
        "FIXED_LIGHTING_OUTLETS_COUNT": fixed_lighting,
    }

    if submitted:
        class_row, reg_row = build_model_input(user_values)
        classifier_band = normalise_rating_prediction(classification_model.predict(class_row)[0])
        predicted_score = float(np.clip(regression_model.predict(reg_row)[0], 1, 100))
        score_band = score_to_rating(predicted_score)
        confidence = np.nan
        try:
            probabilities = classification_model.predict_proba(class_row)[0]
            confidence = float(np.max(probabilities))
        except Exception:
            pass

        st.markdown("<h3 style='margin-top:22px;'>2. Prediction Results</h3>", unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        band_colour = RATING_COLORS.get(classifier_band, "#7C3AED")
        r1.markdown(f"""<div class="kpi-card"><div class="kpi-label">Classifier EPC Band</div><div class="kpi-value" style="color:{band_colour};">{classifier_band}</div><div class="kpi-note">Random Forest classification</div></div>""", unsafe_allow_html=True)
        r2.markdown(f"""<div class="kpi-card"><div class="kpi-label">Predicted Efficiency Score</div><div class="kpi-value" style="color:#059669;">{predicted_score:.1f}</div><div class="kpi-note">Linear Regression score</div></div>""", unsafe_allow_html=True)
        r3.markdown(f"""<div class="kpi-card"><div class="kpi-label">Score-Derived Band</div><div class="kpi-value" style="color:{RATING_COLORS.get(score_band, '#7C3AED')};">{score_band}</div><div class="kpi-note">Band mapped from predicted score</div></div>""", unsafe_allow_html=True)
        confidence_text = "N/A" if pd.isna(confidence) else f"{confidence:.1%}"
        r4.markdown(f"""<div class="kpi-card"><div class="kpi-label">Classifier Confidence</div><div class="kpi-value" style="color:#DB2777;">{confidence_text}</div><div class="kpi-note">Maximum class probability</div></div>""", unsafe_allow_html=True)

        with st.expander("View the exact 38-feature model row", expanded=False):
            display_row = class_row.T.reset_index()
            display_row.columns = ["Model Feature", "Value"]
            display_row["Value"] = display_row["Value"].astype(str)
            st.dataframe(display_row, use_container_width=True, hide_index=True)

        st.info("This prediction is for screening and programme planning only. It does not replace a professional EPC survey.")

elif page == "Interactive Dashboard":
    page_intro(
        "Current Energy Analytics Dashboard",
        "Explore actual Cambridge EPC records by year, month, property type and EPC band. Select a property type first so every chart updates clearly for that group.",
        icon="📊",
    )
    default_metric = "Energy Efficiency Score"
    property_placeholder = "Select property type"

    dashboard_df, dashboard_date_col = get_dataset_date_view(df)
    available_years = get_dataset_years(df)
    if not available_years:
        st.error("No valid dataset years were found in INSPECTION_DATE, LODGEMENT_DATE, or INSPECTION_YEAR.")
        st.stop()

    default_year = max(available_years)
    if "dashboard_year" not in st.session_state or st.session_state["dashboard_year"] not in available_years:
        st.session_state["dashboard_year"] = default_year

    selected_year_pre = int(st.session_state["dashboard_year"])
    months_for_year_pre = get_dataset_months_for_year(df, selected_year_pre)
    if not months_for_year_pre:
        months_for_year_pre = sorted(dashboard_df["__DATASET_MONTH"].dropna().astype(int).unique().tolist())
    default_month = max(months_for_year_pre) if months_for_year_pre else 1
    if "dashboard_month" not in st.session_state or st.session_state["dashboard_month"] not in months_for_year_pre:
        st.session_state["dashboard_month"] = default_month

    if "dashboard_property_type" not in st.session_state:
        st.session_state["dashboard_property_type"] = property_placeholder
    if "dashboard_current_epc" not in st.session_state:
        st.session_state["dashboard_current_epc"] = "All"
    if "dashboard_potential_epc" not in st.session_state:
        st.session_state["dashboard_potential_epc"] = "All"
    if "dashboard_metric" not in st.session_state:
        st.session_state["dashboard_metric"] = default_metric

    def reset_dashboard_filters():
        st.session_state["dashboard_year"] = default_year
        reset_months = get_dataset_months_for_year(df, default_year)
        st.session_state["dashboard_month"] = max(reset_months) if reset_months else 1
        st.session_state["dashboard_property_type"] = property_placeholder
        st.session_state["dashboard_current_epc"] = "All"
        st.session_state["dashboard_potential_epc"] = "All"
        st.session_state["dashboard_metric"] = default_metric

    # Main filters: property type is intentionally not selected by default.
    property_options = [property_placeholder, "All property types"] + get_options(COL_PROPERTY)
    filter_top = st.columns([.9, 1.05, 1.45, 1.25, .72])
    selected_year = filter_top[0].selectbox(
        "Year",
        available_years,
        key="dashboard_year",
        help="Only years present in the uploaded dataset are shown.",
    )

    months_for_year = get_dataset_months_for_year(df, selected_year)
    if not months_for_year:
        months_for_year = sorted(dashboard_df["__DATASET_MONTH"].dropna().astype(int).unique().tolist())
    if st.session_state.get("dashboard_month") not in months_for_year:
        st.session_state["dashboard_month"] = max(months_for_year) if months_for_year else 1
    selected_month = filter_top[1].selectbox(
        "Month",
        months_for_year,
        key="dashboard_month",
        format_func=month_label,
        help="Only months available for the selected dataset year are shown.",
    )

    # Keep old state safe if a previous version stored a value that is not in the new options.
    if st.session_state.get("dashboard_property_type") not in property_options:
        st.session_state["dashboard_property_type"] = property_placeholder
    selected_property_type = filter_top[2].selectbox(
        "Property Type",
        property_options,
        key="dashboard_property_type",
        help="Select a property type to activate the dashboard. Choose All property types for an overall comparison.",
    )
    metric_label = filter_top[3].selectbox(
        "Metric",
        ["Energy Efficiency Score", "CO₂ Emissions", "Energy Consumption", "Heating Cost", "Floor Area"],
        key="dashboard_metric",
    )
    filter_top[4].button("Reset", use_container_width=True, on_click=reset_dashboard_filters)

    filter_bottom = st.columns([1, 1, 2.8])
    current_filter = filter_bottom[0].selectbox("Current EPC Rating", ["All"] + RATING_ORDER, key="dashboard_current_epc")
    potential_filter = filter_bottom[1].selectbox("Potential EPC Rating", ["All"] + RATING_ORDER, key="dashboard_potential_epc")
    with filter_bottom[2]:
        st.markdown(
            f"""
            <div class="dashboard-note" style="margin-top:24px;">
                <b>Dataset-only date filter:</b> using <b>{dashboard_date_col}</b>. Selected period: <b>{MONTH_NAMES.get(int(selected_month), selected_month)} {int(selected_year)}</b>. Property type must be selected before charts are shown.
            </div>
            """,
            unsafe_allow_html=True,
        )

    metric_cols = {
        "Energy Efficiency Score": COL_SCORE,
        "CO₂ Emissions": COL_CO2,
        "Energy Consumption": COL_ENERGY,
        "Heating Cost": COL_HEATING,
        "Floor Area": COL_AREA,
    }
    selected_metric = metric_cols[metric_label]

    period_data = dashboard_df[
        (dashboard_df["__DATASET_YEAR"].astype("Int64") == int(selected_year))
        & (dashboard_df["__DATASET_MONTH"].astype("Int64") == int(selected_month))
    ].copy()

    if selected_property_type == property_placeholder:
        st.markdown(
            """
            <div class="card" style="margin-top:18px; border-left:5px solid #7C3AED;">
                <h3>Choose a property type to start</h3>
                <p class="muted" style="line-height:1.6; margin-bottom:0;">
                    This dashboard is for actual dataset exploration. Select <b>House</b>, <b>Flat</b>, <b>Bungalow</b>, <b>Maisonette</b>, or <b>All property types</b>. After selection, the EPC distribution, current vs potential ratings, metric charts and summary will update for that group.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        demo_cols = st.columns(3)
        demo_cols[0].markdown("<div class='card-small'><div class='small-label'>Step 1</div><b>Select property type</b><br><span class='muted'>No default type is selected.</span></div>", unsafe_allow_html=True)
        demo_cols[1].markdown("<div class='card-small'><div class='small-label'>Step 2</div><b>Choose metric</b><br><span class='muted'>Score, CO₂, energy, cost or floor area.</span></div>", unsafe_allow_html=True)
        demo_cols[2].markdown("<div class='card-small'><div class='small-label'>Step 3</div><b>Read graphs</b><br><span class='muted'>Charts update only for the selected group.</span></div>", unsafe_allow_html=True)
        st.stop()

    property_filter = "All" if selected_property_type == "All property types" else selected_property_type
    filtered = apply_filters(period_data, property_filter, current_filter, potential_filter)
    period_label = f"{MONTH_NAMES.get(int(selected_month), selected_month)} {int(selected_year)}"
    property_label = "All property types" if property_filter == "All" else selected_property_type

    if filtered.empty:
        st.warning("No records match the selected year, month, property type and EPC filters. Choose another dataset period or remove filters.")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matching Records", f"{len(filtered):,}")
    m2.metric("Average Score", f"{filtered[COL_SCORE].mean():.1f}" if len(filtered) else "0")
    m3.metric("Average CO₂", f"{filtered[COL_CO2].mean():.2f}" if len(filtered) else "0")
    m4.metric("Most Common Rating", filtered[COL_CURRENT].mode()[0] if len(filtered) else "-")

    st.markdown(
        f"""
        <div class="dashboard-note">
            <b>Active view:</b> {property_label} · {period_label} · Current EPC: <b>{current_filter}</b> · Potential EPC: <b>{potential_filter}</b> · Metric: <b>{metric_label}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top1, top2 = st.columns(2)
    with top1:
        st.plotly_chart(rating_ring_chart(filtered), use_container_width=True)
        st.caption(f"Current EPC rating mix for {property_label} in {period_label}.")
    with top2:
        st.plotly_chart(current_vs_potential_chart(filtered), use_container_width=True)
        st.caption(f"Current vs potential EPC ratings for {property_label} in {period_label}.")

    def selected_vs_all_metric_chart(all_period_data, selected_data, metric_col, metric_name, selected_name):
        all_value = pd.to_numeric(all_period_data[metric_col], errors="coerce").mean() if metric_col in all_period_data.columns and len(all_period_data) else np.nan
        selected_value = pd.to_numeric(selected_data[metric_col], errors="coerce").mean() if metric_col in selected_data.columns and len(selected_data) else np.nan
        chart_df = pd.DataFrame({
            "Group": ["All property types", selected_name],
            "Average": [all_value, selected_value],
        }).dropna(subset=["Average"])
        fig = px.bar(
            chart_df,
            x="Group",
            y="Average",
            color="Group",
            text="Average",
            title=f"{selected_name} vs All Properties — Average {metric_name}",
            labels={"Average": f"Average {metric_name}"},
            color_discrete_sequence=["#7C3AED", "#0891B2", "#DB2777"],
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(showlegend=False)
        return dashboard_plot_layout(fig, 420)

    st.markdown("<h3>Property-Type Analysis</h3>", unsafe_allow_html=True)
    if property_filter == "All":
        st.plotly_chart(property_type_comparison_chart(filtered, selected_metric, metric_label), use_container_width=True)
        st.caption("Comparison across all property types for the selected dataset period and metric.")
        bottom1, bottom2 = st.columns(2)
        with bottom1:
            st.plotly_chart(property_type_chart(filtered), use_container_width=True)
            st.caption("Record count by property type in the selected period.")
        with bottom2:
            st.plotly_chart(metric_box_chart(filtered, selected_metric, metric_label), use_container_width=True)
            st.caption("Selected metric spread by EPC rating for all property types.")
    else:
        bottom1, bottom2 = st.columns(2)
        with bottom1:
            st.plotly_chart(selected_vs_all_metric_chart(period_data, filtered, selected_metric, metric_label, selected_property_type), use_container_width=True)
            st.caption("This shows whether the selected property type is above or below the overall period average.")
        with bottom2:
            st.plotly_chart(metric_box_chart(filtered, selected_metric, metric_label), use_container_width=True)
            st.caption(f"Selected metric spread by EPC rating for {selected_property_type} only.")

    most_common = filtered[COL_CURRENT].mode()[0] if len(filtered) else "-"
    avg_score = filtered[COL_SCORE].mean() if len(filtered) else 0
    avg_co2 = filtered[COL_CO2].mean() if len(filtered) else 0
    pct_above_c = (filtered[COL_CURRENT].isin(["A", "B", "C"]).sum() / max(len(filtered), 1) * 100)

    one_step_to_c = filtered[filtered[COL_CURRENT] == "D"]
    two_step_to_c = filtered[filtered[COL_CURRENT] == "E"]
    already_c_or_above = filtered[filtered[COL_CURRENT].isin(["A", "B", "C"])]

    st.markdown(
        f"""
        <div class="card" style="margin-top:14px;">
            <h3>📊 Interactive Dashboard Summary</h3>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:8px;">
                <div>
                    <p>• <b style="color:#7C3AED;">{len(filtered):,}</b> records match <b>{property_label}</b> in <b>{period_label}</b>.</p>
                    <p>• Most common EPC rating: <b style="color:{DASHBOARD_RATING_COLORS.get(most_common, "#1E1B4B")};">Band {most_common}</b>.</p>
                    <p>• Average energy efficiency score: <b style="color:#7C3AED;">{avg_score:.1f} / 100</b>.</p>
                </div>
                <div>
                    <p>• Average CO₂ emissions: <b style="color:#DB2777;">{avg_co2:.2f} tonnes/year</b> per property.</p>
                    <p>• <b>{pct_above_c:.1f}%</b> of selected records are rated Band C or above.</p>
                    <p>• Selected comparison metric: <b>{metric_label}</b>.</p>
                </div>
            </div>
            <hr style="margin:12px 0; border-color:rgba(109,40,217,.13);">
            <h3 style="margin-top:4px;">🎯 1-Band-Away Analysis</h3>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-top:10px;">
                <div class="card-small" style="border-left:4px solid #059669;">
                    <div class="kpi-label">Already Band C or above</div>
                    <div class="kpi-value" style="color:#059669;">{len(already_c_or_above):,}</div>
                    <div class="kpi-note">{len(already_c_or_above)/max(len(filtered),1)*100:.1f}% of selected records</div>
                </div>
                <div class="card-small" style="border-left:4px solid #D97706;">
                    <div class="kpi-label">1 step away (Band D)</div>
                    <div class="kpi-value" style="color:#D97706;">{len(one_step_to_c):,}</div>
                    <div class="kpi-note">{len(one_step_to_c)/max(len(filtered),1)*100:.1f}% of selected records</div>
                </div>
                <div class="card-small" style="border-left:4px solid #DC2626;">
                    <div class="kpi-label">2 steps away (Band E)</div>
                    <div class="kpi-value" style="color:#DC2626;">{len(two_step_to_c):,}</div>
                    <div class="kpi-note">{len(two_step_to_c)/max(len(filtered),1)*100:.1f}% of selected records</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("<h3>🔥 Feature Correlation Heatmap</h3>", unsafe_allow_html=True)
    st.caption("Shows numeric relationships for the selected property type and month/year.")
    numeric_cols_for_corr = [c for c in [COL_SCORE, COL_POT_SCORE, COL_CO2, COL_CO2_PER, COL_ENERGY, COL_HEATING, COL_HOT_WATER, COL_LIGHTING, COL_AREA] if c in filtered.columns]
    if len(numeric_cols_for_corr) >= 3 and len(filtered) > 5:
        corr_matrix = filtered[numeric_cols_for_corr].corr().round(2)
        short_labels = [c.replace("CURRENT_ENERGY_", "").replace("_CURRENT", "").replace("CO2_EMISSIONS", "CO₂").replace("POTENTIAL_ENERGY_", "").replace("_", "_")[:18] for c in numeric_cols_for_corr]
        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=short_labels, y=short_labels,
            colorscale=[[0, "#DB2777"], [0.5, "#FFFFFF"], [1, "#7C3AED"]],
            zmin=-1, zmax=1,
            text=corr_matrix.values.round(2),
            texttemplate="%{text:.2f}",
            textfont=dict(size=11),
            hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
        ))
        fig_corr.update_layout(title=f"Pearson Correlation Matrix — {property_label}, {period_label}")
        st.plotly_chart(dashboard_plot_layout(fig_corr, 440), use_container_width=True)
    else:
        st.info("The selected property type and month/year needs at least 6 records to show the correlation heatmap.")


elif page == "Feature Explorer":
    kpi_row()
    st.write("")
    page_intro(
        "Feature Explorer and Custom Visualization",
        "Create simple, supervisor-friendly charts for engineered and numeric features, with filters for property type and EPC rating.",
        icon="🧭",
    )
    feature_df = df.reset_index(drop=True).copy()
    engineered_cols_to_add = [
        "CURRENT_RATING_SCORE",
        "POTENTIAL_RATING_SCORE",
        "PROPERTY_TYPE_Target",
        "FLOOR_AREA_GROUP_SCORE",
        "CONSTRUCTION_AGE_GROUP_SCORE",
        "ENERGY_CONSUMPTION_GROUP_SCORE",
        "MAIN_FUEL_GROUP_SCORE",
        "MAINS_GAS_FLAG_Label",
    ]
    for col in engineered_cols_to_add:
        if col in model_df.columns and col not in feature_df.columns:
            feature_df[col] = model_df[col].reset_index(drop=True)

    feature_labels = {
        COL_SCORE: "Current energy efficiency score",
        COL_POT_SCORE: "Potential energy efficiency score",
        "CURRENT_RATING_SCORE": "Current rating encoded score",
        "POTENTIAL_RATING_SCORE": "Potential rating encoded score",
        "ENVIRONMENT_IMPACT_CURRENT": "Current environment impact",
        "ENVIRONMENT_IMPACT_POTENTIAL": "Potential environment impact",
        COL_ENERGY: "Current energy consumption",
        "ENERGY_CONSUMPTION_POTENTIAL": "Potential energy consumption",
        "ENERGY_CONSUMPTION_GROUP_SCORE": "Energy consumption group score",
        COL_CO2: "Current CO₂ emissions",
        "CO2_EMISSIONS_POTENTIAL": "Potential CO₂ emissions",
        COL_CO2_PER: "CO₂ per floor area",
        COL_HEATING: "Current heating cost",
        "HEATING_COST_POTENTIAL": "Potential heating cost",
        COL_HOT_WATER: "Current hot water cost",
        "HOT_WATER_COST_POTENTIAL": "Potential hot water cost",
        COL_LIGHTING: "Current lighting cost",
        "LIGHTING_COST_POTENTIAL": "Potential lighting cost",
        COL_AREA: "Total floor area",
        "FLOOR_AREA_GROUP_SCORE": "Floor area group score",
        "PROPERTY_TYPE_Target": "Property type target encoded score",
        "CONSTRUCTION_AGE_GROUP_SCORE": "Construction age group score",
        "MAIN_FUEL_GROUP_SCORE": "Main fuel group score",
        "MAINS_GAS_FLAG_Label": "Mains gas encoded flag",
        "LOW_ENERGY_LIGHTING": "Low energy lighting",
        "MULTI_GLAZE_PROPORTION": "Multi-glaze proportion",
        "NUMBER_HABITABLE_ROOMS": "Number of habitable rooms",
        "NUMBER_HEATED_ROOMS": "Number of heated rooms",
        "FLOOR_HEIGHT": "Floor height",
        "EXTENSION_COUNT": "Extension count",
    }
    y_feature_cols = [col for col in feature_labels if col in feature_df.columns and pd.api.types.is_numeric_dtype(feature_df[col])]
    if not y_feature_cols:
        y_feature_cols = feature_df.select_dtypes(include=np.number).columns.tolist()

    category_labels = {
        COL_PROPERTY: "Property type",
        COL_CURRENT: "Current EPC rating",
        COL_POTENTIAL: "Potential EPC rating",
        COL_BUILT: "Built form",
        COL_AGE_GROUP: "Construction age group",
        COL_FUEL_GROUP: "Main fuel group",
        COL_GLAZED: "Glazed type",
        COL_TARIFF: "Energy tariff",
    }
    category_cols = [col for col in category_labels if col in feature_df.columns]
    colour_cols = [col for col in [COL_PROPERTY, COL_CURRENT, COL_POTENTIAL, COL_BUILT, COL_AGE_GROUP, COL_FUEL_GROUP] if col in feature_df.columns]

    def clean_label(col):
        return feature_labels.get(col, category_labels.get(col, str(col).replace("_", " ").title()))

    def colour_map(col):
        if col in [COL_CURRENT, COL_POTENTIAL]:
            return RATING_COLORS
        return None

    def axis_order(col):
        if col in [COL_CURRENT, COL_POTENTIAL]:
            return RATING_ORDER
        if col == COL_PROPERTY:
            return get_options(COL_PROPERTY)
        return sorted(feature_df[col].dropna().astype(str).unique().tolist())

    st.markdown(
        """
        <div class="dashboard-note">
            <b>How to use:</b> Property Type is the default comparison axis, and the Y-axis contains the main feature-engineering / numeric project features. Use simple charts such as bar, line, box, scatter, histogram and pie to compare the property types available in the dataset clearly.
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_options = ["Bar Chart", "Line Chart", "Box Plot", "Scatter Plot", "Histogram", "Pie Chart"]
    top_c1, top_c2, top_c3, top_c4 = st.columns([1.0, 1.15, 1.35, 1.15])
    chart_type = top_c1.selectbox("Chart Type", chart_options, index=0, key="feature_chart_type")

    default_y = COL_ENERGY if COL_ENERGY in y_feature_cols else y_feature_cols[0]
    if chart_type == "Scatter Plot":
        default_x = COL_AREA if COL_AREA in y_feature_cols else y_feature_cols[0]
        x_axis = top_c2.selectbox(
            "X-Axis Feature",
            y_feature_cols,
            index=y_feature_cols.index(default_x),
            format_func=clean_label,
            key="feature_x_numeric",
        )
    elif chart_type == "Histogram":
        x_axis = top_c2.selectbox(
            "Feature to Plot",
            y_feature_cols,
            index=y_feature_cols.index(default_y),
            format_func=clean_label,
            key="feature_hist_x",
        )
    else:
        default_x = COL_PROPERTY if COL_PROPERTY in category_cols else category_cols[0]
        x_axis = top_c2.selectbox(
            "X-Axis / Compare By",
            category_cols,
            index=category_cols.index(default_x),
            format_func=clean_label,
            key="feature_x_category",
        )

    y_axis = top_c3.selectbox(
        "Y-Axis Feature",
        y_feature_cols,
        index=y_feature_cols.index(default_y),
        format_func=clean_label,
        key="feature_y_metric",
        help="These are the important numeric and feature-engineered fields used in the project.",
    )
    color_by = top_c4.selectbox(
        "Colour By",
        colour_cols,
        index=colour_cols.index(COL_PROPERTY) if COL_PROPERTY in colour_cols else 0,
        format_func=clean_label,
        key="feature_colour_by",
    )

    filter_c1, filter_c2, filter_c3 = st.columns([1.25, 1.25, .7])
    property_options = get_options(COL_PROPERTY)
    selected_properties = filter_c1.multiselect(
        "Property Type Filter",
        property_options,
        default=property_options,
        key="feature_property_filter",
        help="Keep all selected to compare all dataset property types together.",
    )
    selected_ratings = filter_c2.multiselect(
        "Current EPC Rating Filter",
        RATING_ORDER,
        default=RATING_ORDER,
        key="feature_rating_filter",
    )
    show_trend = filter_c3.toggle("Trendline", value=True, key="feature_trendline", disabled=(chart_type != "Scatter Plot"))

    custom_df = feature_df[
        feature_df[COL_PROPERTY].astype(str).isin(selected_properties)
        & feature_df[COL_CURRENT].astype(str).isin(selected_ratings)
    ].copy()

    if custom_df.empty:
        st.warning("No records match the selected filters. Select more property types or EPC ratings to show the charts.")
    else:
        x_label = clean_label(x_axis)
        y_label = clean_label(y_axis)

        if chart_type in ["Bar Chart", "Line Chart"]:
            group_cols = [x_axis]
            if color_by != x_axis and color_by in custom_df.columns:
                group_cols.append(color_by)
            plot_df = custom_df.groupby(group_cols, as_index=False)[y_axis].mean()
            order = axis_order(x_axis)
            plot_df[x_axis] = pd.Categorical(plot_df[x_axis].astype(str), categories=order, ordered=True)
            plot_df = plot_df.sort_values(x_axis)
            if chart_type == "Bar Chart":
                fig = px.bar(
                    plot_df,
                    x=x_axis,
                    y=y_axis,
                    color=color_by if color_by in plot_df.columns else x_axis,
                    barmode="group",
                    text_auto=".1f",
                    color_discrete_map=colour_map(color_by),
                    title=f"Average {y_label} by {x_label}",
                    labels={x_axis: x_label, y_axis: y_label},
                )
                fig.update_traces(textposition="outside")
            else:
                fig = px.line(
                    plot_df,
                    x=x_axis,
                    y=y_axis,
                    color=color_by if color_by in plot_df.columns else None,
                    markers=True,
                    color_discrete_map=colour_map(color_by),
                    title=f"Average {y_label} trend by {x_label}",
                    labels={x_axis: x_label, y_axis: y_label},
                )
                fig.update_traces(line=dict(width=3), marker=dict(size=8))
        elif chart_type == "Box Plot":
            fig = px.box(
                custom_df,
                x=x_axis,
                y=y_axis,
                color=color_by,
                points="outliers",
                color_discrete_map=colour_map(color_by),
                title=f"{y_label} spread by {x_label}",
                labels={x_axis: x_label, y_axis: y_label},
            )
        elif chart_type == "Scatter Plot":
            fig = px.scatter(
                custom_df,
                x=x_axis,
                y=y_axis,
                color=color_by,
                opacity=.68,
                color_discrete_map=colour_map(color_by),
                title=f"{y_label} vs {x_label}",
                labels={x_axis: x_label, y_axis: y_label},
            )
            if show_trend and x_axis != y_axis and len(custom_df) > 2:
                x = pd.to_numeric(custom_df[x_axis], errors="coerce").astype(float).values
                y = pd.to_numeric(custom_df[y_axis], errors="coerce").astype(float).values
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() > 2:
                    coef = np.polyfit(x[mask], y[mask], 1)
                    xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
                    ys = coef[0] * xs + coef[1]
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Trendline", line=dict(color="#0F172A", dash="dash", width=3)))
        elif chart_type == "Histogram":
            fig = px.histogram(
                custom_df,
                x=x_axis,
                color=color_by,
                nbins=34,
                marginal="box",
                color_discrete_map=colour_map(color_by),
                title=f"{clean_label(x_axis)} distribution",
                labels={x_axis: clean_label(x_axis)},
            )
        else:
            pie_df = custom_df[x_axis].astype(str).value_counts().reset_index()
            pie_df.columns = [x_axis, "Records"]
            fig = px.pie(
                pie_df,
                names=x_axis,
                values="Records",
                hole=.46,
                title=f"Record share by {x_label}",
                color=x_axis if x_axis in [COL_CURRENT, COL_POTENTIAL] else None,
                color_discrete_map=RATING_COLORS if x_axis in [COL_CURRENT, COL_POTENTIAL] else None,
            )

        st.plotly_chart(plot_layout(fig, 520), use_container_width=True)

        # ── Metrics strip ────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Data Points", f"{len(custom_df):,}")
        k2.metric(f"Avg {y_label[:28]}", f"{custom_df[y_axis].mean():.1f}")
        k3.metric("Top Property Type", custom_df[COL_PROPERTY].mode()[0] if len(custom_df) else "-")
        corr_value = 0
        if chart_type == "Scatter Plot" and x_axis != y_axis and len(custom_df) > 1:
            corr_value = custom_df[[x_axis, y_axis]].corr().iloc[0, 1]
        k4.metric("Correlation", f"{corr_value:.2f}" if chart_type == "Scatter Plot" and np.isfinite(corr_value) else "N/A")

        st.write("")
        # ── Two comparison charts side by side ───────────────────────────
        lower_left, lower_right = st.columns(2)
        with lower_left:
            prop_df = custom_df.groupby(COL_PROPERTY, as_index=False)[y_axis].mean()
            prop_order = get_options(COL_PROPERTY)
            prop_df[COL_PROPERTY] = pd.Categorical(prop_df[COL_PROPERTY].astype(str), categories=prop_order, ordered=True)
            prop_df = prop_df.sort_values(COL_PROPERTY)
            prop_fig = px.bar(
                prop_df,
                x=COL_PROPERTY,
                y=y_axis,
                color=COL_PROPERTY,
                color_discrete_sequence=DASHBOARD_PROPERTY_COLORS,
                text_auto=".1f",
                title=f"By property type — {y_label}",
                labels={COL_PROPERTY: "Property type", y_axis: y_label},
            )
            prop_fig.update_traces(textposition="outside", marker_line_color="rgba(255,255,255,0.5)", marker_line_width=1)
            prop_fig.update_layout(showlegend=False)
            st.plotly_chart(plot_layout(prop_fig, 360), use_container_width=True)

        with lower_right:
            rating_df = custom_df.groupby(COL_CURRENT, as_index=False)[y_axis].mean()
            rating_df[COL_CURRENT] = pd.Categorical(rating_df[COL_CURRENT].astype(str), categories=RATING_ORDER, ordered=True)
            rating_df = rating_df.sort_values(COL_CURRENT)
            rating_fig = px.bar(
                rating_df,
                x=COL_CURRENT,
                y=y_axis,
                color=COL_CURRENT,
                color_discrete_map=RATING_COLORS,
                text_auto=".1f",
                title=f"By EPC band — {y_label}",
                labels={COL_CURRENT: "EPC rating", y_axis: y_label},
            )
            rating_fig.update_traces(textposition="outside", marker_line_color="rgba(255,255,255,0.5)", marker_line_width=1)
            rating_fig.update_layout(showlegend=False)
            st.plotly_chart(plot_layout(rating_fig, 360), use_container_width=True)

        # ── Insights summary card at the bottom ──────────────────────────
        avg_val = custom_df[y_axis].mean()
        top_prop = custom_df.groupby(COL_PROPERTY)[y_axis].mean().idxmax() if len(custom_df) else "-"
        top_band = custom_df.groupby(COL_CURRENT)[y_axis].mean().idxmax() if len(custom_df) else "-"
        st.markdown(
            f"""
            <div class="card" style="margin-top:8px;">
                <h3>📊 Dashboard Insights</h3>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:8px;">
                    <div>
                        <p>• Chart type selected: <b>{chart_type}</b> comparing <b>{y_label}</b> across <b>{x_label}</b>.</p>
                        <p>• Dataset contains <b style="color:#2563EB;">{len(custom_df):,}</b> matching properties after filters.</p>
                        <p>• Average <b>{y_label}</b> across all records: <b style="color:#7C3AED;">{avg_val:.1f}</b>.</p>
                    </div>
                    <div>
                        <p>• Property type with highest average <b>{y_label}</b>: <b style="color:#DB2777;">{top_prop}</b>.</p>
                        <p>• EPC band with highest average <b>{y_label}</b>: <b style="color:#D97706;">Band {top_band}</b>.</p>
                        <p>• Use <b>Bar Chart</b> or <b>Box Plot</b> for clearest presentation to supervisor.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

elif page == "Model Performance":
    kpi_row()
    st.write("")
    page_intro(
        "Model Performance",
        "Review the fixed held-out test results, compare classification/regression models, and evaluate selected dataset periods by month and year.",
        icon="🤖",
    )
    left, right = st.columns([1.08, .92])
    with left:
        cm = np.array(metrics["confusion_matrix"])
        fig_cm = px.imshow(
            cm, x=RATING_ORDER, y=RATING_ORDER, text_auto=True,
            color_continuous_scale=[[0, "#F5F3FF"], [0.3, "#C4B5FD"], [0.65, "#7C3AED"], [1.0, "#3B0764"]],
            labels=dict(x="Predicted Rating", y="Actual Rating", color="Number of properties"),
            title=f"Confusion Matrix — {metrics['best_classification_model']}",
        )
        fig_cm.update_coloraxes(showscale=False)
        st.plotly_chart(plot_layout(fig_cm, 480), use_container_width=True)

    with right:
        st.subheader("Classification Report")
        report = pd.DataFrame(metrics["classification_report"])
        st.dataframe(report, use_container_width=True, hide_index=True)

    # ── Final test-set result cards ───────────────────────────────────────
    st.markdown("<h3 style='margin-top:18px;'>Final Test-Set Results</h3>", unsafe_allow_html=True)
    cv1, cv2, cv3, cv4 = st.columns(4)
    cv1.markdown(f"""<div class="kpi-card"><div class="kpi-label">Test F1 Score</div><div class="kpi-value" style="color:#7C3AED;">{metrics["weighted_f1"]:.4f}</div><div class="kpi-note">Weighted across all EPC bands</div></div>""", unsafe_allow_html=True)
    cv2.markdown(f"""<div class="kpi-card"><div class="kpi-label">Test Accuracy</div><div class="kpi-value" style="color:#DB2777;">{metrics["accuracy"]:.4f}</div><div class="kpi-note">Proportion correctly predicted</div></div>""", unsafe_allow_html=True)
    cv3.markdown(f"""<div class="kpi-card"><div class="kpi-label">Regression R²</div><div class="kpi-value" style="color:#059669;">{metrics["regression_r2"]:.4f}</div><div class="kpi-note">Variance explained by model</div></div>""", unsafe_allow_html=True)
    regression_rmse_value = metrics.get("regression_rmse", metrics.get("rmse", None))
    if regression_rmse_value is not None:
        cv4.markdown(f"""<div class="kpi-card"><div class="kpi-label">Regression RMSE</div><div class="kpi-value" style="color:#D97706;">{float(regression_rmse_value):.4f}</div><div class="kpi-note">Typical score error; lower is better</div></div>""", unsafe_allow_html=True)
    else:
        cv4.markdown(f"""<div class="kpi-card"><div class="kpi-label">Best Regressor</div><div class="kpi-value" style="font-size:1rem; line-height:1.3; color:#D97706;">{metrics["best_regression_model"]}</div><div class="kpi-note">Top regression model</div></div>""", unsafe_allow_html=True)
    st.write("")

    # ── Clearer model comparison visuals ─────────────────────────────────
    st.markdown("<h3 style='margin-top:10px;'>Model Comparison Charts</h3>", unsafe_allow_html=True)
    st.caption("Four key metrics are shown separately so the results are easy to explain: Accuracy and F1 for classification; R² and RMSE for regression.")

    st.markdown(
        """
        <div class="dashboard-note" style="display:grid; grid-template-columns:repeat(4, 1fr); gap:10px;">
            <div><b>Accuracy</b><br><span style="font-size:.82rem;">Classification metric; higher is better.</span></div>
            <div><b>F1 Score</b><br><span style="font-size:.82rem;">Balances precision and recall; higher is better.</span></div>
            <div><b>R² Score</b><br><span style="font-size:.82rem;">Regression fit; higher is better.</span></div>
            <div><b>RMSE</b><br><span style="font-size:.82rem;">Regression error; lower is better.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    class_results = pd.DataFrame(metrics["classification_results"]).copy()
    reg_results = pd.DataFrame(metrics["regression_results"]).copy()

    def _bar_metric_chart(dataframe, metric_col, title, y_title, higher_is_better=True, colors=None):
        """Reusable clean bar chart for one model metric."""
        if dataframe.empty or metric_col not in dataframe.columns or "Model" not in dataframe.columns:
            fig = go.Figure()
            fig.update_layout(title=title)
            return plot_layout(fig, 360)

        plot_df = dataframe[["Model", metric_col]].copy()
        plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[metric_col])
        if plot_df.empty:
            fig = go.Figure()
            fig.update_layout(title=title)
            return plot_layout(fig, 360)

        plot_df = plot_df.sort_values(metric_col, ascending=not higher_is_better)
        if colors is None:
            colors = ["#7C3AED", "#DB2777", "#059669", "#D97706", "#0891B2"]
        fig = px.bar(
            plot_df,
            x="Model",
            y=metric_col,
            color="Model",
            text=metric_col,
            title=title,
            color_discrete_sequence=colors,
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_color="rgba(255,255,255,0.55)", marker_line_width=1)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title_text=y_title)
        return plot_layout(fig, 380)

    c_left, c_right = st.columns(2)
    with c_left:
        if "Accuracy" in class_results.columns:
            acc_min = max(0, float(pd.to_numeric(class_results["Accuracy"], errors="coerce").min()) - 0.05)
            fig_acc = _bar_metric_chart(class_results, "Accuracy", "Classification Accuracy Comparison", "Accuracy", higher_is_better=True, colors=["#7C3AED", "#5B21B6", "#DB2777"])
            fig_acc.update_yaxes(range=[acc_min, 1.02])
            st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info("Accuracy values are not available in the classification results table.")
    with c_right:
        f1_col = next((c for c in ["F1 Score", "Weighted F1", "F1"] if c in class_results.columns), None)
        if f1_col:
            f1_min = max(0, float(pd.to_numeric(class_results[f1_col], errors="coerce").min()) - 0.05)
            fig_f1 = _bar_metric_chart(class_results, f1_col, "Classification F1 Score Comparison", "F1 score", higher_is_better=True, colors=["#DB2777", "#7C3AED", "#5B21B6"])
            fig_f1.update_yaxes(range=[f1_min, 1.02])
            st.plotly_chart(fig_f1, use_container_width=True)
        else:
            st.info("F1 score values are not available in the classification results table.")

    r_left, r_right = st.columns(2)
    with r_left:
        r2_col = next((c for c in ["R_squared", "R-square", "R²"] if c in reg_results.columns), None)
        if r2_col:
            r2_min = max(0, float(pd.to_numeric(reg_results[r2_col], errors="coerce").min()) - 0.05)
            fig_r2 = _bar_metric_chart(reg_results, r2_col, "Regression Cross-Validation R² Comparison", "CV R² score", higher_is_better=True, colors=["#059669", "#0891B2", "#D97706"])
            fig_r2.update_yaxes(range=[r2_min, 1.02])
            st.plotly_chart(fig_r2, use_container_width=True)
        else:
            st.info("R² values are not available in the regression results table.")
    with r_right:
        rmse_col = next((c for c in ["RMSE", "Root Mean Squared Error", "Regression RMSE"] if c in reg_results.columns), None)
        if rmse_col:
            fig_rmse = _bar_metric_chart(reg_results, rmse_col, "Regression RMSE Error Comparison", "RMSE (lower is better)", higher_is_better=False, colors=["#059669", "#0891B2", "#D97706"])
            st.plotly_chart(fig_rmse, use_container_width=True)
        else:
            st.markdown(
                f"""
                <div class="card" style="min-height:250px; display:flex; align-items:center; justify-content:center; text-align:center;">
                    <div>
                        <div class="small-label">Regression error metric</div>
                        <h3 style="margin:8px 0;">RMSE not stored in app bundle</h3>
                        <p class="muted">R² comparison is shown. RMSE can be added from the notebook export if required.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("📋 Full model comparison tables", expanded=False):
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Classification models**")
            st.dataframe(class_results, use_container_width=True, hide_index=True)
        with t2:
            st.markdown("**Regression models**")
            st.dataframe(reg_results, use_container_width=True, hide_index=True)

    # Dynamic month/year evaluation for supervisor review
    render_model_selected_period_evaluation()

    st.markdown(
        f"""
        <div class="card" style="margin-top:8px;">
            <h3>📊 Model Performance Summary</h3>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:8px;">
                <div>
                    <p>• Best classification model: <b style="color:#7C3AED;">{metrics["best_classification_model"]}</b> — selected based on highest weighted F1 score.</p>
                    <p>• Weighted F1 Score: <b style="color:#5B21B6;">{metrics["weighted_f1"]:.4f}</b> — measures precision and recall across all EPC bands.</p>
                    <p>• Overall accuracy: <b style="color:#059669;">{metrics["accuracy"]:.3f}</b> — proportion of correctly predicted EPC ratings.</p>
                    <p>• Macro F1: <b style="color:#0891B2;">{metrics.get("macro_f1", float("nan")):.4f}</b> · Balanced accuracy: <b style="color:#D97706;">{metrics.get("balanced_accuracy", float("nan")):.4f}</b>.</p>
                </div>
                <div>
                    <p>• Best regression model: <b style="color:#DB2777;">{metrics["best_regression_model"]}</b> — selected based on highest R² score.</p>
                    <p>• Regression R²: <b style="color:#D97706;">{metrics["regression_r2"]:.4f}</b> — proportion of variance explained (1.0 = perfect).</p>
                    <p>• Confusion matrix shows the model predicts Band C and D most accurately, matching the dataset distribution.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Learning Curve and ROC sections are intentionally hidden in the final app.
    # They require extra target columns in model_ready_data and were removed to keep
    # the dashboard clean and presentation-ready.

elif page == "Feature Importance":
    kpi_row()
    st.write("")
    page_intro(
        "Feature Importance",
        "Review leakage-free permutation importance for the trained Random Forest classifier using independent physical and structural predictors.",
        icon="🔍",
    )

    fi_left, fi_right = st.columns([1.6, 1.0])
    with fi_left:
        top_features = feature_importance.head(15).sort_values("Importance")
        fig = px.bar(
            top_features, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=[[0, "#93C5FD"], [0.4, "#2563EB"], [0.7, "#7C3AED"], [1.0, "#DB2777"]],
            title="Top 15 Most Important Features",
        )
        fig.update_traces(marker_line_color="rgba(255,255,255,0.3)", marker_line_width=0.5)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(plot_layout(fig, 520), use_container_width=True)

    with fi_right:
        strongest = feature_importance.head(5)["Feature"].tolist()
        top1_score = feature_importance.iloc[0]["Importance"]
        top1_feat = feature_importance.iloc[0]["Feature"]
        st.markdown(
            f"""
            <div class="card" style="margin-top:0px; height:100%;">
                <h3>🔍 Top Feature Drivers</h3>
                <p>Permutation importance ranked these as the <b>5 strongest predictive associations</b> with EPC rating:</p>
                {''.join([f'<div class="card-small" style="margin-bottom:8px;"><b style="color:#7C3AED;">#{i+1}</b> {feat}</div>' for i, feat in enumerate(strongest)])}
                <hr style="margin:12px 0; border-color:rgba(109,40,217,.13);">
                <p class="muted" style="font-size:.84rem;">Top feature <b>{top1_feat}</b> has mean permutation importance <b style="color:#7C3AED;">{top1_score:.4f}</b>. This is predictive association, not causation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="card" style="margin-top:14px;">
            <h3>📊 Interpretation Summary</h3>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:8px;">
                <div>
                    <p>• <b>Permutation importance</b> measures the loss in held-out predictive performance after shuffling a feature — higher means a stronger predictive association.</p>
                    <p>• The top 5 features are: <b style="color:#7C3AED;">{", ".join(strongest)}</b>.</p>
                    <p>• <b>Walls, roofs, construction age, hot-water systems, built form, heating controls, heating systems and glazing</b> are the strongest notebook predictors.</p>
                </div>
                <div>
                    <p>• <b>Low-scoring features</b> may still contribute jointly; importance values must not be interpreted as causal effects.</p>
                    <p>• This analysis supports <b>academic transparency</b> — it shows the model is using physically meaningful variables, not noise.</p>
                    <p>• Useful for explaining <b>why</b> a property gets a particular rating during your supervisor review.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Recommendations":
    kpi_row()
    st.write("")
    page_intro(
        "Energy Improvement Recommendations",
        "Select an EPC band to view targeted recommendations, priority level, and real dataset context for improving energy efficiency.",
        icon="💡",
    )

    sel_col, info_col = st.columns([1, 2.5])
    with sel_col:
        selected_rating = st.selectbox("Select EPC Rating Band", RATING_ORDER, index=3)

    if selected_rating in ["A", "B"]:
        priority = "Low"
        priority_color = "#059669"
        priority_bg = "rgba(5,150,105,.10)"
        summary = "This property already performs well. Focus on maintaining standards and minor optimisations."
        recommendations = [
            ("🔧 Maintain Systems", "Keep heating, glazing and insulation systems regularly serviced to preserve efficiency."),
            ("📊 Monitor Usage", "Track energy consumption monthly to detect any unexpected increases early."),
            ("💡 Lighting Check", "Ensure 100% of lighting uses LED or low-energy bulbs for maximum savings."),
        ]
    elif selected_rating in ["C", "D"]:
        priority = "Medium"
        priority_color = "#D97706"
        priority_bg = "rgba(217,119,6,.10)"
        summary = "Moderate improvements can move this property into Band B or A. Insulation and heating upgrades give the best return."
        recommendations = [
            ("🏠 Loft Insulation", "Improving loft and cavity wall insulation reduces heat loss significantly and improves the EPC score."),
            ("🌡️ Heating Controls", "Install smart heating controls and thermostats — heating controls are among the strongest physical predictors in the notebook."),
            ("💡 Low-Energy Lighting", "Switch remaining lighting to low-energy LEDs. Efficient lighting is a practical retrofit measure; cost outputs are not model inputs."),
            ("🪟 Glazing Upgrade", "Upgrading single glazing to double or triple glazing reduces heat loss and improves the EPC band."),
        ]
    else:
        priority = "High"
        priority_color = "#DC2626"
        priority_bg = "rgba(220,38,38,.10)"
        summary = "Significant improvements are needed. Prioritise building fabric, glazing, hot-water systems, heating systems and heating controls, supported by a professional survey."
        recommendations = [
            ("🏗️ Insulation Priority", "Prioritise wall, floor and loft insulation — this single change has the highest impact on moving from F/G to D/E."),
            ("♨️ Heating System", "Replace outdated boilers with modern high-efficiency or heat-pump systems to cut heating costs and CO₂."),
            ("🌿 Reduce CO₂", "Consider appropriate low-carbon heating and renewable systems after fabric improvements and a professional assessment."),
            ("🪟 Glazing & Draught", "Improve glazing type and seal draughts — multi-glaze proportion is an important feature in the model."),
            ("💧 Hot Water", "Install a more efficient hot water system or solar thermal panel — hot-water system description is one of the stronger physical predictors in the notebook."),
        ]

    with info_col:
        st.markdown(
            f"""
            <div class="card" style="background:linear-gradient(145deg,#FFFFFF,{priority_bg}); border-left:5px solid {priority_color};">
                <div style="display:flex; align-items:center; gap:16px;">
                    <div class="big-rating" style="background:{RATING_COLORS[selected_rating]}; min-width:80px;">{selected_rating}</div>
                    <div>
                        <div class="small-label">Selected Band</div>
                        <h2 style="margin:4px 0;">Band {selected_rating} — Priority: <span style="color:{priority_color};">{priority}</span></h2>
                        <p style="color:#334155; margin:0;">{summary}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    cols = st.columns(min(len(recommendations), 3))
    for i, (title, desc) in enumerate(recommendations):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="card" style="min-height:140px; border-top:4px solid {priority_color};">
                    <h3 style="font-size:1rem; margin-bottom:8px;">{title}</h3>
                    <p class="muted" style="font-size:.9rem; line-height:1.55;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    if len(recommendations) > 3:
        st.write("")
        extra_cols = st.columns(len(recommendations) - 3)
        for i, (title, desc) in enumerate(recommendations[3:]):
            with extra_cols[i]:
                st.markdown(
                    f"""
                    <div class="card" style="min-height:140px; border-top:4px solid {priority_color};">
                        <h3 style="font-size:1rem; margin-bottom:8px;">{title}</h3>
                        <p class="muted" style="font-size:.9rem; line-height:1.55;">{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.write("")
    # Context from actual data
    band_data = df[df[COL_CURRENT] == selected_rating]
    if len(band_data) > 0:
        avg_band_score = band_data[COL_SCORE].mean()
        avg_band_co2 = band_data[COL_CO2].mean()
        avg_band_energy = band_data[COL_ENERGY].mean()
        avg_pot_score = band_data[COL_POT_SCORE].mean()
        gap = avg_pot_score - avg_band_score
        st.markdown(
            f"""
            <div class="card" style="margin-top:4px;">
                <h3>📊 Real Data Insights — Band {selected_rating} Properties in Dataset</h3>
                <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:12px; margin-top:10px;">
                    <div class="card-small">
                        <div class="kpi-label">Properties in Dataset</div>
                        <div class="kpi-value" style="font-size:1.4rem;">{len(band_data):,}</div>
                    </div>
                    <div class="card-small">
                        <div class="kpi-label">Avg Efficiency Score</div>
                        <div class="kpi-value" style="font-size:1.4rem; color:#7C3AED;">{avg_band_score:.1f}/100</div>
                    </div>
                    <div class="card-small">
                        <div class="kpi-label">Avg CO₂ Emissions</div>
                        <div class="kpi-value" style="font-size:1.4rem; color:#DC2626;">{avg_band_co2:.2f}</div>
                    </div>
                    <div class="card-small">
                        <div class="kpi-label">Potential Score Gap</div>
                        <div class="kpi-value" style="font-size:1.4rem; color:#059669;">+{gap:.1f}</div>
                        <div class="kpi-note">Points improvable if all actions taken</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

elif page == "Property Type Comparison":
    kpi_row()
    st.write("")
    page_intro(
        "Property Type Comparison",
        "Compare all property types from the uploaded dataset using the selected year, month and metric. This page is designed for clean marker-friendly comparison.",
        icon="🏘️",
    )

    dated_compare, compare_date_col = get_dataset_date_view(df)
    compare_years = get_dataset_years(df)
    if compare_years:
        cy = st.selectbox("Year", compare_years, index=len(compare_years)-1, key="compare_year_dataset")
        cm_options = get_dataset_months_for_year(df, cy)
        cm = st.selectbox("Month", cm_options, index=len(cm_options)-1, key="compare_month_dataset", format_func=month_label) if cm_options else None
        compare_df = dated_compare[
            (dated_compare["__DATASET_YEAR"].astype("Int64") == int(cy))
            & (dated_compare["__DATASET_MONTH"].astype("Int64") == int(cm))
        ].copy() if cm is not None else dated_compare.copy()
    else:
        cy, cm = None, None
        compare_df = dated_compare.copy()

    compare_metric_map = {
        "Energy Efficiency Score": COL_SCORE,
        "CO₂ Emissions": COL_CO2,
        "Energy Consumption": COL_ENERGY,
        "Heating Cost": COL_HEATING,
        "Floor Area": COL_AREA,
    }
    compare_metric_label = st.selectbox("Metric", list(compare_metric_map.keys()), key="compare_metric_dataset")
    compare_metric_col = compare_metric_map[compare_metric_label]

    if compare_df.empty:
        st.warning("No records found for the selected dataset period.")
    else:
        st.plotly_chart(property_type_comparison_chart(compare_df, compare_metric_col, compare_metric_label), use_container_width=True)
        summary = compare_df.groupby(COL_PROPERTY, as_index=False).agg(
            Records=(COL_PROPERTY, "size"),
            Average_Score=(COL_SCORE, "mean"),
            Average_CO2=(COL_CO2, "mean"),
            Average_Energy=(COL_ENERGY, "mean"),
            Average_Heating_Cost=(COL_HEATING, "mean"),
        )
        st.dataframe(summary.round(2), use_container_width=True, hide_index=True)

elif page == "About & Glossary":
    kpi_row()
    st.write("")
    page_intro(
        "About This Project & Glossary",
        "Academic context, stakeholder information, methodology notes and simple definitions for the technical terms used in the application.",
        icon="📘",
    )

    # ── About / Methodology card ──────────────────────────────────────────
    st.markdown("<h3>📋 About This Project</h3>", unsafe_allow_html=True)
    about1, about2 = st.columns(2)
    with about1:
        st.markdown(
            f"""<div class="card">
            <h3>Project Details</h3>
            <p><b>Title:</b> {PROJECT_TITLE}</p>
            <p><b>Student:</b> {STUDENT_NAME}</p>
            <p><b>Type:</b> {PROJECT_TYPE}</p>
            <p><b>Dataset:</b> Cambridge EPC property records — DS-BY-PINKY.xlsx</p>
            <p><b>Records:</b> {len(df):,} actual EPC records from Cambridge</p>
            <p><b>Target variable:</b> EPC Rating (A–G classification) + Energy Efficiency Score (regression)</p>
            <hr style="border-color:rgba(109,40,217,.13);">
            <h3>Data Science Pipeline</h3>
            <p>1. Raw data loading from Excel</p>
            <p>2. Exploratory Data Analysis (EDA)</p>
            <p>3. Leakage-safe feature engineering</p>
            <p>4. Train-only imputation + one-hot encoding</p>
            <p>5. Train/test split (80/20)</p>
            <p>6. Model training: Logistic Regression, Decision Tree, Random Forest and Linear Regression</p>
            <p>7. Evaluation: F1, R², confusion matrix, cross-validation</p>
            <p>8. Streamlit deployment</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with about2:
        st.markdown(
            f"""<div class="card">
            <h3>Model Summary</h3>
            <p><b>Best Classification Model:</b> <span style="color:#7C3AED;">{metrics["best_classification_model"]}</span></p>
            <p><b>Best Regression Model:</b> <span style="color:#059669;">{metrics["best_regression_model"]}</span></p>
            <p><b>Weighted F1 Score:</b> <span style="color:#5B21B6;">{metrics["weighted_f1"]:.4f}</span></p>
            <p><b>Regression R²:</b> <span style="color:#D97706;">{metrics["regression_r2"]:.4f}</span></p>
            <p><b>Accuracy:</b> <span style="color:#DB2777;">{metrics["accuracy"]:.3f}</span></p>
            <hr style="border-color:rgba(109,40,217,.13);">
            <h3>Limitations</h3>
            <p>• Dataset limited to Cambridge — results may not generalise to other UK regions</p>
            <p>• EPC predictions are indicative only — not an official EPC certificate</p>
            <p>• Future predictions use the selected year and 3-month range — actual improvement depends on physical interventions</p>
            <p>• Class imbalance (Band C/D majority) may affect rare band (A, G) prediction accuracy</p>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Stakeholder Personas ──────────────────────────────────────────────
    st.write("")
    st.markdown("<h3>👥 Who Uses This Application?</h3>", unsafe_allow_html=True)
    st.caption("This tool was designed with three main stakeholder groups in mind.")

    sp1, sp2, sp3 = st.columns(3)
    personas = [
        ("🏠", "Homeowner", "#7C3AED",
         "Wants to understand their current EPC rating and what improvements may raise it. Uses the Prediction page to check their score, compares predicted results by property type, and uses the Recommendations page for actionable steps.",
         ["Prediction page", "Recommendations", "Property Type Prediction"]),
        ("🏢", "Estate Agent / Surveyor", "#0891B2",
         "Needs to quickly assess and compare energy performance by property type. Uses the current and prediction dashboards for dataset-based property-type comparison.",
         ["Interactive Dashboard", "Prediction", "Feature Explorer"]),
        ("🏛️", "Local Council / Policy Maker", "#DB2777",
         "Interested in how many Cambridge properties are close to the Band C government target. Uses the 1-band-away analysis and geographic map to prioritise retrofit funding and understand CO₂ impact at scale.",
         ["Interactive Dashboard", "1-Band-Away Analysis", "Feature Importance"]),
    ]
    for col, (icon, role, color, desc, features) in zip([sp1, sp2, sp3], personas):
        with col:
            st.markdown(
                f"""<div class="card" style="border-top:4px solid {color}; min-height:280px;">
                <div style="font-size:2rem; margin-bottom:8px;">{icon}</div>
                <h3 style="color:{color}; margin-bottom:8px;">{role}</h3>
                <p style="font-size:.88rem; color:#334155; line-height:1.55;">{desc}</p>
                <div style="margin-top:10px;">
                    <div class="small-label">Key pages</div>
                    {"".join([f'<span style="display:inline-block; margin:3px 4px 0 0; padding:3px 9px; background:{color}18; border:1px solid {color}44; border-radius:20px; font-size:.78rem; color:{color}; font-weight:600;">{f}</span>' for f in features])}
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Glossary ──────────────────────────────────────────────────────────
    st.write("")
    st.markdown("<h3>📖 Glossary of Technical Terms</h3>", unsafe_allow_html=True)
    st.caption("Definitions for all technical terms used in this application — useful for non-technical supervisors and markers.")

    glossary = [
        ("EPC", "Energy Performance Certificate", "A certificate that rates a property's energy efficiency from A (best) to G (worst). Required by law when selling or renting in the UK."),
        ("EPC Band", "A–G Rating Band", "The letter grade assigned based on the energy efficiency score. A = 92–100 (excellent), B = 81–91, C = 69–80, D = 55–68, E = 39–54, F = 21–38, G = 1–20 (poor)."),
        ("F1 Score", "Weighted F1 Score", "A model evaluation metric that balances precision (correct positives) and recall (found positives). Score ranges 0–1; higher is better. Weighted F1 accounts for class imbalance."),
        ("R²", "R-Squared / Coefficient of Determination", "A regression metric showing what proportion of variance in the target variable the model explains. R² = 1.0 is perfect prediction; R² = 0 means the model is no better than the mean."),
        ("Confusion Matrix", "Prediction Error Table", "A table showing how many predictions were correct per EPC band. The diagonal (top-left to bottom-right) shows correct predictions; off-diagonal shows errors."),
        ("Random Forest", "Ensemble Machine Learning Model", "A model that builds many decision trees and combines their votes. More accurate than a single tree and resistant to overfitting. Used here as the selected classifier; Linear Regression is the selected efficiency-score regressor."),
        ("Cross-Validation (CV)", "K-Fold Model Validation", "A technique that trains and tests the model on different data splits (folds) to check it generalises well — not just memorising the training data."),
        ("ROC Curve", "Receiver Operating Characteristic Curve", "A chart showing how well the model separates one EPC band from all others. The closer the curve is to the top-left corner, the better. AUC = area under the curve (1.0 = perfect)."),
        ("Feature Importance", "Variable Influence Score", "A held-out permutation score showing how prediction performance changes when one physical or structural input is shuffled. It indicates association, not causation."),
        ("Multicollinearity", "Feature Correlation", "When two or more input features are strongly related to each other (e.g. heating cost and energy consumption both rise together). The correlation heatmap shows this visually."),
        ("CO₂ Emissions", "Carbon Dioxide Output", "Estimated annual CO₂ released by a property's energy use. It is retained for descriptive analysis but excluded from the leakage-free prediction inputs."),
        ("Energy Efficiency Score", "SAP Score", "A numeric score from 1–100 derived from the Standard Assessment Procedure. Used to classify properties into EPC bands. Higher = more efficient."),
        ("One-Hot Encoding", "Categorical Variable Encoding", "A method of converting categorical variables (like Property Type or Built Form) into binary 0/1 columns that machine learning models can understand."),
        ("Learning Curve", "Training Size vs Accuracy Chart", "Shows how model performance changes as more training data is used. If training and validation scores converge, the model generalises well."),
    ]

    g1, g2 = st.columns(2)
    for i, (term, full, definition) in enumerate(glossary):
        col = g1 if i % 2 == 0 else g2
        with col:
            st.markdown(
                f"""<div class="card-small" style="margin-bottom:10px; border-left:4px solid #7C3AED;">
                <div style="display:flex; align-items:baseline; gap:8px; margin-bottom:4px;">
                    <span style="font-size:1rem; font-weight:800; color:#7C3AED;">{term}</span>
                    <span style="font-size:.82rem; color:#5B4B8A;">{full}</span>
                </div>
                <p style="font-size:.87rem; color:#334155; line-height:1.5; margin:0;">{definition}</p>
                </div>""",
                unsafe_allow_html=True,
            )

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    f"""
    <div class="footer-box">
        <b>{PROJECT_TITLE}</b> | {PROJECT_TYPE} | Prepared by <b>{STUDENT_NAME}</b><br>
        Cambridge EPC Energy Rating Predictor using actual project data.
    </div>
    """,
    unsafe_allow_html=True,
)

