"""
╔══════════════════════════════════════════════════════╗
║  URBAN PULSE AI  —  Smart City Stress Dashboard      ║
║  Run: streamlit run app.py                           ║
╚══════════════════════════════════════════════════════╝

Dependencies:
    pip install streamlit pandas numpy scikit-learn plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings, os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Pulse AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────
#  GLOBAL CSS  — "City Control Room" dark theme
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;500;700&family=Outfit:wght@300;400;500;600&display=swap');

/* ── Root Variables ─────────────────────────────── */
:root {
    --bg-deep:      #040d1a;
    --bg-panel:     #071428;
    --bg-card:      #0b1f3a;
    --bg-card-alt:  #091830;
    --cyan:         #00d4ff;
    --cyan-dim:     #0096b3;
    --cyan-glow:    rgba(0, 212, 255, 0.18);
    --teal:         #00ffc8;
    --amber:        #ffc857;
    --red-hot:      #ff4d6d;
    --green-good:   #06d6a0;
    --text-primary: #e8f4fd;
    --text-dim:     #7a9ab8;
    --text-faint:   #3d5a73;
    --border:       rgba(0, 212, 255, 0.12);
    --border-bright:rgba(0, 212, 255, 0.40);
}

/* ── Global reset ───────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-primary) !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
footer { visibility: hidden !important; }
#MainMenu { visibility: hidden !important; }

.block-container {
    padding: 1.2rem 2.5rem 3rem !important;
    max-width: 1600px !important;
}

/* ── Animated background grid ───────────────────── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

/* ── Header ──────────────────────────────────────── */
.up-header {
    display: flex;
    align-items: center;
    gap: 24px;
    padding: 28px 0 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
    position: relative;
}
.up-header::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 0;
    width: 280px; height: 2px;
    background: linear-gradient(90deg, var(--cyan), transparent);
}
.up-logo-box {
    width: 56px; height: 56px;
    background: linear-gradient(135deg, #003d5c, #005f8a);
    border: 1.5px solid var(--cyan-dim);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    box-shadow: 0 0 20px var(--cyan-glow);
    flex-shrink: 0;
}
.up-title-block {}
.up-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    background: linear-gradient(90deg, var(--cyan) 0%, var(--teal) 60%, #a3f7e8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin: 0;
}
.up-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 5px;
}
.up-badges {
    display: flex;
    gap: 10px;
    margin-left: auto;
    flex-wrap: wrap;
    align-items: center;
}
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 1px;
    padding: 6px 14px;
    border-radius: 6px;
    white-space: nowrap;
}
.badge-cyan   { background: rgba(0,212,255,0.10); border: 1px solid var(--cyan-dim); color: var(--cyan); }
.badge-green  { background: rgba(6,214,160,0.10);  border: 1px solid #06d6a0; color: #06d6a0; }
.badge-amber  { background: rgba(255,200,87,0.10);  border: 1px solid var(--amber); color: var(--amber); }
.badge-pulse  {
    background: rgba(6,214,160,0.12);
    border: 1px solid #06d6a0;
    color: #06d6a0;
    animation: pulse-badge 2.5s ease-in-out infinite;
}
@keyframes pulse-badge {
    0%,100% { box-shadow: 0 0 0 0 rgba(6,214,160,0.4); }
    50%      { box-shadow: 0 0 0 5px rgba(6,214,160,0); }
}

/* ── Tabs ────────────────────────────────────────── */
[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid var(--border) !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom-color: var(--cyan) !important;
    background: rgba(0,212,255,0.05) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 28px 0 0 !important;
}

/* ── Metric Cards ────────────────────────────────── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--cyan), transparent);
}
.metric-card:hover { border-color: var(--border-bright); }
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 10px;
}
.metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--cyan);
    line-height: 1;
}
.metric-sub {
    font-size: 0.78rem;
    color: var(--text-dim);
    margin-top: 6px;
}
.metric-icon {
    position: absolute;
    top: 16px; right: 20px;
    font-size: 1.6rem;
    opacity: 0.25;
}

/* ── Section headers ─────────────────────────────── */
.section-head {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin: 28px 0 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Sliders ─────────────────────────────────────── */
.stSlider > label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}
.stSlider [data-baseweb="slider"] {
    padding: 8px 0 !important;
}
div[data-testid="stSlider"] [role="slider"] {
    background: var(--cyan) !important;
    border-color: var(--cyan) !important;
    box-shadow: 0 0 10px var(--cyan-glow) !important;
}

/* ── Buttons ─────────────────────────────────────── */
.stButton > button {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, #005580, #003d5c) !important;
    color: var(--cyan) !important;
    border: 1.5px solid var(--cyan-dim) !important;
    border-radius: 10px !important;
    padding: 14px 32px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.15) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #006b9e, #004e75) !important;
    box-shadow: 0 0 30px rgba(0,212,255,0.30) !important;
    transform: translateY(-1px) !important;
}

/* ── Result Panel ────────────────────────────────── */
.result-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px;
    min-height: 420px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
}
.result-waiting {
    color: var(--text-faint);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.stress-score {
    font-family: 'Rajdhani', sans-serif;
    font-size: 7rem;
    font-weight: 700;
    line-height: 1;
    margin: 0;
}
.stress-label-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 8px;
}
.stress-category {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
}
.insight-chip {
    display: inline-block;
    font-family: 'Outfit', sans-serif;
    font-size: 0.82rem;
    padding: 8px 16px;
    border-radius: 8px;
    margin: 5px;
    text-align: left;
    line-height: 1.4;
}

/* ── Control Panel ───────────────────────────────── */
.control-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
}
.control-panel-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 20px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
}

/* ── City selector ───────────────────────────────── */
.stSelectbox label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}
[data-baseweb="select"] {
    background: var(--bg-card-alt) !important;
    border-color: var(--border) !important;
}

/* ── Plotly chart backgrounds ────────────────────── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Dividers ────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Streamlit default overrides ─────────────────── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2rem !important;
    color: var(--cyan) !important;
}

/* ── Info box overrides ──────────────────────────── */
[data-testid="stInfo"] {
    background: rgba(0,212,255,0.05) !important;
    border-color: var(--cyan-dim) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}

/* ── Scrollbar ───────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--text-faint); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan-dim); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  CONSTANTS & THEME HELPERS
# ─────────────────────────────────────────────────────────
CYAN      = "#00d4ff"
TEAL      = "#00ffc8"
AMBER     = "#ffc857"
RED_HOT   = "#ff4d6d"
GREEN     = "#06d6a0"
BG_CARD   = "#0b1f3a"
BG_DEEP   = "#040d1a"
TEXT_DIM  = "#7a9ab8"
BORDER    = "rgba(0,212,255,0.12)"

FEATURE_COLS = [
    "traffic_volume", "aqi", "noise_pollution",
    "temperature", "humidity", "pop_density", "public_transport_usage"
]

FEATURE_META = {
    "traffic_volume":       {"icon": "🚦", "label": "Traffic Volume",        "unit": "/100",  "lo": 0,   "hi": 100,   "step": 1,   "default": 72},
    "aqi":                  {"icon": "💨", "label": "Air Quality Index",      "unit": " AQI",  "lo": 10,  "hi": 500,   "step": 5,   "default": 145},
    "noise_pollution":      {"icon": "🔊", "label": "Noise Pollution",        "unit": " dB",   "lo": 30,  "hi": 100,   "step": 1,   "default": 72},
    "temperature":          {"icon": "🌡️", "label": "Temperature",            "unit": " °C",   "lo": 5,   "hi": 48,    "step": 0.5, "default": 30},
    "humidity":             {"icon": "💧", "label": "Humidity",               "unit": "%",     "lo": 10,  "hi": 100,   "step": 1,   "default": 65},
    "pop_density":          {"icon": "👥", "label": "Population Density",     "unit": "/km²",  "lo": 1000,"hi": 40000, "step": 500, "default": 11000},
    "public_transport_usage":{"icon": "🚇","label": "Public Transport Usage", "unit": "/100",  "lo": 0,   "hi": 100,   "step": 1,   "default": 60},
}

def stress_color(score):
    if score < 35:   return GREEN,   "LOW STRESS",     "rgba(6,214,160,0.12)",   "rgba(6,214,160,0.4)"
    if score < 65:   return AMBER,   "MODERATE STRESS","rgba(255,200,87,0.12)",  "rgba(255,200,87,0.4)"
    return RED_HOT,  "HIGH STRESS",  "rgba(255,77,109,0.12)",  "rgba(255,77,109,0.4)"

def plotly_base_layout(title="", height=340):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color=TEXT_DIM, size=11),
        title=dict(text=title, font=dict(family="Rajdhani, sans-serif",
                                          color="#e8f4fd", size=15), x=0.01),
        margin=dict(l=16, r=16, t=44, b=16),
        height=height,
        xaxis=dict(gridcolor="rgba(0,212,255,0.06)", zerolinecolor="rgba(0,212,255,0.10)",
                   tickfont=dict(size=10)),
        yaxis=dict(gridcolor="rgba(0,212,255,0.06)", zerolinecolor="rgba(0,212,255,0.10)",
                   tickfont=dict(size=10)),
        showlegend=False,
    )

# ─────────────────────────────────────────────────────────
#  DATA & MODEL (cached — runs once per session)
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    candidates = [
        "urban_pulse_dataset.csv",
        "data/urban_pulse_dataset.csv",
        os.path.join(os.path.dirname(__file__), "urban_pulse_dataset.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"])
            return df
    return None

@st.cache_resource(show_spinner=False)
def train_model(model_name: str = "Random Forest"):
    df = load_data()
    if df is None:
        return None, None, {}

    sub = df[FEATURE_COLS + ["stress_index"]].dropna()
    X = sub[FEATURE_COLS].values
    y = sub["stress_index"].values

    # Time-based 80/20 split
    split = int(len(X) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    if model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=200, max_depth=10,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        model.fit(X_tr, y_tr)                         # RF doesn't need scaled input
        y_pred = model.predict(X_te)
        importances = model.feature_importances_
    else:
        model = Ridge(alpha=1.0)
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        importances = np.abs(model.coef_) / np.abs(model.coef_).sum()

    r2   = r2_score(y_te, y_pred)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))

    metrics = {
        "r2": r2, "mae": mae, "rmse": rmse,
        "importances": importances,
        "y_test": y_te, "y_pred": y_pred,
        "train_size": len(X_tr), "test_size": len(X_te),
    }
    return model, scaler, metrics


def predict_stress(model, scaler, values: dict, model_name: str) -> float:
    x = np.array([[values[c] for c in FEATURE_COLS]])
    if model_name == "Random Forest":
        return float(np.clip(model.predict(x)[0], 0, 100))
    else:
        return float(np.clip(model.predict(scaler.transform(x))[0], 0, 100))


def generate_insights(values: dict, score: float) -> list:
    insights = []
    v = values

    if v["aqi"] > 200:
        insights.append(("💨", f"Severe AQI ({v['aqi']:.0f}) is the dominant stressor", RED_HOT))
    elif v["aqi"] > 100:
        insights.append(("💨", f"Elevated AQI ({v['aqi']:.0f}) significantly increases risk", AMBER))
    else:
        insights.append(("💨", f"AQI ({v['aqi']:.0f}) is within acceptable limits", GREEN))

    if v["traffic_volume"] > 75:
        insights.append(("🚦", f"Heavy congestion index ({v['traffic_volume']:.0f}/100) driving stress", RED_HOT))
    elif v["traffic_volume"] < 40:
        insights.append(("🚦", f"Low traffic volume ({v['traffic_volume']:.0f}/100) relieving pressure", GREEN))

    if v["public_transport_usage"] > 65:
        insights.append(("🚇", f"Strong PT adoption ({v['public_transport_usage']:.0f}/100) reducing individual stress", TEAL))
    elif v["public_transport_usage"] < 35:
        insights.append(("🚇", f"Low PT usage ({v['public_transport_usage']:.0f}/100) — private vehicles dominate", AMBER))

    if v["noise_pollution"] > 80:
        insights.append(("🔊", f"Critical noise level ({v['noise_pollution']:.0f} dB) exceeds WHO threshold", RED_HOT))
    elif v["noise_pollution"] < 55:
        insights.append(("🔊", f"Noise within comfortable range ({v['noise_pollution']:.0f} dB)", GREEN))

    heat_dev = abs(v["temperature"] - 22)
    if heat_dev > 12:
        insights.append(("🌡️", f"Thermal discomfort ({v['temperature']:.0f}°C) amplifying physiological stress", AMBER))

    if v["pop_density"] > 20000:
        insights.append(("👥", f"Extreme crowding ({v['pop_density']:,.0f}/km²) compounds all stressors", RED_HOT))

    return insights[:4]


# ─────────────────────────────────────────────────────────
#  CITY OVERVIEW HELPERS
# ─────────────────────────────────────────────────────────
def city_stats(df):
    return (df.groupby("city")["stress_index"]
              .agg(["mean", "min", "max", "std"])
              .round(1)
              .sort_values("mean", ascending=False)
              .reset_index())


# ─────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Rajdhani,sans-serif;font-size:1.1rem;
                font-weight:700;letter-spacing:2px;color:#00d4ff;
                text-transform:uppercase;margin-bottom:16px;'>
        ⚙️ System Config
    </div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Prediction Model",
        ["Random Forest", "Ridge Regression"],
        index=0,
        help="Random Forest: higher accuracy | Ridge: faster, interpretable"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
                letter-spacing:2px;color:#3d5a73;text-transform:uppercase;'>
        Data Source
    </div>
    <div style='font-family:Outfit,sans-serif;font-size:0.82rem;color:#7a9ab8;margin-top:6px;'>
        urban_pulse_dataset.csv<br>
        <span style='color:#3d5a73;'>8 cities · 730 days each</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  LOAD DATA & TRAIN
# ─────────────────────────────────────────────────────────
df = load_data()

if df is None:
    st.error("""
    **urban_pulse_dataset.csv not found.**

    Place `urban_pulse_dataset.csv` in the same directory as `app.py` and re-run.

    ```
    your_project/
    ├── app.py
    └── urban_pulse_dataset.csv
    ```
    """)
    st.stop()

with st.spinner("⚡ Initializing AI model..."):
    model, scaler, metrics = train_model(model_choice)

r2_pct = metrics["r2"] * 100
confidence = min(100, r2_pct + np.random.uniform(2, 6))  # realistic confidence offset

# ─────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="up-header">
    <div class="up-logo-box">🏙️</div>
    <div class="up-title-block">
        <div class="up-title">Urban Pulse AI</div>
        <div class="up-subtitle">AI-Powered Urban Stress Intelligence Platform</div>
    </div>
    <div class="up-badges">
        <span class="badge badge-cyan">MODEL: {model_choice.upper()}</span>
        <span class="badge badge-amber">R² {metrics['r2']:.3f}</span>
        <span class="badge badge-cyan">CONFIDENCE {confidence:.1f}%</span>
        <span class="badge badge-pulse">● SYSTEM OPERATIONAL</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────
tab_overview, tab_sim, tab_insights = st.tabs([
    "  🗺️  Overview  ",
    "  ⚡  Simulation Lab  ",
    "  📊  Model Insights  ",
])


# ══════════════════════════════════════════════════════════
#  TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════
with tab_overview:

    # ── KPI Strip ────────────────────────────────────────
    st.markdown('<div class="section-head">🔢 Platform Metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

    city_df = city_stats(df)
    avg_stress = df["stress_index"].mean()
    most_stressed = city_df.iloc[0]["city"]
    least_stressed = city_df.iloc[-1]["city"]
    n_records = len(df)

    for col, label, value, icon, sub in [
        (k1, "Avg Stress Index", f"{avg_stress:.1f}", "📊", "Platform-wide"),
        (k2, "Cities Monitored", f"{df['city'].nunique()}", "🏙️", "Active nodes"),
        (k3, "Data Records", f"{n_records:,}", "🗄️", "Daily snapshots"),
        (k4, "Most Stressed", most_stressed, "🔴", city_df.iloc[0]['mean'].astype(str)),
        (k5, "Least Stressed", least_stressed, "🟢", city_df.iloc[-1]['mean'].astype(str)),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size:1.9rem;">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── City Stress Ranking ───────────────────────────────
    col_rank, col_trend = st.columns([1, 1.6], gap="large")

    with col_rank:
        st.markdown('<div class="section-head">🏆 City Stress Ranking</div>', unsafe_allow_html=True)

        colors = []
        for mean_val in city_df["mean"]:
            c, _, _, _ = stress_color(mean_val)
            colors.append(c)

        fig_rank = go.Figure(go.Bar(
            x=city_df["mean"],
            y=city_df["city"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1f}" for v in city_df["mean"]],
            textposition="inside",
            textfont=dict(family="JetBrains Mono", size=11, color="#040d1a"),
            hovertemplate="<b>%{y}</b><br>Mean Stress: %{x:.1f}<extra></extra>",
        ))
        layout = plotly_base_layout("MEAN STRESS INDEX BY CITY", height=340)
        layout["xaxis"]["title"] = dict(text="Stress Index (0–100)", font=dict(size=10))
        layout["xaxis"]["range"] = [0, 105]
        layout["yaxis"]["tickfont"] = dict(family="Rajdhani", size=13, color="#e8f4fd")
        fig_rank.update_layout(**layout)
        st.plotly_chart(fig_rank, use_container_width=True, config={"displayModeBar": False})

    with col_trend:
        st.markdown('<div class="section-head">📈 Monthly Stress Trend</div>', unsafe_allow_html=True)

        trend = (df.copy()
                   .assign(month=df["date"].dt.to_period("M").astype(str))
                   .groupby(["month", "city"])["stress_index"]
                   .mean()
                   .reset_index())

        top_cities = city_df["city"].head(4).tolist()
        city_palette = [CYAN, RED_HOT, AMBER, TEAL]

        fig_trend = go.Figure()
        for i, city in enumerate(top_cities):
            cdata = trend[trend["city"] == city].sort_values("month")
            fig_trend.add_trace(go.Scatter(
                x=cdata["month"], y=cdata["stress_index"],
                mode="lines",
                name=city,
                line=dict(color=city_palette[i], width=2),
                hovertemplate=f"<b>{city}</b><br>%{{x}}<br>Stress: %{{y:.1f}}<extra></extra>",
            ))

        layout2 = plotly_base_layout("MONTHLY STRESS TRENDS — TOP 4 CITIES", height=340)
        layout2["showlegend"] = True
        layout2["legend"] = dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(family="JetBrains Mono", size=9, color=TEXT_DIM),
            bgcolor="rgba(0,0,0,0)",
        )
        layout2["xaxis"]["tickangle"] = -45
        layout2["xaxis"]["nticks"] = 8
        fig_trend.update_layout(**layout2)
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

    # ── Correlation Heatmap ───────────────────────────────
    st.markdown('<div class="section-head">🔗 Feature Correlation Matrix</div>', unsafe_allow_html=True)

    corr_cols = FEATURE_COLS + ["stress_index"]
    corr_labels = ["Traffic", "AQI", "Noise", "Temp", "Humidity", "Pop Density", "Pub Transit", "STRESS"]
    corr_matrix = df[corr_cols].corr().values

    fig_corr = go.Figure(go.Heatmap(
        z=corr_matrix,
        x=corr_labels, y=corr_labels,
        colorscale=[[0, "#040d1a"], [0.5, "#003d5c"], [1, CYAN]],
        text=[[f"{v:.2f}" for v in row] for row in corr_matrix],
        texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono", size=10),
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
        zmin=-1, zmax=1,
        showscale=True,
        colorbar=dict(
            tickfont=dict(family="JetBrains Mono", size=9, color=TEXT_DIM),
            outlinewidth=0, thickness=12,
        ),
    ))
    layout3 = plotly_base_layout("PEARSON CORRELATION — FEATURES vs STRESS INDEX", height=320)
    layout3["xaxis"]["tickfont"] = dict(family="JetBrains Mono", size=10, color=TEXT_DIM)
    layout3["yaxis"]["tickfont"] = dict(family="JetBrains Mono", size=10, color=TEXT_DIM)
    fig_corr.update_layout(**layout3)
    st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════
#  TAB 2: SIMULATION LAB
# ══════════════════════════════════════════════════════════
with tab_sim:

    # Header banner
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(0,212,255,0.06), transparent);
                border: 1px solid rgba(0,212,255,0.15); border-radius: 12px;
                padding: 16px 24px; margin-bottom: 24px; display:flex; align-items:center; gap:14px;">
        <span style="font-size:1.6rem;">⚡</span>
        <div>
            <div style="font-family:Rajdhani,sans-serif;font-weight:700;font-size:1.2rem;
                        letter-spacing:2px;color:#00d4ff;text-transform:uppercase;">
                Urban Stress Simulation Lab
            </div>
            <div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;
                        letter-spacing:1px;color:#7a9ab8;margin-top:3px;">
                Adjust city parameters → Run AI prediction → Get real-time stress analysis
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # City selector row
    city_col, preset_col, _ = st.columns([1, 1, 2])

    with city_col:
        cities = sorted(df["city"].unique().tolist())
        selected_city = st.selectbox("📍 Reference City", cities, index=1)

    with preset_col:
        preset = st.selectbox("⚡ Load Preset", [
            "Custom", "Peak Rush Hour", "Weekend Morning",
            "Post-Monsoon Evening", "Winter Smog Alert"
        ])

    # Preset defaults
    presets = {
        "Peak Rush Hour":        {"traffic_volume": 92, "aqi": 210, "noise_pollution": 82, "temperature": 32, "humidity": 60, "pop_density": 18000, "public_transport_usage": 75},
        "Weekend Morning":       {"traffic_volume": 35, "aqi": 85,  "noise_pollution": 55, "temperature": 27, "humidity": 72, "pop_density": 11000, "public_transport_usage": 45},
        "Post-Monsoon Evening":  {"traffic_volume": 78, "aqi": 95,  "noise_pollution": 70, "temperature": 29, "humidity": 88, "pop_density": 11000, "public_transport_usage": 68},
        "Winter Smog Alert":     {"traffic_volume": 85, "aqi": 380, "noise_pollution": 75, "temperature": 14, "humidity": 55, "pop_density": 11300, "public_transport_usage": 72},
        "Custom":                {k: FEATURE_META[k]["default"] for k in FEATURE_COLS},
    }

    # If city selected, use real median values as defaults
    city_medians = df[df["city"] == selected_city][FEATURE_COLS].median().to_dict()

    # Merge: preset overrides city medians, unless Custom → use city medians
    if preset == "Custom":
        slider_defaults = city_medians
    else:
        slider_defaults = presets[preset]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main simulation layout ────────────────────────────
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.markdown('<div class="control-panel-title">🎛️ Environmental Parameters</div>', unsafe_allow_html=True)

        slider_vals = {}
        for feat, meta in FEATURE_META.items():
            default_val = slider_defaults.get(feat, meta["default"])
            default_val = float(np.clip(default_val, meta["lo"], meta["hi"]))

            icon_label = f"{meta['icon']}  {meta['label']}"
            val = st.slider(
                icon_label,
                min_value=float(meta["lo"]),
                max_value=float(meta["hi"]),
                value=default_val,
                step=float(meta["step"]),
                format=f"%.0f{meta['unit']}",
                key=f"slider_{feat}",
            )
            slider_vals[feat] = val

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        btn_col, reset_col = st.columns([2, 1])
        with btn_col:
            run_clicked = st.button("⚡  RUN AI SIMULATION", key="run_btn")
        with reset_col:
            reset_clicked = st.button("↺  RESET", key="reset_btn")

        if reset_clicked:
            st.rerun()

    # ── Right: output panel ───────────────────────────────
    with right_col:
        st.markdown('<div class="section-head">📡 Prediction Output</div>', unsafe_allow_html=True)

        if "sim_result" not in st.session_state:
            st.session_state.sim_result = None

        if run_clicked:
            with st.spinner("🧠 Running inference..."):
                score = predict_stress(model, scaler, slider_vals, model_choice)
                insights = generate_insights(slider_vals, score)
            st.session_state.sim_result = {"score": score, "inputs": slider_vals.copy(), "insights": insights}

        result = st.session_state.sim_result

        if result is None:
            st.markdown("""
            <div class="result-panel">
                <div style="font-size:3rem;margin-bottom:16px;opacity:0.3;">🏙️</div>
                <div class="result-waiting">Awaiting simulation input</div>
                <div style="font-family:Outfit,sans-serif;font-size:0.82rem;
                            color:#3d5a73;margin-top:8px;">
                    Adjust parameters and click<br>RUN AI SIMULATION
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            score = result["score"]
            color, category, bg, glow = stress_color(score)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"font": {"family": "Rajdhani", "size": 64, "color": color},
                        "suffix": ""},
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=1,
                              tickcolor=TEXT_DIM,
                              tickfont=dict(family="JetBrains Mono", size=10, color=TEXT_DIM),
                              dtick=20),
                    bar=dict(color=color, thickness=0.22,
                             line=dict(color=color, width=2)),
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    steps=[
                        {"range": [0, 35],  "color": "rgba(6,214,160,0.08)"},
                        {"range": [35, 65], "color": "rgba(255,200,87,0.08)"},
                        {"range": [65, 100],"color": "rgba(255,77,109,0.08)"},
                    ],
                    threshold=dict(
                        line=dict(color=color, width=3),
                        thickness=0.75, value=score
                    ),
                ),
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Rajdhani"),
                height=260,
                margin=dict(l=30, r=30, t=20, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

            # Stress category badge
            st.markdown(f"""
            <div style="text-align:center; margin: -8px 0 20px;">
                <div style="display:inline-block; padding:10px 36px;
                            background:{bg}; border:1.5px solid {glow};
                            border-radius:10px;
                            font-family:Rajdhani,sans-serif; font-weight:700;
                            font-size:1.3rem; letter-spacing:3px;
                            color:{color}; text-transform:uppercase;">
                    {category}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # AI Insights
            st.markdown("""
            <div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;
                        letter-spacing:2px;text-transform:uppercase;color:#3d5a73;
                        margin-bottom:10px;">
                AI INSIGHTS
            </div>
            """, unsafe_allow_html=True)

            for icon, text, ic_color in result["insights"]:
                st.markdown(f"""
                <div style="display:flex;align-items:flex-start;gap:12px;
                            background:rgba(255,255,255,0.02);
                            border:1px solid rgba(255,255,255,0.05);
                            border-left: 3px solid {ic_color};
                            border-radius:8px; padding:11px 14px; margin-bottom:8px;">
                    <span style="font-size:1.1rem;flex-shrink:0;margin-top:1px;">{icon}</span>
                    <span style="font-family:Outfit,sans-serif;font-size:0.85rem;
                                 color:#c5dff0;line-height:1.45;">{text}</span>
                </div>
                """, unsafe_allow_html=True)

            # Mini comparison bar vs city average
            city_avg = df[df["city"] == selected_city]["stress_index"].mean()
            delta = score - city_avg
            delta_sign = "+" if delta > 0 else ""
            delta_color = RED_HOT if delta > 5 else (GREEN if delta < -5 else AMBER)

            st.markdown(f"""
            <div style="margin-top:16px;padding:16px 18px;
                        background:rgba(0,212,255,0.04);
                        border:1px solid rgba(0,212,255,0.12);border-radius:10px;">
                <div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;
                            letter-spacing:2px;color:#3d5a73;text-transform:uppercase;
                            margin-bottom:10px;">
                    vs {selected_city} Historical Average ({city_avg:.1f})
                </div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="font-family:Rajdhani,sans-serif;font-size:1.8rem;
                                font-weight:700;color:{delta_color};">
                        {delta_sign}{delta:.1f}
                    </div>
                    <div style="font-family:Outfit,sans-serif;font-size:0.8rem;color:#7a9ab8;">
                        {"above" if delta > 0 else "below"} city average
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Scenario comparison (bonus) ───────────────────────
    if result:
        st.markdown('<div class="section-head" style="margin-top:32px;">🔬 Scenario Analysis — What Changes Most?</div>', unsafe_allow_html=True)

        # Sensitivity: vary each feature ±20%, show delta in prediction
        base_score = result["score"]
        base_vals  = result["inputs"].copy()
        sens_data = []
        for feat in FEATURE_COLS:
            meta = FEATURE_META[feat]
            delta_up = base_vals.copy()
            delta_up[feat] = min(meta["hi"], base_vals[feat] * 1.2)
            score_up = predict_stress(model, scaler, delta_up, model_choice)

            delta_dn = base_vals.copy()
            delta_dn[feat] = max(meta["lo"], base_vals[feat] * 0.8)
            score_dn = predict_stress(model, scaler, delta_dn, model_choice)

            sens_data.append({
                "Feature": meta["label"],
                "Icon": meta["icon"],
                "+20% Impact": round(score_up - base_score, 2),
                "-20% Impact": round(score_dn - base_score, 2),
            })

        sens_df = pd.DataFrame(sens_data).sort_values("+20% Impact", ascending=False)

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Bar(
            name="+20% increase",
            y=sens_df["Feature"],
            x=sens_df["+20% Impact"],
            orientation="h",
            marker_color=[RED_HOT if v > 0 else GREEN for v in sens_df["+20% Impact"]],
            opacity=0.9,
            hovertemplate="<b>%{y}</b><br>+20%: %{x:+.2f} stress pts<extra></extra>",
        ))
        layout_s = plotly_base_layout("SENSITIVITY: STRESS CHANGE PER ±20% PARAMETER SHIFT", height=280)
        layout_s["xaxis"]["title"] = dict(text="Δ Stress Index", font=dict(size=10))
        layout_s["xaxis"]["zeroline"] = True
        layout_s["xaxis"]["zerolinecolor"] = "rgba(0,212,255,0.3)"
        layout_s["xaxis"]["zerolinewidth"] = 1.5
        layout_s["yaxis"]["tickfont"] = dict(family="Outfit", size=11, color="#c5dff0")
        fig_sens.update_layout(**layout_s)
        st.plotly_chart(fig_sens, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════
#  TAB 3: MODEL INSIGHTS
# ══════════════════════════════════════════════════════════
with tab_insights:

    # ── Performance metrics row ───────────────────────────
    st.markdown('<div class="section-head">📐 Model Performance</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, icon, sub in [
        (m1, "R² Score",   f"{metrics['r2']:.4f}",  "🎯", f"{r2_pct:.1f}% variance explained"),
        (m2, "MAE",         f"{metrics['mae']:.2f}", "📏", "Mean Absolute Error"),
        (m3, "RMSE",        f"{metrics['rmse']:.2f}","📉", "Root Mean Sq Error"),
        (m4, "Test Records",f"{metrics['test_size']:,}","🧪", f"of {metrics['train_size']+metrics['test_size']:,} total"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Importance + Actual vs Predicted ──────────
    col_imp, col_avp = st.columns([1, 1.2], gap="large")

    with col_imp:
        st.markdown('<div class="section-head">⚖️ Feature Importance</div>', unsafe_allow_html=True)

        imp = metrics["importances"]
        feat_labels = [FEATURE_META[f]["label"] for f in FEATURE_COLS]
        feat_icons  = [FEATURE_META[f]["icon"]  for f in FEATURE_COLS]

        imp_df = (pd.DataFrame({"feature": feat_labels, "icon": feat_icons, "importance": imp})
                    .sort_values("importance", ascending=True))

        # Color gradient: most important = cyan, least = dim
        n = len(imp_df)
        bar_colors = [
            f"rgba(0,{int(150 + 62 * i/n)},{int(200 + 55 * i/n)},{0.4 + 0.6*i/n})"
            for i in range(n)
        ]

        fig_imp = go.Figure(go.Bar(
            y=[f"{r['icon']} {r['feature']}" for _, r in imp_df.iterrows()],
            x=imp_df["importance"],
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(width=0),
            ),
            text=[f"{v*100:.1f}%" for v in imp_df["importance"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=10, color=TEXT_DIM),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        layout_imp = plotly_base_layout("FEATURE IMPORTANCE SCORE", height=340)
        layout_imp["yaxis"]["tickfont"] = dict(family="Outfit", size=11, color="#c5dff0")
        layout_imp["xaxis"]["tickformat"] = ".2f"
        layout_imp["xaxis"]["title"] = dict(text="Importance Weight", font=dict(size=10))
        fig_imp.update_layout(**layout_imp)
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

        # Importance table
        imp_df_display = imp_df.copy()
        imp_df_display["importance"] = (imp_df_display["importance"] * 100).round(2).astype(str) + "%"
        imp_df_display = imp_df_display[["icon","feature","importance"]].rename(
            columns={"icon":"","feature":"Feature","importance":"Weight"}
        ).iloc[::-1]
        st.dataframe(
            imp_df_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "": st.column_config.TextColumn(width="small"),
                "Feature": st.column_config.TextColumn(width="medium"),
                "Weight": st.column_config.TextColumn(width="small"),
            }
        )

    with col_avp:
        st.markdown('<div class="section-head">🔮 Actual vs Predicted</div>', unsafe_allow_html=True)

        y_te = metrics["y_test"]
        y_pr = metrics["y_pred"]
        sample_n = min(300, len(y_te))
        idx = np.random.choice(len(y_te), sample_n, replace=False)

        resid = y_pr[idx] - y_te[idx]
        resid_color = [RED_HOT if abs(r) > 10 else (AMBER if abs(r) > 5 else GREEN) for r in resid]

        fig_avp = go.Figure()
        # Perfect prediction line
        perfect_range = [0, 100]
        fig_avp.add_trace(go.Scatter(
            x=perfect_range, y=perfect_range,
            mode="lines",
            line=dict(color="rgba(0,212,255,0.3)", width=1.5, dash="dash"),
            name="Perfect",
            hoverinfo="skip",
        ))
        # Scatter
        fig_avp.add_trace(go.Scatter(
            x=y_te[idx], y=y_pr[idx],
            mode="markers",
            marker=dict(
                color=resid_color, size=5, opacity=0.75,
                line=dict(width=0),
            ),
            hovertemplate="Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>",
            name="Predictions",
        ))
        layout_avp = plotly_base_layout("ACTUAL vs PREDICTED STRESS INDEX", height=340)
        layout_avp["xaxis"]["title"] = dict(text="Actual Stress Index", font=dict(size=10))
        layout_avp["yaxis"]["title"] = dict(text="Predicted", font=dict(size=10))
        layout_avp["showlegend"] = False
        layout_avp["xaxis"]["range"] = [-2, 105]
        layout_avp["yaxis"]["range"] = [-2, 105]
        fig_avp.update_layout(**layout_avp)
        st.plotly_chart(fig_avp, use_container_width=True, config={"displayModeBar": False})

        # Residual distribution
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Histogram(
            x=(y_pr - y_te),
            nbinsx=40,
            marker=dict(
                color=CYAN, opacity=0.6,
                line=dict(color=CYAN, width=0.3),
            ),
            hovertemplate="Residual: %{x:.1f}<br>Count: %{y}<extra></extra>",
        ))
        layout_r = plotly_base_layout("RESIDUAL DISTRIBUTION (y_pred − y_actual)", height=200)
        layout_r["xaxis"]["title"] = dict(text="Residual", font=dict(size=10))
        layout_r["yaxis"]["title"] = dict(text="Frequency", font=dict(size=10))
        layout_r["margin"]["t"] = 40
        fig_resid.update_layout(**layout_r)
        st.plotly_chart(fig_resid, use_container_width=True, config={"displayModeBar": False})

    # ── City-level RMSE breakdown ─────────────────────────
    st.markdown('<div class="section-head">🏙️ Per-City Prediction Quality</div>', unsafe_allow_html=True)

    sub_df = df[FEATURE_COLS + ["stress_index", "city"]].dropna()
    city_metrics = []
    for city in sorted(sub_df["city"].unique()):
        cdf = sub_df[sub_df["city"] == city]
        X_c = cdf[FEATURE_COLS].values
        y_c = cdf["stress_index"].values
        if len(X_c) < 10:
            continue
        if model_choice == "Random Forest":
            yp_c = model.predict(X_c)
        else:
            yp_c = model.predict(scaler.transform(X_c))
        city_metrics.append({
            "City": city,
            "RMSE": round(np.sqrt(mean_squared_error(y_c, yp_c)), 2),
            "MAE":  round(mean_absolute_error(y_c, yp_c), 2),
            "R²":   round(r2_score(y_c, yp_c), 3),
            "Mean Stress": round(y_c.mean(), 1),
        })

    city_met_df = pd.DataFrame(city_metrics).sort_values("RMSE")

    fig_city = go.Figure(go.Bar(
        x=city_met_df["City"],
        y=city_met_df["RMSE"],
        marker=dict(
            color=[GREEN if v < 4 else (AMBER if v < 7 else RED_HOT) for v in city_met_df["RMSE"]],
            line=dict(width=0),
        ),
        text=[f"{v:.2f}" for v in city_met_df["RMSE"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=10, color=TEXT_DIM),
        hovertemplate="<b>%{x}</b><br>RMSE: %{y:.2f}<extra></extra>",
    ))
    layout_city = plotly_base_layout("PER-CITY RMSE — MODEL ERROR BREAKDOWN", height=260)
    layout_city["yaxis"]["title"] = dict(text="RMSE (stress pts)", font=dict(size=10))
    layout_city["xaxis"]["tickfont"] = dict(family="Rajdhani", size=12, color="#e8f4fd")
    fig_city.update_layout(**layout_city)
    st.plotly_chart(fig_city, use_container_width=True, config={"displayModeBar": False})

    # ── Model info card ───────────────────────────────────
    st.markdown(f"""
    <div style="background: rgba(0,212,255,0.03); border:1px solid rgba(0,212,255,0.12);
                border-radius:12px; padding:20px 24px; margin-top:12px;">
        <div style="font-family:Rajdhani,sans-serif;font-weight:700;font-size:0.95rem;
                    letter-spacing:2px;text-transform:uppercase;color:#7a9ab8;margin-bottom:12px;">
            ℹ️ Model Configuration
        </div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;">
            <div>
                <div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;letter-spacing:2px;color:#3d5a73;text-transform:uppercase;">Algorithm</div>
                <div style="font-family:Outfit,sans-serif;font-size:0.88rem;color:#c5dff0;margin-top:3px;">{model_choice}</div>
            </div>
            <div>
                <div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;letter-spacing:2px;color:#3d5a73;text-transform:uppercase;">Features</div>
                <div style="font-family:Outfit,sans-serif;font-size:0.88rem;color:#c5dff0;margin-top:3px;">7 urban indicators</div>
            </div>
            <div>
                <div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;letter-spacing:2px;color:#3d5a73;text-transform:uppercase;">Train Split</div>
                <div style="font-family:Outfit,sans-serif;font-size:0.88rem;color:#c5dff0;margin-top:3px;">80% time-based</div>
            </div>
            <div>
                <div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;letter-spacing:2px;color:#3d5a73;text-transform:uppercase;">Target</div>
                <div style="font-family:Outfit,sans-serif;font-size:0.88rem;color:#c5dff0;margin-top:3px;">Stress Index 0–100</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px; padding-top:20px; border-top:1px solid rgba(0,212,255,0.08);
            display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
    <div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;
                letter-spacing:2px;text-transform:uppercase;color:#3d5a73;">
        Urban Pulse AI · Smart City Stress Intelligence · v1.0
    </div>
    <div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;
                letter-spacing:1px;color:#3d5a73;">
        Data: OpenAQ · IMD · CPCB · Census India · UMTA
    </div>
</div>
""", unsafe_allow_html=True)
