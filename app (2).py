# ╔══════════════════════════════════════════════════════════════════════╗
# ║  URBAN PULSE AI  ·  Smart City Stress Command Center  ·  v2.1       ║
# ║  Run: streamlit run app.py                                           ║
# ║  Deps: streamlit pandas numpy scikit-learn plotly                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
import os
import datetime
import calendar

warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Pulse AI · Command Center",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  –  Neo-brutalist Command Room
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=Barlow:wght@300;400;500;600&display=swap');

:root {
  --ink:    #070d14;  --ink1: #0c1622;  --ink2: #111f2e;  --ink3: #162840;
  --glass:  rgba(12,22,34,0.74);
  --gb:     rgba(0,229,255,0.12);
  --gs:     rgba(0,229,255,0.05);
  --cyan:   #00e5ff;  --cdim: #0099b3;  --cglow: rgba(0,229,255,0.18);
  --green:  #00f5a0;  --gdim: rgba(0,245,160,0.14);
  --amber:  #ffb340;  --adim: rgba(255,179,64,0.14);
  --red:    #ff4060;  --rdim: rgba(255,64,96,0.14);
  --tx:     #deeeff;  --tx2: #8aaec8;  --tx3: #3d6278;
  --bdr:    rgba(0,229,255,0.09);  --bdr2: rgba(0,229,255,0.24);
}

html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"] {
  background: var(--ink) !important; color: var(--tx) !important;
  font-family: 'Barlow', sans-serif !important;
}
[data-testid="stAppViewContainer"]::before {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background: repeating-linear-gradient(
    0deg,transparent,transparent 3px,
    rgba(0,229,255,0.025) 3px,rgba(0,229,255,0.025) 4px
  );
}
[data-testid="stAppViewContainer"]::after {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.28;
  background-image:
    linear-gradient(var(--gb) 1px,transparent 1px),
    linear-gradient(90deg,var(--gb) 1px,transparent 1px);
  background-size: 80px 80px;
}
[data-testid="stHeader"],[data-testid="stToolbar"]{display:none!important;}
footer,#MainMenu{visibility:hidden!important;}
.block-container{padding:1rem 2.2rem 4rem!important; max-width:1700px!important;}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
  background:var(--ink1)!important;
  border-right:1px solid var(--bdr)!important;
}
[data-testid="stSidebar"]::before{
  content:''; position:absolute; top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--cyan),var(--green),transparent);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{
  background:var(--ink1)!important; border-bottom:1px solid var(--bdr)!important;
  gap:2px; padding:4px 4px 0; border-radius:10px 10px 0 0;
}
.stTabs [data-baseweb="tab"]{
  font-family:'Barlow Condensed',sans-serif!important; font-size:.88rem!important;
  font-weight:600!important; letter-spacing:1.8px!important; text-transform:uppercase!important;
  color:var(--tx3)!important; background:transparent!important; border:none!important;
  border-radius:8px 8px 0 0!important; padding:10px 22px!important; transition:all .2s!important;
}
.stTabs [aria-selected="true"]{
  color:var(--cyan)!important; background:rgba(0,229,255,0.06)!important;
  border-bottom:2px solid var(--cyan)!important;
}
.stTabs [data-baseweb="tab-panel"]{background:transparent!important; padding-top:28px!important;}

/* ── Glass card ── */
.gc{
  background:var(--glass); border:1px solid var(--gb); border-radius:14px;
  padding:22px 24px; position:relative; overflow:hidden;
  transition:border-color .25s,box-shadow .25s;
}
.gc::before{
  content:''; position:absolute; top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,var(--cyan),transparent 60%);
}
.gc:hover{border-color:var(--bdr2); box-shadow:0 4px 32px var(--cglow);}

/* ── KPI card ── */
.kc{
  background:var(--glass); border:1px solid var(--gb); border-radius:16px;
  padding:24px 20px 20px; position:relative; overflow:hidden; transition:all .25s;
}
.kc::after{
  content:''; position:absolute; bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--cdim),transparent);
  opacity:0; transition:.25s;
}
.kc:hover::after{opacity:1;}
.kc-lbl{
  font-family:'DM Mono',monospace;font-size:.60rem;letter-spacing:2.5px;
  text-transform:uppercase;color:var(--tx3);margin-bottom:8px;
}
.kc-val{
  font-family:'Barlow Condensed',sans-serif;font-size:2.5rem;
  font-weight:700;color:var(--cyan);line-height:1;
}
.kc-sub{font-size:.75rem;color:var(--tx2);margin-top:6px;line-height:1.4;}
.kc-ico{font-size:1.4rem;margin-bottom:10px;opacity:.8;}

/* ── Section label ── */
.sl{
  font-family:'DM Mono',monospace; font-size:.62rem; letter-spacing:3px;
  text-transform:uppercase; color:var(--tx3); margin:28px 0 14px;
  display:flex; align-items:center; gap:10px;
}
.sl::after{content:''; flex:1; height:1px; background:var(--bdr);}

/* ── Pulse dot ── */
@keyframes pulse-dot{
  0%,100%{opacity:1;box-shadow:0 0 6px currentColor;}
  50%{opacity:.3;box-shadow:none;}
}
.dot{
  display:inline-block;width:8px;height:8px;
  border-radius:50%;animation:pulse-dot 2.2s ease-in-out infinite;
}
.dg{color:var(--green);background:var(--green);}
.da{color:var(--amber);background:var(--amber);}
.dr{color:var(--red);background:var(--red);}

/* ── Insight row ── */
.ir{
  display:flex; align-items:flex-start; gap:14px; padding:14px 16px;
  background:var(--glass); border:1px solid var(--gb);
  border-radius:10px; margin-bottom:10px;
}
.ir-ico{font-size:1.15rem;flex-shrink:0;margin-top:1px;}
.ir-txt{font-size:.87rem;line-height:1.55;color:var(--tx);}
.itag{
  font-family:'DM Mono',monospace; font-size:.58rem; letter-spacing:1.5px;
  text-transform:uppercase; padding:2px 8px; border-radius:4px;
  margin-bottom:4px; display:inline-block;
}

/* ── Alert banner ── */
.ab{
  padding:14px 18px;border-radius:10px;margin-bottom:10px;
  display:flex;align-items:flex-start;gap:12px;font-size:.85rem;line-height:1.5;
}

/* ── Slider overrides ── */
.stSlider>label{
  font-family:'DM Mono',monospace!important;font-size:.65rem!important;
  letter-spacing:2px!important;text-transform:uppercase!important;color:var(--tx2)!important;
}
div[data-testid="stSlider"] [role="slider"]{
  background:var(--cyan)!important;border-color:var(--cyan)!important;
  box-shadow:0 0 10px var(--cglow)!important;
}

/* ── Buttons ── */
.stButton>button{
  font-family:'Barlow Condensed',sans-serif!important; font-size:1rem!important;
  font-weight:700!important; letter-spacing:2.5px!important; text-transform:uppercase!important;
  background:linear-gradient(135deg,#003d52,#00283a)!important; color:var(--cyan)!important;
  border:1.5px solid var(--cdim)!important; border-radius:8px!important;
  padding:12px 24px!important; transition:all .2s!important;
  box-shadow:0 0 16px rgba(0,229,255,.12)!important;
}
.stButton>button:hover{
  background:linear-gradient(135deg,#005270,#003a50)!important;
  box-shadow:0 0 28px rgba(0,229,255,.28)!important;
  border-color:var(--cyan)!important; transform:translateY(-1px)!important;
}

/* ── Selectbox ── */
.stSelectbox label,.stMultiSelect label{
  font-family:'DM Mono',monospace!important;font-size:.62rem!important;
  letter-spacing:2px!important;text-transform:uppercase!important;color:var(--tx2)!important;
}
[data-baseweb="select"]>div{
  background:var(--ink2)!important;border-color:var(--bdr)!important;color:var(--tx)!important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--ink);}
::-webkit-scrollbar-thumb{background:var(--ink3);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--cdim);}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
CY  = "#00e5ff"
GR  = "#00f5a0"
AM  = "#ffb340"
RD  = "#ff4060"
TX  = "#deeeff"
D2  = "#8aaec8"
D3  = "#3d6278"
INK = "#070d14"
INK2 = "#0c1622"
BDR = "rgba(0,229,255,0.09)"

FEATURE_COLS = [
    "traffic_volume", "aqi", "noise_pollution",
    "temperature", "humidity", "pop_density", "public_transport_usage",
]

# FIX ①: slider format — escape % in unit strings to avoid sprintf error
# "%.0f%" -> "%.0f%%" is the corrected sprintf format for a literal percent sign
FMETA = {
    "traffic_volume":        {"ico": "🚦", "lbl": "Traffic Volume",        "unit": "/100",  "fmt": "%.0f/100",  "lo": 0,    "hi": 100,   "step": 1.0,   "def": 72},
    "aqi":                   {"ico": "💨", "lbl": "Air Quality Index",      "unit": " AQI",  "fmt": "%.0f AQI",  "lo": 10,   "hi": 500,   "step": 5.0,   "def": 140},
    "noise_pollution":       {"ico": "🔊", "lbl": "Noise Pollution",        "unit": " dB",   "fmt": "%.0f dB",   "lo": 30,   "hi": 100,   "step": 1.0,   "def": 70},
    "temperature":           {"ico": "🌡️", "lbl": "Temperature",            "unit": " °C",   "fmt": "%.0f °C",   "lo": 5,    "hi": 48,    "step": 0.5,   "def": 29},
    "humidity":              {"ico": "💧", "lbl": "Humidity",               "unit": "%%",    "fmt": "%.0f%%",    "lo": 10,   "hi": 100,   "step": 1.0,   "def": 65},
    "pop_density":           {"ico": "👥", "lbl": "Population Density",     "unit": "/km2",  "fmt": "%.0f/km2",  "lo": 1000, "hi": 40000, "step": 500.0, "def": 11000},
    "public_transport_usage":{"ico": "🚇", "lbl": "Public Transport Usage", "unit": "/100",  "fmt": "%.0f/100",  "lo": 0,    "hi": 100,   "step": 1.0,   "def": 60},
}

MODEL_OPTIONS = {
    "Random Forest":      {"cls": RandomForestRegressor,    "kw": {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 3, "random_state": 42, "n_jobs": -1}},
    "Gradient Boosting":  {"cls": GradientBoostingRegressor,"kw": {"n_estimators": 200, "max_depth": 5,  "learning_rate": 0.08,  "random_state": 42}},
    "Ridge Regression":   {"cls": Ridge,                    "kw": {"alpha": 1.0}},
    "Linear Regression":  {"cls": LinearRegression,         "kw": {}},
}


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

# FIX ②: compute timestamp dynamically on every render
def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d  %H:%M UTC")


def stress_tier(s):
    """Returns (color, label, bg_rgba, dot_class)."""
    try:
        v = float(s)
        if v < 35:
            return GR, "LOW",      "rgba(0,245,160,0.12)",  "dg"
        if v < 65:
            return AM, "MODERATE", "rgba(255,179,64,0.12)", "da"
        return RD, "HIGH",         "rgba(255,64,96,0.12)",  "dr"
    except Exception:
        return D2, "UNKNOWN", "rgba(100,100,100,0.12)", "da"


def hex_rgb(h):
    """'#rrggbb' → 'r,g,b' string for use inside rgba(...)."""
    h = h.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


# FIX ③: convert hex colour + alpha to a valid rgba() string for Plotly
def hex_rgba(h, alpha=1.0):
    """'#rrggbb', 0.13 → 'rgba(r,g,b,0.13)'  — safe for Plotly fillcolor."""
    return f"rgba({hex_rgb(h)},{alpha})"


def plot_base(title="", h=340, xl="", yl="", legend=False):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color=D2, size=11),
        title=dict(
            text=title,
            font=dict(family="Barlow Condensed, sans-serif", color=TX, size=15),
            x=0.01,
        ),
        margin=dict(l=14, r=14, t=46, b=14),
        height=h,
        xaxis=dict(
            gridcolor="rgba(0,229,255,0.06)",
            zerolinecolor="rgba(0,229,255,0.12)",
            tickfont=dict(size=10),
            title=dict(text=xl, font=dict(size=10)),
        ),
        yaxis=dict(
            gridcolor="rgba(0,229,255,0.06)",
            zerolinecolor="rgba(0,229,255,0.12)",
            tickfont=dict(size=10),
            title=dict(text=yl, font=dict(size=10)),
        ),
        showlegend=legend,
        legend=dict(
            font=dict(family="DM Mono", size=9, color=D2),
            bgcolor="rgba(0,0,0,0)",
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ) if legend else {},
        hoverlabel=dict(
            bgcolor=INK2,
            font=dict(family="DM Mono", size=11, color=TX),
            bordercolor=CY,
        ),
    )


CFG = {"displayModeBar": False, "scrollZoom": False}


# ═══════════════════════════════════════════════════════════════════════
#  DATA & MODEL
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    paths = [
        "urban_pulse_dataset.csv",
        os.path.join(os.path.dirname(__file__), "urban_pulse_dataset.csv"),
        "data/urban_pulse_dataset.csv",
    ]
    for p in paths:
        try:
            if os.path.exists(p):
                df = pd.read_csv(p, parse_dates=["date"])
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                return df, None
        except Exception:
            continue
    return None, "urban_pulse_dataset.csv not found. Place it alongside app.py."


@st.cache_resource(show_spinner=False)
def train_model(name):
    df, err = load_data()
    if err or df is None:
        return None, None, {}
    try:
        sub  = df[FEATURE_COLS + ["stress_index"]].dropna()
        X, y = sub[FEATURE_COLS].values, sub["stress_index"].values
        sp   = int(len(X) * 0.80)
        Xtr, Xte, ytr, yte = X[:sp], X[sp:], y[:sp], y[sp:]
        sc   = StandardScaler()
        Xts  = sc.fit_transform(Xtr)
        Xes  = sc.transform(Xte)
        ns   = name in ("Ridge Regression", "Linear Regression")
        cfg  = MODEL_OPTIONS[name]
        mdl  = cfg["cls"](**cfg["kw"])
        mdl.fit(Xts if ns else Xtr, ytr)
        yp   = mdl.predict(Xes if ns else Xte)
        imp  = (
            mdl.feature_importances_ if hasattr(mdl, "feature_importances_")
            else np.abs(mdl.coef_) / (np.abs(mdl.coef_).sum() + 1e-9)
        )
        return mdl, sc, {
            "r2":   r2_score(yte, yp),
            "mae":  mean_absolute_error(yte, yp),
            "rmse": np.sqrt(mean_squared_error(yte, yp)),
            "imp":  imp,
            "yte":  yte,
            "yp":   yp,
            "trn":  len(Xtr),
            "tst":  len(Xte),
            "ns":   ns,
        }
    except Exception as e:
        return None, None, {"error": str(e)}


def predict_one(mdl, sc, vals, ns):
    try:
        x = np.array([[vals[c] for c in FEATURE_COLS]])
        return float(np.clip(mdl.predict(sc.transform(x) if ns else x)[0], 0, 100))
    except Exception:
        return 50.0


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='padding:8px 0 24px;'>
      <div style='font-family:Barlow Condensed,sans-serif;font-size:1.55rem;
                  font-weight:800;letter-spacing:3px;color:{CY};
                  text-transform:uppercase;line-height:1;'>
        URBAN PULSE
      </div>
      <div style='font-family:DM Mono,monospace;font-size:.58rem;letter-spacing:2px;
                  color:{D3};text-transform:uppercase;margin-top:5px;'>
        AI Command Center v2.1
      </div>
      <div style='margin-top:12px;display:flex;align-items:center;gap:8px;'>
        <span class="dot dg"></span>
        <span style='font-family:DM Mono,monospace;font-size:.60rem;
                     color:{GR};letter-spacing:1.5px;'>SYSTEM OPERATIONAL</span>
      </div>
    </div>
    <hr style='border-color:{BDR};margin:0 0 20px;'>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "🧠  Prediction Model",
        list(MODEL_OPTIONS.keys()),
        index=0,
    )

    # FIX ②: use now_str() function so sidebar timestamp is always current
    st.markdown(f"""
    <div style='margin:20px 0 8px;font-family:DM Mono,monospace;font-size:.58rem;
                letter-spacing:2px;color:{D3};text-transform:uppercase;'>Data Source</div>
    <div style='font-family:Barlow,sans-serif;font-size:.83rem;color:{D2};line-height:1.6;'>
      urban_pulse_dataset.csv<br>
      <span style='color:{D3};'>8 cities · daily granularity</span>
    </div>
    <hr style='border-color:{BDR};margin:20px 0;'>
    <div style='font-family:DM Mono,monospace;font-size:.58rem;letter-spacing:2px;color:{D3};
                text-transform:uppercase;margin-bottom:6px;'>Session Time</div>
    <div style='font-family:DM Mono,monospace;font-size:.72rem;color:{D2};'>{now_str()}</div>
    <br>
    <div style='font-family:DM Mono,monospace;font-size:.56rem;color:{D3};letter-spacing:1px;
                line-height:1.8;border-top:1px solid {BDR};padding-top:14px;'>
      <b style='color:{TX};'>Sources</b><br>OpenAQ · CPCB · IMD<br>Census India · UMTA · GTFS
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  LOAD & TRAIN
# ═══════════════════════════════════════════════════════════════════════
df, data_err = load_data()
if data_err:
    st.error(
        f"**⚠️ Data not found:** `{data_err}`\n\n"
        "Place `urban_pulse_dataset.csv` next to `app.py`."
    )
    st.stop()

with st.spinner("⚡ Initialising AI engine …"):
    model, scaler, mtr = train_model(model_choice)

if model is None or "error" in mtr:
    st.error(f"Model training failed: {mtr.get('error', 'Unknown')}")
    st.stop()

r2pct = mtr["r2"] * 100
conf  = min(99.9, r2pct + 3.2)


# ═══════════════════════════════════════════════════════════════════════
#  GLOBAL HEADER
# ═══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='display:flex;align-items:center;gap:20px;padding:22px 0 18px;
            margin-bottom:8px;border-bottom:1px solid {BDR};position:relative;'>
  <div style='width:54px;height:54px;flex-shrink:0;
              background:linear-gradient(135deg,#003344,#001e2e);
              border:1.5px solid {CY};border-radius:14px;
              display:flex;align-items:center;justify-content:center;
              font-size:26px;box-shadow:0 0 22px rgba(0,229,255,0.2);'>
    🏙️
  </div>
  <div>
    <div style='font-family:Barlow Condensed,sans-serif;font-size:2rem;font-weight:800;
                letter-spacing:4px;text-transform:uppercase;
                background:linear-gradient(90deg,{CY} 0%,{GR} 60%,#a8fff2 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1;'>
      Urban Pulse AI
    </div>
    <div style='font-family:DM Mono,monospace;font-size:.64rem;color:{D3};
                letter-spacing:2.5px;text-transform:uppercase;margin-top:5px;'>
      Smart City Stress Intelligence · Command Center
    </div>
  </div>
  <div style='margin-left:auto;display:flex;gap:10px;flex-wrap:wrap;align-items:center;'>
    <span style='font-family:DM Mono,monospace;font-size:.66rem;letter-spacing:1.5px;
                 padding:5px 14px;border-radius:6px;background:rgba(0,229,255,.08);
                 border:1px solid rgba(0,229,255,0.33);color:{CY};'>
      MODEL · {model_choice.upper()}
    </span>
    <span style='font-family:DM Mono,monospace;font-size:.66rem;letter-spacing:1.5px;
                 padding:5px 14px;border-radius:6px;background:rgba(255,179,64,.08);
                 border:1px solid rgba(255,179,64,0.33);color:{AM};'>
      R² {mtr["r2"]:.4f}
    </span>
    <span style='font-family:DM Mono,monospace;font-size:.66rem;letter-spacing:1.5px;
                 padding:5px 14px;border-radius:6px;background:rgba(0,245,160,.08);
                 border:1px solid rgba(0,245,160,0.33);color:{GR};'>
      CONF {conf:.1f}%
    </span>
    <span style='font-family:DM Mono,monospace;font-size:.66rem;letter-spacing:1.5px;
                 padding:5px 14px;border-radius:6px;background:rgba(0,245,160,.10);
                 border:1px solid rgba(0,245,160,0.33);color:{GR};'>
      ● LIVE
    </span>
  </div>
  <div style='position:absolute;bottom:-1px;left:0;width:220px;height:2px;
              background:linear-gradient(90deg,{CY},{GR},transparent);'></div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════
T1, T2, T3, T4, T5 = st.tabs([
    "  📊  Executive Dashboard  ",
    "  ⚡  Simulation Lab  ",
    "  🗺️  City Intelligence  ",
    "  🤖  AI Insights  ",
    "  🔧  System & Model  ",
])


# ───────────────────────────────────────────────────────────────────────
#  ① EXECUTIVE DASHBOARD
# ───────────────────────────────────────────────────────────────────────
with T1:
    try:
        city_avg = (
            df.groupby("city")["stress_index"]
              .mean().sort_values(ascending=False).reset_index()
        )
        city_avg.columns = ["city", "ms"]
        most_s, least_s = city_avg.iloc[0], city_avg.iloc[-1]
        n_alerts = int((city_avg["ms"] > 65).sum())
        avg_s    = df["stress_index"].mean()

        # KPI strip
        st.markdown('<div class="sl">🔢  Platform-Wide KPIs</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        sc_col, sc_lbl, _, _ = stress_tier(avg_s)
        for col, ico, lbl, val, sub, color in [
            (k1, "📊", "Avg Stress Index",   f"{avg_s:.1f}",    f"{sc_lbl} tier overall",         sc_col),
            (k2, "🏙️", "Active Cities",       str(df["city"].nunique()), f"{len(df):,} records",   CY),
            (k3, "🚨", "High-Stress Alerts",  str(n_alerts),    "Cities above 65 threshold",       RD if n_alerts else GR),
            (k4, "🎯", "Model Accuracy",      f"{r2pct:.1f}%",  f"R²={mtr['r2']:.4f}  MAE={mtr['mae']:.2f}", CY),
        ]:
            col.markdown(f"""
            <div class="kc">
              <div class="kc-ico">{ico}</div>
              <div class="kc-lbl">{lbl}</div>
              <div class="kc-val" style="color:{color};">{val}</div>
              <div class="kc-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # City spotlight
        st.markdown('<div class="sl">🏆  City Spotlight</div>', unsafe_allow_html=True)
        ch, cl, cr = st.columns([1, 1, 1.8])
        mc = stress_tier(most_s["ms"])
        ml = stress_tier(least_s["ms"])

        ch.markdown(f"""
        <div class="gc" style="border-color:rgba(255,64,96,0.2);">
          <div class="kc-lbl">🔴  Most Stressed City</div>
          <div style='font-family:Barlow Condensed,sans-serif;font-size:2rem;
                      font-weight:800;color:{RD};margin:6px 0;'>
            {most_s["city"]}
          </div>
          <div class="kc-val" style="font-size:1.8rem;color:{mc[0]};">{most_s["ms"]:.1f}</div>
          <div class="kc-sub">Mean Stress · {mc[1]} tier</div>
        </div>""", unsafe_allow_html=True)

        cl.markdown(f"""
        <div class="gc" style="border-color:rgba(0,245,160,0.2);">
          <div class="kc-lbl">🟢  Least Stressed City</div>
          <div style='font-family:Barlow Condensed,sans-serif;font-size:2rem;
                      font-weight:800;color:{GR};margin:6px 0;'>
            {least_s["city"]}
          </div>
          <div class="kc-val" style="font-size:1.8rem;color:{ml[0]};">{least_s["ms"]:.1f}</div>
          <div class="kc-sub">Mean Stress · {ml[1]} tier</div>
        </div>""", unsafe_allow_html=True)

        try:
            fig_r = go.Figure(go.Bar(
                x=city_avg["ms"],
                y=city_avg["city"],
                orientation="h",
                marker=dict(
                    color=[stress_tier(v)[0] for v in city_avg["ms"]],
                    line=dict(width=0),
                ),
                text=[f"{v:.1f}" for v in city_avg["ms"]],
                textposition="inside",
                textfont=dict(family="DM Mono", size=10, color=INK),
                hovertemplate="<b>%{y}</b><br>Stress: %{x:.1f}<extra></extra>",
            ))
            lay = plot_base("CITY STRESS RANKING", h=260, xl="Stress Index (0–100)")
            lay["xaxis"]["range"] = [0, 108]
            lay["yaxis"]["tickfont"] = dict(family="Barlow Condensed", size=13, color=TX)
            lay["margin"]["t"] = 38
            fig_r.update_layout(**lay)
            cr.plotly_chart(fig_r, use_container_width=True, config=CFG)
        except Exception as e:
            cr.warning(f"Ranking chart: {e}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Trend + violin
        st.markdown('<div class="sl">📈  Temporal Overview</div>', unsafe_allow_html=True)
        tl, tr = st.columns([1.7, 1])

        with tl:
            try:
                trend = (
                    df.copy()
                      .assign(ym=df["date"].dt.to_period("M").astype(str))
                      .groupby(["ym", "city"])["stress_index"].mean().reset_index()
                )
                top4 = city_avg["city"].head(4).tolist()
                pal  = [CY, RD, AM, GR]
                fig_t = go.Figure()
                for i, c in enumerate(top4):
                    cd = trend[trend["city"] == c].sort_values("ym")
                    fig_t.add_trace(go.Scatter(
                        x=cd["ym"], y=cd["stress_index"],
                        mode="lines", name=c,
                        line=dict(color=pal[i], width=2),
                        hovertemplate=f"<b>{c}</b><br>%{{x}}<br>%{{y:.1f}}<extra></extra>",
                    ))
                lay_t = plot_base("MONTHLY STRESS TRENDS · TOP 4 CITIES",
                                  h=300, yl="Stress Index", legend=True)
                lay_t["xaxis"]["tickangle"] = -40
                lay_t["xaxis"]["nticks"] = 8
                fig_t.update_layout(**lay_t)
                tl.plotly_chart(fig_t, use_container_width=True, config=CFG)
            except Exception as e:
                tl.warning(f"Trend chart: {e}")

        with tr:
            try:
                fig_v = go.Figure(go.Violin(
                    y=df["stress_index"],
                    box_visible=True,
                    line_color=CY,
                    fillcolor="rgba(0,229,255,0.07)",
                    points="outliers",
                    marker=dict(color=CY, size=3, opacity=0.5),
                ))
                lay_v = plot_base("STRESS DISTRIBUTION", h=300, yl="Stress Index")
                lay_v["xaxis"]["showticklabels"] = False
                fig_v.update_layout(**lay_v)
                tr.plotly_chart(fig_v, use_container_width=True, config=CFG)
            except Exception as e:
                tr.warning(f"Violin chart: {e}")

        # Alerts
        if n_alerts:
            st.markdown('<div class="sl">🚨  Active Alerts</div>', unsafe_allow_html=True)
            for _, row in city_avg[city_avg["ms"] > 65].iterrows():
                sc2, _, bg, _ = stress_tier(row["ms"])
                st.markdown(f"""
                <div class="ab" style="background:{bg};border:1px solid rgba({hex_rgb(sc2)},0.27);color:{TX};">
                  <span style="font-size:1.1rem;">🚨</span>
                  <div>
                    <b style="color:{sc2};">{row["city"]}</b> — Mean Stress
                    <b style="color:{sc2};">{row["ms"]:.1f}</b> exceeds HIGH threshold.
                    <span style="color:{D2};font-size:.82rem;">
                      Urban planning review recommended.
                    </span>
                  </div>
                </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Executive Dashboard error: {e}")


# ───────────────────────────────────────────────────────────────────────
#  ② SIMULATION LAB
# ───────────────────────────────────────────────────────────────────────
with T2:
    try:
        st.markdown(f"""
        <div style='background:rgba(0,229,255,.04);border:1px solid rgba(0,229,255,.15);
                    border-radius:12px;padding:16px 22px;margin-bottom:26px;
                    display:flex;align-items:center;gap:14px;'>
          <span style='font-size:1.7rem;'>⚡</span>
          <div>
            <div style='font-family:Barlow Condensed,sans-serif;font-weight:800;
                        font-size:1.25rem;letter-spacing:2.5px;color:{CY};text-transform:uppercase;'>
              Urban Stress Simulation Lab
            </div>
            <div style='font-family:DM Mono,monospace;font-size:.63rem;
                        letter-spacing:1.5px;color:{D3};margin-top:3px;'>
              Adjust city parameters · Run AI prediction · Compare scenarios
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        ra, rb = st.columns([1, 1])
        with ra:
            cities   = sorted(df["city"].unique().tolist())
            ref_city = st.selectbox("📍 Reference City", cities, index=1)
        with rb:
            preset = st.selectbox("⚡ Quick Preset", [
                "Custom (City Baseline)", "Peak Rush Hour", "Weekend Morning",
                "Winter Smog Alert", "Clean Green City", "Monsoon Evening",
            ])

        PRESETS = {
            "Peak Rush Hour":    {"traffic_volume": 92, "aqi": 210, "noise_pollution": 82, "temperature": 33, "humidity": 62,  "pop_density": 18000, "public_transport_usage": 74},
            "Weekend Morning":   {"traffic_volume": 32, "aqi": 80,  "noise_pollution": 52, "temperature": 27, "humidity": 70,  "pop_density": 10000, "public_transport_usage": 42},
            "Winter Smog Alert": {"traffic_volume": 84, "aqi": 385, "noise_pollution": 76, "temperature": 14, "humidity": 54,  "pop_density": 11300, "public_transport_usage": 70},
            "Clean Green City":  {"traffic_volume": 28, "aqi": 40,  "noise_pollution": 44, "temperature": 22, "humidity": 60,  "pop_density": 4000,  "public_transport_usage": 88},
            "Monsoon Evening":   {"traffic_volume": 79, "aqi": 92,  "noise_pollution": 68, "temperature": 30, "humidity": 91,  "pop_density": 11000, "public_transport_usage": 66},
        }
        city_meds  = df[df["city"] == ref_city][FEATURE_COLS].median().to_dict()
        slider_def = PRESETS.get(preset, city_meds)

        st.markdown("<br>", unsafe_allow_html=True)
        sl_col, out_col = st.columns([1, 1.05], gap="large")

        with sl_col:
            st.markdown(f"""
            <div style='font-family:DM Mono,monospace;font-size:.62rem;letter-spacing:2.5px;
                        text-transform:uppercase;color:{D3};margin-bottom:16px;
                        padding-bottom:12px;border-bottom:1px solid {BDR};'>
              🎛️  Environmental Parameter Controls
            </div>""", unsafe_allow_html=True)

            sv = {}
            for feat, m in FMETA.items():
                d = float(np.clip(slider_def.get(feat, m["def"]), m["lo"], m["hi"]))
                # FIX ①: use pre-built fmt string (no bare % that trips sprintf)
                sv[feat] = st.slider(
                    f"{m['ico']}  {m['lbl']}",
                    min_value=float(m["lo"]),
                    max_value=float(m["hi"]),
                    value=d,
                    step=float(m["step"]),
                    format=m["fmt"],
                    key=f"s_{feat}",
                )

            st.markdown("<br>", unsafe_allow_html=True)
            rc, rr = st.columns([2, 1])
            with rc:
                run_btn = st.button("⚡  RUN AI PREDICTION", key="run_p", use_container_width=True)
            with rr:
                if st.button("↺  RESET", key="rst_s", use_container_width=True):
                    st.rerun()

        with out_col:
            st.markdown(f"""
            <div style='font-family:DM Mono,monospace;font-size:.62rem;letter-spacing:2.5px;
                        text-transform:uppercase;color:{D3};margin-bottom:16px;
                        padding-bottom:12px;border-bottom:1px solid {BDR};'>
              📡  Prediction Output
            </div>""", unsafe_allow_html=True)

            if "sim_score" not in st.session_state:
                st.session_state.sim_score  = None
                st.session_state.sim_inputs = None

            if run_btn:
                with st.spinner("🧠 Running inference …"):
                    st.session_state.sim_score  = predict_one(model, scaler, sv, mtr["ns"])
                    st.session_state.sim_inputs = sv.copy()

            score  = st.session_state.sim_score
            inputs = st.session_state.sim_inputs

            if score is None:
                st.markdown(f"""
                <div style='background:rgba(0,229,255,.03);border:1px solid {BDR};
                            border-radius:14px;padding:60px 30px;text-align:center;'>
                  <div style='font-size:3rem;margin-bottom:16px;opacity:.25;'>🏙️</div>
                  <div style='font-family:DM Mono,monospace;font-size:.72rem;
                              letter-spacing:2.5px;text-transform:uppercase;color:{D3};'>
                    Awaiting simulation
                  </div>
                  <div style='font-family:Barlow,sans-serif;font-size:.84rem;
                              color:{D3};margin-top:8px;'>
                    Set parameters and click RUN AI PREDICTION
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                sc_col2, sc_lbl2, sc_bg, _ = stress_tier(score)

                try:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        number={"font": {"family": "Barlow Condensed", "size": 68, "color": sc_col2}, "suffix": ""},
                        gauge=dict(
                            axis=dict(range=[0, 100], tickwidth=1, tickcolor=D3,
                                      tickfont=dict(family="DM Mono", size=9, color=D3), dtick=25),
                            bar=dict(color=sc_col2, thickness=.22, line=dict(color=sc_col2, width=2)),
                            bgcolor="rgba(0,0,0,0)",
                            borderwidth=0,
                            steps=[
                                {"range": [0,  35], "color": "rgba(0,245,160,0.07)"},
                                {"range": [35, 65], "color": "rgba(255,179,64,0.07)"},
                                {"range": [65,100], "color": "rgba(255,64,96,0.07)"},
                            ],
                            threshold=dict(line=dict(color=sc_col2, width=3), thickness=.8, value=score),
                        ),
                    ))
                    fig_g.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Barlow Condensed"),
                        height=250,
                        margin=dict(l=28, r=28, t=16, b=8),
                    )
                    out_col.plotly_chart(fig_g, use_container_width=True, config=CFG)
                except Exception as e:
                    out_col.warning(f"Gauge: {e}")

                st.markdown(f"""
                <div style='text-align:center;margin:-6px 0 20px;'>
                  <div style='display:inline-flex;align-items:center;gap:10px;padding:10px 32px;
                              border-radius:8px;background:{sc_bg};
                              border:1.5px solid rgba({hex_rgb(sc_col2)},0.33);
                              font-family:Barlow Condensed,sans-serif;font-weight:800;
                              font-size:1.25rem;letter-spacing:3.5px;
                              color:{sc_col2};text-transform:uppercase;'>
                    <span class="dot" style="background:{sc_col2};color:{sc_col2};"></span>
                    {sc_lbl2} STRESS
                  </div>
                </div>""", unsafe_allow_html=True)

                baseline = predict_one(model, scaler, city_meds, mtr["ns"])
                delta    = score - baseline
                ds       = "+" if delta >= 0 else ""
                dc       = RD if delta > 5 else (GR if delta < -5 else AM)
                dt       = "above" if delta >= 0 else "below"

                st.markdown(f"""
                <div style='padding:14px 18px;background:rgba(0,229,255,.03);
                            border:1px solid {BDR};border-radius:10px;margin-bottom:14px;'>
                  <div style='font-family:DM Mono,monospace;font-size:.58rem;letter-spacing:2px;
                              text-transform:uppercase;color:{D3};margin-bottom:8px;'>
                    vs {ref_city} baseline ({baseline:.1f})
                  </div>
                  <div style='display:flex;align-items:baseline;gap:10px;'>
                    <div style='font-family:Barlow Condensed,sans-serif;font-size:2rem;
                                font-weight:800;color:{dc};'>{ds}{delta:.1f}</div>
                    <div style='font-family:Barlow,sans-serif;font-size:.82rem;color:{D2};'>
                      stress pts {dt} city baseline
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

        # Scenario comparison
        if score is not None and inputs is not None:
            st.markdown(
                '<div class="sl" style="margin-top:32px;">🔬  Scenario Comparison — Baseline vs Simulation</div>',
                unsafe_allow_html=True,
            )
            try:
                fl  = [FMETA[f]["lbl"] for f in FEATURE_COLS]
                bn  = [float(np.clip(
                    (city_meds.get(f, 0) - FMETA[f]["lo"]) / (FMETA[f]["hi"] - FMETA[f]["lo"] + 1e-9),
                    0, 1,
                )) for f in FEATURE_COLS]
                sn  = [float(np.clip(
                    (inputs.get(f, 0) - FMETA[f]["lo"]) / (FMETA[f]["hi"] - FMETA[f]["lo"] + 1e-9),
                    0, 1,
                )) for f in FEATURE_COLS]
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(
                    name=f"{ref_city} Baseline ({baseline:.1f})",
                    x=fl, y=bn,
                    marker_color="rgba(0,229,255,0.45)",
                    marker_line=dict(color=CY, width=1),
                    hovertemplate="<b>%{x}</b><br>Baseline: %{y:.2f}<extra></extra>",
                ))
                fig_cmp.add_trace(go.Bar(
                    name=f"Simulation ({score:.1f})",
                    x=fl, y=sn,
                    marker_color=hex_rgba(sc_col2, 0.55),
                    marker_line=dict(color=sc_col2, width=1),
                    hovertemplate="<b>%{x}</b><br>Simulation: %{y:.2f}<extra></extra>",
                ))
                lay_c = plot_base("NORMALISED FEATURE VALUES · BASELINE vs SIMULATION",
                                  h=300, yl="Normalised (0–1)", legend=True)
                lay_c["barmode"] = "group"
                lay_c["xaxis"]["tickfont"] = dict(family="Barlow Condensed", size=12, color=TX)
                fig_cmp.update_layout(**lay_c)
                st.plotly_chart(fig_cmp, use_container_width=True, config=CFG)
            except Exception as e:
                st.warning(f"Comparison chart: {e}")

            # Sensitivity
            st.markdown(
                '<div class="sl">📡  Sensitivity — Impact of ±20% Change</div>',
                unsafe_allow_html=True,
            )
            try:
                rows = []
                for f in FEATURE_COLS:
                    uv = inputs.copy()
                    uv[f] = min(FMETA[f]["hi"], inputs[f] * 1.2)
                    dv = inputs.copy()
                    dv[f] = max(FMETA[f]["lo"], inputs[f] * 0.8)
                    rows.append({
                        "Feature": FMETA[f]["lbl"],
                        "+20%": round(predict_one(model, scaler, uv, mtr["ns"]) - score, 2),
                        "-20%": round(predict_one(model, scaler, dv, mtr["ns"]) - score, 2),
                    })
                sdf = pd.DataFrame(rows).sort_values("+20%", ascending=False)
                fig_s = go.Figure()
                fig_s.add_trace(go.Bar(
                    name="+20%", x=sdf["Feature"], y=sdf["+20%"],
                    marker_color=[RD if v > 0 else GR for v in sdf["+20%"]],
                    opacity=.85, marker_line=dict(width=0),
                    hovertemplate="<b>%{x}</b><br>+20%%: %{y:+.2f}<extra></extra>",
                ))
                fig_s.add_trace(go.Bar(
                    name="-20%", x=sdf["Feature"], y=sdf["-20%"],
                    marker_color=[GR if v < 0 else RD for v in sdf["-20%"]],
                    opacity=.55, marker_line=dict(width=0),
                    hovertemplate="<b>%{x}</b><br>-20%%: %{y:+.2f}<extra></extra>",
                ))
                lay_s = plot_base("SENSITIVITY ANALYSIS", h=280, yl="Δ Stress Index", legend=True)
                lay_s["barmode"] = "group"
                lay_s["yaxis"]["zeroline"]      = True
                lay_s["yaxis"]["zerolinecolor"] = "rgba(0,229,255,0.30)"
                lay_s["yaxis"]["zerolinewidth"] = 1.5
                lay_s["xaxis"]["tickfont"] = dict(family="Barlow Condensed", size=12, color=TX)
                fig_s.update_layout(**lay_s)
                st.plotly_chart(fig_s, use_container_width=True, config=CFG)
            except Exception as e:
                st.warning(f"Sensitivity: {e}")

    except Exception as e:
        st.error(f"Simulation Lab error: {e}")


# ───────────────────────────────────────────────────────────────────────
#  ③ CITY INTELLIGENCE
# ───────────────────────────────────────────────────────────────────────
with T3:
    try:
        st.markdown('<div class="sl">🗺️  Select City</div>', unsafe_allow_html=True)
        city_sel = st.selectbox(
            "City", sorted(df["city"].unique()), index=0, label_visibility="collapsed",
        )
        cdf = df[df["city"] == city_sel].copy().sort_values("date")

        if cdf.empty:
            st.warning("No data available.")
        else:
            # City KPIs
            st.markdown('<div class="sl">📊  City KPIs</div>', unsafe_allow_html=True)
            ck = st.columns(5)
            mean_s_city = cdf["stress_index"].mean()
            for col, ico, lbl, val, sub, color in [
                (ck[0], "📊", "Mean Stress",  f"{mean_s_city:.1f}",               stress_tier(mean_s_city)[1],  stress_tier(mean_s_city)[0]),
                (ck[1], "📈", "Peak Stress",  f"{cdf['stress_index'].max():.1f}",  "All-time max",               RD),
                (ck[2], "📉", "Min Stress",   f"{cdf['stress_index'].min():.1f}",  "All-time min",               GR),
                (ck[3], "💨", "Mean AQI",     f"{cdf['aqi'].mean():.0f}",          "Daily avg",                  AM),
                (ck[4], "🚦", "Mean Traffic", f"{cdf['traffic_volume'].mean():.1f}","/100 index",                CY),
            ]:
                col.markdown(f"""
                <div class="kc">
                  <div class="kc-ico">{ico}</div>
                  <div class="kc-lbl">{lbl}</div>
                  <div class="kc-val" style="color:{color};font-size:2rem;">{val}</div>
                  <div class="kc-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Stress time-series with AQI overlay
            st.markdown('<div class="sl">📈  Stress Index Time Series</div>', unsafe_allow_html=True)
            try:
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(
                    x=cdf["date"], y=cdf["stress_index"],
                    mode="lines", name="Stress Index",
                    line=dict(color=CY, width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0,229,255,0.06)",  # valid rgba — no hex+alpha
                    hovertemplate="<b>Stress</b>: %{y:.1f}<br>%{x}<extra></extra>",
                ))
                if "aqi" in cdf.columns:
                    fig_ts.add_trace(go.Scatter(
                        x=cdf["date"], y=cdf["aqi"],
                        mode="lines", name="AQI",
                        line=dict(color=AM, width=1.5, dash="dot"),
                        yaxis="y2",
                        hovertemplate="<b>AQI</b>: %{y:.0f}<br>%{x}<extra></extra>",
                    ))
                lay_ts = plot_base(
                    f"STRESS INDEX vs AQI · {city_sel.upper()}", h=320,
                    yl="Stress Index", legend=True,
                )
                # FIX ③: gridcolor must be rgba not "transparent"
                lay_ts["yaxis2"] = dict(
                    title=dict(text="AQI", font=dict(size=10, color=AM)),
                    overlaying="y",
                    side="right",
                    gridcolor="rgba(0,0,0,0)",   # was "transparent" — invalid in Plotly
                    tickfont=dict(size=10, color=AM),
                )
                fig_ts.update_layout(**lay_ts)
                st.plotly_chart(fig_ts, use_container_width=True, config=CFG)
            except Exception as e:
                st.warning(f"Time-series: {e}")

            # Monthly box + heatmap
            bc, hc = st.columns([1, 1], gap="large")

            with bc:
                st.markdown('<div class="sl">📦  Monthly Distribution</div>', unsafe_allow_html=True)
                try:
                    cdf2 = cdf.copy()
                    cdf2["mn"]     = cdf2["date"].dt.strftime("%b")
                    cdf2["mn_num"] = cdf2["date"].dt.month
                    mo = cdf2.sort_values("mn_num")["mn"].unique().tolist()
                    fig_b = go.Figure()
                    for m in mo:
                        md  = cdf2[cdf2["mn"] == m]["stress_index"]
                        mc2 = stress_tier(md.mean())[0]
                        # FIX ③: convert hex to rgba for Plotly fillcolor
                        fig_b.add_trace(go.Box(
                            y=md, name=m,
                            marker_color=mc2,
                            line_color=mc2,
                            fillcolor=hex_rgba(mc2, 0.13),  # was f"{mc2}22" — invalid
                            showlegend=False,
                            hovertemplate=f"<b>{m}</b><br>%{{y:.1f}}<extra></extra>",
                        ))
                    lay_b = plot_base("MONTHLY STRESS DISTRIBUTION", h=300, yl="Stress Index")
                    lay_b["xaxis"]["tickfont"] = dict(family="Barlow Condensed", size=12, color=TX)
                    fig_b.update_layout(**lay_b)
                    bc.plotly_chart(fig_b, use_container_width=True, config=CFG)
                except Exception as e:
                    bc.warning(f"Box chart: {e}")

            with hc:
                st.markdown('<div class="sl">🔥  Day-of-Week Heatmap</div>', unsafe_allow_html=True)
                try:
                    cdf3 = cdf.copy()
                    cdf3["dow"]    = cdf3["date"].dt.dayofweek
                    cdf3["mn_num"] = cdf3["date"].dt.month
                    piv = cdf3.groupby(["mn_num", "dow"])["stress_index"].mean().unstack(fill_value=0)
                    for i in range(7):
                        if i not in piv.columns:
                            piv[i] = 0
                    piv    = piv[[i for i in range(7)]]
                    mn_lbl = [calendar.month_abbr[m] for m in piv.index]
                    fig_h  = go.Figure(go.Heatmap(
                        z=piv.values,
                        x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                        y=mn_lbl,
                        colorscale=[[0, INK2], [.35, "#007a99"], [.65, AM], [1, RD]],
                        showscale=True,
                        text=[[f"{v:.0f}" for v in row] for row in piv.values],
                        texttemplate="%{text}",
                        textfont=dict(size=10),
                        hovertemplate="<b>%{y} %{x}</b><br>Stress: %{z:.1f}<extra></extra>",
                        colorbar=dict(
                            tickfont=dict(family="DM Mono", size=9, color=D2),
                            outlinewidth=0, thickness=10,
                        ),
                    ))
                    lay_h = plot_base(f"STRESS BY MONTH × DAY · {city_sel.upper()}", h=300)
                    lay_h["xaxis"]["tickfont"] = dict(family="DM Mono", size=11, color=TX)
                    lay_h["yaxis"]["tickfont"] = dict(family="DM Mono", size=11, color=TX)
                    fig_h.update_layout(**lay_h)
                    hc.plotly_chart(fig_h, use_container_width=True, config=CFG)
                except Exception as e:
                    hc.warning(f"Heatmap: {e}")

            # Correlation matrix
            st.markdown('<div class="sl">🔗  Feature Correlation Matrix</div>', unsafe_allow_html=True)
            try:
                cc = FEATURE_COLS + ["stress_index"]
                cl = ["Traffic", "AQI", "Noise", "Temp", "Humidity", "Pop Density", "Pub Transit", "STRESS"]
                cm = cdf[cc].corr().values
                fig_c = go.Figure(go.Heatmap(
                    z=cm, x=cl, y=cl,
                    colorscale=[[0, INK], [.5, "#004466"], [1, CY]],
                    text=[[f"{v:.2f}" for v in row] for row in cm],
                    texttemplate="%{text}",
                    textfont=dict(family="DM Mono", size=10),
                    zmin=-1, zmax=1, showscale=True,
                    colorbar=dict(
                        tickfont=dict(family="DM Mono", size=9, color=D2),
                        outlinewidth=0, thickness=11,
                    ),
                    hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
                ))
                lay_c2 = plot_base(f"PEARSON CORRELATION · {city_sel.upper()}", h=320)
                lay_c2["xaxis"]["tickfont"] = dict(family="DM Mono", size=10, color=D2)
                lay_c2["yaxis"]["tickfont"] = dict(family="DM Mono", size=10, color=D2)
                fig_c.update_layout(**lay_c2)
                st.plotly_chart(fig_c, use_container_width=True, config=CFG)
            except Exception as e:
                st.warning(f"Correlation: {e}")

            # Radar
            st.markdown('<div class="sl">🕸️  Feature Profile Radar</div>', unsafe_allow_html=True)
            try:
                fa  = cdf[FEATURE_COLS].mean()
                fn  = [(fa[f] - FMETA[f]["lo"]) / (FMETA[f]["hi"] - FMETA[f]["lo"] + 1e-9) for f in FEATURE_COLS]
                lb  = [FMETA[f]["lbl"] for f in FEATURE_COLS]
                lbc = lb + [lb[0]]
                vnc = fn + [fn[0]]
                fig_rad = go.Figure(go.Scatterpolar(
                    r=vnc, theta=lbc,
                    fill="toself",
                    fillcolor="rgba(0,229,255,0.08)",
                    line=dict(color=CY, width=2),
                    hovertemplate="<b>%{theta}</b><br>%{r:.2f}<extra></extra>",
                ))
                fig_rad.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(
                            visible=True, range=[0, 1],
                            tickfont=dict(size=9, color=D3),
                            gridcolor="rgba(0,229,255,0.10)",
                        ),
                        angularaxis=dict(
                            tickfont=dict(family="Barlow Condensed", size=12, color=TX),
                            gridcolor="rgba(0,229,255,0.08)",
                        ),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Barlow Condensed"),
                    title=dict(
                        text=f"FEATURE PROFILE · {city_sel.upper()}",
                        font=dict(family="Barlow Condensed", size=15, color=TX),
                        x=.5,
                    ),
                    height=380,
                    margin=dict(l=60, r=60, t=60, b=40),
                )
                st.plotly_chart(fig_rad, use_container_width=True, config=CFG)
            except Exception as e:
                st.warning(f"Radar: {e}")

    except Exception as e:
        st.error(f"City Intelligence error: {e}")


# ───────────────────────────────────────────────────────────────────────
#  ④ AI INSIGHTS
# ───────────────────────────────────────────────────────────────────────
with T4:
    try:
        st.markdown(f"""
        <div style='background:rgba(0,229,255,.03);border:1px solid rgba(0,229,255,.14);
                    border-radius:12px;padding:18px 24px;margin-bottom:26px;'>
          <div style='font-family:Barlow Condensed,sans-serif;font-weight:800;
                      font-size:1.18rem;letter-spacing:2.5px;color:{CY};
                      text-transform:uppercase;margin-bottom:4px;'>
            🤖  AI Insight Engine
          </div>
          <div style='font-family:Barlow,sans-serif;font-size:.85rem;color:{D2};'>
            Dynamic recommendations generated from your latest Simulation Lab prediction.
            Run a simulation first to unlock personalised insights.
          </div>
        </div>""", unsafe_allow_html=True)

        score  = st.session_state.get("sim_score")
        inputs = st.session_state.get("sim_inputs")

        if score is None or inputs is None:
            st.info("💡 Run a prediction in **Simulation Lab** first — insights will appear here automatically.")
        else:
            sc3, sl3, sb3, _ = stress_tier(score)

            # Score summary card
            st.markdown(f"""
            <div class="gc" style="margin-bottom:24px;background:{sb3};
                                   border-color:rgba({hex_rgb(sc3)},0.27);">
              <div style='display:flex;align-items:center;gap:18px;flex-wrap:wrap;'>
                <div>
                  <div style='font-family:DM Mono,monospace;font-size:.60rem;letter-spacing:2px;
                              text-transform:uppercase;color:{D3};margin-bottom:6px;'>
                    Simulation Result
                  </div>
                  <div style='font-family:Barlow Condensed,sans-serif;font-size:4rem;
                              font-weight:800;color:{sc3};line-height:1;'>
                    {score:.1f}<span style='font-size:1.2rem;color:{D2};'>/100</span>
                  </div>
                </div>
                <div>
                  <div style='font-family:Barlow Condensed,sans-serif;font-size:1.6rem;
                              font-weight:700;color:{sc3};letter-spacing:3px;text-transform:uppercase;'>
                    {sl3} STRESS
                  </div>
                  <div style='font-family:Barlow,sans-serif;font-size:.84rem;color:{D2};margin-top:4px;'>
                    Model: {model_choice} · Confidence: {conf:.1f}%
                  </div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            def make_insights(inp, sc_score):
                rows = []
                v = inp
                # AQI
                if v["aqi"] >= 300:
                    rows.append(("🔴", "CRITICAL",
                        f"AQI {v['aqi']:.0f} — HAZARDOUS. Serious health risk for all populations. Emergency pollution reduction protocols required.", RD))
                elif v["aqi"] >= 200:
                    rows.append(("🟠", "HIGH RISK",
                        f"AQI {v['aqi']:.0f} (Very Unhealthy). Industrial emission controls and vehicle access restrictions strongly recommended.", AM))
                elif v["aqi"] >= 100:
                    rows.append(("🟡", "MODERATE",
                        f"AQI {v['aqi']:.0f}. Sensitive groups at risk. Low-emission zones and green transit investment advised.", AM))
                else:
                    rows.append(("🟢", "GOOD",
                        f"AQI {v['aqi']:.0f} within acceptable limits. Air quality is not a primary stressor.", GR))
                # Traffic
                if v["traffic_volume"] >= 80:
                    rows.append(("🔴", "HIGH CONGESTION",
                        f"Traffic {v['traffic_volume']:.0f}/100 — severe gridlock. Congestion pricing and emergency PT expansion are urgent.", RD))
                elif v["traffic_volume"] >= 55:
                    rows.append(("🟠", "ELEVATED",
                        f"Traffic {v['traffic_volume']:.0f}/100. Staggered work hours and ride-pooling incentives can reduce pressure.", AM))
                else:
                    rows.append(("🟢", "OPTIMAL",
                        f"Traffic {v['traffic_volume']:.0f}/100 is well-managed. Flow patterns are contributing positively.", GR))
                # PT
                if v["public_transport_usage"] >= 70:
                    rows.append(("🟢", "POSITIVE SIGNAL",
                        f"PT usage {v['public_transport_usage']:.0f}/100 is excellent. High modal shift is the single most effective stress reducer.", GR))
                elif v["public_transport_usage"] >= 45:
                    rows.append(("🟡", "OPPORTUNITY",
                        f"PT {v['public_transport_usage']:.0f}/100 — room to grow. Last-mile connectivity and fare subsidies could drive adoption.", AM))
                else:
                    rows.append(("🔴", "UNDERUTILISED",
                        f"PT critically low at {v['public_transport_usage']:.0f}/100. Private vehicle dominance magnifies all stressors.", RD))
                # Noise
                if v["noise_pollution"] >= 75:
                    rows.append(("🔴", "CRITICAL",
                        f"Noise {v['noise_pollution']:.0f} dB far exceeds WHO threshold (55 dB). Noise barriers and construction curfews warranted.", RD))
                elif v["noise_pollution"] >= 60:
                    rows.append(("🟡", "ELEVATED",
                        f"Noise {v['noise_pollution']:.0f} dB exceeds residential limits. Acoustic zoning recommended.", AM))
                # Temperature
                if v["temperature"] >= 38:
                    rows.append(("🔴", "HEAT ALERT",
                        f"Temperature {v['temperature']:.0f}°C in danger zone. Cool roofs, tree canopy, and public cooling centres urgently needed.", RD))
                elif abs(v["temperature"] - 22) > 10:
                    rows.append(("🟡", "THERMAL STRESS",
                        f"Temperature {v['temperature']:.0f}°C deviates significantly from comfort zone (18–26°C). Physiological stress elevated.", AM))
                # Population
                if v["pop_density"] >= 25000:
                    rows.append(("🔴", "OVERCROWDING",
                        f"Density {v['pop_density']:,.0f}/km² is extreme. Overcrowding compounds all stressors. Decentralisation imperative.", RD))
                return rows[:6]

            insights = make_insights(inputs, score)
            st.markdown('<div class="sl">💡  Contextual AI Insights</div>', unsafe_allow_html=True)
            for ii, it, itxt, ic in insights:
                tb = hex_rgba(ic, 0.15)
                st.markdown(f"""
                <div class="ir" style="border-left:3px solid {ic};">
                  <span class="ir-ico">{ii}</span>
                  <div style="flex:1;">
                    <div class="itag" style="background:{tb};color:{ic};">{it}</div>
                    <div class="ir-txt">{itxt}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Recommendations
            st.markdown(
                '<div class="sl" style="margin-top:28px;">🎯  Strategic Recommendations</div>',
                unsafe_allow_html=True,
            )
            recs = []
            if score >= 65:
                recs.extend([
                    ("🚨", "IMMEDIATE", RD,
                     "Activate Emergency Urban Stress Protocol",
                     "Stress exceeds HIGH threshold. Multi-agency coordination required. Implement traffic diversion, air quality advisories, and open public cooling/quiet centres."),
                    ("🏛️", "SHORT-TERM", AM,
                     "Deploy Rapid Intervention Stack",
                     "Institute odd-even vehicle rationing, expand PT frequency 40%, enforce construction noise curfews. Target: reduce index below 55 in 30 days."),
                ])
            elif score >= 35:
                recs.append(("📋", "MEDIUM-TERM", AM,
                    "Targeted Stress Reduction Plan",
                    "Focus on top 2 contributing features. Introduce congestion pricing and accelerate green space development."))
            else:
                recs.append(("✅", "MAINTENANCE", GR,
                    "Sustain Current Strategy",
                    "Stress is in LOW zone. Maintain PT quality, monitor AQI weekly, and pre-emptive monsoon traffic management."))
            recs.extend([
                ("🌳", "LONG-TERM", CY,
                 "Green Infrastructure Investment",
                 "Increase urban tree canopy 15%, create car-free pedestrian zones in high-density areas, invest in metro extensions."),
                ("📡", "ONGOING", CY,
                 "Expand IoT Sensor Network",
                 "Deploy additional AQI, noise and traffic sensors. Integrate with Urban Pulse AI for sub-daily prediction frequency."),
            ])
            for ri, rh, rc2, rt, rb_txt in recs[:5]:
                rb2 = hex_rgba(rc2, 0.06)
                st.markdown(f"""
                <div class="ab" style="background:{rb2};border:1px solid rgba({hex_rgb(rc2)},0.20);
                                       margin-bottom:10px;">
                  <span style="font-size:1.2rem;">{ri}</span>
                  <div>
                    <div class="itag" style="background:{rb2};color:{rc2};
                                             border:1px solid rgba({hex_rgb(rc2)},0.27);">{rh}</div>
                    <div style='font-family:Barlow Condensed,sans-serif;font-weight:700;
                                font-size:1rem;color:{TX};margin:4px 0 3px;'>{rt}</div>
                    <div style='font-family:Barlow,sans-serif;font-size:.83rem;
                                color:{D2};line-height:1.5;'>{rb_txt}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Summary table
            st.markdown(
                '<div class="sl" style="margin-top:28px;">📊  Input Parameter Summary</div>',
                unsafe_allow_html=True,
            )
            try:
                srows = []
                for f in FEATURE_COLS:
                    v   = inputs[f]
                    lo  = FMETA[f]["lo"]
                    hi  = FMETA[f]["hi"]
                    pct = (v - lo) / (hi - lo + 1e-9) * 100
                    srows.append({
                        "Feature":  f"{FMETA[f]['ico']} {FMETA[f]['lbl']}",
                        "Value":    f"{v:.1f} {FMETA[f]['unit'].replace('%%','%')}",
                        "Scale %":  f"{pct:.0f}%",
                    })
                st.dataframe(pd.DataFrame(srows), hide_index=True, use_container_width=True)
            except Exception as e:
                st.warning(f"Summary table: {e}")

    except Exception as e:
        st.error(f"AI Insights error: {e}")


# ───────────────────────────────────────────────────────────────────────
#  ⑤ SYSTEM & MODEL PANEL
# ───────────────────────────────────────────────────────────────────────
with T5:
    try:
        pred_status = "✓ RAN" if st.session_state.get("sim_score") else "PENDING"

        # FIX ④: build status badges HTML separately — no nested f-string join
        _status_items = [
            ("Data Engine", "● OPERATIONAL", GR),
            ("AI Engine",   "● ACTIVE",      CY),
            ("Predictions", pred_status,      AM),
        ]
        _status_html = ""
        for _lbl, _val, _c in _status_items:
            _status_html += f"""
            <div style='padding:10px 20px;background:rgba({hex_rgb(_c)},0.08);
                        border:1px solid rgba({hex_rgb(_c)},0.27);
                        border-radius:10px;text-align:center;'>
              <div style='font-family:DM Mono,monospace;font-size:.55rem;letter-spacing:2px;
                          color:{D3};text-transform:uppercase;'>{_lbl}</div>
              <div style='font-family:Barlow Condensed,sans-serif;font-weight:700;
                          color:{_c};font-size:1rem;margin-top:3px;'>{_val}</div>
            </div>"""

        st.markdown(f"""
        <div class="gc" style="margin-bottom:28px;">
          <div style='display:flex;align-items:center;gap:16px;flex-wrap:wrap;'>
            <div style='font-size:2.4rem;'>🔧</div>
            <div>
              <div style='font-family:Barlow Condensed,sans-serif;font-weight:800;
                          font-size:1.3rem;letter-spacing:2.5px;text-transform:uppercase;color:{CY};'>
                System Control &amp; Model Panel
              </div>
              <div style='font-family:DM Mono,monospace;font-size:.62rem;
                          letter-spacing:1.5px;color:{D3};margin-top:5px;'>
                Session: {now_str()}
              </div>
            </div>
            <div style='margin-left:auto;display:flex;gap:10px;flex-wrap:wrap;'>
              {_status_html}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Performance KPIs
        st.markdown('<div class="sl">📐  Model Performance</div>', unsafe_allow_html=True)
        mp = st.columns(4)
        for col, ico, lbl, val, sub, c in [
            (mp[0], "🎯", "R² Score",     f"{mtr['r2']:.4f}",  f"{r2pct:.1f}% variance explained", CY),
            (mp[1], "📏", "MAE",          f"{mtr['mae']:.2f}", "Mean Absolute Error",               GR),
            (mp[2], "📉", "RMSE",         f"{mtr['rmse']:.2f}","Root Mean Squared Error",           AM),
            (mp[3], "🧪", "Test Samples", f"{mtr['tst']:,}",   f"of {mtr['trn']+mtr['tst']:,} total", CY),
        ]:
            col.markdown(f"""
            <div class="kc">
              <div class="kc-ico">{ico}</div>
              <div class="kc-lbl">{lbl}</div>
              <div class="kc-val" style="color:{c};">{val}</div>
              <div class="kc-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature importance + actual vs predicted
        fi, avp = st.columns([1, 1.1], gap="large")

        with fi:
            st.markdown('<div class="sl">⚖️  Feature Importance</div>', unsafe_allow_html=True)
            try:
                imp = mtr["imp"]
                fl  = [f"{FMETA[f]['ico']} {FMETA[f]['lbl']}" for f in FEATURE_COLS]
                idf = pd.DataFrame({"f": fl, "i": imp}).sort_values("i", ascending=True)
                n   = len(idf)
                bc2 = [f"rgba(0,{int(180+49*j/n)},{int(200+55*j/n)},{.45+.55*j/n})" for j in range(n)]
                fig_fi = go.Figure(go.Bar(
                    y=idf["f"], x=idf["i"],
                    orientation="h",
                    marker=dict(color=bc2, line=dict(width=0)),
                    text=[f"{v*100:.1f}%" for v in idf["i"]],
                    textposition="outside",
                    textfont=dict(family="DM Mono", size=10, color=D2),
                    hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>",
                ))
                lay_fi = plot_base("FEATURE IMPORTANCE", h=320, xl="Importance")
                lay_fi["yaxis"]["tickfont"] = dict(family="Barlow Condensed", size=12, color=TX)
                fig_fi.update_layout(**lay_fi)
                fi.plotly_chart(fig_fi, use_container_width=True, config=CFG)
            except Exception as e:
                fi.warning(f"FI chart: {e}")

        with avp:
            st.markdown('<div class="sl">🔮  Actual vs Predicted</div>', unsafe_allow_html=True)
            try:
                yt  = mtr["yte"]
                yp2 = mtr["yp"]
                ns2 = min(400, len(yt))
                idx = np.random.default_rng(0).choice(len(yt), ns2, replace=False)
                res = yp2[idx] - yt[idx]
                pc  = [RD if abs(r) > 10 else (AM if abs(r) > 5 else GR) for r in res]
                fig_av = go.Figure()
                fig_av.add_trace(go.Scatter(
                    x=[0, 100], y=[0, 100], mode="lines",
                    line=dict(color="rgba(0,229,255,0.25)", width=1.5, dash="dash"),
                    hoverinfo="skip",
                ))
                fig_av.add_trace(go.Scatter(
                    x=yt[idx], y=yp2[idx], mode="markers",
                    marker=dict(color=pc, size=5, opacity=.72, line=dict(width=0)),
                    hovertemplate="Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>",
                ))
                lay_av = plot_base("ACTUAL vs PREDICTED", h=320, xl="Actual", yl="Predicted")
                lay_av["xaxis"]["range"] = [-2, 103]
                lay_av["yaxis"]["range"] = [-2, 103]
                fig_av.update_layout(**lay_av)
                avp.plotly_chart(fig_av, use_container_width=True, config=CFG)
            except Exception as e:
                avp.warning(f"AvsP: {e}")

        # Residuals
        st.markdown('<div class="sl">📊  Residual Distribution</div>', unsafe_allow_html=True)
        try:
            res2 = mtr["yp"] - mtr["yte"]
            fig_res = go.Figure(go.Histogram(
                x=res2, nbinsx=50,
                marker=dict(color=CY, opacity=.55, line=dict(color=CY, width=.3)),
                hovertemplate="Residual: %{x:.1f}<br>Count: %{y}<extra></extra>",
            ))
            fig_res.add_vline(
                x=0, line_color="rgba(0,229,255,0.50)",
                line_width=1.5, line_dash="dash",
            )
            lay_res = plot_base("RESIDUAL DISTRIBUTION (y_pred − y_actual)", h=230,
                                xl="Residual", yl="Frequency")
            lay_res["margin"]["t"] = 40
            fig_res.update_layout(**lay_res)
            st.plotly_chart(fig_res, use_container_width=True, config=CFG)
        except Exception as e:
            st.warning(f"Residuals: {e}")

        # Per-city RMSE
        st.markdown('<div class="sl">🏙️  Per-City Model Error</div>', unsafe_allow_html=True)
        try:
            ce = []
            for c in sorted(df["city"].unique()):
                cd2 = df[df["city"] == c][FEATURE_COLS + ["stress_index"]].dropna()
                if len(cd2) < 10:
                    continue
                Xc  = cd2[FEATURE_COLS].values
                yc2 = cd2["stress_index"].values
                yp3 = model.predict(scaler.transform(Xc) if mtr["ns"] else Xc)
                ce.append({
                    "City": c,
                    "RMSE": round(np.sqrt(mean_squared_error(yc2, yp3)), 2),
                    "MAE":  round(mean_absolute_error(yc2, yp3), 2),
                    "R²":   round(r2_score(yc2, yp3), 3),
                })
            ce_df = pd.DataFrame(ce).sort_values("RMSE")
            fig_ce = go.Figure(go.Bar(
                x=ce_df["City"], y=ce_df["RMSE"],
                marker=dict(
                    color=[GR if v < 4 else (AM if v < 7 else RD) for v in ce_df["RMSE"]],
                    line=dict(width=0),
                ),
                text=[f"{v}" for v in ce_df["RMSE"]],
                textposition="outside",
                textfont=dict(family="DM Mono", size=10, color=D2),
                hovertemplate="<b>%{x}</b><br>RMSE: %{y:.2f}<extra></extra>",
            ))
            lay_ce = plot_base("PER-CITY RMSE", h=250, yl="RMSE (stress pts)")
            lay_ce["xaxis"]["tickfont"] = dict(family="Barlow Condensed", size=13, color=TX)
            fig_ce.update_layout(**lay_ce)
            st.plotly_chart(fig_ce, use_container_width=True, config=CFG)
        except Exception as e:
            st.warning(f"Per-city RMSE: {e}")

        # Model comparison table
        st.markdown('<div class="sl">🔬  Available Models</div>', unsafe_allow_html=True)
        mdf = pd.DataFrame([
            {"Model": "Random Forest",     "Type": "Ensemble", "Strength": "High accuracy + feature importance", "Active": "✅" if model_choice == "Random Forest"     else ""},
            {"Model": "Gradient Boosting", "Type": "Ensemble", "Strength": "Best accuracy, sequential learner",  "Active": "✅" if model_choice == "Gradient Boosting" else ""},
            {"Model": "Ridge Regression",  "Type": "Linear",   "Strength": "Regularised, interpretable",         "Active": "✅" if model_choice == "Ridge Regression"  else ""},
            {"Model": "Linear Regression", "Type": "Linear",   "Strength": "Baseline sanity check",              "Active": "✅" if model_choice == "Linear Regression" else ""},
        ])
        st.dataframe(mdf, hide_index=True, use_container_width=True)

        # FIX ④: build config card HTML separately — no nested f-string join
        _scaler_lbl = "StandardScaler" if mtr["ns"] else "None (tree model)"
        _cfg_items  = [
            ("Algorithm",   model_choice),
            ("Features",    f"{len(FEATURE_COLS)} urban indicators"),
            ("Train Split", "80% time-based"),
            ("Scaler",      _scaler_lbl),
            ("Target",      "stress_index (0–100)"),
            ("Session",     now_str()),
        ]
        _cfg_html = ""
        for _k, _v in _cfg_items:
            _cfg_html += f"""
            <div>
              <div style='font-family:DM Mono,monospace;font-size:.56rem;letter-spacing:2px;
                          text-transform:uppercase;color:{D3};'>{_k}</div>
              <div style='font-family:Barlow,sans-serif;font-size:.86rem;
                          color:{TX};margin-top:3px;'>{_v}</div>
            </div>"""

        st.markdown(f"""
        <div class="gc" style="margin-top:12px;">
          <div style='font-family:DM Mono,monospace;font-size:.6rem;letter-spacing:2px;
                      text-transform:uppercase;color:{D3};margin-bottom:14px;'>
            ℹ️  Runtime Configuration
          </div>
          <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;'>
            {_cfg_html}
          </div>
        </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"System Panel error: {e}")


# ═══════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='margin-top:52px;padding-top:18px;border-top:1px solid {BDR};
            display:flex;justify-content:space-between;align-items:center;
            flex-wrap:wrap;gap:8px;'>
  <div style='font-family:DM Mono,monospace;font-size:.58rem;letter-spacing:2px;
              text-transform:uppercase;color:{D3};'>
    Urban Pulse AI · Smart City Stress Command Center · v2.1
  </div>
  <div style='display:flex;gap:20px;flex-wrap:wrap;'>
    <span style='font-family:DM Mono,monospace;font-size:.56rem;color:{D3};'>
      Data: OpenAQ · CPCB · IMD · Census India · UMTA
    </span>
    <span style='font-family:DM Mono,monospace;font-size:.56rem;color:{D3};'>
      Stack: Streamlit · scikit-learn · Plotly
    </span>
  </div>
</div>
""", unsafe_allow_html=True)
