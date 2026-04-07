"""
╔══════════════════════════════════════════════════════════════════╗
║          URBAN PULSE AI — Global Smart City Intelligence         ║
║          Production-grade real-time urban monitoring system      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import time
import json
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium

# ── Local modules ─────────────────────────────────────────────────────────────
from config.cities import CITIES, UPI_COLORS, ALERT_THRESHOLD
from modules.data_engine import fetch_aqi, fetch_weather, fetch_historical_aqi, fetch_hourly_data
from modules.intelligence import compute_upi, compute_crowd_score, detect_peak_hours, compute_zone_stress
from modules.prediction import forecast_timeseries, classify_stress, forecast_alerts
from modules.visualization import (
    build_map, chart_hourly, chart_7day_trend,
    chart_forecast, chart_city_comparison, chart_upi_gauge,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Urban Pulse AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — Premium dark theme with glassmorphism
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Global reset ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    background-color: #040d1a !important;
    color: #e2e8f0 !important;
}

/* ── App bg ─── */
.stApp {
    background: radial-gradient(ellipse at 20% 20%, #0a1628 0%, #040d1a 60%) !important;
}
.main .block-container {
    padding: 1rem 1.5rem 2rem 1.5rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #040d1a 100%) !important;
    border-right: 1px solid rgba(99,132,255,0.15) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1rem !important; }

/* ── Metric cards ─── */
[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(99,132,255,0.15) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 22px !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* ── Selectbox / input ─── */
.stSelectbox select, .stSelectbox [data-baseweb="select"] {
    background: #0a1628 !important;
    border: 1px solid rgba(99,132,255,0.2) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
[data-baseweb="select"] > div { background: #0a1628 !important; border-color: rgba(99,132,255,0.25) !important; }
[data-baseweb="popover"] { background: #0f172a !important; }

/* ── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(10,22,40,0.8) !important;
    border-radius: 10px !important;
    gap: 4px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #64748b !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 6px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,132,255,0.15) !important;
    color: #818cf8 !important;
}

/* ── Expander ─── */
.streamlit-expanderHeader {
    background: rgba(15,23,42,0.6) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
}
.streamlit-expanderContent { background: rgba(15,23,42,0.4) !important; }

/* ── Scrollbar ─── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #040d1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

/* ── Button ─── */
.stButton > button {
    background: linear-gradient(135deg, #1e40af, #6366f1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

/* ── Text input ─── */
.stTextInput > div > div > input {
    background: #0a1628 !important;
    border: 1px solid rgba(99,132,255,0.2) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Divider ─── */
hr { border-color: rgba(99,132,255,0.1) !important; margin: 1rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HTML COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def glass_card(content: str, border_color: str = "rgba(99,132,255,0.2)", padding: str = "16px 20px") -> str:
    return f"""
    <div style="
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid {border_color};
        border-radius: 14px;
        padding: {padding};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 12px;
    ">{content}</div>"""


def render_header():
    st.markdown("""
    <div style="
        display:flex; align-items:center; justify-content:space-between;
        padding: 14px 0 8px 0;
        border-bottom: 1px solid rgba(99,132,255,0.12);
        margin-bottom: 18px;
    ">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="
                background: linear-gradient(135deg, #1e40af, #6366f1);
                border-radius: 10px;
                padding: 8px 12px;
                font-size: 20px;
            ">🌐</div>
            <div>
                <div style="font-size:20px; font-weight:800; color:#e2e8f0; letter-spacing:-0.5px;">
                    URBAN PULSE AI
                </div>
                <div style="font-size:11px; color:#475569; font-weight:500; letter-spacing:1px;">
                    GLOBAL SMART CITY INTELLIGENCE PLATFORM
                </div>
            </div>
        </div>
        <div style="display:flex; gap:20px; align-items:center;">
            <div style="text-align:right;">
                <div style="font-size:11px; color:#475569;">SYSTEM STATUS</div>
                <div style="font-size:12px; color:#22c55e; font-weight:600;">● OPERATIONAL</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:11px; color:#475569;">LAST SYNC</div>
                <div style="font-size:12px; color:#94a3b8; font-weight:600;" id="last-sync">
                    {datetime.now().strftime("%H:%M:%S")}
                </div>
            </div>
        </div>
    </div>
    """.format(datetime=datetime), unsafe_allow_html=True)


def render_upi_badge(score: float, category: str) -> str:
    color = UPI_COLORS.get(category, "#eab308")
    bg = {
        "Low": "rgba(34,197,94,0.12)",
        "Moderate": "rgba(234,179,8,0.12)",
        "High": "rgba(249,115,22,0.12)",
        "Extreme": "rgba(239,68,68,0.15)",
    }.get(category, "rgba(99,132,255,0.12)")
    pulse = "animation: pulse 1.5s infinite;" if category == "Extreme" else ""
    return f"""
    <div style="
        display:inline-flex; align-items:center; gap:8px;
        background:{bg}; border:1px solid {color};
        border-radius:8px; padding:6px 14px; {pulse}
    ">
        <div style="width:8px; height:8px; border-radius:50%; background:{color};"></div>
        <span style="font-weight:700; color:{color}; font-size:16px;">{score}</span>
        <span style="color:#94a3b8; font-size:13px;">/ 100 — {category}</span>
    </div>
    <style>
    @keyframes pulse {{
        0%,100% {{ box-shadow: 0 0 0 0 rgba(239,68,68,0.3); }}
        50% {{ box-shadow: 0 0 0 8px rgba(239,68,68,0); }}
    }}
    </style>
    """


def render_alert_banner(message: str, city: str, upi: float, color: str = "#ef4444"):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
        border: 1px solid rgba(239,68,68,0.5);
        border-left: 4px solid #ef4444;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        display:flex; align-items:center; gap:14px;
        animation: alertSlide 0.4s ease;
    ">
        <span style="font-size:24px;">⚠️</span>
        <div>
            <div style="color:#fca5a5; font-weight:700; font-size:14px;">
                URBAN STRESS ALERT — {city.upper()}
            </div>
            <div style="color:#94a3b8; font-size:12px; margin-top:3px;">
                {message} | UPI Score: <span style="color:#ef4444; font-weight:700;">{upi}</span>
            </div>
        </div>
        <div style="margin-left:auto; text-align:right;">
            <div style="color:#ef4444; font-size:11px; font-weight:600;">IMMEDIATE ACTION REQUIRED</div>
            <div style="color:#475569; font-size:10px;">{datetime.now().strftime('%H:%M:%S')}</div>
        </div>
    </div>
    <style>
    @keyframes alertSlide {{ from {{ opacity:0; transform:translateY(-10px); }} to {{ opacity:1; transform:translateY(0); }} }}
    </style>
    """, unsafe_allow_html=True)


def render_zone_card(zone: dict):
    color = UPI_COLORS.get(zone["category"], "#eab308")
    st.markdown(glass_card(f"""
    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <div style="
                    background: linear-gradient(135deg, {color}33, {color}11);
                    border: 1px solid {color}66;
                    border-radius: 6px;
                    padding: 2px 10px;
                    font-size:11px; font-weight:700; color:{color};
                ">Zone {zone['zone']}</div>
                <span style="color:#94a3b8; font-weight:600; font-size:13px;">{zone['name']}</span>
            </div>
            <div style="display:flex; gap:16px; font-size:12px; color:#64748b; margin-top:6px;">
                <span>💨 AQI: <b style="color:#94a3b8">{zone['aqi']:.0f}</b></span>
                <span>👥 Crowd: <b style="color:#94a3b8">{zone['crowd']:.0f}%</b></span>
                <span>⚙️ Infra: <b style="color:#94a3b8">{zone['infra_load']:.0f}%</b></span>
            </div>
            <div style="font-size:11px; color:#475569; margin-top:5px;">
                🎯 Driver: <span style="color:{color}">{zone['dominant_driver']}</span>
            </div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:22px; font-weight:800; color:{color};">{zone['upi']:.0f}</div>
            <div style="font-size:10px; color:#475569;">{zone['category'].upper()}</div>
        </div>
    </div>
    """, border_color=f"{color}33"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE — fetch + compute everything for selected city
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def compute_city_pulse(city_key: str, owm_key: str, _ts: int):
    """Full pipeline for a single city. Cache TTL = 5 min."""
    cfg = CITIES[city_key]
    now = datetime.now()
    hour = now.hour
    dow = now.weekday()

    # ── Data fetch ────
    aqi_data = fetch_aqi(cfg["openaq_city"])
    weather_data = fetch_weather(cfg["owm_city"], owm_key, cfg["lat"], cfg["lon"])

    aqi = aqi_data["aqi"]
    temp_c = weather_data["temp_c"]
    humidity = weather_data["humidity"]

    # ── Intelligence ──
    crowd_data = compute_crowd_score(cfg["area_type"], cfg["population_density"], hour, dow)
    upi_data = compute_upi(
        aqi, temp_c, humidity,
        crowd_data["score"],
        cfg["area_type"],
        cfg["population_density"],
        cfg["infrastructure_load_base"],
        hour,
    )

    # ── Historical + hourly ──
    hist = fetch_historical_aqi(cfg["openaq_city"])
    hourly = fetch_hourly_data(cfg["openaq_city"])
    hourly_crowd_vals = [h["crowd"] for h in hourly]
    peaks = detect_peak_hours(cfg["area_type"], hourly_crowd_vals)

    # ── Stress model ──
    stress = classify_stress(upi_data["score"], aqi, crowd_data["score"], upi_data["dominant_factor"])

    # ── Zones ──
    zones = compute_zone_stress(cfg, upi_data["score"], crowd_data, aqi)

    # ── Forecast ──
    forecasts_24 = forecast_timeseries(
        cfg["openaq_city"], cfg["area_type"],
        upi_data["score"], aqi, crowd_data["score"],
        horizon_hours=24,
    )
    forecasts_48 = forecast_timeseries(
        cfg["openaq_city"], cfg["area_type"],
        upi_data["score"], aqi, crowd_data["score"],
        horizon_hours=48,
    )
    future_alerts = forecast_alerts(forecasts_48, cfg["display"])

    return {
        "cfg": cfg,
        "aqi_data": aqi_data,
        "weather": weather_data,
        "crowd": crowd_data,
        "upi": upi_data,
        "hist": hist,
        "hourly": hourly,
        "peaks": peaks,
        "stress": stress,
        "zones": zones,
        "forecasts_24": forecasts_24,
        "forecasts_48": forecasts_48,
        "future_alerts": future_alerts,
        "computed_at": now.isoformat(),
    }


@st.cache_data(ttl=300, show_spinner=False)
def compute_all_cities(owm_key: str, _ts: int):
    """Compute UPI for ALL cities for map + comparison."""
    now = datetime.now()
    hour = now.hour
    dow = now.weekday()
    results = []
    for city_key, cfg in CITIES.items():
        try:
            aqi_data = fetch_aqi(cfg["openaq_city"])
            crowd_data = compute_crowd_score(cfg["area_type"], cfg["population_density"], hour, dow)
            upi_data = compute_upi(
                aqi_data["aqi"], 28, 65,  # Weather approximated for bulk load
                crowd_data["score"],
                cfg["area_type"],
                cfg["population_density"],
                cfg["infrastructure_load_base"],
                hour,
            )
            results.append({
                "city_key": city_key,
                "display": cfg["display"],
                "lat": cfg["lat"],
                "lon": cfg["lon"],
                "upi": upi_data["score"],
                "category": upi_data["category"],
                "aqi": aqi_data["aqi"],
                "crowd_score": crowd_data["score"],
                "is_peak": crowd_data["is_peak"],
                "dominant_factor": upi_data["dominant_factor"],
                "state": cfg.get("state", ""),
                "country": cfg.get("country", ""),
            })
        except Exception:
            continue
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "selected_city" not in st.session_state:
    st.session_state.selected_city = "Mumbai_Dadar"
if "owm_key" not in st.session_state:
    st.session_state.owm_key = ""
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False
if "comparison_cities" not in st.session_state:
    st.session_state.comparison_cities = ["Mumbai_Dadar", "Delhi_CP", "Bengaluru", "Kolkata", "Dhaka"]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 16px 0;">
        <div style="font-size:22px;">🌐</div>
        <div style="font-size:15px; font-weight:700; color:#818cf8;">URBAN PULSE AI</div>
        <div style="font-size:10px; color:#475569; letter-spacing:1px;">SMART CITY INTELLIGENCE</div>
    </div>
    <hr style="border-color:rgba(99,132,255,0.15);">
    """, unsafe_allow_html=True)

    # ── City Selector ─────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:11px; color:#475569; font-weight:600; letter-spacing:1px; margin-bottom:6px;">CITY SELECTION</div>', unsafe_allow_html=True)
    city_options = {k: f"{v['display']} ({v.get('country','India')})" for k, v in CITIES.items()}
    city_keys = list(city_options.keys())
    selected_idx = city_keys.index(st.session_state.selected_city) if st.session_state.selected_city in city_keys else 0
    chosen = st.selectbox(
        "Select City",
        options=city_keys,
        format_func=lambda k: city_options[k],
        index=selected_idx,
        label_visibility="collapsed",
    )
    st.session_state.selected_city = chosen

    st.markdown('<hr style="border-color:rgba(99,132,255,0.1);">', unsafe_allow_html=True)

    # ── API Config ────────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:11px; color:#475569; font-weight:600; letter-spacing:1px; margin-bottom:6px;">API CONFIGURATION</div>', unsafe_allow_html=True)
    owm_key = st.text_input(
        "OpenWeatherMap API Key",
        value=st.session_state.owm_key,
        type="password",
        placeholder="Paste your OWM key (optional)",
        help="Get free key at openweathermap.org. Leave blank for simulation.",
    )
    st.session_state.owm_key = owm_key
    data_src_color = "#22c55e" if owm_key else "#f97316"
    data_src_label = "Live + Simulated" if owm_key else "Simulated Mode"
    st.markdown(f'<div style="font-size:11px; color:{data_src_color}; margin-top:4px;">● {data_src_label}</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(99,132,255,0.1);">', unsafe_allow_html=True)

    # ── Comparison Cities ─────────────────────────────────────────────────────
    st.markdown('<div style="font-size:11px; color:#475569; font-weight:600; letter-spacing:1px; margin-bottom:6px;">COMPARISON CITIES</div>', unsafe_allow_html=True)
    comp_chosen = st.multiselect(
        "Select cities",
        options=city_keys,
        default=st.session_state.comparison_cities,
        format_func=lambda k: city_options[k],
        max_selections=8,
        label_visibility="collapsed",
    )
    if comp_chosen:
        st.session_state.comparison_cities = comp_chosen

    st.markdown('<hr style="border-color:rgba(99,132,255,0.1);">', unsafe_allow_html=True)

    # ── Refresh ───────────────────────────────────────────────────────────────
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown('<hr style="border-color:rgba(99,132,255,0.1);">', unsafe_allow_html=True)

    # ── Legend ────────────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:11px; color:#475569; font-weight:600; letter-spacing:1px; margin-bottom:8px;">UPI LEGEND</div>', unsafe_allow_html=True)
    for cat, color in UPI_COLORS.items():
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:5px;">
            <div style="width:10px; height:10px; border-radius:50%; background:{color};"></div>
            <span style="font-size:12px; color:#94a3b8;">{cat}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:20px; padding:10px; background:rgba(10,22,40,0.6); border-radius:8px; border:1px solid rgba(99,132,255,0.1);">
        <div style="font-size:10px; color:#475569; line-height:1.6;">
            ⚡ Data: OpenAQ (AQI) + OpenWeatherMap<br>
            🧠 2 Prediction Models Active<br>
            🗺 {n} Cities Monitored<br>
            🕐 Cache TTL: 5 minutes
        </div>
    </div>
    """.format(n=len(CITIES)), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
render_header()

# ── Cache buster (5-min buckets) ─────────────────────────────────────────────
_ts = int(time.time() / 300)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("🔄 Synchronizing urban intelligence feeds..."):
    pulse = compute_city_pulse(st.session_state.selected_city, st.session_state.owm_key, _ts)
    all_cities = compute_all_cities(st.session_state.owm_key, _ts)

cfg = pulse["cfg"]
upi = pulse["upi"]
crowd = pulse["crowd"]
weather = pulse["weather"]
aqi_data = pulse["aqi_data"]
stress = pulse["stress"]
peaks = pulse["peaks"]

# ── ALERTS (top level) ────────────────────────────────────────────────────────
if upi["alert"]:
    render_alert_banner(
        f"Extreme Urban Stress detected at {cfg['display']} — Immediate action required",
        cfg["display"],
        upi["score"],
    )
elif upi["score"] >= 55:
    st.markdown(f"""
    <div style="background:rgba(249,115,22,0.1); border:1px solid rgba(249,115,22,0.4); border-left:3px solid #f97316;
        border-radius:8px; padding:10px 16px; margin-bottom:10px; font-size:13px; color:#fed7aa;">
        ⚠ High Urban Stress detected at <b>{cfg['display']}</b> — UPI: {upi['score']} | Monitor closely
    </div>
    """, unsafe_allow_html=True)

# Future alert previews
if pulse["future_alerts"]:
    for fa in pulse["future_alerts"][:2]:
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.07); border:1px solid rgba(239,68,68,0.25);
            border-radius:8px; padding:8px 14px; margin-bottom:6px; font-size:12px; color:#fca5a5;">
            🔮 FORECAST ALERT: {fa['message']}
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT: Left (Map) | Right (Analytics)
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1.35, 1], gap="medium")

# ════════════════════════════════════
# LEFT COLUMN: Map + Overview Cards
# ════════════════════════════════════
with col_left:
    # ── City title ────────────────────────────────────────────────────────────
    col_city_info, col_upi_badge = st.columns([1.4, 1])
    with col_city_info:
        st.markdown(f"""
        <div style="margin-bottom:10px;">
            <div style="font-size:20px; font-weight:800; color:#e2e8f0;">{cfg['display']}</div>
            <div style="font-size:12px; color:#475569;">{cfg.get('state','')}, {cfg.get('country','India')} &nbsp;|&nbsp; 
            <span style="color:#64748b;">{cfg['area_type'].replace('_',' ').title()}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col_upi_badge:
        st.markdown(render_upi_badge(upi["score"], upi["category"]), unsafe_allow_html=True)

    # ── Quick metrics ─────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("AQI", f"{aqi_data['aqi']:.0f}", delta=None)
    with m2:
        st.metric("Temp", f"{weather['temp_c']}°C", f"{weather['desc']}")
    with m3:
        st.metric("Humidity", f"{weather['humidity']}%")
    with m4:
        crowd_delta = "🔴 Peak" if crowd["is_peak"] else "🟢 Normal"
        st.metric("Crowd", f"{crowd['score']:.0f}%", crowd_delta)

    # ── Map ───────────────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:11px; color:#475569; font-weight:600; letter-spacing:1px; margin:12px 0 6px 0;">GLOBAL MONITORING MAP</div>', unsafe_allow_html=True)
    selected_cfg = CITIES[st.session_state.selected_city]
    fmap = build_map(
        all_cities,
        center_lat=selected_cfg["lat"],
        center_lon=selected_cfg["lon"],
        zoom=5 if selected_cfg.get("country") == "India" else 3,
        selected_city_key=st.session_state.selected_city,
    )
    st_folium(fmap, height=420, use_container_width=True, returned_objects=[])

    # ── Dominant factor + stress model output ─────────────────────────────────
    col_dom, col_stress = st.columns(2)
    with col_dom:
        st.markdown(glass_card(f"""
        <div style="font-size:11px; color:#64748b; margin-bottom:6px; font-weight:600;">DOMINANT STRESS FACTOR</div>
        <div style="font-size:17px; font-weight:700; color:{upi['color']};">{upi['dominant_factor']}</div>
        <div style="font-size:11px; color:#475569; margin-top:4px;">AQI source: {aqi_data['source']}</div>
        """), unsafe_allow_html=True)
    with col_stress:
        st.markdown(glass_card(f"""
        <div style="font-size:11px; color:#64748b; margin-bottom:6px; font-weight:600;">STRESS MODEL OUTPUT</div>
        <div style="font-size:15px; font-weight:700; color:{stress['color']};">{stress['icon']} {stress['level']}</div>
        <div style="font-size:11px; color:#475569; margin-top:4px;">{stress['name']}</div>
        """, border_color=f"{stress['color']}44"), unsafe_allow_html=True)

    # ── Response recommendations ───────────────────────────────────────────────
    with st.expander("📋 Response Recommendations", expanded=upi["alert"]):
        st.markdown(f"""
        <div style="font-size:12px; color:#94a3b8; margin-bottom:8px; font-style:italic;">
            {stress['response']}
        </div>
        """, unsafe_allow_html=True)
        for rec in stress["recommendations"]:
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap:8px; margin-bottom:6px;
                background:rgba(15,23,42,0.6); border-radius:6px; padding:8px 12px;
                border-left:2px solid {stress['color']}55;">
                <span style="color:{stress['color']};">›</span>
                <span style="font-size:12px; color:#94a3b8;">{rec}</span>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════
# RIGHT COLUMN: Analytics Panel
# ════════════════════════════════════
with col_right:
    tabs = st.tabs(["📊 Analytics", "🔮 Forecast", "🏙 Zones", "🌍 Compare"])

    # ── TAB 1: Analytics ──────────────────────────────────────────────────────
    with tabs[0]:
        # UPI Gauge
        st.plotly_chart(
            chart_upi_gauge(upi["score"], upi["category"], upi["color"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # UPI components breakdown
        st.markdown('<div style="font-size:11px; color:#475569; font-weight:600; letter-spacing:1px; margin:6px 0 8px 0;">UPI COMPONENT BREAKDOWN</div>', unsafe_allow_html=True)
        comp = upi["components"]
        comp_labels = {
            "aqi_score": ("💨 Air Quality", "#818cf8"),
            "temp_score": ("🌡 Temperature", "#f97316"),
            "humidity_score": ("💧 Humidity", "#38bdf8"),
            "crowd_score": ("👥 Crowd", "#f43f5e"),
            "traffic_score": ("🚦 Traffic", "#eab308"),
            "infra_load_score": ("⚙️ Infrastructure", "#a3e635"),
        }
        for k, (label, color) in comp_labels.items():
            val = comp.get(k, 0)
            pct = int(val)
            st.markdown(f"""
            <div style="margin-bottom:7px;">
                <div style="display:flex; justify-content:space-between; font-size:11px; color:#64748b; margin-bottom:3px;">
                    <span>{label}</span><span style="color:{color}; font-weight:600;">{val:.0f}</span>
                </div>
                <div style="background:rgba(15,23,42,0.8); border-radius:4px; height:5px; overflow:hidden;">
                    <div style="height:100%; width:{pct}%; background:linear-gradient(90deg,{color}88,{color}); border-radius:4px; transition:width 0.5s;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Peak hours
        st.markdown('<div style="font-size:11px; color:#475569; font-weight:600; letter-spacing:1px; margin-bottom:8px;">PEAK HOUR DETECTION</div>', unsafe_allow_html=True)
        for pw in peaks["windows"]:
            intensity_color = {"Extreme": "#ef4444", "High": "#f97316", "Moderate": "#eab308"}.get(pw["intensity"], "#64748b")
            now_badge = '<span style="background:rgba(239,68,68,0.2); color:#ef4444; font-size:10px; padding:1px 7px; border-radius:4px; margin-left:6px;">NOW</span>' if peaks["current_is_peak"] else ""
            st.markdown(f"""
            <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(99,132,255,0.1); border-radius:8px; padding:10px 14px; margin-bottom:8px;">
                <div style="display:flex; align-items:center; justify-content:space-between;">
                    <div>
                        <div style="font-size:13px; font-weight:700; color:#e2e8f0;">{pw['label']}{now_badge}</div>
                        <div style="font-size:11px; color:#475569; margin-top:3px;">{pw['reason']}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:13px; font-weight:700; color:{intensity_color};">{pw['intensity']}</div>
                        <div style="font-size:11px; color:#475569;">{pw['avg_crowd']:.0f}% avg</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Hourly chart
        st.plotly_chart(
            chart_hourly(pulse["hourly"], peaks["windows"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # 7-day trend
        st.plotly_chart(
            chart_7day_trend(pulse["hist"], cfg["display"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── TAB 2: Forecast ───────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("""
        <div style="display:flex; gap:10px; margin-bottom:12px;">
            <div style="background:rgba(129,140,248,0.1); border:1px solid rgba(129,140,248,0.25); border-radius:8px; padding:8px 14px; font-size:12px; color:#818cf8;">
                🔮 <b>Model 1:</b> Holt's Exponential Smoothing
            </div>
            <div style="background:rgba(249,115,22,0.1); border:1px solid rgba(249,115,22,0.25); border-radius:8px; padding:8px 14px; font-size:12px; color:#f97316;">
                📐 <b>Model 2:</b> Rule-Based Stress Scoring
            </div>
        </div>
        """, unsafe_allow_html=True)

        horizon_choice = st.radio(
            "Forecast Window",
            ["24 Hours", "48 Hours"],
            horizontal=True,
            label_visibility="collapsed",
        )
        forecasts = pulse["forecasts_24"] if horizon_choice == "24 Hours" else pulse["forecasts_48"]
        st.plotly_chart(
            chart_forecast(forecasts, horizon_choice),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Upcoming forecast alerts
        if pulse["future_alerts"]:
            st.markdown('<div style="font-size:11px; color:#475569; font-weight:600; letter-spacing:1px; margin:8px 0 6px;">PREDICTED STRESS WINDOWS</div>', unsafe_allow_html=True)
            for fa in pulse["future_alerts"]:
                st.markdown(f"""
                <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.3); border-radius:8px;
                    padding:10px 14px; margin-bottom:6px; font-size:12px; color:#fca5a5;">
                    ⚠ {fa['message']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.25); border-radius:8px;
                padding:10px 14px; font-size:12px; color:#86efac;">
                ✅ No extreme stress events predicted in forecast window
            </div>
            """, unsafe_allow_html=True)

        # Forecast table (top 12 hours)
        st.markdown('<div style="font-size:11px; color:#475569; font-weight:600; letter-spacing:1px; margin:10px 0 6px;">FORECAST DETAIL TABLE</div>', unsafe_allow_html=True)
        table_rows = ""
        for f in forecasts[:12]:
            color = UPI_COLORS.get(f["category"], "#eab308")
            peak_tag = '<span style="font-size:9px; background:rgba(249,115,22,0.2); color:#f97316; padding:1px 5px; border-radius:3px; margin-left:4px;">PEAK</span>' if f["is_peak"] else ""
            table_rows += f"""
            <tr style="border-bottom:1px solid rgba(99,132,255,0.06);">
                <td style="padding:6px 8px; font-size:11px; color:#64748b;">{f['hour_label']}{peak_tag}</td>
                <td style="padding:6px 8px; font-size:12px; font-weight:700; color:{color}; text-align:center;">{f['upi']:.0f}</td>
                <td style="padding:6px 8px; font-size:11px; color:#94a3b8; text-align:center;">{f['aqi']:.0f}</td>
                <td style="padding:6px 8px; font-size:11px; color:#94a3b8; text-align:center;">{f['crowd']:.0f}%</td>
                <td style="padding:6px 8px; font-size:11px; color:{color}; text-align:right;">{f['category']}</td>
            </tr>"""
        st.markdown(f"""
        <table style="width:100%; border-collapse:collapse; background:rgba(15,23,42,0.5); border-radius:10px; overflow:hidden;">
            <thead>
                <tr style="background:rgba(10,22,40,0.8);">
                    <th style="padding:8px; font-size:10px; color:#475569; font-weight:600; text-align:left;">TIME</th>
                    <th style="padding:8px; font-size:10px; color:#475569; font-weight:600; text-align:center;">UPI</th>
                    <th style="padding:8px; font-size:10px; color:#475569; font-weight:600; text-align:center;">AQI</th>
                    <th style="padding:8px; font-size:10px; color:#475569; font-weight:600; text-align:center;">CROWD</th>
                    <th style="padding:8px; font-size:10px; color:#475569; font-weight:600; text-align:right;">STATUS</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

    # ── TAB 3: Zone Intelligence ───────────────────────────────────────────────
    with tabs[2]:
        st.markdown(f"""
        <div style="font-size:12px; color:#64748b; margin-bottom:12px;">
            {cfg['display']} is divided into <b style="color:#94a3b8">{len(pulse['zones'])} zones</b> 
            based on administrative boundaries and land use patterns.
        </div>
        """, unsafe_allow_html=True)
        for zone in pulse["zones"]:
            render_zone_card(zone)

        # Weather details
        st.markdown('<div style="font-size:11px; color:#475569; font-weight:600; letter-spacing:1px; margin:12px 0 8px;">ENVIRONMENTAL CONDITIONS</div>', unsafe_allow_html=True)
        st.markdown(glass_card(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
            <div>
                <div style="font-size:11px; color:#475569;">Temperature</div>
                <div style="font-size:18px; font-weight:700; color:#f97316;">{weather['temp_c']}°C</div>
                <div style="font-size:10px; color:#64748b;">Feels like {weather.get('feels_like', weather['temp_c'])}°C</div>
            </div>
            <div>
                <div style="font-size:11px; color:#475569;">Humidity</div>
                <div style="font-size:18px; font-weight:700; color:#38bdf8;">{weather['humidity']}%</div>
                <div style="font-size:10px; color:#64748b;">{weather['desc']}</div>
            </div>
            <div>
                <div style="font-size:11px; color:#475569;">Wind Speed</div>
                <div style="font-size:18px; font-weight:700; color:#a3e635;">{weather['wind_kmh']} km/h</div>
            </div>
            <div>
                <div style="font-size:11px; color:#475569;">AQI (PM2.5)</div>
                <div style="font-size:18px; font-weight:700; color:#818cf8;">{aqi_data['aqi']:.0f}</div>
                <div style="font-size:10px; color:#64748b;">PM2.5: {aqi_data['pm25']} µg/m³</div>
            </div>
        </div>
        <div style="margin-top:10px; font-size:10px; color:#334155;">Source: {weather['source']} | {aqi_data['source']}</div>
        """), unsafe_allow_html=True)

    # ── TAB 4: City Comparison ────────────────────────────────────────────────
    with tabs[3]:
        comp_keys = st.session_state.comparison_cities
        comp_data = [c for c in all_cities if c["city_key"] in comp_keys]
        comp_data.sort(key=lambda x: x["upi"], reverse=True)

        if comp_data:
            st.plotly_chart(
                chart_city_comparison(comp_data),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            # Table
            st.markdown('<div style="font-size:11px; color:#475569; font-weight:600; letter-spacing:1px; margin:10px 0 6px;">CITY RANKING TABLE</div>', unsafe_allow_html=True)
            rows = ""
            for i, cd in enumerate(comp_data):
                color = UPI_COLORS.get(cd["category"], "#eab308")
                medal = ["🥇","🥈","🥉"][i] if i < 3 else f"#{i+1}"
                peak_tag = "🔴" if cd["is_peak"] else "🟢"
                rows += f"""
                <tr style="border-bottom:1px solid rgba(99,132,255,0.06);">
                    <td style="padding:7px 8px; font-size:12px; color:#64748b;">{medal}</td>
                    <td style="padding:7px 8px; font-size:12px; color:#e2e8f0; font-weight:600;">{cd['display']}</td>
                    <td style="padding:7px 8px; font-size:13px; font-weight:800; color:{color}; text-align:center;">{cd['upi']:.0f}</td>
                    <td style="padding:7px 8px; font-size:11px; color:#94a3b8; text-align:center;">{cd['aqi']:.0f}</td>
                    <td style="padding:7px 8px; font-size:11px; color:#94a3b8; text-align:center;">{cd['crowd_score']:.0f}%</td>
                    <td style="padding:7px 8px; font-size:11px; text-align:center;">{peak_tag}</td>
                    <td style="padding:7px 8px; font-size:11px; color:{color}; text-align:right;">{cd['category']}</td>
                </tr>"""
            st.markdown(f"""
            <table style="width:100%; border-collapse:collapse; background:rgba(15,23,42,0.5); border-radius:10px; overflow:hidden;">
                <thead>
                    <tr style="background:rgba(10,22,40,0.8);">
                        <th style="padding:8px; font-size:10px; color:#475569; text-align:left;">#</th>
                        <th style="padding:8px; font-size:10px; color:#475569; text-align:left;">CITY</th>
                        <th style="padding:8px; font-size:10px; color:#475569; text-align:center;">UPI</th>
                        <th style="padding:8px; font-size:10px; color:#475569; text-align:center;">AQI</th>
                        <th style="padding:8px; font-size:10px; color:#475569; text-align:center;">CROWD</th>
                        <th style="padding:8px; font-size:10px; color:#475569; text-align:center;">STATUS</th>
                        <th style="padding:8px; font-size:10px; color:#475569; text-align:right;">CATEGORY</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)
        else:
            st.info("Select cities in the sidebar to compare.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    margin-top: 24px;
    border-top: 1px solid rgba(99,132,255,0.1);
    padding: 14px 0 8px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
">
    <div style="font-size:11px; color:#334155;">
        🌐 <b style="color:#475569;">Urban Pulse AI</b> &nbsp;|&nbsp; 
        Global Smart City Intelligence Platform &nbsp;|&nbsp;
        Powered by OpenAQ + OpenWeatherMap
    </div>
    <div style="display:flex; gap:16px; font-size:10px; color:#334155;">
        <span>📡 {n_cities} Cities</span>
        <span>🧠 2 Prediction Models</span>
        <span>⚡ Real-time Intelligence</span>
        <span>🔄 5-min Cache</span>
    </div>
</div>
""".format(n_cities=len(CITIES)), unsafe_allow_html=True)
