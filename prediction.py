"""
Urban Pulse AI — Prediction Layer
STRICT: Only 2 models used:
  1. Time-series forecasting (next 24–48 hours) — exponential smoothing
  2. Stress scoring model — rule-based weighted composite
"""

import math
import random
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict

from config.cities import AREA_PEAK_PROFILES


# ═══════════════════════════════════════════════════════════════
# MODEL 1: Time-Series Forecasting (Exponential Smoothing)
#          Forecasts UPI, AQI, crowd for next 24–48 hours
# ═══════════════════════════════════════════════════════════════

def _double_exponential_smoothing(values: List[float], alpha: float = 0.4, beta: float = 0.2) -> List[float]:
    """
    Holt's Double Exponential Smoothing for trend-aware forecasting.
    Simple, efficient, no heavy ML stack.
    """
    if not values:
        return []
    level = values[0]
    trend = values[1] - values[0] if len(values) > 1 else 0
    smoothed = [level]
    for v in values[1:]:
        prev_level = level
        level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        smoothed.append(level)
    return smoothed, level, trend


def forecast_timeseries(
    city_name: str,
    area_type: str,
    current_upi: float,
    current_aqi: float,
    current_crowd: float,
    horizon_hours: int = 24,
) -> List[Dict]:
    """
    Forecast UPI, AQI, and crowd for the next `horizon_hours` hours.
    Uses double exponential smoothing + area-type peak patterns.
    Returns list of {hour, hour_label, upi, aqi, crowd, category}
    """
    # Seed for reproducible variation within a day
    seed = int(hashlib.md5(f"{city_name}{datetime.now().date()}".encode()).hexdigest(), 16) % 10000
    rng = random.Random(seed)

    current_hour = datetime.now().hour
    profile = AREA_PEAK_PROFILES.get(area_type, AREA_PEAK_PROFILES["mixed_zone"])
    forecasts = []

    # Build 24-h base pattern for the area type
    upi_history = []
    for h in range(24):
        in_peak = any(s <= h < e for s, e in profile["peaks"])
        factor = profile["peak_multiplier"] if in_peak else profile["off_peak_multiplier"]
        base = current_upi * (0.5 + factor * 0.5)
        upi_history.append(max(5, min(100, base + rng.uniform(-4, 4))))

    # Smooth the pattern
    _, level, trend = _double_exponential_smoothing(upi_history)

    for h_offset in range(1, horizon_hours + 1):
        future_hour = (current_hour + h_offset) % 24
        future_dt = datetime.now() + timedelta(hours=h_offset)

        # Pattern-based multiplier
        in_peak = any(s <= future_hour < e for s, e in profile["peaks"])
        factor = profile["peak_multiplier"] if in_peak else profile["off_peak_multiplier"]

        # Forecast with trend decay (mean-reverting after 24h)
        decay = math.exp(-h_offset / 36)  # Trend decays over ~36h
        forecast_upi = level + trend * h_offset * decay
        forecast_upi *= (0.6 + factor * 0.4)
        forecast_upi = max(5, min(100, forecast_upi + rng.gauss(0, 2.5)))

        # AQI correlated but slightly independent
        aqi_base = current_aqi * factor * (0.8 + rng.uniform(0, 0.4))
        forecast_aqi = max(5, min(500, aqi_base + rng.gauss(0, 8)))

        # Crowd
        forecast_crowd = current_crowd * factor + rng.gauss(0, 3)
        forecast_crowd = max(5, min(100, forecast_crowd))

        # Classify
        category = "Low"
        if forecast_upi >= 75:   category = "Extreme"
        elif forecast_upi >= 55: category = "High"
        elif forecast_upi >= 30: category = "Moderate"

        forecasts.append({
            "h_offset": h_offset,
            "hour": future_hour,
            "hour_label": future_dt.strftime("%b %d %H:%M"),
            "upi": round(forecast_upi, 1),
            "aqi": round(forecast_aqi, 1),
            "crowd": round(forecast_crowd, 1),
            "category": category,
            "is_peak": in_peak,
        })

    return forecasts


# ═══════════════════════════════════════════════════════════════
# MODEL 2: Stress Scoring Model
#          Rule-based + weighted logic for alert classification
#          and recommended response level
# ═══════════════════════════════════════════════════════════════

# Stress escalation rules (ordered, first match wins)
STRESS_RULES = [
    {
        "name": "Pandemic-Level Urban Collapse",
        "condition": lambda d: d["upi"] >= 92 and d["aqi"] > 300 and d["crowd"] > 85,
        "level": "CRITICAL",
        "color": "#dc2626",
        "response": "Immediate evacuation protocols. Emergency services on full alert.",
        "icon": "🔴",
    },
    {
        "name": "Extreme Urban Stress",
        "condition": lambda d: d["upi"] >= 75,
        "level": "EXTREME",
        "color": "#ef4444",
        "response": "Alert emergency response teams. Recommend restricting vehicle access.",
        "icon": "🔴",
    },
    {
        "name": "High Urban Stress",
        "condition": lambda d: d["upi"] >= 55,
        "level": "HIGH",
        "color": "#f97316",
        "response": "Activate traffic management protocols. Advisory for vulnerable populations.",
        "icon": "🟠",
    },
    {
        "name": "Elevated Crowd Risk",
        "condition": lambda d: d["crowd"] > 80 and d["upi"] < 55,
        "level": "ELEVATED",
        "color": "#f59e0b",
        "response": "Deploy crowd management personnel at key chokepoints.",
        "icon": "🟡",
    },
    {
        "name": "Air Quality Alert",
        "condition": lambda d: d["aqi"] > 200 and d["upi"] < 55,
        "level": "AIR ALERT",
        "color": "#a855f7",
        "response": "Issue air quality advisory. Advise limiting outdoor activity.",
        "icon": "🟣",
    },
    {
        "name": "Moderate Stress",
        "condition": lambda d: d["upi"] >= 30,
        "level": "MODERATE",
        "color": "#eab308",
        "response": "Standard monitoring. Prepare contingency if trends worsen.",
        "icon": "🟡",
    },
    {
        "name": "Normal Operations",
        "condition": lambda d: True,
        "level": "NORMAL",
        "color": "#22c55e",
        "response": "City operating within normal parameters.",
        "icon": "🟢",
    },
]


def classify_stress(upi: float, aqi: float, crowd: float, dominant_factor: str) -> Dict:
    """
    Apply stress rules to classify current city state and generate recommendations.
    Returns: {level, name, color, response, icon, recommendations}
    """
    data = {"upi": upi, "aqi": aqi, "crowd": crowd}

    for rule in STRESS_RULES:
        if rule["condition"](data):
            matched = rule
            break

    # Generate specific recommendations based on dominant factor
    recommendations = _generate_recommendations(dominant_factor, matched["level"])

    return {
        "level": matched["level"],
        "name": matched["name"],
        "color": matched["color"],
        "response": matched["response"],
        "icon": matched["icon"],
        "recommendations": recommendations,
    }


def _generate_recommendations(dominant_factor: str, stress_level: str) -> List[str]:
    """Rule-based recommendation engine."""
    base_recs = {
        "Air pollution": [
            "Deploy air quality monitoring teams to hotspots",
            "Restrict heavy vehicle movement in affected zones",
            "Issue health advisory to residents — limit outdoor exposure",
            "Activate emergency ventilation systems in public spaces",
        ],
        "Crowd congestion": [
            "Deploy additional crowd management personnel",
            "Open auxiliary entry/exit points at transit hubs",
            "Activate real-time crowd flow alerts on public displays",
            "Coordinate with transport authority for frequency increase",
        ],
        "Traffic overload": [
            "Activate adaptive signal control systems",
            "Divert routes via dynamic signage",
            "Alert traffic police for manual override at critical junctions",
            "Enable emergency lane protocols if applicable",
        ],
        "Temperature stress": [
            "Open cooling/hydration centers in public spaces",
            "Issue heat wave advisory for outdoor workers",
            "Increase water availability at transport nodes",
            "Alert hospitals to prepare for heat-related cases",
        ],
        "High humidity": [
            "Enhance ventilation in crowded enclosed spaces",
            "Issue advisory for elderly and vulnerable populations",
            "Monitor for heat + humidity combination risk",
        ],
        "Infrastructure strain": [
            "Alert infrastructure maintenance teams",
            "Initiate load-shedding protocols if electrical grid overloaded",
            "Inspect critical nodes for stress fractures",
            "Activate backup systems for key infrastructure",
        ],
    }

    recs = base_recs.get(dominant_factor, ["Monitor situation continuously"])

    # Escalate number of recommendations by stress level
    if stress_level in ("EXTREME", "CRITICAL"):
        return recs
    elif stress_level == "HIGH":
        return recs[:3]
    else:
        return recs[:2]


def forecast_alerts(
    forecasts: List[Dict],
    city_display: str,
) -> List[Dict]:
    """
    Scan forecast to identify upcoming alert windows.
    Returns list of future alerts.
    """
    alerts = []
    for f in forecasts:
        if f["upi"] >= 75:
            alerts.append({
                "city": city_display,
                "time": f["hour_label"],
                "upi": f["upi"],
                "category": f["category"],
                "message": f"⚠ Extreme Urban Stress forecast — {city_display} at {f['hour_label']} (UPI: {f['upi']})",
            })
    return alerts[:3]  # Return top 3 upcoming alerts
