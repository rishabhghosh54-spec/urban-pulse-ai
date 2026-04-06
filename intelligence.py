"""
Urban Pulse AI — Intelligence Layer
- Urban Pulse Index (UPI) computation
- Crowd Intelligence Engine
- Peak Hour Detection Engine
- Stress Classification System
"""

import math
import random
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple

from config.cities import AREA_PEAK_PROFILES, UPI_THRESHOLDS, UPI_COLORS

# ─── Weights for UPI components ───────────────────────────────────────────────
UPI_WEIGHTS = {
    "aqi_score":         0.25,
    "temp_score":        0.12,
    "humidity_score":    0.08,
    "crowd_score":       0.28,
    "traffic_score":     0.17,
    "infra_load_score":  0.10,
}


# ─── AQI → normalized score (0–100) ───────────────────────────────────────────
def _normalize_aqi(aqi: float) -> float:
    """AQI 0–500 → stress score 0–100"""
    return min(100, (aqi / 500) * 100)


# ─── Temperature stress ────────────────────────────────────────────────────────
def _temp_stress(temp_c: float) -> float:
    """Temperature stress score. Comfortable 18–26°C = low stress."""
    if 18 <= temp_c <= 26:
        return max(0, 10 - (26 - abs(temp_c - 22)) * 2)
    elif temp_c > 26:
        return min(100, ((temp_c - 26) / 20) * 100)
    else:
        return min(100, ((18 - temp_c) / 18) * 100)


# ─── Humidity stress ───────────────────────────────────────────────────────────
def _humidity_stress(humidity: int) -> float:
    """High humidity (>70%) increases discomfort exponentially."""
    if humidity <= 50:
        return 10
    elif humidity <= 70:
        return 10 + (humidity - 50) * 1.5
    else:
        return 40 + (humidity - 70) * 2.0


# ─── Crowd Intelligence Engine ────────────────────────────────────────────────
def compute_crowd_score(
    area_type: str,
    population_density: int,
    hour: int,
    day_of_week: int,
) -> Dict:
    """
    Simulate realistic crowd density based on:
    - Area type (railway, office, residential, etc.)
    - Time of day (hour)
    - Day of week (0=Mon, 6=Sun)
    - Population density baseline
    Returns: {score, category, level_label, is_peak}
    """
    profile = AREA_PEAK_PROFILES.get(area_type, AREA_PEAK_PROFILES["mixed_zone"])
    base = profile["base_crowd"]
    
    # Determine if current hour is in any peak window
    in_peak = False
    peak_proximity = 0.0
    for (peak_start, peak_end) in profile["peaks"]:
        if peak_start <= hour < peak_end:
            in_peak = True
            # Gaussian-like shape within peak window
            peak_center = (peak_start + peak_end) / 2
            spread = (peak_end - peak_start) / 2
            peak_proximity = math.exp(-((hour - peak_center) ** 2) / (2 * spread ** 2))
            break
        else:
            # Proximity to nearest peak
            dist = min(abs(hour - peak_start), abs(hour - peak_end))
            peak_proximity = max(peak_proximity, math.exp(-(dist ** 2) / 4))

    # Apply multiplier
    if in_peak:
        crowd = base + (100 - base) * peak_proximity * (profile["peak_multiplier"] - 1)
    else:
        crowd = base * (profile["off_peak_multiplier"] + peak_proximity * 0.4)

    # Weekend factor — less in offices, more in malls/residential
    if day_of_week >= 5:  # Sat/Sun
        if area_type in ("business_zone", "industrial"):
            crowd *= 0.55
        elif area_type in ("mixed_zone", "planned_zone"):
            crowd *= 1.15

    # Population density scaling
    density_factor = 0.7 + (population_density / 100) * 0.6
    crowd *= density_factor

    # Add seed-based deterministic noise
    seed = int(hashlib.md5(f"{area_type}{datetime.now().date()}{hour}".encode()).hexdigest(), 16) % 1000
    noise = (seed % 100) / 1000 * 8 - 4  # ±4
    crowd = max(5, min(100, crowd + noise))

    # Classify
    if crowd < 25:
        category = "Low"
    elif crowd < 50:
        category = "Moderate"
    elif crowd < 75:
        category = "High"
    else:
        category = "Extreme"

    return {
        "score": round(crowd, 1),
        "category": category,
        "is_peak": in_peak,
        "peak_proximity": round(peak_proximity, 2),
    }


# ─── Traffic proxy ────────────────────────────────────────────────────────────
def compute_traffic_score(crowd_score: float, area_type: str, hour: int) -> float:
    """
    Traffic load as a proxy derived from crowd + area type + time.
    Returns 0–100.
    """
    base_traffic = crowd_score * 0.85  # Traffic highly correlated with crowd

    # Area type boosts
    traffic_boost = {
        "railway_station": 15,
        "business_zone": 12,
        "mixed_zone": 8,
        "industrial": 10,
        "residential": 3,
        "planned_zone": 5,
    }
    base_traffic += traffic_boost.get(area_type, 7)

    # Night reduction
    if 0 <= hour < 5:
        base_traffic *= 0.3

    return min(100, round(base_traffic, 1))


# ─── Peak Hour Detection ──────────────────────────────────────────────────────
def detect_peak_hours(area_type: str, hourly_crowd: List[float]) -> Dict:
    """
    Analyze 24h crowd data to identify peak windows.
    Returns: {windows: [{start, end, reason, intensity}], current_is_peak: bool}
    """
    profile = AREA_PEAK_PROFILES.get(area_type, AREA_PEAK_PROFILES["mixed_zone"])
    peak_threshold = 65

    windows = []
    for (start, end) in profile["peaks"]:
        window_avg = sum(hourly_crowd[start:end]) / max(1, end - start)
        intensity = "Extreme" if window_avg > 80 else "High" if window_avg > 65 else "Moderate"

        reasons_map = {
            "railway_station": "Rail commuter surge + platform congestion",
            "business_zone": "Office rush hour + road congestion",
            "residential": "School + residential movement",
            "mixed_zone": "Mixed commuter + commercial activity",
            "industrial": "Factory shift change + freight movement",
            "planned_zone": "Planned zone commuter + residential activity",
        }
        reason = reasons_map.get(area_type, "High urban activity")

        windows.append({
            "start": start,
            "end": end,
            "label": f"{start:02d}:00 – {end:02d}:00",
            "reason": reason,
            "intensity": intensity,
            "avg_crowd": round(window_avg, 1),
        })

    current_hour = datetime.now().hour
    current_is_peak = any(w["start"] <= current_hour < w["end"] for w in windows)

    return {
        "windows": windows,
        "current_is_peak": current_is_peak,
        "peak_threshold": peak_threshold,
    }


# ─── Urban Pulse Index ────────────────────────────────────────────────────────
def compute_upi(
    aqi: float,
    temp_c: float,
    humidity: int,
    crowd_score: float,
    area_type: str,
    population_density: int,
    infra_load_base: int,
    hour: int,
) -> Dict:
    """
    Compute Urban Pulse Index (0–100) from all urban stress signals.
    Returns: {score, category, color, dominant_factor, components, alert}
    """
    traffic_score = compute_traffic_score(crowd_score, area_type, hour)

    # Component normalization (all → 0–100 stress scores)
    components = {
        "aqi_score":        round(_normalize_aqi(aqi), 1),
        "temp_score":       round(_temp_stress(temp_c), 1),
        "humidity_score":   round(min(100, _humidity_stress(humidity)), 1),
        "crowd_score":      round(crowd_score, 1),
        "traffic_score":    round(traffic_score, 1),
        "infra_load_score": round(min(100, infra_load_base + crowd_score * 0.15), 1),
    }

    # Weighted sum
    upi = sum(components[k] * UPI_WEIGHTS[k] for k in UPI_WEIGHTS)

    # Population density modifier (+up to 8 points for ultra-dense cities)
    density_boost = (population_density - 50) / 50 * 8
    upi = min(100, max(0, upi + density_boost))

    # Classify
    category = "Low"
    for cat, (lo, hi) in UPI_THRESHOLDS.items():
        if lo <= upi < hi:
            category = cat
            break
    if upi >= 100:
        category = "Extreme"

    # Dominant stress factor
    factor_labels = {
        "aqi_score": "Air pollution",
        "temp_score": "Temperature stress",
        "humidity_score": "High humidity",
        "crowd_score": "Crowd congestion",
        "traffic_score": "Traffic overload",
        "infra_load_score": "Infrastructure strain",
    }
    # Weighted contribution of each component
    contributions = {k: components[k] * UPI_WEIGHTS[k] for k in UPI_WEIGHTS}
    dominant_key = max(contributions, key=contributions.get)
    dominant_factor = factor_labels[dominant_key]

    return {
        "score": round(upi, 1),
        "category": category,
        "color": UPI_COLORS[category],
        "dominant_factor": dominant_factor,
        "components": components,
        "alert": upi >= 75,
        "traffic_score": traffic_score,
    }


# ─── Zone Intelligence ────────────────────────────────────────────────────────
def compute_zone_stress(city_cfg: Dict, upi_score: float, crowd_data: Dict, aqi: float) -> List[Dict]:
    """
    Divide city into zones A/B/C and assign stress levels.
    """
    zones_raw = city_cfg.get("zones", ["Zone A", "Zone B", "Zone C"])
    result = []
    zone_labels = ["A", "B", "C", "D"]

    # Generate slightly varied stress per zone
    for i, zone_name in enumerate(zones_raw[:4]):
        seed = int(hashlib.md5(f"{zone_name}{datetime.now().date()}".encode()).hexdigest(), 16) % 1000
        variation = (seed % 100) / 100 * 20 - 10  # ±10
        zone_upi = max(5, min(100, upi_score + variation + i * 3))

        cat = "Low"
        for c, (lo, hi) in UPI_THRESHOLDS.items():
            if lo <= zone_upi < hi:
                cat = c
                break

        zone_crowd = max(5, min(100, crowd_data["score"] + variation * 0.8))
        zone_aqi = max(5, min(500, aqi + (seed % 50) - 25))
        zone_infra = max(10, min(100, city_cfg["infrastructure_load_base"] + variation))

        drivers = {
            "Air pollution": zone_aqi / 500 * 0.25,
            "Crowd pressure": zone_crowd / 100 * 0.28,
            "Infrastructure load": zone_infra / 100 * 0.22,
        }
        dominant_driver = max(drivers, key=drivers.get)

        result.append({
            "zone": zone_labels[i],
            "name": zone_name,
            "upi": round(zone_upi, 1),
            "category": cat,
            "color": UPI_COLORS[cat],
            "crowd": round(zone_crowd, 1),
            "aqi": round(zone_aqi, 1),
            "infra_load": round(zone_infra, 1),
            "dominant_driver": dominant_driver,
        })
    return result
