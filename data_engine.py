"""
Urban Pulse AI — Data Engine
Real-time data fetching: OpenAQ (AQI) + OpenWeatherMap
Caching layer for performance. Dynamic by city + time.
"""

import time
import math
import random
import hashlib
import requests
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# ─── In-memory cache ───────────────────────────────────────────────────────────
_cache: Dict[str, Dict] = {}
_cache_lock = threading.Lock()
CACHE_TTL = 600  # 10 minutes


def _cache_key(*args) -> str:
    return hashlib.md5("|".join(str(a) for a in args).encode()).hexdigest()


def _cache_get(key: str) -> Optional[Any]:
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.time() - entry["ts"] < CACHE_TTL:
            return entry["data"]
    return None


def _cache_set(key: str, data: Any):
    with _cache_lock:
        _cache[key] = {"data": data, "ts": time.time()}


# ─── OpenAQ  ───────────────────────────────────────────────────────────────────
OPENAQ_BASE = "https://api.openaq.org/v3"
OPENAQ_HEADERS = {"accept": "application/json"}


def fetch_aqi(city_name: str) -> Dict:
    """
    Fetch latest PM2.5/AQI-equivalent from OpenAQ v3.
    Returns: {"aqi": float, "pm25": float, "source": str, "ts": str}
    Falls back to deterministic simulation on failure.
    """
    key = _cache_key("aqi", city_name)
    cached = _cache_get(key)
    if cached:
        return cached

    try:
        # Search for locations in the city
        resp = requests.get(
            f"{OPENAQ_BASE}/locations",
            params={"city": city_name, "limit": 5, "order_by": "lastUpdated", "sort": "desc"},
            headers=OPENAQ_HEADERS,
            timeout=6,
        )
        if resp.status_code == 200:
            locs = resp.json().get("results", [])
            for loc in locs:
                loc_id = loc.get("id")
                if not loc_id:
                    continue
                # Get latest measurements for this location
                meas = requests.get(
                    f"{OPENAQ_BASE}/locations/{loc_id}/latest",
                    headers=OPENAQ_HEADERS,
                    timeout=5,
                )
                if meas.status_code == 200:
                    results = meas.json().get("results", [])
                    for r in results:
                        for param in r.get("parameters", []):
                            if param.get("parameter") in ("pm25", "pm2.5"):
                                pm25 = float(param.get("value", 0))
                                if pm25 > 0:
                                    aqi = _pm25_to_aqi(pm25)
                                    result = {
                                        "aqi": round(aqi, 1),
                                        "pm25": round(pm25, 2),
                                        "source": "OpenAQ Live",
                                        "ts": datetime.utcnow().isoformat(),
                                    }
                                    _cache_set(key, result)
                                    return result
    except Exception:
        pass  # Fall through to simulation

    # ── Deterministic simulation (seeded on city + hour for consistency) ───────
    result = _simulate_aqi(city_name)
    _cache_set(key, result)
    return result


def _pm25_to_aqi(pm25: float) -> float:
    """Convert PM2.5 µg/m³ → US AQI using standard breakpoints."""
    breakpoints = [
        (0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo
    return 500.0


def _simulate_aqi(city_name: str) -> Dict:
    """Realistic AQI simulation — seeded by city + current hour for consistency."""
    hour = datetime.now().hour
    seed = int(hashlib.md5(f"{city_name}{datetime.now().date()}".encode()).hexdigest(), 16) % 10000
    rng = random.Random(seed + hour)

    # City pollution tier
    high_pollution = ["Delhi", "Kanpur", "Dhaka", "Karachi", "Beijing", "Cairo", "Lagos", "Jakarta"]
    moderate_pollution = ["Mumbai", "Kolkata", "Chennai", "Hyderabad", "Lucknow", "Bangkok", "Seoul"]

    if any(p.lower() in city_name.lower() for p in high_pollution):
        base_pm25 = rng.uniform(80, 220)
    elif any(p.lower() in city_name.lower() for p in moderate_pollution):
        base_pm25 = rng.uniform(35, 90)
    else:
        base_pm25 = rng.uniform(10, 55)

    # Diurnal variation — worst in morning rush & evening
    hour_factor = 1.0
    if 7 <= hour <= 10 or 17 <= hour <= 21:
        hour_factor = 1.3
    elif 1 <= hour <= 5:
        hour_factor = 0.75

    pm25 = base_pm25 * hour_factor
    pm25 += rng.uniform(-5, 5)  # micro-variation
    pm25 = max(5, min(pm25, 500))

    return {
        "aqi": round(_pm25_to_aqi(pm25), 1),
        "pm25": round(pm25, 2),
        "source": "Simulated (OpenAQ fallback)",
        "ts": datetime.utcnow().isoformat(),
    }


# ─── OpenWeatherMap  ───────────────────────────────────────────────────────────
OWM_BASE = "https://api.openweathermap.org/data/2.5"


def fetch_weather(city_name: str, api_key: Optional[str], lat: float, lon: float) -> Dict:
    """
    Fetch real-time weather. Uses lat/lon for accuracy.
    Returns: {"temp_c": float, "humidity": int, "desc": str, "wind_kmh": float, "source": str}
    """
    key = _cache_key("weather", lat, lon)
    cached = _cache_get(key)
    if cached:
        return cached

    if api_key and api_key.strip() and api_key != "YOUR_OWM_KEY":
        try:
            resp = requests.get(
                f"{OWM_BASE}/weather",
                params={
                    "lat": lat,
                    "lon": lon,
                    "appid": api_key,
                    "units": "metric",
                },
                timeout=6,
            )
            if resp.status_code == 200:
                data = resp.json()
                result = {
                    "temp_c": round(data["main"]["temp"], 1),
                    "humidity": data["main"]["humidity"],
                    "desc": data["weather"][0]["description"].title(),
                    "wind_kmh": round(data["wind"]["speed"] * 3.6, 1),
                    "feels_like": round(data["main"]["feels_like"], 1),
                    "source": "OpenWeatherMap Live",
                    "ts": datetime.utcnow().isoformat(),
                }
                _cache_set(key, result)
                return result
        except Exception:
            pass

    result = _simulate_weather(city_name, lat, lon)
    _cache_set(key, result)
    return result


def _simulate_weather(city_name: str, lat: float, lon: float) -> Dict:
    """Climate-aware weather simulation seeded by city + hour."""
    hour = datetime.now().hour
    month = datetime.now().month
    seed = int(hashlib.md5(f"{city_name}{datetime.now().date()}".encode()).hexdigest(), 16) % 10000
    rng = random.Random(seed + hour)

    # Latitude-based base temperature
    abs_lat = abs(lat)
    if abs_lat < 15:           # Tropical
        base_temp = 30 + rng.uniform(-3, 5)
    elif abs_lat < 30:         # Subtropical
        base_temp = 26 + rng.uniform(-6, 8)
    elif abs_lat < 45:         # Temperate
        base_temp = 15 + rng.uniform(-10, 15)
    else:                      # Cold
        base_temp = 5 + rng.uniform(-15, 20)

    # Seasonal adjustment (NH/SH aware)
    if lat >= 0:  # Northern Hemisphere
        seasonal = math.sin((month - 3) * math.pi / 6) * 8
    else:         # Southern Hemisphere
        seasonal = math.sin((month + 3) * math.pi / 6) * 8
    base_temp += seasonal

    # Diurnal variation: peak ~14:00, min ~05:00
    diurnal = 6 * math.sin((hour - 5) * math.pi / 12)
    temp = base_temp + diurnal + rng.uniform(-1.5, 1.5)

    # Humidity — inverse of temp for simple model
    if abs_lat < 20:  # Tropical: high humidity
        humidity = int(rng.uniform(70, 95))
    elif abs_lat < 35:
        humidity = int(rng.uniform(40, 80))
    else:
        humidity = int(rng.uniform(35, 70))

    descs = ["Clear Sky", "Partly Cloudy", "Overcast", "Light Rain", "Sunny", "Haze", "Mist"]
    if humidity > 80:
        descs = ["Mist", "Light Rain", "Overcast", "Haze"]
    desc = rng.choice(descs)

    wind = round(rng.uniform(5, 35), 1)

    return {
        "temp_c": round(temp, 1),
        "humidity": humidity,
        "desc": desc,
        "wind_kmh": wind,
        "feels_like": round(temp - wind * 0.1 + humidity * 0.05 - 3, 1),
        "source": "Simulated (OWM fallback)",
        "ts": datetime.utcnow().isoformat(),
    }


# ─── Historical trend (7 days) ─────────────────────────────────────────────────
def fetch_historical_aqi(city_name: str, days: int = 7) -> list:
    """
    Returns a list of {date, aqi, pm25} for the past `days` days.
    Uses simulation with plausible day-to-day variation.
    """
    seed = int(hashlib.md5(city_name.encode()).hexdigest(), 16) % 10000
    rng = random.Random(seed)
    records = []
    base = _simulate_aqi(city_name)["pm25"]
    today = datetime.now().date()
    for i in range(days, -1, -1):
        d = today - timedelta(days=i)
        # Correlated random walk
        base += rng.uniform(-8, 8)
        base = max(8, min(base, 450))
        aqi = _pm25_to_aqi(base)
        records.append({"date": str(d), "pm25": round(base, 1), "aqi": round(aqi, 1)})
    return records


def fetch_hourly_data(city_name: str) -> list:
    """
    Returns hourly UPI-component data for the past 24 hours.
    Used for analytics charts.
    """
    seed = int(hashlib.md5(f"{city_name}{datetime.now().date()}".encode()).hexdigest(), 16) % 10000
    rng = random.Random(seed)
    current_hour = datetime.now().hour
    records = []
    for h in range(24):
        factor = 1.0
        if 7 <= h <= 10 or 17 <= h <= 21:
            factor = 1.4
        elif 0 <= h <= 5:
            factor = 0.6

        noise = rng.uniform(0.85, 1.15)
        base_aqi_sim = _simulate_aqi(city_name)["pm25"] * factor * noise
        aqi = round(_pm25_to_aqi(max(5, base_aqi_sim)), 1)

        traffic = round(min(100, 20 + 80 * factor * rng.uniform(0.8, 1.2)), 1)
        crowd = round(min(100, 15 + 85 * factor * rng.uniform(0.7, 1.2)), 1)

        records.append({
            "hour": h,
            "hour_label": f"{h:02d}:00",
            "aqi": aqi,
            "traffic": traffic,
            "crowd": crowd,
            "is_current": h == current_hour,
        })
    return records
