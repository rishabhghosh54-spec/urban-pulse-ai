# 🌐 URBAN PULSE AI — Global Smart City Intelligence Platform

> Production-grade real-time urban monitoring, prediction, and alerting system for governments, urban planners, and emergency response teams.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
cd urban_pulse
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. (Optional) Add API keys in the sidebar
- **OpenWeatherMap**: Free at [openweathermap.org](https://openweathermap.org/api) — enables live weather data
- **OpenAQ**: No key required — free AQI data fetched automatically
- Without keys, the system runs in **simulation mode** with realistic, seeded data

---

## 🏗 Architecture

```
urban_pulse/
├── app.py                    # Main Streamlit application
├── requirements.txt
├── config/
│   └── cities.py             # 40+ city dataset (deep Maharashtra modeling)
└── modules/
    ├── data_engine.py        # Real-time data: OpenAQ + OpenWeatherMap + caching
    ├── intelligence.py       # UPI computation, crowd engine, peak detection
    ├── prediction.py         # 2-model prediction layer
    └── visualization.py      # Folium map, Plotly charts
```

---

## 🧠 System Components

### Urban Pulse Index (UPI)
Composite score (0–100) weighted across:
- Air Quality (AQI) — 25%
- Crowd Density — 28%
- Traffic Load (proxy) — 17%
- Temperature — 12%
- Infrastructure Load — 10%
- Humidity — 8%

**Categories:** Low → Moderate → High → Extreme

### Crowd Intelligence Engine
Time-of-day + area-type based crowd simulation:
- Railway stations: extreme during 7–10 AM, 5–9 PM
- Business zones: peak at 8–10 AM, 6–8 PM
- Residential: moderate peaks morning/evening

### Peak Hour Detection
Analyzes 24h crowd data → identifies peak windows with reasons

### Prediction Models (STRICT: 2 only)
1. **Time-series**: Holt's Double Exponential Smoothing (24h/48h)
2. **Stress scoring**: Rule-based weighted composite

### Alert System
- Real-time alerts when UPI ≥ 75
- Forecast alerts for upcoming stress windows
- Stress level classification with response recommendations

---

## 🗺 Cities Monitored

**Maharashtra (Deep):** Mumbai CST, Dadar, Andheri, Bandra, Thane, Kalyan, Dombivli, Pune, Nagpur, Nashik, Aurangabad, Navi Mumbai

**India:** Delhi, Bengaluru, Kolkata, Chennai, Hyderabad, Ahmedabad, Jaipur, Lucknow, Kanpur, Surat, Bhopal, Patna

**Global:** Tokyo, London, New York, Beijing, Shanghai, Singapore, Dubai, Sydney, Paris, Berlin, Toronto, Jakarta, Bangkok, Seoul, Dhaka, Karachi, Lagos, Cairo, São Paulo, Mexico City

---

## ⚡ Performance
- 5-minute data cache (no redundant API calls)
- Seeded simulation for consistent results
- Lazy loading per city on demand

---

## 📡 Data Sources
| Source | Data | Auth |
|--------|------|------|
| OpenAQ v3 | PM2.5, AQI | None (free) |
| OpenWeatherMap | Temp, Humidity, Wind | API Key (free tier) |
| Simulation | Crowd, Traffic, Trends | None (deterministic) |
