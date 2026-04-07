"""
Urban Pulse AI — Visualization Layer
- Interactive folium map with styled markers
- Plotly analytics charts
"""

import json
import math
from typing import Dict, List, Optional

import folium
from folium.plugins import MarkerCluster
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config.cities import UPI_COLORS


# ─── Map ──────────────────────────────────────────────────────────────────────
def build_map(
    cities_data: List[Dict],
    center_lat: float = 20.5937,
    center_lon: float = 78.9629,
    zoom: int = 5,
    selected_city_key: Optional[str] = None,
) -> folium.Map:
    """
    Build interactive folium map with styled UPI markers.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=None,
    )

    # Dark tile layer (premium look)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr='© OpenStreetMap © CARTO',
        name="Dark Mode",
        overlay=False,
        control=True,
    ).add_to(m)

    for cd in cities_data:
        lat = cd["lat"]
        lon = cd["lon"]
        upi = cd.get("upi", 50)
        category = cd.get("category", "Moderate")
        color = UPI_COLORS.get(category, "#eab308")
        city_display = cd.get("display", "City")

        is_selected = cd.get("city_key") == selected_city_key

        # Marker size by UPI
        radius = 8 + (upi / 100) * 12
        if is_selected:
            radius += 6

        # Pulse ring for high/extreme
        if category in ("High", "Extreme"):
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius + 6,
                color=color,
                fill=False,
                weight=1.5,
                opacity=0.4,
            ).add_to(m)

        # Popup HTML
        crowd = cd.get("crowd_score", 0)
        aqi = cd.get("aqi", 0)
        peak_label = "🔴 Peak" if cd.get("is_peak") else "🟢 Off-Peak"

        popup_html = f"""
        <div style="
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            border: 1px solid {color};
            border-radius: 10px;
            padding: 14px 16px;
            min-width: 220px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.6);
        ">
            <div style="font-size:15px; font-weight:700; color:{color}; margin-bottom:8px;">
                📍 {city_display}
            </div>
            <table style="width:100%; font-size:12px; border-collapse:collapse;">
                <tr>
                    <td style="padding:3px 0; color:#94a3b8;">Urban Pulse Index</td>
                    <td style="text-align:right; font-weight:700; color:{color};">{upi} — {category}</td>
                </tr>
                <tr>
                    <td style="padding:3px 0; color:#94a3b8;">AQI</td>
                    <td style="text-align:right; font-weight:600;">{aqi}</td>
                </tr>
                <tr>
                    <td style="padding:3px 0; color:#94a3b8;">Crowd Level</td>
                    <td style="text-align:right; font-weight:600;">{crowd}%</td>
                </tr>
                <tr>
                    <td style="padding:3px 0; color:#94a3b8;">Status</td>
                    <td style="text-align:right;">{peak_label}</td>
                </tr>
            </table>
            <div style="margin-top:8px; font-size:11px; color:#64748b;">
                🎯 {cd.get('dominant_factor', '—')}
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=2.5 if is_selected else 1.5,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=folium.Tooltip(
                f"<b style='color:{color}'>{city_display}</b> | UPI: {upi} | {category}",
                sticky=True,
            ),
        ).add_to(m)

    return m


# ─── Hourly Chart ─────────────────────────────────────────────────────────────
def chart_hourly(hourly_data: List[Dict], peak_windows: List[Dict]) -> go.Figure:
    hours = [h["hour_label"] for h in hourly_data]
    crowd = [h["crowd"] for h in hourly_data]
    traffic = [h["traffic"] for h in hourly_data]
    aqi = [h["aqi"] for h in hourly_data]
    current_idx = next((i for i, h in enumerate(hourly_data) if h["is_current"]), 0)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.08,
    )

    # Peak zone shading
    for pw in peak_windows:
        start_idx = pw["start"]
        end_idx = min(pw["end"], 23)
        fig.add_vrect(
            x0=start_idx, x1=end_idx,
            fillcolor="rgba(249,115,22,0.12)",
            layer="below", line_width=0,
            row=1, col=1,
        )

    # Crowd
    fig.add_trace(go.Scatter(
        x=list(range(24)), y=crowd,
        name="Crowd %",
        line=dict(color="#f97316", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(249,115,22,0.15)",
        mode="lines",
    ), row=1, col=1)

    # Traffic
    fig.add_trace(go.Scatter(
        x=list(range(24)), y=traffic,
        name="Traffic %",
        line=dict(color="#818cf8", width=2, dash="dot"),
        mode="lines",
    ), row=1, col=1)

    # Current hour marker
    fig.add_vline(
        x=current_idx,
        line=dict(color="#22c55e", width=1.5, dash="dash"),
        row=1, col=1,
        annotation_text="Now",
        annotation_font_color="#22c55e",
    )

    # AQI
    fig.add_trace(go.Bar(
        x=list(range(24)), y=aqi,
        name="AQI",
        marker=dict(
            color=aqi,
            colorscale=[[0, "#22c55e"], [0.3, "#eab308"], [0.6, "#f97316"], [1, "#ef4444"]],
            colorbar=dict(thickness=8, title="AQI"),
        ),
        opacity=0.85,
    ), row=2, col=1)

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        legend=dict(
            orientation="h", x=0, y=1.08,
            font=dict(color="#94a3b8", size=11),
        ),
        font=dict(color="#94a3b8", family="Segoe UI"),
        title=dict(text="24-Hour Urban Signals", font=dict(color="#e2e8f0", size=13), x=0),
        xaxis2=dict(
            tickmode="array",
            tickvals=list(range(0, 24, 3)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)],
            gridcolor="rgba(255,255,255,0.06)",
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Score (0–100)", title_font=dict(size=11)),
        yaxis2=dict(gridcolor="rgba(255,255,255,0.06)", title="AQI", title_font=dict(size=11)),
    )
    return fig


# ─── 7-day Trend ─────────────────────────────────────────────────────────────
def chart_7day_trend(hist_data: List[Dict], city_display: str) -> go.Figure:
    dates = [d["date"] for d in hist_data]
    aqi_vals = [d["aqi"] for d in hist_data]

    fig = go.Figure()

    # Gradient fill
    fig.add_trace(go.Scatter(
        x=dates, y=aqi_vals,
        mode="lines+markers",
        name="AQI",
        line=dict(color="#818cf8", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(129,140,248,0.12)",
        marker=dict(size=6, color="#818cf8"),
    ))

    # Threshold lines
    for threshold, label, color in [(100, "Unhealthy", "#f97316"), (150, "Very Unhealthy", "#ef4444"), (50, "Good", "#22c55e")]:
        fig.add_hline(
            y=threshold,
            line=dict(color=color, width=1, dash="dot"),
            annotation_text=label,
            annotation_font=dict(color=color, size=10),
        )

    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        title=dict(text=f"7-Day AQI Trend — {city_display}", font=dict(color="#e2e8f0", size=12), x=0),
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="AQI"),
        showlegend=False,
    )
    return fig


# ─── Forecast Chart ───────────────────────────────────────────────────────────
def chart_forecast(forecasts: List[Dict], horizon: str = "24h") -> go.Figure:
    labels = [f["hour_label"] for f in forecasts]
    upi_vals = [f["upi"] for f in forecasts]
    aqi_vals = [f["aqi"] for f in forecasts]
    crowd_vals = [f.get("crowd", f.get("crowd_score", 0)) for f in forecasts]
    colors = [UPI_COLORS.get(f["category"], "#eab308") for f in forecasts]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08)

    fig.add_trace(go.Scatter(
        x=labels, y=upi_vals,
        mode="lines+markers",
        name="UPI Forecast",
        line=dict(color="#818cf8", width=2.5),
        marker=dict(color=colors, size=7, line=dict(color="#0f172a", width=1)),
        fill="tozeroy",
        fillcolor="rgba(129,140,248,0.1)",
    ), row=1, col=1)

    # Threshold band
    fig.add_hrect(y0=75, y1=100, fillcolor="rgba(239,68,68,0.08)", line_width=0, row=1, col=1)
    fig.add_hrect(y0=55, y1=75, fillcolor="rgba(249,115,22,0.06)", line_width=0, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=labels, y=crowd_vals,
        mode="lines",
        name="Crowd Forecast",
        line=dict(color="#f97316", width=1.8, dash="dot"),
    ), row=2, col=1)

    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        title=dict(text=f"Predictive Forecast — Next {horizon}", font=dict(color="#e2e8f0", size=13), x=0),
        font=dict(color="#94a3b8"),
        legend=dict(orientation="h", x=0, y=1.09, font=dict(size=11)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", showticklabels=False),
        xaxis2=dict(gridcolor="rgba(255,255,255,0.06)", tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="UPI"),
        yaxis2=dict(gridcolor="rgba(255,255,255,0.06)", title="Crowd %"),
    )
    return fig


# ─── City Comparison Chart ────────────────────────────────────────────────────
def chart_city_comparison(comparison_data: List[Dict]) -> go.Figure:
    cities = [d.get("display", "Unknown") for d in comparison_data]
    upi_vals = [d.get("upi", 0) for d in comparison_data]
    aqi_vals = [d.get("aqi", 0) for d in comparison_data]
    crowd_vals = [d.get("crowd_score", 0) for d in comparison_data]
    colors = [UPI_COLORS.get(d["category"], "#eab308") for d in comparison_data]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Urban Pulse Index", "AQI", "Crowd %"],
        horizontal_spacing=0.08,
    )

    fig.add_trace(go.Bar(
        x=cities, y=upi_vals,
        marker_color=colors,
        name="UPI", text=upi_vals, textposition="outside",
        textfont=dict(color="#e2e8f0", size=10),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=cities, y=aqi_vals,
        marker_color="#818cf8",
        name="AQI", text=[f"{v:.0f}" for v in aqi_vals], textposition="outside",
        textfont=dict(color="#e2e8f0", size=10),
        opacity=0.8,
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=cities, y=crowd_vals,
        marker_color="#f97316",
        name="Crowd", text=[f"{v:.0f}%" for v in crowd_vals], textposition="outside",
        textfont=dict(color="#e2e8f0", size=10),
        opacity=0.8,
    ), row=1, col=3)

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=50, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(color="#94a3b8"),
        showlegend=False,
        bargap=0.25,
    )
    for ax in ["xaxis", "xaxis2", "xaxis3"]:
        fig.update_layout(**{ax: dict(tickangle=30, tickfont=dict(size=9), gridcolor="rgba(255,255,255,0.04)")})
    for ax in ["yaxis", "yaxis2", "yaxis3"]:
        fig.update_layout(**{ax: dict(gridcolor="rgba(255,255,255,0.06)")})
    for ann in fig.layout.annotations:
        ann.update(font=dict(color="#94a3b8", size=12))
    return fig


# ─── UPI Gauge ────────────────────────────────────────────────────────────────
def chart_upi_gauge(upi: float, category: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=upi,
        number=dict(font=dict(color=color, size=36, family="Segoe UI"), suffix=""),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#475569", tickfont=dict(color="#475569")),
            bar=dict(color=color, thickness=0.28),
            bgcolor="rgba(15,23,42,0.8)",
            borderwidth=0,
            steps=[
                dict(range=[0, 30], color="rgba(34,197,94,0.12)"),
                dict(range=[30, 55], color="rgba(234,179,8,0.12)"),
                dict(range=[55, 75], color="rgba(249,115,22,0.12)"),
                dict(range=[75, 100], color="rgba(239,68,68,0.12)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=upi),
        ),
        title=dict(text=f"Urban Pulse Index<br><span style='color:{color};font-size:14px'>{category}</span>", font=dict(color="#e2e8f0", size=13)),
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
    )
    return fig
