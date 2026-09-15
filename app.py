#!/usr/bin/env python3
"""Streamlit dashboard — replays cached habitat episodes with ML overlay."""

from __future__ import annotations

import json
import os
import pickle
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from sim import (
    BATTERY_CAPACITY_KWH,
    BREACH_PPCO2_KPA,
    BREACH_PPO2_KPA,
    BREACH_WATER_RESERVE_H,
    ECLSS_LOAD_KW,
)
from train import CRITICAL_THRESHOLD, status_label
from train_habitat import (
    CAUTION_BATTERY_FRAC,
    CAUTION_PPCO2,
    CAUTION_PPO2,
    CAUTION_WATER_RESERVE_H,
    ML_ALARM_TTB,
    PlaceholderFaultClassifier,  # noqa: F401 — required for pickle unpickling
    PlaceholderTTBRegressor,  # noqa: F401
    build_features,
    ml_alarm_time,
    threshold_alarm_time,
)

DATA_PATH = Path("data/episodes.parquet")
MODEL_PATH = Path("models/ttb.pkl")
EVA_DEMO_PATH = Path("predictions/eva_demo.json")
BLEND_PATH = Path("Astronaut Health Simulation.blend")

# ElevenLabs Conversational AI — agent ID is public; API key stays in secrets.
ELEVENLABS_AGENT_ID = "agent_5701m2k902y3fybs2ttq3mejjhdj"
ELEVENLABS_WIDGET_SCRIPT = "https://unpkg.com/@elevenlabs/convai-widget-embed"

SCENARIO_LABELS = {
    "nominal": "Nominal",
    "scrubber_degradation": "Scrubber degradation",
    "hull_leak": "Hull leak",
    "dust_storm": "Dust storm",
    "water_loop_fouling": "Water loop fouling",
}

# Demo episode picks (one per fault type; breaching where available).
DEMO_EPISODE_IDS = {
    "nominal": 11,
    "scrubber_degradation": 120,
    "hull_leak": 1,
    "dust_storm": 4,
    "water_loop_fouling": 0,
}

FAULT_TO_MODULE = {
    "scrubber_degradation": "eclss",
    "hull_leak": "airlock",
    "dust_storm": "power",
    "water_loop_fouling": "water",
}

STATUS_COLORS = {
    "green": "#14532d",
    "amber": "#92400e",
    "red": "#7f1d1d",
}

# Design tokens
THEME = {
    "bg": "#0a0e14",
    "surface": "#121820",
    "surface_alt": "#1a2230",
    "border": "#2a3544",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "accent": "#38bdf8",
    "green": "#22c55e",
    "amber": "#f59e0b",
    "red": "#ef4444",
}

# --- habitat schematic layout (top-down plan, fixed geometry) ---
MODULE_GEOM = {
    "airlock": {"x": 30, "y": 60, "w": 100, "h": 72, "rx": 10},
    "crew": {"x": 190, "y": 60, "w": 120, "h": 72, "rx": 10},
    "greenhouse": {"x": 30, "y": 170, "w": 100, "h": 72, "rx": 10},
    "eclss": {"x": 190, "y": 170, "w": 120, "h": 72, "rx": 10},
    "water": {"x": 30, "y": 280, "w": 100, "h": 72, "rx": 10},
    "power": {"x": 190, "y": 280, "w": 120, "h": 72, "rx": 10},
}

CORRIDORS = [
    (130, 88, 60, 16),
    (80, 132, 16, 38),
    (240, 132, 16, 38),
    (130, 206, 60, 16),
    (80, 242, 16, 38),
    (240, 242, 16, 38),
]


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model_bundle() -> dict:
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


@st.cache_data
def load_all_episodes() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


@st.cache_data
def prepare_episode(episode_id: int) -> pd.DataFrame:
    """Load episode rows and attach ML predictions."""
    df = load_all_episodes()
    ep = df[df["episode_id"] == episode_id].sort_values("t_min").reset_index(drop=True)
    if ep.empty:
        raise ValueError(f"Episode {episode_id} not found")

    bundle = load_model_bundle()
    ttb_model = bundle["ttb_model"]
    fault_model = bundle["fault_model"]
    feat_cols = bundle["feature_cols"]

    feat = build_features(ep)
    if feat.empty:
        ep["pred_ttb"] = 720.0
        ep["pred_fault"] = ep["fault_type"].iloc[0]
        ep["pred_fault_conf"] = 0.0
        return ep

    merged = feat.merge(
        ep[["t_min", "fault_type", "minutes_to_breach"]],
        on="t_min",
        how="left",
    )
    X = merged[feat_cols]
    merged["pred_ttb"] = ttb_model.predict(X)
    merged["pred_fault"] = fault_model.predict(X)
    proba = fault_model.predict_proba(X)
    merged["pred_fault_conf"] = proba.max(axis=1)

    ep = ep.merge(
        merged[["t_min", "pred_ttb", "pred_fault", "pred_fault_conf"]],
        on="t_min",
        how="left",
    )
    ep["pred_ttb"] = ep["pred_ttb"].fillna(720.0)
    ep["pred_fault"] = ep["pred_fault"].fillna(ep["fault_type"].iloc[0])
    ep["pred_fault_conf"] = ep["pred_fault_conf"].fillna(0.0)
    return ep


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


def _battery_breach(row: pd.Series) -> bool:
    return row["battery_kwh"] <= 0 and row["solar_input_kw"] < ECLSS_LOAD_KW


def gas_status(row: pd.Series) -> str:
    if row["ppO2_kpa"] < BREACH_PPO2_KPA or row["ppCO2_kpa"] > BREACH_PPCO2_KPA:
        return "red"
    if row["ppO2_kpa"] < CAUTION_PPO2 or row["ppCO2_kpa"] > CAUTION_PPCO2:
        return "amber"
    return "green"


def water_status(row: pd.Series) -> str:
    reserve = row["water_reserve_h"]
    if reserve < BREACH_WATER_RESERVE_H:
        return "red"
    if reserve < CAUTION_WATER_RESERVE_H:
        return "amber"
    return "green"


def battery_status(row: pd.Series) -> str:
    if _battery_breach(row):
        return "red"
    soc = row["battery_kwh"] / BATTERY_CAPACITY_KWH
    if soc < CAUTION_BATTERY_FRAC:
        return "amber"
    return "green"


def pressure_status(row: pd.Series) -> str:
    p = row["cabin_pressure_kpa"]
    if p < 65.0:
        return "red"
    if p < 68.0:
        return "amber"
    return "green"


def airlock_status(row: pd.Series, fault_type: str) -> str:
    if fault_type == "hull_leak":
        return pressure_status(row)
    return "green"


def module_statuses(row: pd.Series, fault_type: str) -> dict[str, str]:
    gas = gas_status(row)
    return {
        "airlock": airlock_status(row, fault_type),
        "crew": pressure_status(row),
        "greenhouse": gas,
        "eclss": gas,
        "water": water_status(row),
        "power": battery_status(row),
    }


def module_values(row: pd.Series, fault_type: str) -> dict[str, str]:
    soc = 100.0 * row["battery_kwh"] / BATTERY_CAPACITY_KWH
    airlock_val = (
        f"{row['cabin_pressure_kpa']:.1f} kPa"
        if fault_type == "hull_leak"
        else "SEAL OK"
    )
    return {
        "airlock": airlock_val,
        "crew": f"{row['cabin_pressure_kpa']:.1f} kPa",
        "greenhouse": f"{row['ppO2_kpa']:.1f} kPa O₂",
        "eclss": f"{row['ppCO2_kpa']:.2f} kPa CO₂",
        "water": f"{row['water_reserve_h']:.0f} h reserve",
        "power": f"{soc:.0f}% SOC",
    }


def overall_status(row: pd.Series, pred_ttb: float) -> tuple[str, str]:
    statuses = [
        gas_status(row),
        water_status(row),
        battery_status(row),
        pressure_status(row),
    ]
    if "red" in statuses or _battery_breach(row):
        return "ALARM", THEME["red"]
    if pred_ttb < ML_ALARM_TTB or "amber" in statuses:
        return "WATCH", THEME["amber"]
    return "GREEN", THEME["green"]


def ml_alarm_active(row: pd.Series) -> bool:
    return float(row["pred_ttb"]) < ML_ALARM_TTB


# ---------------------------------------------------------------------------
# Habitat SVG
# ---------------------------------------------------------------------------


def _module_svg(
    key: str,
    title: str,
    fill: str,
    value: str,
    pulse: bool,
) -> str:
    g = MODULE_GEOM[key]
    x, y, w, h, rx = g["x"], g["y"], g["w"], g["h"], g["rx"]
    pulse_rect = ""
    if pulse:
        pulse_rect = (
            f'<rect class="pulse-outline" x="{x - 3}" y="{y - 3}" '
            f'width="{w + 6}" height="{h + 6}" rx="{rx + 2}"/>'
        )
    return (
        f'<g id="mod-{key}">'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{THEME["border"]}" stroke-width="1.5"/>'
        f"{pulse_rect}"
        f'<text x="{x + w / 2}" y="{y + 22}" class="mod-label">{title}</text>'
        f'<text x="{x + w / 2}" y="{y + 48}" class="mod-value">{value}</text>'
        f"</g>"
    )


def build_habitat_svg(
    row: pd.Series,
    fault_type: str,
    pred_fault: str,
    ml_active: bool,
) -> str:
    statuses = module_statuses(row, fault_type)
    values = module_values(row, fault_type)
    pulse_module = FAULT_TO_MODULE.get(pred_fault) if ml_active else None

    titles = {
        "airlock": "Airlock",
        "crew": "Crew quarters",
        "greenhouse": "Greenhouse",
        "eclss": "ECLSS bay",
        "water": "Water recl.",
        "power": "Power / battery",
    }

    corridors = "".join(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{THEME["surface_alt"]}" rx="4"/>'
        for x, y, w, h in CORRIDORS
    )

    modules = ""
    for key, title in titles.items():
        fill = STATUS_COLORS[statuses[key]]
        pulse = ml_active and pulse_module == key
        modules += _module_svg(key, title, fill, values[key], pulse)

    return (
        '<svg class="habitat-svg" viewBox="0 0 340 370" '
        'xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:520px;background:{THEME["surface_alt"]};border-radius:12px;">'
        "<defs><style>"
        "@keyframes habitat-pulse{0%,100%{stroke-opacity:1}50%{stroke-opacity:.25}}"
        ".mod-label{fill:#cbd5e1;font-family:system-ui,sans-serif;font-size:11px;text-anchor:middle}"
        ".mod-value{fill:#f8fafc;font-family:system-ui,sans-serif;font-size:12px;font-weight:600;text-anchor:middle}"
        f".pulse-outline{{animation:habitat-pulse 1.4s ease-in-out infinite;stroke:{THEME['accent']};stroke-width:3;fill:none}}"
        ".plan-title{fill:#64748b;font-family:system-ui,sans-serif;font-size:12px;letter-spacing:0.08em}"
        "</style></defs>"
        '<text x="170" y="24" class="plan-title" text-anchor="middle">HABITAT PLAN · TOP-DOWN</text>'
        f"{corridors}{modules}"
        "</svg>"
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def build_charts(
    ep: pd.DataFrame,
    tick_idx: int,
    ml_t: int | None,
    thr_t: int | None,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("ppO₂ (kPa)", "ppCO₂ (kPa)", "Battery (kWh)", "Water reserve (h)"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    t = ep["t_min"]
    current_t = ep.iloc[tick_idx]["t_min"]

    def add_trace(row: int, col: int, data_col: str, color: str) -> None:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=ep[data_col],
                line=dict(color=color, width=2),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    add_trace(1, 1, "ppO2_kpa", THEME["accent"])
    add_trace(1, 2, "ppCO2_kpa", THEME["amber"])
    add_trace(2, 1, "battery_kwh", THEME["green"])
    add_trace(2, 2, "water_reserve_h", "#818cf8")

    # Caution / breach bands
    band_style = dict(type="rect", layer="below", line_width=0)
    fig.add_hrect(y0=0, y1=CAUTION_PPO2, row=1, col=1, fillcolor="rgba(192,57,43,0.15)", **band_style)
    fig.add_hrect(y0=0, y1=BREACH_PPO2_KPA, row=1, col=1, fillcolor="rgba(192,57,43,0.35)", **band_style)
    fig.add_hrect(y0=CAUTION_PPCO2, y1=2, row=1, col=2, fillcolor="rgba(192,57,43,0.15)", **band_style)
    fig.add_hrect(y0=BREACH_PPCO2_KPA, y1=2, row=1, col=2, fillcolor="rgba(192,57,43,0.35)", **band_style)
    fig.add_hrect(y0=0, y1=BATTERY_CAPACITY_KWH * CAUTION_BATTERY_FRAC, row=2, col=1, fillcolor="rgba(192,57,43,0.15)", **band_style)
    fig.add_hrect(y0=0, y1=CAUTION_WATER_RESERVE_H, row=2, col=2, fillcolor="rgba(192,57,43,0.15)", **band_style)
    fig.add_hrect(y0=0, y1=BREACH_WATER_RESERVE_H, row=2, col=2, fillcolor="rgba(192,57,43,0.35)", **band_style)

    # Playhead
    for row in (1, 2):
        for col in (1, 2):
            fig.add_vline(x=current_t, row=row, col=col, line=dict(color="#ffffff", width=1, dash="dot"))

    # Alarm markers on ppO2 panel
    annotations = []
    if ml_t is not None:
        fig.add_vline(x=ml_t, row=1, col=1, line=dict(color=THEME["accent"], width=2, dash="dash"))
        annotations.append(dict(x=ml_t, y=1.08, xref="x", yref="paper", text="ML", showarrow=False, font=dict(color=THEME["accent"], size=10)))
    if thr_t is not None:
        fig.add_vline(x=thr_t, row=1, col=1, line=dict(color=THEME["red"], width=2, dash="dot"))
        annotations.append(dict(x=thr_t, y=1.02, xref="x", yref="paper", text="THR", showarrow=False, font=dict(color=THEME["red"], size=10)))
    if ml_t is not None and thr_t is not None and thr_t > ml_t:
        gap_h = (thr_t - ml_t) / 60.0
        mid = (ml_t + thr_t) / 2
        annotations.append(dict(
            x=mid, y=1.14, xref="x", yref="paper",
            text=f"Δ {gap_h:.1f} h lead",
            showarrow=False,
            font=dict(color="#ccc", family="monospace", size=11),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=THEME["surface_alt"],
        height=480,
        margin=dict(t=48, b=36, l=48, r=24),
        annotations=annotations,
        font=dict(family="Inter, system-ui, sans-serif", color=THEME["muted"], size=11),
    )
    fig.update_xaxes(title_text="minutes", gridcolor=THEME["border"], zeroline=False)
    fig.update_yaxes(gridcolor=THEME["border"], zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# EVA suit HUD (train.py + Blender scene)
# ---------------------------------------------------------------------------


@st.cache_data
def load_eva_demo() -> dict:
    if not EVA_DEMO_PATH.exists():
        raise FileNotFoundError(
            f"{EVA_DEMO_PATH} not found. Run `python train.py` to export the HUD timeline."
        )
    return json.loads(EVA_DEMO_PATH.read_text(encoding="utf-8"))


def build_helmet_hud(frame: dict, meta: dict) -> str:
    risk = frame["predicted_pct"]
    critical = frame["is_critical"]
    status = status_label(risk)
    accent = THEME["red"] if critical else THEME["green"]
    bar_w = min(100, max(0, risk))
    bg, surf, border, text, muted = (
        THEME["bg"], THEME["surface"], THEME["border"], THEME["text"], THEME["muted"],
    )
    return (
        f'<style>'
        f'body{{margin:0;font-family:system-ui,sans-serif;background:{bg};color:{text};}}'
        f'.helmet-hud{{background:{surf};border:1px solid {border};border-left:4px solid {accent};'
        f'border-radius:12px;padding:20px 22px;}}'
        f'.helmet-label{{color:{muted};font-size:10px;letter-spacing:0.14em;text-transform:uppercase;}}'
        f'.helmet-status{{font-size:1.5rem;font-weight:700;margin:6px 0 2px;color:{accent};}}'
        f'.helmet-risk{{font-size:2.4rem;font-weight:700;line-height:1.1;}}'
        f'.helmet-risk span{{font-size:0.85rem;color:{muted};font-weight:400;margin-left:4px;}}'
        f'.helmet-bar-track{{background:{border};border-radius:6px;height:8px;margin:14px 0 8px;overflow:hidden;}}'
        f'.helmet-bar-fill{{height:100%;border-radius:6px;transition:width .2s;}}'
        f'.helmet-threshold{{color:{muted};font-size:11px;margin-bottom:14px;}}'
        f'.helmet-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px 16px;font-size:13px;}}'
        f'.helmet-grid span{{display:block;color:{muted};font-size:10px;text-transform:uppercase;letter-spacing:0.06em;}}'
        f'.helmet-grid b{{font-weight:600;color:{text};}}'
        f'</style>'
        f'<div class="helmet-hud">'
        '<div class="helmet-label">Helmet HUD · Live telemetry</div>'
        f'<div class="helmet-status">{status}</div>'
        f'<div class="helmet-risk">{risk:.1f}<span>% risk</span></div>'
        f'<div class="helmet-bar-track"><div class="helmet-bar-fill" style="width:{bar_w}%;background:{accent};"></div></div>'
        f'<div class="helmet-threshold">Critical threshold · {meta["critical_threshold"]:.0f}%</div>'
        '<div class="helmet-grid">'
        f'<div><span>HR</span><b>{frame["hr_bpm"]}</b> bpm</div>'
        f'<div><span>RESP</span><b>{frame["resp_rpm"]}</b> rpm</div>'
        f'<div><span>SpO₂</span><b>{frame["spo2_pct"]}</b> %</div>'
        f'<div><span>Core</span><b>{frame["core_temp_c"]:.1f}</b> °C</div>'
        f'<div><span>CO₂</span><b>{frame["helmet_co2_ppm"]}</b> ppm</div>'
        f'<div><span>Press</span><b>{frame["suit_press_kpa"]:.1f}</b> kPa</div>'
        f'<div><span>Ext</span><b>{frame["ext_temp_c"]:.1f}</b> °C</div>'
        f'<div><span>Time</span><b>{frame["t_min"]}</b> min</div>'
        "</div></div>"
    )


def build_eva_charts(timeline: list[dict], tick_idx: int) -> go.Figure:
    df = pd.DataFrame(timeline)
    current_t = df.iloc[tick_idx]["t_min"]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Health risk (%)", "Heart rate", "Helmet CO₂", "Suit pressure"),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(x=df["t_min"], y=df["predicted_pct"], line=dict(color="#ff6b6b", width=2)),
        row=1,
        col=1,
    )
    fig.add_hline(y=CRITICAL_THRESHOLD, row=1, col=1, line=dict(color="#888", dash="dash"))
    fig.add_trace(
        go.Scatter(x=df["t_min"], y=df["hr_bpm"], line=dict(color="#4cc9f0", width=2)),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=df["t_min"], y=df["helmet_co2_ppm"], line=dict(color="#f4a261", width=2)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["t_min"], y=df["suit_press_kpa"], line=dict(color="#90be6d", width=2)),
        row=2,
        col=2,
    )
    for row in (1, 2):
        for col in (1, 2):
            fig.add_vline(x=current_t, row=row, col=col, line=dict(color="#fff", width=1, dash="dot"))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=THEME["surface_alt"],
        height=440,
        showlegend=False,
        margin=dict(t=48, b=36, l=48, r=24),
        font=dict(family="Inter, system-ui, sans-serif", color=THEME["muted"], size=11),
    )
    fig.update_xaxes(title_text="minutes", gridcolor=THEME["border"], zeroline=False)
    fig.update_yaxes(gridcolor=THEME["border"], zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------


def render_page_header() -> None:
    st.markdown(
        """
        <div class="aegis-hero">
          <div class="aegis-hero-badge">MARS MISSION CONTROL</div>
          <h1 class="aegis-title">AEGIS</h1>
          <p class="aegis-subtitle">
            Adaptive Early-warning Gateway for Integrated Systems —
            habitat life support, EVA health, and voice-assisted monitoring.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title: str, subtitle: str = "") -> None:
    sub = f'<p class="section-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="section-head"><h2 class="section-title">{title}</h2>{sub}</div>',
        unsafe_allow_html=True,
    )


def render_replay_bar(
    *,
    playing_key: str,
    play_label: str,
    pause_label: str,
    slider_key: str,
    slider_label: str,
    max_idx: int,
    tick_idx: int,
    time_label: str,
) -> int:
    """Unified replay controls — returns updated tick index."""
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 1, 3, 1.2])
        with c1:
            if st.button(play_label, use_container_width=True, type="primary", key=f"{playing_key}_play"):
                st.session_state[playing_key] = True
        with c2:
            if st.button(pause_label, use_container_width=True, key=f"{playing_key}_pause"):
                st.session_state[playing_key] = False
        with c3:
            tick_idx = st.slider(slider_label, 0, max_idx, tick_idx, key=slider_key)
        with c4:
            st.markdown(
                f'<div class="time-pill"><span class="time-label">ELAPSED</span>'
                f'<span class="time-value">{time_label}</span></div>',
                unsafe_allow_html=True,
            )
    return tick_idx


def render_status_banner(status: str) -> None:
    css_class = {
        "GREEN": "banner-green",
        "NOMINAL": "banner-green",
        "WATCH": "banner-watch",
        "ALARM": "banner-alarm",
        "CRITICAL": "banner-alarm",
    }.get(status, "banner-green")
    st.markdown(
        f'<div class="status-banner {css_class}">'
        f'<span class="status-dot"></span><span class="status-text">{status}</span></div>',
        unsafe_allow_html=True,
    )


def render_metric_cards(cards: list[tuple[str, str, str]]) -> None:
    """Render a row of metric cards: (label, value, hint)."""
    cols = st.columns(len(cards))
    for col, (label, value, hint) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-hint">{hint}</div></div>',
                unsafe_allow_html=True,
            )


def render_eva_tab() -> None:
    try:
        demo = load_eva_demo()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    render_section(
        "EVA Suit HUD",
        f"{demo['eva_id']} · fault: {demo.get('fault', 'unknown')} · synced with Blender scene",
    )

    timeline = demo["timeline"]
    max_idx = len(timeline) - 1

    if "eva_tick_idx" not in st.session_state:
        st.session_state.eva_tick_idx = 0
    if "eva_playing" not in st.session_state:
        st.session_state.eva_playing = False

    frame = timeline[st.session_state.eva_tick_idx]
    time_label = f"{frame['t_min']} min"
    st.session_state.eva_tick_idx = render_replay_bar(
        playing_key="eva_playing",
        play_label="▶ Play",
        pause_label="⏸ Pause",
        slider_key="eva_slider",
        slider_label="Timeline",
        max_idx=max_idx,
        tick_idx=st.session_state.eva_tick_idx,
        time_label=time_label,
    )
    frame = timeline[st.session_state.eva_tick_idx]

    render_status_banner("CRITICAL" if frame["is_critical"] else "NOMINAL")
    render_metric_cards([
        ("Risk score", f"{frame['predicted_pct']:.1f}%", "ML predicted criticality"),
        ("Heart rate", f"{frame['hr_bpm']} bpm", "Suit telemetry"),
        ("SpO₂", f"{frame['spo2_pct']}%", "Blood oxygen"),
        ("Suit pressure", f"{frame['suit_press_kpa']:.1f} kPa", "Pressure integrity"),
    ])

    col_hud, col_dl = st.columns([1.4, 1])
    with col_hud:
        with st.container(border=True):
            components.html(build_helmet_hud(frame, demo), height=340, scrolling=False)
    with col_dl:
        with st.container(border=True):
            if BLEND_PATH.exists():
                st.download_button(
                    "Download Blender scene",
                    data=BLEND_PATH.read_bytes(),
                    file_name=BLEND_PATH.name,
                    mime="application/octet-stream",
                    use_container_width=True,
                    type="primary",
                )
            with st.expander("Blender setup", expanded=False):
                st.markdown(
                    "1. Open **Astronaut Health Simulation.blend** in Blender 3.x+\n\n"
                    "2. Point the HUD script at `predictions/eva_demo.json`\n\n"
                    "3. This dashboard replays the same JSON timeline"
                )

    with st.container(border=True):
        st.plotly_chart(build_eva_charts(timeline, st.session_state.eva_tick_idx), use_container_width=True)

    if st.session_state.eva_playing and st.session_state.eva_tick_idx < max_idx:
        st.session_state.eva_tick_idx += 1
        st.rerun()


def render_habitat_tab() -> None:
    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        st.error("Run `python generate.py` and `python train_habitat.py` first.")
        st.stop()

    render_section("Habitat Life Support", "Replay cached episodes with ML early-warning overlay")

    scenario = st.selectbox(
        "Fault scenario",
        options=list(SCENARIO_LABELS.keys()),
        format_func=lambda k: SCENARIO_LABELS[k],
        key="habitat_scenario",
    )
    episode_id = DEMO_EPISODE_IDS[scenario]
    ep = prepare_episode(episode_id)
    max_idx = len(ep) - 1

    if "tick_idx" not in st.session_state:
        st.session_state.tick_idx = 0
    if st.session_state.get("last_scenario") != scenario:
        st.session_state.tick_idx = 0
        st.session_state.last_scenario = scenario
        st.session_state.playing = False

    row = ep.iloc[st.session_state.tick_idx]
    time_label = f"{int(row['t_min'])} min · {row['t_min'] / 60:.1f} h"
    st.session_state.tick_idx = render_replay_bar(
        playing_key="playing",
        play_label="▶ Play",
        pause_label="⏸ Pause",
        slider_key="hab_slider",
        slider_label=f"Timeline · episode #{episode_id}",
        max_idx=max_idx,
        tick_idx=st.session_state.tick_idx,
        time_label=time_label,
    )

    row = ep.iloc[st.session_state.tick_idx]
    pred_ttb = float(row["pred_ttb"])
    pred_fault = str(row["pred_fault"])
    pred_conf = float(row["pred_fault_conf"])
    ml_active = ml_alarm_active(row)
    fault_type = row["fault_type"]

    feat_ep = ep.dropna(subset=["pred_ttb"])
    ml_t = ml_alarm_time(feat_ep, feat_ep["pred_ttb"].values)
    thr_t = threshold_alarm_time(ep)
    status_label_hab, _ = overall_status(row, pred_ttb)

    render_status_banner(status_label_hab)
    render_metric_cards([
        ("Time to breach", f"{pred_ttb:.0f} min", "ML prediction"),
        ("Predicted fault", pred_fault.replace("_", " "), f"{pred_conf:.0%} confidence"),
        ("Actual scenario", fault_type.replace("_", " "), "Episode ground truth"),
        ("ML alarm", "ACTIVE" if ml_active else "OFF", f"Threshold &lt; {ML_ALARM_TTB} min"),
    ])

    col_map, col_alarms = st.columns([1.1, 1])
    with col_map:
        with st.container(border=True):
            svg = build_habitat_svg(row, fault_type, pred_fault, ml_active)
            components.html(
                f'<div style="background:{THEME["surface"]};padding:8px 0;">{svg}</div>',
                height=400,
                scrolling=False,
            )
    with col_alarms:
        with st.container(border=True):
            st.markdown("**Alarm timeline**")
            if ml_t is not None:
                st.markdown(
                    f'<div class="alarm-row ml"><span>ML alarm</span><b>{ml_t} min</b></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("ML alarm: not triggered")
            if thr_t is not None:
                st.markdown(
                    f'<div class="alarm-row thr"><span>Threshold</span><b>{thr_t} min</b></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Threshold alarm: not triggered")
            if ml_t is not None and thr_t is not None and thr_t > ml_t:
                lead_h = (thr_t - ml_t) / 60.0
                st.success(f"ML lead time: **{lead_h:.1f} hours**")
            st.markdown(
                '<div class="legend">'
                '<span class="leg-item"><i class="dot green"></i> Nominal</span>'
                '<span class="leg-item"><i class="dot amber"></i> Caution</span>'
                '<span class="leg-item"><i class="dot red"></i> Breach</span>'
                "</div>",
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        st.plotly_chart(build_charts(ep, st.session_state.tick_idx, ml_t, thr_t), use_container_width=True)

    if st.session_state.get("playing") and st.session_state.tick_idx < max_idx:
        st.session_state.tick_idx += 1
        st.rerun()


# ---------------------------------------------------------------------------
# ElevenLabs voice assistant
# ---------------------------------------------------------------------------


def get_elevenlabs_config() -> tuple[str, str | None]:
    """
    Returns (agent_id, api_key_or_none).

    API key is read from Streamlit secrets or ELEVENLABS_API_KEY env var.
    Never hard-code the real key in this file.
    """
    agent_id = ELEVENLABS_AGENT_ID
    api_key: str | None = None

    try:
        if "elevenlabs" in st.secrets:
            agent_id = st.secrets.elevenlabs.get("agent_id", agent_id)
            api_key = st.secrets.elevenlabs.get("api_key")
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("ELEVENLABS_API_KEY")

    return agent_id, api_key


@st.cache_data(ttl=300, show_spinner=False)
def fetch_elevenlabs_signed_url(api_key: str, agent_id: str) -> str | None:
    """For agents with auth enabled — signed URL is generated server-side."""
    url = (
        "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url"
        f"?agent_id={agent_id}"
    )
    req = urllib.request.Request(url, headers={"xi-api-key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode())
            return payload.get("signed_url")
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError):
        return None


def build_elevenlabs_widget_html(agent_id: str, signed_url: str | None) -> str:
    if signed_url:
        auth_attr = f'signed-url="{signed_url}"'
    else:
        auth_attr = f'agent-id="{agent_id}"'
    return (
        f'<div style="min-height:520px;background:#0e1117;padding:12px;">'
        f'<elevenlabs-convai {auth_attr} dismissible="true" variant="expanded"></elevenlabs-convai>'
        f'<script src="{ELEVENLABS_WIDGET_SCRIPT}" async type="text/javascript"></script>'
        "</div>"
    )


def render_voice_assistant_tab() -> None:
    agent_id, api_key = get_elevenlabs_config()
    signed_url = fetch_elevenlabs_signed_url(api_key, agent_id) if api_key else None

    render_section(
        "Voice Assistant",
        "Ask about habitat status, EVA health, and alarm timelines",
    )

    if api_key:
        if signed_url:
            st.caption("Authenticated session via signed URL.")
        else:
            st.warning(
                "API key is set but signed URL fetch failed. "
                "Falling back to public embed — disable auth on the agent, or check the key."
            )
    else:
        st.info(
            "No API key configured — using public widget. "
            "Disable authentication on the agent in ElevenLabs, or add your key to Streamlit secrets."
        )

    with st.container(border=True):
        components.html(
            build_elevenlabs_widget_html(agent_id, signed_url),
            height=560,
            scrolling=True,
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def apply_dark_theme() -> None:
    t = THEME
    st.markdown(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

          :root {{
            --bg: {t["bg"]};
            --surface: {t["surface"]};
            --surface-alt: {t["surface_alt"]};
            --border: {t["border"]};
            --text: {t["text"]};
            --muted: {t["muted"]};
            --accent: {t["accent"]};
            --green: {t["green"]};
            --amber: {t["amber"]};
            --red: {t["red"]};
          }}

          .stApp {{
            background: radial-gradient(ellipse 120% 80% at 50% -20%, #1a2744 0%, var(--bg) 55%);
            font-family: 'Inter', system-ui, sans-serif;
          }}

          .block-container {{
            padding-top: 1.5rem;
            max-width: 1280px;
          }}

          h1, h2, h3, .stMarkdown p {{
            font-family: 'Inter', system-ui, sans-serif;
          }}

          /* Tabs */
          .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 6px;
          }}
          .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 500;
            color: var(--muted);
          }}
          .stTabs [aria-selected="true"] {{
            background: var(--surface-alt) !important;
            color: var(--text) !important;
            box-shadow: inset 0 0 0 1px var(--border);
          }}

          /* Buttons */
          .stButton > button {{
            border-radius: 8px;
            font-weight: 500;
            transition: transform 0.12s ease, box-shadow 0.12s ease;
          }}
          .stButton > button:hover {{
            transform: translateY(-1px);
          }}
          .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, #0ea5e9, #0284c7);
            border: none;
          }}

          /* Sliders & inputs */
          .stSlider label, .stSelectbox label {{
            font-size: 0.85rem;
            color: var(--muted) !important;
          }}

          /* Hero */
          .aegis-hero {{
            margin-bottom: 1.75rem;
            padding-bottom: 1.25rem;
            border-bottom: 1px solid var(--border);
          }}
          .aegis-hero-badge {{
            display: inline-block;
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 0.16em;
            color: var(--accent);
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 999px;
            padding: 4px 12px;
            margin-bottom: 0.6rem;
          }}
          .aegis-title {{
            font-size: 2.25rem;
            font-weight: 700;
            color: var(--text);
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.02em;
          }}
          .aegis-subtitle {{
            color: var(--muted);
            font-size: 0.95rem;
            max-width: 640px;
            margin: 0;
            line-height: 1.55;
          }}

          /* Sections */
          .section-head {{ margin: 0.5rem 0 1rem 0; }}
          .section-title {{
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--text);
            margin: 0;
          }}
          .section-sub {{
            color: var(--muted);
            font-size: 0.85rem;
            margin: 0.25rem 0 0 0;
          }}

          /* Bordered containers */
          [data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--surface);
            border-color: var(--border) !important;
            border-radius: 12px;
            padding: 0.65rem 0.85rem;
            margin-bottom: 0.75rem;
          }}

          /* Metric cards */
          .metric-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
          }}

          /* Time pill */
          .time-pill {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            justify-content: center;
            height: 100%;
            padding: 6px 0;
          }}
          .time-label {{
            font-size: 0.6rem;
            letter-spacing: 0.12em;
            color: var(--muted);
          }}
          .time-value {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--text);
            font-variant-numeric: tabular-nums;
          }}

          /* Status banner */
          .status-banner {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0.75rem 1.1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.04em;
          }}
          .banner-green {{
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.35);
            color: #86efac;
          }}
          .banner-watch {{
            background: rgba(245, 158, 11, 0.12);
            border: 1px solid rgba(245, 158, 11, 0.35);
            color: #fcd34d;
          }}
          .banner-alarm {{
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.35);
            color: #fca5a5;
          }}
          .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
            box-shadow: 0 0 8px currentColor;
          }}

          /* Metric cards */
          .metric-card {{
            padding: 0.9rem 1rem;
            margin-bottom: 1rem;
            transition: border-color 0.15s ease;
          }}
          .metric-card:hover {{ border-color: rgba(56, 189, 248, 0.35); }}
          .metric-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 0.35rem;
          }}
          .metric-value {{
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text);
            font-variant-numeric: tabular-nums;
            line-height: 1.2;
          }}
          .metric-hint {{
            font-size: 0.72rem;
            color: var(--muted);
            margin-top: 0.3rem;
          }}

          /* Alarm panel */
          .alarm-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.55rem 0.75rem;
            border-radius: 8px;
            margin: 0.4rem 0;
            font-size: 0.88rem;
          }}
          .alarm-row span {{ color: var(--muted); }}
          .alarm-row b {{ font-variant-numeric: tabular-nums; }}
          .alarm-row.ml {{
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.25);
          }}
          .alarm-row.ml b {{ color: var(--accent); }}
          .alarm-row.thr {{
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.25);
          }}
          .alarm-row.thr b {{ color: var(--red); }}
          .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border);
            font-size: 0.78rem;
            color: var(--muted);
          }}
          .leg-item {{ display: flex; align-items: center; gap: 6px; }}
          .dot {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 3px;
          }}
          .dot.green {{ background: #14532d; }}
          .dot.amber {{ background: #92400e; }}
          .dot.red {{ background: #7f1d1d; }}

          /* Hide Streamlit chrome noise */
          #MainMenu {{ visibility: hidden; }}
          footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="AEGIS — Mars Life Support",
        page_icon="🛰",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_dark_theme()
    render_page_header()

    tab_habitat, tab_eva, tab_voice = st.tabs(
        ["🏠 Habitat", "🧑‍🚀 EVA Suit", "🎙 Voice"]
    )

    with tab_habitat:
        render_habitat_tab()
    with tab_eva:
        render_eva_tab()
    with tab_voice:
        render_voice_assistant_tab()


if __name__ == "__main__":
    main()
