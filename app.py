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
    "green": "#2d6a4f",
    "amber": "#b8860b",
    "red": "#c0392b",
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
        return "ALARM", "#c0392b"
    if pred_ttb < ML_ALARM_TTB or "amber" in statuses:
        return "WATCH", "#b8860b"
    return "GREEN", "#2d6a4f"


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
        f'fill="{fill}" stroke="#555" stroke-width="1.5"/>'
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
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#333" rx="4"/>'
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
        'style="width:100%;max-width:520px;background:#1a1a2e;border-radius:8px;">'
        "<defs><style>"
        "@keyframes habitat-pulse{0%,100%{stroke-opacity:1}50%{stroke-opacity:.25}}"
        ".mod-label{fill:#e0e0e0;font-family:monospace;font-size:11px;text-anchor:middle}"
        ".mod-value{fill:#fff;font-family:monospace;font-size:12px;font-weight:bold;text-anchor:middle}"
        ".pulse-outline{animation:habitat-pulse 1.4s ease-in-out infinite;stroke:#00d4ff;stroke-width:3;fill:none}"
        ".plan-title{fill:#888;font-family:monospace;font-size:13px}"
        "</style></defs>"
        '<text x="170" y="24" class="plan-title" text-anchor="middle">HABITAT PLAN (TOP-DOWN)</text>'
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

    add_trace(1, 1, "ppO2_kpa", "#4cc9f0")
    add_trace(1, 2, "ppCO2_kpa", "#f4a261")
    add_trace(2, 1, "battery_kwh", "#90be6d")
    add_trace(2, 2, "water_reserve_h", "#577590")

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
        fig.add_vline(x=ml_t, row=1, col=1, line=dict(color="#00d4ff", width=2, dash="dash"))
        annotations.append(dict(x=ml_t, y=1.08, xref="x", yref="paper", text="ML", showarrow=False, font=dict(color="#00d4ff", size=10)))
    if thr_t is not None:
        fig.add_vline(x=thr_t, row=1, col=1, line=dict(color="#ff6b6b", width=2, dash="dot"))
        annotations.append(dict(x=thr_t, y=1.02, xref="x", yref="paper", text="THR", showarrow=False, font=dict(color="#ff6b6b", size=10)))
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
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1a2e",
        height=520,
        margin=dict(t=40, b=30),
        annotations=annotations,
    )
    fig.update_xaxes(title_text="minutes", gridcolor="#333")
    fig.update_yaxes(gridcolor="#333")
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
    accent = "#ff3333" if critical else "#33ff99"
    bar_w = min(100, max(0, risk))
    return (
        '<div style="font-family:monospace;background:#050810;border:2px solid '
        f'{accent};border-radius:12px;padding:18px;color:#e8e8e8;max-width:640px;">'
        '<div style="color:#888;font-size:11px;letter-spacing:2px;">AEGIS HELMET HUD</div>'
        f'<div style="font-size:2rem;font-weight:bold;color:{accent};margin:8px 0;">{status}</div>'
        f'<div style="font-size:2.8rem;font-weight:bold;">{risk:.1f}<span style="font-size:1rem;color:#888;"> % RISK</span></div>'
        f'<div style="background:#222;border-radius:4px;height:10px;margin:12px 0;">'
        f'<div style="background:{accent};width:{bar_w}%;height:10px;border-radius:4px;"></div></div>'
        f'<div style="color:#666;font-size:11px;margin-bottom:10px;">threshold {meta["critical_threshold"]:.0f}%</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px;">'
        f'<div>HR <b>{frame["hr_bpm"]}</b> bpm</div>'
        f'<div>RESP <b>{frame["resp_rpm"]}</b> rpm</div>'
        f'<div>SpO₂ <b>{frame["spo2_pct"]}</b> %</div>'
        f'<div>Core <b>{frame["core_temp_c"]:.1f}</b> °C</div>'
        f'<div>CO₂ <b>{frame["helmet_co2_ppm"]}</b> ppm</div>'
        f'<div>Press <b>{frame["suit_press_kpa"]:.1f}</b> kPa</div>'
        f'<div>Ext <b>{frame["ext_temp_c"]:.1f}</b> °C</div>'
        f'<div>t = <b>{frame["t_min"]}</b> min</div>'
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
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1a2e",
        height=480,
        showlegend=False,
        margin=dict(t=40, b=30),
    )
    fig.update_xaxes(title_text="minutes", gridcolor="#333")
    fig.update_yaxes(gridcolor="#333")
    return fig


def render_eva_tab() -> None:
    try:
        demo = load_eva_demo()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    timeline = demo["timeline"]
    max_idx = len(timeline) - 1

    if "eva_tick_idx" not in st.session_state:
        st.session_state.eva_tick_idx = 0
    if "eva_playing" not in st.session_state:
        st.session_state.eva_playing = False

    st.markdown(
        f"**EVA {demo['eva_id']}** — fault: `{demo.get('fault', 'unknown')}`  "
        f"· same timeline drives the Blender helmet HUD"
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("▶ Play EVA", use_container_width=True, key="eva_play"):
            st.session_state.eva_playing = True
    with c2:
        if st.button("⏸ Pause EVA", use_container_width=True, key="eva_pause"):
            st.session_state.eva_playing = False
    with c3:
        if BLEND_PATH.exists():
            st.download_button(
                "⬇ Download Blender scene (.blend)",
                data=BLEND_PATH.read_bytes(),
                file_name=BLEND_PATH.name,
                mime="application/octet-stream",
                use_container_width=True,
            )
        else:
            st.caption("Blender scene not found in repo.")

    st.session_state.eva_tick_idx = st.slider(
        "EVA replay (index)",
        0,
        max_idx,
        st.session_state.eva_tick_idx,
        key="eva_slider",
    )
    frame = timeline[st.session_state.eva_tick_idx]

    col_hud, col_info = st.columns([1, 1])
    with col_hud:
        components.html(build_helmet_hud(frame, demo), height=320, scrolling=False)
    with col_info:
        st.markdown(
            """
            **Blender integration**

            1. Download `Astronaut Health Simulation.blend`
            2. Open in Blender 3.x+
            3. Point the HUD script at `predictions/eva_demo.json`
               (exported by `python train.py`)

            The web HUD above replays the same JSON timeline the helmet
            display uses during the EVA walkthrough.
            """
        )

    st.plotly_chart(build_eva_charts(timeline, st.session_state.eva_tick_idx), use_container_width=True)

    if st.session_state.eva_playing and st.session_state.eva_tick_idx < max_idx:
        st.session_state.eva_tick_idx += 1
        st.rerun()


def render_habitat_tab() -> None:
    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        st.error("Run `python generate.py` and `python train_habitat.py` first.")
        st.stop()

    # --- sidebar content moved inline for tab context ---
    scenario = st.selectbox(
        "Fault scenario",
        options=list(SCENARIO_LABELS.keys()),
        format_func=lambda k: SCENARIO_LABELS[k],
        key="habitat_scenario",
    )
    episode_id = DEMO_EPISODE_IDS[scenario]
    st.caption(f"Replay episode #{episode_id}")

    ep = prepare_episode(episode_id)
    max_idx = len(ep) - 1

    if "tick_idx" not in st.session_state:
        st.session_state.tick_idx = 0
    if st.session_state.get("last_scenario") != scenario:
        st.session_state.tick_idx = 0
        st.session_state.last_scenario = scenario
        st.session_state.playing = False

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("▶ Play", use_container_width=True, key="hab_play"):
            st.session_state.playing = True
    with col2:
        if st.button("⏸ Pause", use_container_width=True, key="hab_pause"):
            st.session_state.playing = False
    with col3:
        st.session_state.tick_idx = st.slider(
            "Replay time (index)",
            0,
            max_idx,
            st.session_state.tick_idx,
            key="hab_slider",
        )

    row = ep.iloc[st.session_state.tick_idx]
    st.markdown(f"**t = {int(row['t_min'])} min** ({row['t_min'] / 60:.1f} h)")

    pred_ttb = float(row["pred_ttb"])
    pred_fault = str(row["pred_fault"])
    pred_conf = float(row["pred_fault_conf"])
    ml_active = ml_alarm_active(row)
    fault_type = row["fault_type"]

    feat_ep = ep.dropna(subset=["pred_ttb"])
    ml_t = ml_alarm_time(feat_ep, feat_ep["pred_ttb"].values)
    thr_t = threshold_alarm_time(ep)

    status_label_hab, _ = overall_status(row, pred_ttb)
    status_class = {
        "GREEN": "status-green",
        "WATCH": "status-watch",
        "ALARM": "status-alarm",
    }[status_label_hab]

    st.markdown(
        f"""
        <div class="status-panel">
          <div class="{status_class}">{status_label_hab}</div>
          <div class="metric-mono">Predicted time to breach: <b>{pred_ttb:.0f} min</b></div>
          <div class="metric-mono">Predicted fault: <b>{pred_fault}</b>
            (confidence {pred_conf:.0%})</div>
          <div class="metric-mono">Actual scenario: <b>{fault_type}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    svg = build_habitat_svg(row, fault_type, pred_fault, ml_active)
    components.html(
        f'<div style="background:#0e1117;padding:4px 0;">{svg}</div>',
        height=400,
        scrolling=False,
    )

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

    st.markdown(
        "Talk to the **AEGIS mission assistant** about habitat status, EVA health, and alarms."
    )

    if api_key:
        if signed_url:
            st.caption("Authenticated agent session (signed URL from your API key).")
        else:
            st.warning(
                "API key is set but signed URL fetch failed. "
                "Falling back to public agent-id embed — disable auth on the agent, or check the key."
            )
    else:
        st.info(
            "No API key configured — using public widget embed. "
            "Ensure **authentication is disabled** on this agent in the ElevenLabs dashboard, "
            "or add your key to Streamlit secrets (see README)."
        )

    components.html(
        build_elevenlabs_widget_html(agent_id, signed_url),
        height=560,
        scrolling=True,
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def apply_dark_theme() -> None:
    st.markdown(
        """
        <style>
          .stApp { background-color: #0e1117; }
          [data-testid="stSidebar"] { background-color: #1a1a2e; }
          .status-panel {
            background: #1a1a2e;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
            font-family: monospace;
            margin-bottom: 1rem;
          }
          .status-green { color: #2d6a4f; font-size: 2.4rem; font-weight: bold; }
          .status-watch { color: #b8860b; font-size: 2.4rem; font-weight: bold; }
          .status-alarm { color: #c0392b; font-size: 2.4rem; font-weight: bold; }
          .metric-mono { font-family: monospace; color: #ccc; font-size: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="AEGIS — Mars Life Support",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_dark_theme()

    st.title("AEGIS — Mars Life Support & EVA Health")

    tab_habitat, tab_eva, tab_voice = st.tabs(
        ["Habitat monitor", "EVA suit HUD", "Voice assistant"]
    )

    with tab_habitat:
        render_habitat_tab()
    with tab_eva:
        render_eva_tab()
    with tab_voice:
        render_voice_assistant_tab()


if __name__ == "__main__":
    main()
