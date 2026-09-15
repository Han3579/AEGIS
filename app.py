#!/usr/bin/env python3
"""Streamlit dashboard — replays cached habitat episodes with ML overlay."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sim import (
    BATTERY_CAPACITY_KWH,
    BREACH_PPCO2_KPA,
    BREACH_PPO2_KPA,
    BREACH_WATER_RESERVE_H,
    ECLSS_LOAD_KW,
)
from train import (
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
    return f"""
    <g id="mod-{key}">
      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"
            fill="{fill}" stroke="#555" stroke-width="1.5"/>
      {pulse_rect}
      <text x="{x + w / 2}" y="{y + 22}" class="mod-label">{title}</text>
      <text x="{x + w / 2}" y="{y + 48}" class="mod-value">{value}</text>
    </g>"""


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

    return f"""
<style>
  @keyframes habitat-pulse {{
    0%, 100% {{ stroke-opacity: 1; }}
    50% {{ stroke-opacity: 0.25; }}
  }}
  .habitat-svg .mod-label {{
    fill: #e0e0e0;
    font-family: monospace;
    font-size: 11px;
    text-anchor: middle;
  }}
  .habitat-svg .mod-value {{
    fill: #ffffff;
    font-family: monospace;
    font-size: 12px;
    font-weight: bold;
    text-anchor: middle;
  }}
  .habitat-svg .pulse-outline {{
    animation: habitat-pulse 1.4s ease-in-out infinite;
    stroke: #00d4ff;
    stroke-width: 3;
    fill: none;
  }}
  .habitat-svg .plan-title {{
    fill: #888;
    font-family: monospace;
    font-size: 13px;
  }}
</style>
<svg class="habitat-svg" viewBox="0 0 340 370" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:520px;background:#1a1a2e;border-radius:8px;">
  <text x="170" y="24" class="plan-title" text-anchor="middle">HABITAT PLAN (TOP-DOWN)</text>
  {corridors}
  {modules}
</svg>
"""


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
    st.set_page_config(page_title="AEGIS Habitat Monitor", layout="wide", initial_sidebar_state="expanded")
    apply_dark_theme()

    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        st.error("Run `python generate.py` and `python train.py` first.")
        st.stop()

    st.title("AEGIS — Mars Habitat Life Support Monitor")

    # --- sidebar ---
    with st.sidebar:
        st.header("Scenario")
        scenario = st.selectbox(
            "Fault scenario",
            options=list(SCENARIO_LABELS.keys()),
            format_func=lambda k: SCENARIO_LABELS[k],
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

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Play", use_container_width=True):
                st.session_state.playing = True
        with col2:
            if st.button("⏸ Pause", use_container_width=True):
                st.session_state.playing = False

        st.session_state.tick_idx = st.slider(
            "Replay time (index)",
            0,
            max_idx,
            st.session_state.tick_idx,
        )

        row = ep.iloc[st.session_state.tick_idx]
        st.markdown(f"**t = {int(row['t_min'])} min** ({row['t_min'] / 60:.1f} h)")

    # --- predictions & alarms ---
    row = ep.iloc[st.session_state.tick_idx]
    fault_type = row["fault_type"]
    pred_ttb = float(row["pred_ttb"])
    pred_fault = str(row["pred_fault"])
    pred_conf = float(row["pred_fault_conf"])
    ml_active = ml_alarm_active(row)

    feat_ep = ep.dropna(subset=["pred_ttb"])
    pred_ttb_arr = feat_ep["pred_ttb"].values
    ml_t = ml_alarm_time(feat_ep, pred_ttb_arr)
    thr_t = threshold_alarm_time(ep)

    status_label, status_color = overall_status(row, pred_ttb)
    status_class = {
        "GREEN": "status-green",
        "WATCH": "status-watch",
        "ALARM": "status-alarm",
    }[status_label]

    # --- status panel ---
    st.markdown(
        f"""
        <div class="status-panel">
          <div class="{status_class}">{status_label}</div>
          <div class="metric-mono">Predicted time to breach: <b>{pred_ttb:.0f} min</b></div>
          <div class="metric-mono">Predicted fault: <b>{pred_fault}</b>
            (confidence {pred_conf:.0%})</div>
          <div class="metric-mono">Actual scenario: <b>{fault_type}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- habitat schematic ---
    svg = build_habitat_svg(row, fault_type, pred_fault, ml_active)
    st.markdown(svg, unsafe_allow_html=True)

    # --- charts ---
    fig = build_charts(ep, st.session_state.tick_idx, ml_t, thr_t)
    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.get("playing") and st.session_state.tick_idx < max_idx:
        st.session_state.tick_idx += 1
        st.rerun()


if __name__ == "__main__":
    main()
