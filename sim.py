"""Headless Mars habitat life-support simulation. No UI imports."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# --- crew & habitat ---
CREW = 4
VOLUME_M3 = 400.0
TARGET_PRESSURE_KPA = 70.0
TARGET_PPO2_KPA = 20.0

# --- time ---
SOL_MINUTES = 24.66 * 60  # 1479.6
EPISODE_MINUTES = 48 * 60  # 2880

# --- metabolic rates (per crew per sol) ---
O2_KG_PER_CREW_SOL = 0.84
CO2_KG_PER_CREW_SOL = 1.0
WATER_KG_PER_CREW_SOL = 3.6
WATER_RECOVERY_NOMINAL = 0.93

# --- power ---
SOLAR_AREA_M2 = 120.0
PEAK_IRRADIANCE_W_M2 = 590.0
PANEL_EFF = 0.22
PEAK_SOLAR_KW = SOLAR_AREA_M2 * PEAK_IRRADIANCE_W_M2 * PANEL_EFF / 1000.0
BATTERY_CAPACITY_KWH = 60.0
BASELINE_LOAD_KW = 4.5
ECLSS_LOAD_KW = 2.0

# --- gas constants ---
R = 8.314
T_K = 293.0
M_O2 = 0.032
M_CO2 = 0.044

# --- per-minute crew totals ---
O2_USE_KG_MIN = O2_KG_PER_CREW_SOL * CREW / SOL_MINUTES
CO2_PROD_KG_MIN = CO2_KG_PER_CREW_SOL * CREW / SOL_MINUTES
WATER_USE_KG_MIN = WATER_KG_PER_CREW_SOL * CREW / SOL_MINUTES

# --- breach thresholds ---
BREACH_PPO2_KPA = 15.5
BREACH_PPCO2_KPA = 1.0
BREACH_WATER_RESERVE_H = 12.0

# --- fault timing ---
ONSET_MIN = 2 * 60
ONSET_MAX = 30 * 60

FAULT_TYPES = [
    "nominal",
    "scrubber_degradation",
    "hull_leak",
    "dust_storm",
    "water_loop_fouling",
]

# Gaussian sensor noise (1-sigma) on logged telemetry
SENSOR_NOISE: dict[str, float] = {
    "ppO2_kpa": 0.05,
    "ppCO2_kpa": 0.02,
    "cabin_pressure_kpa": 0.1,
    "cabin_humidity_pct": 0.5,
    "potable_water_kg": 0.2,
    "grey_water_kg": 0.2,
    "battery_kwh": 0.05,
    "solar_input_kw": 0.1,
    "scrubber_efficiency": 0.005,
    "electrolyser_o2_kg_h": 0.01,
    "eclss_power_kw": 0.05,
}


def _pp_from_mass(m_kg: float, molar_mass: float) -> float:
    return (m_kg / molar_mass) * R * T_K / VOLUME_M3 / 1000.0


def _mass_from_pp(pp_kpa: float, molar_mass: float) -> float:
    return pp_kpa * 1000.0 * VOLUME_M3 * molar_mass / (R * T_K)


def _solar_kw(t_min: int) -> float:
    phase = (t_min % SOL_MINUTES) / SOL_MINUTES
    if phase <= 0.5:
        return PEAK_SOLAR_KW * np.sin(np.pi * phase / 0.5)
    return 0.0


def _water_reserve_hours(potable_kg: float) -> float:
    use_kg_h = WATER_USE_KG_MIN * 60.0
    if use_kg_h <= 0:
        return float("inf")
    return potable_kg / use_kg_h


def _check_breach(
    ppO2: float,
    ppCO2: float,
    potable_kg: float,
    battery_kwh: float,
    solar_kw: float,
) -> bool:
    if ppO2 < BREACH_PPO2_KPA:
        return True
    if ppCO2 > BREACH_PPCO2_KPA:
        return True
    if _water_reserve_hours(potable_kg) < BREACH_WATER_RESERVE_H:
        return True
    # Life support unmet: empty battery and insufficient solar for critical ECLSS load.
    if battery_kwh <= 0.0 and solar_kw < ECLSS_LOAD_KW:
        return True
    return False


def _sample_fault_params(fault_type: str, rng: np.random.Generator) -> dict[str, float]:
    if fault_type == "scrubber_degradation":
        return {"decay_pct_per_hour": rng.uniform(0.5, 3.0)}
    if fault_type == "hull_leak":
        return {"leak_kpa_per_hour": rng.uniform(0.02, 0.2)}
    if fault_type == "dust_storm":
        return {
            "target_multiplier": rng.uniform(0.1, 0.4),
            "ramp_hours": rng.uniform(2.0, 6.0),
        }
    if fault_type == "water_loop_fouling":
        return {"recovery_floor": 0.6, "decay_pct_per_hour": rng.uniform(1.0, 4.0)}
    return {}


def _dust_multiplier(
    t_min: int, onset: int, ramp_hours: float, target: float
) -> float:
    if t_min < onset:
        return 1.0
    elapsed_h = (t_min - onset) / 60.0
    if elapsed_h >= ramp_hours:
        return target
    frac = elapsed_h / ramp_hours
    return 1.0 - frac * (1.0 - target)


def _initial_state() -> dict[str, float]:
    ppO2 = TARGET_PPO2_KPA
    ppCO2 = 0.45
    cabin_pressure = TARGET_PRESSURE_KPA
    o2_mass = _mass_from_pp(ppO2, M_O2)
    co2_mass = _mass_from_pp(ppCO2, M_CO2)

    return {
        "ppO2": ppO2,
        "ppCO2": ppCO2,
        "cabin_pressure": cabin_pressure,
        "o2_mass": o2_mass,
        "co2_mass": co2_mass,
        "humidity": 45.0,
        "potable_kg": 80.0,
        "grey_kg": 25.0,
        "battery_kwh": BATTERY_CAPACITY_KWH,
        "scrubber_eff": 1.0,
        "recovery_rate": WATER_RECOVERY_NOMINAL,
    }


def _apply_faults(
    fault_type: str,
    fault_params: dict[str, float],
    t_min: int,
    onset: int,
    state: dict[str, float],
) -> tuple[float, float]:
    """Return (solar_multiplier, electrolyser_factor). Mutates state in place."""
    solar_mult = 1.0
    electrolyser_factor = 1.0

    if t_min < onset:
        return solar_mult, electrolyser_factor

    if fault_type == "scrubber_degradation":
        # Efficiency drops by N percentage points per hour (e.g. 2%/h -> 0.98 -> 0.96 ...).
        decay = fault_params["decay_pct_per_hour"] / 100.0 / 60.0
        state["scrubber_eff"] = max(0.0, state["scrubber_eff"] - decay)

    elif fault_type == "hull_leak":
        leak_per_min = fault_params["leak_kpa_per_hour"] / 60.0
        old_p = state["cabin_pressure"]
        new_p = max(10.0, old_p - leak_per_min)
        ratio = new_p / old_p if old_p > 0 else 1.0
        state["cabin_pressure"] = new_p
        state["ppO2"] *= ratio
        state["ppCO2"] *= ratio
        state["o2_mass"] = _mass_from_pp(state["ppO2"], M_O2)
        state["co2_mass"] = _mass_from_pp(state["ppCO2"], M_CO2)

    elif fault_type == "dust_storm":
        solar_mult = _dust_multiplier(
            t_min,
            onset,
            fault_params["ramp_hours"],
            fault_params["target_multiplier"],
        )

    elif fault_type == "water_loop_fouling":
        decay = fault_params["decay_pct_per_hour"] / 100.0 / 60.0
        floor = fault_params["recovery_floor"]
        state["recovery_rate"] = max(
            floor, state["recovery_rate"] * (1.0 - decay)
        )
        # electrolyser starves when potable drops below ~36 h reserve
        reserve_h = _water_reserve_hours(state["potable_kg"])
        electrolyser_factor = min(1.0, reserve_h / 36.0)

    return solar_mult, electrolyser_factor


def _physics_tick(
    state: dict[str, float],
    solar_mult: float,
    electrolyser_factor: float,
) -> dict[str, float]:
    """Advance physics one minute. Returns telemetry dict (clean values)."""
    solar_kw = _solar_kw(state["t_min"]) * solar_mult

    # --- power (non-critical load sheds when SOC is low) ---
    soc = state["battery_kwh"] / BATTERY_CAPACITY_KWH
    non_critical = BASELINE_LOAD_KW - ECLSS_LOAD_KW
    if soc >= 0.25:
        load_kw = BASELINE_LOAD_KW
    else:
        load_kw = ECLSS_LOAD_KW + non_critical * (soc / 0.25)

    net_kw = solar_kw - load_kw
    battery = state["battery_kwh"] + net_kw / 60.0
    battery = min(BATTERY_CAPACITY_KWH, max(0.0, battery))
    state["battery_kwh"] = battery

    # --- water ---
    potable = state["potable_kg"]
    grey = state["grey_kg"]
    potable -= WATER_USE_KG_MIN
    grey += WATER_USE_KG_MIN
    recycled = min(grey, WATER_USE_KG_MIN * 3.0) * state["recovery_rate"]
    grey -= recycled / state["recovery_rate"] if state["recovery_rate"] > 0 else 0.0
    grey = max(0.0, grey)
    potable += recycled
    state["potable_kg"] = max(0.0, potable)
    state["grey_kg"] = grey

    # --- atmosphere ---
    o2_mass = state["o2_mass"]
    co2_mass = state["co2_mass"]

    o2_mass -= O2_USE_KG_MIN
    co2_mass += CO2_PROD_KG_MIN

    # electrolyser replenishes O2 (nominal matches consumption)
    o2_prod_kg_min = O2_USE_KG_MIN * electrolyser_factor
    o2_mass += o2_prod_kg_min

    # scrubber removes CO2 (nonlinear: degraded beds lose capacity faster)
    eff = state["scrubber_eff"]
    co2_removed = CO2_PROD_KG_MIN * (eff ** 2)
    co2_mass = max(0.0, co2_mass - co2_removed)

    state["o2_mass"] = max(0.0, o2_mass)
    state["co2_mass"] = max(0.0, co2_mass)
    state["ppO2"] = _pp_from_mass(state["o2_mass"], M_O2)
    state["ppCO2"] = _pp_from_mass(state["co2_mass"], M_CO2)

    electrolyser_o2_kg_h = o2_prod_kg_min * 60.0

    return {
        "ppO2_kpa": state["ppO2"],
        "ppCO2_kpa": state["ppCO2"],
        "cabin_pressure_kpa": state["cabin_pressure"],
        "cabin_humidity_pct": state["humidity"],
        "potable_water_kg": state["potable_kg"],
        "grey_water_kg": state["grey_kg"],
        "battery_kwh": state["battery_kwh"],
        "solar_input_kw": solar_kw,
        "scrubber_efficiency": state["scrubber_eff"],
        "electrolyser_o2_kg_h": electrolyser_o2_kg_h,
        "eclss_power_kw": ECLSS_LOAD_KW,
        "water_reserve_h": _water_reserve_hours(state["potable_kg"]),
    }


def _add_noise(clean: dict[str, float], rng: np.random.Generator) -> dict[str, float]:
    noisy = {}
    for key, val in clean.items():
        if key in SENSOR_NOISE:
            noisy[key] = val + rng.normal(0.0, SENSOR_NOISE[key])
        else:
            noisy[key] = val
    return noisy


def run_episode(
    fault_type: str = "nominal",
    seed: int = 0,
    onset_minute: Optional[int] = None,
) -> dict[str, Any]:
    """
    Run one 48-hour habitat episode.

    Returns dict with fault_type, onset_minute, breach_minute (None if no breach),
    fault_params, and ticks (list of per-minute telemetry rows with noise).
    """
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"Unknown fault_type: {fault_type}")

    rng = np.random.default_rng(seed)
    fault_params = _sample_fault_params(fault_type, rng)

    if onset_minute is None:
        onset = int(rng.integers(ONSET_MIN, ONSET_MAX + 1))
    else:
        onset = onset_minute

    state = _initial_state()
    ticks: list[dict[str, Any]] = []
    breach_minute: Optional[int] = None

    for t_min in range(EPISODE_MINUTES):
        state["t_min"] = t_min
        solar_mult, electrolyser_factor = _apply_faults(
            fault_type, fault_params, t_min, onset, state
        )
        clean = _physics_tick(state, solar_mult, electrolyser_factor)
        noisy = _add_noise(clean, rng)

        row = {
            "t_min": t_min,
            "fault_type": fault_type,
            **noisy,
        }
        ticks.append(row)

        if breach_minute is None and _check_breach(
            state["ppO2"],
            state["ppCO2"],
            state["potable_kg"],
            state["battery_kwh"],
            clean["solar_input_kw"],
        ):
            breach_minute = t_min

    return {
        "fault_type": fault_type,
        "seed": seed,
        "onset_minute": onset,
        "fault_params": fault_params,
        "breach_minute": breach_minute,
        "ticks": ticks,
    }
