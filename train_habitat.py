#!/usr/bin/env python3
"""
Train habitat time-to-breach (TTB) and fault-type models.

Habitat ML pipeline — separate from train.py (EVA suit health model).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

from sim import BATTERY_CAPACITY_KWH, WATER_USE_KG_MIN

DATA_PATH = Path("data/episodes.parquet")
MODEL_PATH = Path("models/ttb.pkl")

TICK_MINUTES = 5
DELTA_10M_STEPS = 10 // TICK_MINUTES
DELTA_30M_STEPS = 30 // TICK_MINUTES
ROLL_30M_WINDOW = 6

TELEMETRY_COLS = [
    "ppO2_kpa",
    "ppCO2_kpa",
    "cabin_pressure_kpa",
    "cabin_humidity_pct",
    "potable_water_kg",
    "grey_water_kg",
    "battery_kwh",
    "solar_input_kw",
    "scrubber_efficiency",
    "electrolyser_o2_kg_h",
    "eclss_power_kw",
    "water_reserve_h",
]

ML_ALARM_TTB = 180
CAUTION_PPO2 = 17.0
CAUTION_PPCO2 = 0.7
CAUTION_WATER_RESERVE_H = 24.0
CAUTION_BATTERY_FRAC = 0.15


class PlaceholderTTBRegressor:
    def __init__(self) -> None:
        self.model = DummyRegressor(strategy="median")
        self.feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PlaceholderTTBRegressor":
        self.feature_cols = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_cols])


class PlaceholderFaultClassifier:
    def __init__(self) -> None:
        self.model = DummyClassifier(strategy="most_frequent")
        self.feature_cols: list[str] = []
        self.classes_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PlaceholderFaultClassifier":
        self.feature_cols = list(X.columns)
        self.model.fit(X, y)
        self.classes_ = list(self.model.classes_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_cols])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_cols])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    parts = [df[["episode_id", "t_min"]].copy()]
    grouped = df.groupby("episode_id", sort=False)
    for col in TELEMETRY_COLS:
        current = grouped[col].transform(lambda s: s)
        delta_10 = grouped[col].transform(lambda s: s - s.shift(DELTA_10M_STEPS))
        delta_30 = grouped[col].transform(lambda s: s - s.shift(DELTA_30M_STEPS))
        roll_std = grouped[col].transform(
            lambda s: s.rolling(ROLL_30M_WINDOW, min_periods=1).std()
        )
        parts.append(current.rename(col))
        parts.append(delta_10.rename(f"{col}_delta10"))
        parts.append(delta_30.rename(f"{col}_delta30"))
        parts.append(roll_std.rename(f"{col}_std30"))
    features = pd.concat(parts, axis=1)
    return features.dropna().reset_index(drop=True)


def align_labels(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    return features.merge(
        df[["episode_id", "t_min", "fault_type", "minutes_to_breach"]],
        on=["episode_id", "t_min"],
        how="left",
    )


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c not in ("episode_id", "t_min")]


def split_episodes(
    episode_ids: list[int], test_frac: float = 0.2, seed: int = 42
) -> tuple[set[int], set[int]]:
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(set(episode_ids)))
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * test_frac))
    test_ids = set(ids[:n_test].tolist())
    train_ids = set(ids[n_test:].tolist())
    return train_ids, test_ids


def _water_reserve_hours(potable_kg: float) -> float:
    use_kg_h = WATER_USE_KG_MIN * 60.0
    return potable_kg / use_kg_h if use_kg_h > 0 else float("inf")


def threshold_alarm_time(ep_df: pd.DataFrame) -> int | None:
    battery_limit = BATTERY_CAPACITY_KWH * CAUTION_BATTERY_FRAC
    for _, row in ep_df.sort_values("t_min").iterrows():
        reserve_h = row.get("water_reserve_h", _water_reserve_hours(row["potable_water_kg"]))
        if row["ppO2_kpa"] < CAUTION_PPO2:
            return int(row["t_min"])
        if row["ppCO2_kpa"] > CAUTION_PPCO2:
            return int(row["t_min"])
        if reserve_h < CAUTION_WATER_RESERVE_H:
            return int(row["t_min"])
        if row["battery_kwh"] < battery_limit:
            return int(row["t_min"])
    return None


def ml_alarm_time(ep_df: pd.DataFrame, pred_ttb: np.ndarray) -> int | None:
    ep_df = ep_df.sort_values("t_min").reset_index(drop=True)
    for i, ttb in enumerate(pred_ttb):
        if ttb < ML_ALARM_TTB:
            return int(ep_df.loc[i, "t_min"])
    return None


def episode_breach_minute(ep_df: pd.DataFrame) -> int | None:
    ep_df = ep_df.sort_values("t_min")
    for _, row in ep_df.iterrows():
        if row["minutes_to_breach"] < 720:
            return int(row["t_min"] + row["minutes_to_breach"])
    return None


def headline_metrics(
    df: pd.DataFrame,
    test_ids: set[int],
    ttb_model: PlaceholderTTBRegressor,
    feature_cols: list[str],
) -> dict[str, Any]:
    lead_times: list[float] = []
    false_alarms = 0
    nominal_count = 0
    feat_df = build_features(df)
    labels = align_labels(df, feat_df)

    for eid in sorted(test_ids):
        ep_rows = labels[labels["episode_id"] == eid].sort_values("t_min")
        if ep_rows.empty:
            continue
        raw_ep = df[df["episode_id"] == eid].sort_values("t_min")
        fault = raw_ep["fault_type"].iloc[0]
        breach = episode_breach_minute(raw_ep)
        pred_ttb = ttb_model.predict(ep_rows[feature_cols])

        if fault == "nominal":
            nominal_count += 1
            if ml_alarm_time(ep_rows, pred_ttb) is not None:
                false_alarms += 1
            continue
        if breach is None:
            continue
        ml_t = ml_alarm_time(ep_rows, pred_ttb)
        thr_t = threshold_alarm_time(raw_ep)
        if ml_t is not None and thr_t is not None:
            lead_times.append(thr_t - ml_t)

    result: dict[str, Any] = {
        "lead_times": lead_times,
        "false_alarm_rate": false_alarms / nominal_count if nominal_count else 0.0,
        "nominal_episodes": nominal_count,
    }
    if lead_times:
        result["lead_time_median"] = float(np.median(lead_times))
        q25, q75 = np.percentile(lead_times, [25, 75])
        result["lead_time_iqr"] = float(q75 - q25)
    else:
        result["lead_time_median"] = None
        result["lead_time_iqr"] = None
    return result


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run `python generate.py` first.")

    df = pd.read_parquet(DATA_PATH)
    features = build_features(df)
    data = align_labels(df, features)
    feat_cols = feature_columns(features)
    train_ids, test_ids = split_episodes(df["episode_id"].unique().tolist())

    train_mask = data["episode_id"].isin(train_ids)
    test_mask = data["episode_id"].isin(test_ids)

    ttb_model = PlaceholderTTBRegressor()
    ttb_model.fit(data.loc[train_mask, feat_cols], data.loc[train_mask, "minutes_to_breach"])
    fault_model = PlaceholderFaultClassifier()
    fault_model.fit(data.loc[train_mask, feat_cols], data.loc[train_mask, "fault_type"])

    mae = mean_absolute_error(
        data.loc[test_mask, "minutes_to_breach"],
        ttb_model.predict(data.loc[test_mask, feat_cols]),
    )
    fault_acc = accuracy_score(
        data.loc[test_mask, "fault_type"],
        fault_model.predict(data.loc[test_mask, feat_cols]),
    )
    headline = headline_metrics(df, test_ids, ttb_model, feat_cols)

    print(f"MAE (minutes_to_breach): {mae:.1f}")
    print(f"Fault classification accuracy: {fault_acc:.3f}")
    print(f"Lead time median: {headline.get('lead_time_median')}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(
            {
                "version": "placeholder",
                "ttb_model": ttb_model,
                "fault_model": fault_model,
                "feature_cols": feat_cols,
                "telemetry_cols": TELEMETRY_COLS,
                "ml_alarm_ttb": ML_ALARM_TTB,
                "caution_bands": {
                    "ppO2_kpa": CAUTION_PPO2,
                    "ppCO2_kpa": CAUTION_PPCO2,
                    "water_reserve_h": CAUTION_WATER_RESERVE_H,
                    "battery_frac": CAUTION_BATTERY_FRAC,
                },
            },
            f,
        )
    print(f"Saved model bundle -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
