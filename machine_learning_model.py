"""
Train a spacesuit health model, then export one EVA for the helmet HUD.

What it does:
1. Load suit_telemetry.csv
2. Train on some spacewalks, test on others (never mix the same EVA)
3. Predict critical_pct (0–100 health risk)
4. Save the model and one demo timeline for Blender
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# --- files ---
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "suit_telemetry.csv"
MODEL_PATH = ROOT / "models" / "health_regressor.joblib"
DEMO_PATH = ROOT / "predictions" / "eva_demo.json"

# Inputs the model is allowed to see (not the fault label).
FEATURES = [
    "t_min",
    "hr_bpm",
    "resp_rpm",
    "core_temp_c",
    "spo2_pct",
    "ext_temp_c",
    "helmet_co2_ppm",
    "suit_press_kpa",
]
TARGET = "critical_pct"
CRITICAL_THRESHOLD = 50.0  # HUD: >= 50% is CRITICAL


def status_label(risk_pct: float) -> str:
    return "CRITICAL" if risk_pct >= CRITICAL_THRESHOLD else "NON-CRITICAL"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    needed = FEATURES + [TARGET, "eva_id"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df


def split_by_eva(df: pd.DataFrame):
    """Keep each whole spacewalk in train *or* test, not both."""
    eva_ids = df["eva_id"].drop_duplicates().sort_values()
    train_ids, test_ids = train_test_split(
        eva_ids.to_numpy(),
        test_size=0.2,
        random_state=42,
    )
    train = df[df["eva_id"].isin(train_ids)].copy()
    test = df[df["eva_id"].isin(test_ids)].copy()
    return train, test, set(test_ids)


def pick_demo_eva(df: pd.DataFrame, predicted: pd.Series, test_ids: set) -> str:
    """
    Pick one EVA the HUD can play: starts safe, later goes critical.
    Prefer a seal_leak walk from the test set.
    """
    work = df.copy()
    work["predicted_pct"] = predicted.to_numpy()

    best_id = None
    best_rank = None  # lower is better: (priority, -rise)

    for eva_id, group in work.groupby("eva_id"):
        group = group.sort_values("t_min")
        preds = group["predicted_pct"]

        starts_safe = preds.iloc[0] < CRITICAL_THRESHOLD
        later_critical = preds.max() >= CRITICAL_THRESHOLD
        if not (starts_safe and later_critical):
            continue

        fault = str(group["fault"].iloc[0]) if "fault" in group.columns else ""
        is_seal = fault == "seal_leak"
        in_test = eva_id in test_ids
        if is_seal and in_test:
            priority = 0
        elif in_test:
            priority = 1
        elif is_seal:
            priority = 2
        else:
            priority = 3

        rise = float(preds.max() - preds.iloc[0])
        rank = (priority, -rise)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_id = eva_id

    if best_id is None:
        raise RuntimeError("No EVA goes from safe to critical. Check the CSV.")
    return best_id


def make_hud_json(df: pd.DataFrame, predicted: pd.Series, eva_id: str) -> dict:
    """One spacewalk as a list of HUD frames."""
    eva = df.loc[df["eva_id"] == eva_id].copy()
    eva["predicted_pct"] = predicted.loc[eva.index].to_numpy()
    eva = eva.sort_values("t_min")

    timeline = []
    for row in eva.itertuples(index=False):
        risk = round(float(row.predicted_pct), 2)
        timeline.append({
            "t_min": int(row.t_min),
            "predicted_pct": risk,
            "is_critical": risk >= CRITICAL_THRESHOLD,
            "hr_bpm": int(row.hr_bpm),
            "resp_rpm": int(row.resp_rpm),
            "core_temp_c": round(float(row.core_temp_c), 2),
            "spo2_pct": int(row.spo2_pct),
            "ext_temp_c": round(float(row.ext_temp_c), 2),
            "helmet_co2_ppm": int(row.helmet_co2_ppm),
            "suit_press_kpa": round(float(row.suit_press_kpa), 2),
        })

    fault = str(eva["fault"].iloc[0]) if "fault" in eva.columns else None
    return {
        "eva_id": eva_id,
        "fault": fault,
        "critical_threshold": CRITICAL_THRESHOLD,
        "n_steps": len(timeline),
        "timeline": timeline,
    }


def main() -> None:
    df = load_data()
    train, test, test_ids = split_by_eva(df)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(train[FEATURES], train[TARGET])

    predicted_risk = pd.Series(model.predict(test[FEATURES])).clip(0, 100)
    true_risk = test[TARGET]

    mae = mean_absolute_error(true_risk, predicted_risk)
    r2 = r2_score(true_risk, predicted_risk)
    acc = accuracy_score(
        true_risk >= CRITICAL_THRESHOLD,
        predicted_risk >= CRITICAL_THRESHOLD,
    )

    print("=== Suit health model ===")
    print(f"Train EVAs: {train['eva_id'].nunique()}  Test EVAs: {test['eva_id'].nunique()}")
    print(f"Train rows: {len(train)}  Test rows: {len(test)}")
    print(f"MAE (average error in %): {mae:.3f}")
    print(f"R2 (1.0 = perfect): {r2:.3f}")
    print(f"CRITICAL vs NON-CRITICAL accuracy: {acc:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURES, "threshold": CRITICAL_THRESHOLD},
        MODEL_PATH,
    )
    print(f"Saved model: {MODEL_PATH}")

    all_pred = pd.Series(model.predict(df[FEATURES]), index=df.index).clip(0, 100)
    demo_eva = pick_demo_eva(df, all_pred, test_ids)
    payload = make_hud_json(df, all_pred, demo_eva)

    DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEMO_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    first, last = payload["timeline"][0], payload["timeline"][-1]
    print(f"Saved HUD demo: {DEMO_PATH}")
    print(f"Demo EVA: {demo_eva}  fault: {payload.get('fault')}")
    print(
        f"{first['t_min']} min  {first['predicted_pct']}%  {status_label(first['predicted_pct'])}"
        f"  ->  "
        f"{last['t_min']} min  {last['predicted_pct']}%  {status_label(last['predicted_pct'])}"
    )


if __name__ == "__main__":
    main()
