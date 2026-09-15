# AEGIS

**Adaptive Early-warning Gateway for Integrated Systems**

Closed-loop life support monitoring for a 4-person Mars habitat, with ML-driven early alarms and an EVA suit health HUD.

**Live demo:** [aegismarshack.streamlit.app](https://aegismarshack.streamlit.app/)

---

## Track

**Life Support & Resource Systems**

AEGIS simulates Mars habitat ECLSS (atmosphere, water, power), trains models to predict time-to-breach before conventional thresholds fire, and pairs that with an astronaut suit health model and Blender helmet HUD.

---

## What we built

| Component | Description |
|-----------|-------------|
| **Habitat simulator** (`sim.py`) | 48 h episodes, 5 fault modes, reproducible physics + noisy telemetry |
| **Data generation** (`generate.py`) | 300 episodes → `data/episodes.parquet` |
| **Habitat ML** (`train_habitat.py`) | TTB regressor + fault classifier (placeholder models ready for real estimators) |
| **EVA suit ML** (`train.py`) | Spacesuit health risk model on `suit_telemetry.csv` → Blender HUD JSON |
| **Dashboard** (`app.py`) | Streamlit replay: habitat schematic, charts, EVA helmet HUD, Blender download |

### ML decisions

1. **Habitat — when to raise a life support alarm**  
   Predict `minutes_to_breach` from telemetry features (current values + 10/30 min deltas + rolling std). Alarm when predicted TTB &lt; 180 min — intended to beat fixed threshold bands on ppO₂, ppCO₂, water reserve, and battery SOC.

2. **EVA suit — astronaut critical risk**  
   Gradient boosting regressor predicts `critical_pct` (0–100) from suit vitals. HUD flags **CRITICAL** at ≥ 50%. Demo timeline exported for the Blender helmet scene.

---

## Quick start

### Requirements

Python 3.11+, dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Run the dashboard (local)

```bash
streamlit run app.py
```

Open the **Habitat monitor** tab for life-support replay, or **EVA suit HUD** for the spacewalk timeline.

### Full pipeline (from scratch)

```bash
# 1 — verify simulator
python smoke_sim.py

# 2 — generate habitat training data (~35 s)
python generate.py

# 3 — train models
python train_habitat.py   # → models/ttb.pkl
python train.py           # → models/health_regressor.joblib, predictions/eva_demo.json

# 4 — launch dashboard
streamlit run app.py
```

---

## Dashboard

Hosted at **[aegismarshack.streamlit.app](https://aegismarshack.streamlit.app/)**

### Habitat monitor

- Pick a fault scenario (nominal, scrubber degradation, hull leak, dust storm, water fouling)
- Replay timeline with play/pause
- Top-down habitat schematic — modules colour by status (green / amber / red)
- ML status panel: GREEN / WATCH / ALARM, predicted TTB, predicted fault
- Plotly charts with caution bands and ML vs threshold alarm markers

### EVA suit HUD

- Replays `predictions/eva_demo.json` (seal-leak EVA, safe → critical)
- Helmet-style HUD with live vitals and risk %
- Download **Astronaut Health Simulation.blend** for the 3D Blender scene

---

## Repository layout

```
sim.py                               Headless habitat physics
smoke_sim.py                         One-line sim sanity check
generate.py                          Episode dataset builder
train_habitat.py                     Habitat TTB + fault models → models/ttb.pkl
machine_learning_model.py            EVA suit health model → models/health_regressor.joblib
                                     + predictions/eva_demo.json (Blender HUD feed)
app.py                               Streamlit dashboard
data/                                Generated episodes (parquet)
models/                              Trained model artifacts
predictions/                         EVA demo timeline for Blender
suit_telemetry.csv                   EVA training data
Astronaut Health Simulation.blend    Blender helmet HUD scene
```

---

## Simulation spec (summary)

- **Tick:** 1 simulated minute · **Episode:** 48 h · **Sol:** 24.66 h
- **Crew:** 4 · **Volume:** 400 m³ · **Battery:** 60 kWh
- **Faults:** nominal, scrubber degradation, hull leak, dust storm, water loop fouling (random onset h 2–30)
- **Breach:** ppO₂ &lt; 15.5 kPa, ppCO₂ &gt; 1.0 kPa, water &lt; 12 h reserve, or battery empty with load unmet

---

## Habitat ML (collaborator notes)

`train_habitat.py` runs end-to-end but uses **placeholder** sklearn dummy models so the pipeline works without tuning. Swap in:

- `HistGradientBoostingRegressor` for `minutes_to_breach`
- `HistGradientBoostingClassifier` for `fault_type`

Search for `PlaceholderTTBRegressor` / `PlaceholderFaultClassifier` in `train_habitat.py`.

Headline metric (printed after training): median lead time of ML alarm over threshold alarm on held-out breaching episodes, plus false alarm rate on nominal episodes.

---

## EVA / Blender workflow

1. Run `python train.py` to refresh the model and `predictions/eva_demo.json`
2. Open **Astronaut Health Simulation.blend** in Blender 3.x+
3. Point the HUD script at `predictions/eva_demo.json`
4. The Streamlit **EVA suit HUD** tab replays the same JSON timeline in the browser

---

## Stack

Python 3.11 · numpy · pandas · scikit-learn · streamlit · plotly · pyarrow · joblib

No PyTorch, no database, no Docker — flat files in repo root.

---

## Team repos

- Upstream: [Han3579/AEGIS](https://github.com/Han3579/AEGIS)
- Deploy mirror: [yasmin-osman/aegis](https://github.com/yasmin-osman/aegis) (Streamlit Cloud)
