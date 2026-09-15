#!/usr/bin/env python3
"""Generate training episodes from the habitat simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sim import EPISODE_MINUTES, FAULT_TYPES, run_episode

N_EPISODES = 300
DOWNSAMPLE_EVERY = 5
MTB_CAP = 720
OUTPUT_PATH = Path("data/episodes.parquet")


def _minutes_to_breach(t_min: int, breach_minute: int | None) -> int:
    if breach_minute is None:
        return MTB_CAP
    return min(MTB_CAP, max(0, breach_minute - t_min))


def generate_episode(episode_id: int) -> pd.DataFrame:
    rng = np.random.default_rng(episode_id)
    fault_type = str(rng.choice(FAULT_TYPES))
    ep = run_episode(fault_type=fault_type, seed=episode_id)
    breach_minute = ep["breach_minute"]

    rows = []
    for tick in ep["ticks"]:
        t_min = tick["t_min"]
        if t_min % DOWNSAMPLE_EVERY != 0:
            continue
        row = {k: v for k, v in tick.items() if k != "fault_type"}
        row["episode_id"] = episode_id
        row["fault_type"] = fault_type
        row["minutes_to_breach"] = _minutes_to_breach(t_min, breach_minute)
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for episode_id in range(N_EPISODES):
        frames.append(generate_episode(episode_id))
        if (episode_id + 1) % 50 == 0:
            print(f"  {episode_id + 1}/{N_EPISODES} episodes")

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    n_breach = (
        df.groupby("episode_id")["minutes_to_breach"].min().lt(MTB_CAP).sum()
    )
    print(f"Wrote {len(df):,} rows ({N_EPISODES} episodes) -> {OUTPUT_PATH}")
    print(f"Episodes with breach: {n_breach}/{N_EPISODES}")
    print(f"Fault mix:\n{df.groupby('episode_id')['fault_type'].first().value_counts().to_string()}")


if __name__ == "__main__":
    main()
