#!/usr/bin/env python3
"""Smoke test: one nominal and one scrubber_degradation episode."""

from sim import run_episode


def fmt_breach(ep: dict) -> str:
    b = ep["breach_minute"]
    if b is None:
        return "no breach within 48 h"
    return f"breach at minute {b} ({b / 60:.1f} h)"


if __name__ == "__main__":
    nominal = run_episode("nominal", seed=42)
    # Early onset guarantees CO2 breach within 48 h for smoke demo (seed 42 random onset is too late).
    scrubber = run_episode("scrubber_degradation", seed=42, onset_minute=120)

    print(f"nominal:              {fmt_breach(nominal)}")
    print(f"scrubber_degradation: {fmt_breach(scrubber)}")
    print(f"  onset minute: {scrubber['onset_minute']}")
