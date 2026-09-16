"""Temporary analysis: plan-level comparison across stage-2 solver outputs."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

RUNS = sys.argv[1:] or [
    "outputs_s2_matching",
    "outputs_s2_adm_fata",
    "outputs_s2_adm_fata60",
    "outputs_s2_hybrid",
    "outputs_s2p_matching",
    "outputs_s2p_adm_fata60",
    "outputs_s2p_hybrid",
]


def _load(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def stats(run: str) -> dict[str, float]:
    root = Path(run)
    initial = _load(root / "initial_plans.pkl")
    final = _load(root / "final_plans_two_stage.pkl")
    by_id = {p.id: p for p in initial}
    delay_threshold = 30.0

    total_delay = sum(max(0.0, p.delay) for p in final)
    abs_delay = sum(abs(p.delay) for p in final)
    delayed = sum(1 for p in final if p.delay >= delay_threshold)
    changed = sum(1 for p in final if p.changed)
    time_changed = sum(1 for p in final if abs(p.etd - by_id[p.id].etd) > 1e-9)
    rerouted = sum(1 for p in final if p.path != by_id[p.id].path)
    speed_changed = sum(
        1
        for p in final
        if p.path == by_id[p.id].path
        and not np.allclose(np.asarray(p.speed_profile, float), np.asarray(by_id[p.id].speed_profile, float))
    )
    air_before = sum(p.total_air_time for p in initial)
    air_after = sum(p.total_air_time for p in final)
    risk_before = sum(p.risk_sum for p in initial)
    risk_after = sum(p.risk_sum for p in final)
    battery = sum(1 for p in final if p.total_air_time > 1200.0)

    soft = (
        0.001 * total_delay
        + 0.0001 * air_after
        + 0.0001 * risk_after
        + 10.0 * changed
        + 25.0 * delayed
    )
    return {
        "changed": changed,
        "delayed>=30s": delayed,
        "time_changed": time_changed,
        "speed_changed": speed_changed,
        "rerouted": rerouted,
        "delay_sum_s": round(total_delay, 1),
        "delay_abs_s": round(abs_delay, 1),
        "delay_max_s": round(max((p.delay for p in final), default=0.0), 1),
        "air_delta_s": round(air_after - air_before, 1),
        "risk_before": round(risk_before, 2),
        "risk_after": round(risk_after, 2),
        "risk_delta": round(risk_after - risk_before, 3),
        "battery>1200s": battery,
        "soft_cost": round(soft, 2),
        "soft_delay": round(0.001 * total_delay, 2),
        "soft_air": round(0.0001 * air_after, 2),
        "soft_risk": round(0.0001 * risk_after, 2),
    }


def main() -> None:
    keys = None
    rows: dict[str, dict[str, float]] = {}
    for run in RUNS:
        rows[run] = stats(run)
        keys = list(rows[run])
    assert keys is not None
    header = f"{'metric':<16}" + "".join(f"{run.replace('outputs_',''):>22}" for run in RUNS)
    print(header)
    print("-" * len(header))
    for key in keys:
        line = f"{key:<16}" + "".join(f"{rows[run][key]:>22}" for run in RUNS)
        print(line)


if __name__ == "__main__":
    main()
