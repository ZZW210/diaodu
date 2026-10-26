"""Temporary experiment: compare the original stage-2 order with the swapped order.

original: greedy -> independent matching -> reroute -> tail greedy -> tail reroute
swapped : independent matching -> greedy -> reroute -> tail greedy -> tail reroute

Both start from the saved stage-1 plans (final_plans_one_stage.pkl) and share the
same config, so only the order and the resulting adjustments differ.
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd

from src.config import load_config
from src.conflict_detection import (
    count_conflict_pairs,
    count_conflict_points,
    detect_conflicts,
)
from src.grid import AirspaceGrid
from src.optimization_model import (
    changed_count,
    greedy_time_deconfliction,
    independent_matching_deconfliction,
    limited_local_reroute,
)


def stats(plans) -> dict[str, float]:
    conflicts = detect_conflicts(plans, cfg, uncertain=True)
    return {
        "pairs": count_conflict_pairs(conflicts),
        "points": count_conflict_points(conflicts),
        "changed": changed_count(plans),
        "delayed": sum(1 for p in plans if p.delay >= 30.0),
        "delay_s": sum(max(0.0, p.delay) for p in plans),
        "air_s": sum(p.total_air_time for p in plans),
    }


def show(label: str, row: dict[str, float], seconds: float | None = None) -> None:
    tail = "" if seconds is None else f"  time={seconds:.1f}s"
    print(
        f"{label:<24} conflicts={row['pairs']:>3.0f}/{row['points']:<3.0f} "
        f"changed={row['changed']:>3.0f} delayed={row['delayed']:>3.0f} "
        f"delay={row['delay_s']:>9.1f}s air={row['air_s']:>9.1f}s{tail}"
    )


cfg = load_config("config.yaml")
grid = AirspaceGrid.from_config(cfg, seed=2025)
risk = np.load("outputs/risk_map.npy")
initial = pickle.load(open("outputs/initial_plans.pkl", "rb"))
stage1 = pickle.load(open("outputs/final_plans_one_stage.pkl", "rb"))
original = pickle.load(open("outputs/final_plans_two_stage.pkl", "rb"))
metrics = pd.read_csv("outputs/metrics_summary.csv").iloc[0]

print("=== reference ===")
show("initial", stats(initial))
show("stage 1", stats(stage1))
orig_stage2 = float(metrics["greedy_repair_time"]) + float(metrics["stage2_independent_matching_time"]) + float(metrics["local_reroute_time"])
show("original order", stats(original), orig_stage2)

conflicts = detect_conflicts(stage1, cfg, uncertain=True)
total_started = time.perf_counter()

started = time.perf_counter()
plans_im, conflicts_im, log_im = independent_matching_deconfliction(
    stage1, conflicts, cfg, grid, risk,
    max_rounds=int(cfg["optimization"].get("independent_matching_rounds", 20)),
    seed=2025 + 206,
)
t_im = time.perf_counter() - started

started = time.perf_counter()
plans_g, conflicts_g, log_g = greedy_time_deconfliction(
    plans_im, conflicts_im, cfg, grid, risk, max_rounds=50
)
t_g = time.perf_counter() - started

started = time.perf_counter()
plans_r2, conflicts_r2, log_r2 = limited_local_reroute(
    plans_g, conflicts_g, cfg, grid, risk, max_attempts=20
)
t_r2 = time.perf_counter() - started

started = time.perf_counter()
plans_g2, conflicts_g2, log_g2 = greedy_time_deconfliction(
    plans_r2, conflicts_r2, cfg, grid, risk, max_rounds=30
)
t_g2 = time.perf_counter() - started

started = time.perf_counter()
plans_r3, conflicts_r3, log_r3 = limited_local_reroute(
    plans_g2, conflicts_g2, cfg, grid, risk, max_attempts=20
)
t_r3 = time.perf_counter() - started
total = time.perf_counter() - total_started

print("")
print("=== swapped order ===")
print(
    f"  steps: IM {t_im:.1f}s, greedy {t_g:.1f}s, reroute {t_r2:.1f}s, "
    f"tail greedy {t_g2:.1f}s, tail reroute {t_r3:.1f}s"
)
show("swapped order", stats(plans_r3), total)
print(
    f"  accepted: IM {sum(1 for r in log_im if r.get('accepted'))}, "
    f"greedy {sum(1 for r in log_g if r.get('accepted'))}, "
    f"reroute {sum(1 for r in log_r2 + log_r3 if r.get('accepted'))}, "
    f"tail greedy {sum(1 for r in log_g2 if r.get('accepted'))}"
)
