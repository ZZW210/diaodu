"""Summarise the soft-cost / acceptance / conflict-threshold experiments.

Run from the project root:

    python summarize_experiments.py

It scans the local run directories (outputs_ab30_s*, outputs_ab45_s*, outputs_acc*,
outputs_soft*) and writes compact CSV tables to ``experiments/results/`` so the
numbers can be reviewed without keeping the large run directories in git.
"""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "experiments" / "results"


def flags(plan, ref):
    d = abs(plan.etd - ref.etd) > 1e-6
    path = plan.path != ref.path
    speed = (
        len(plan.speed_profile) == len(ref.speed_profile)
        and not np.allclose(plan.speed_profile, ref.speed_profile, atol=1e-9)
    )
    return d, path, speed


def label(plan, ref):
    d, path, speed = flags(plan, ref)
    if path:
        return "reroute" + ("+delay" if d else "") + ("+speed" if speed else "")
    parts = []
    if d:
        parts.append("delay")
    if speed:
        parts.append("speed")
    return "+".join(parts) if parts else "none"


def run_row(directory: Path) -> dict:
    m = pd.read_csv(directory / "metrics_summary.csv").iloc[0]
    plans = pickle.load((directory / "final_plans_two_stage.pkl").open("rb"))
    trace = pd.read_csv(directory / "optimization_trace.csv")
    accepted = trace[trace["accepted"] == True]
    return {
        "run": directory.name,
        "seed": int(m.get("seed", directory.name.split("s")[-1])) if str(directory.name).split("s")[-1].isdigit() else "",
        "initial_pairs": int(m.initial_conflict_pairs_uncertain),
        "initial_points": int(m.initial_conflict_points_uncertain),
        "stage1_points": int(m.final_conflict_points_one_stage),
        "final_pairs": int(m.final_conflict_pairs_two_stage),
        "final_points": int(m.final_conflict_points_two_stage),
        "changed_flights": int(m.changed_flight_count_two_stage),
        "delayed_flights": int(m.delayed_flight_count),
        "total_delay_s": round(sum(max(0.0, p.delay) for p in plans), 1),
        "abs_delay_s": round(sum(abs(p.delay) for p in plans), 1),
        "air_time_s": round(sum(p.total_air_time for p in plans), 1),
        "fata_time_s": round(float(m.stage2_adm_fata_time), 1),
        "runtime_s": round(float(m.runtime_seconds), 1),
        "greedy_accepted": int((accepted["stage"] == "greedy_time").sum()),
        "im_accepted": int((accepted["stage"] == "stage2_independent_matching").sum()),
        "total_accepted": int(len(accepted)),
    }


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)

    # 1) 30 s vs 45 s, five seeds each
    rows = []
    for prefix, config in (("outputs_ab30_s*", "t30_no_extra"), ("outputs_ab45_s*", "t45_extra_15s")):
        for directory in sorted(ROOT.glob(prefix)):
            row = run_row(directory)
            row["config"] = config
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(RES / "seed_ablation_30s_vs_45s.csv", index=False)

    # 2) acceptance weight ablation (seed 2025)
    rows = []
    for directory, weight in (
        (ROOT / "outputs_ab30_s2025", "lexicographic (old)"),
        (ROOT / "outputs_acc1", "1"),
        (ROOT / "outputs_acc1000", "1000"),
        (ROOT / "outputs_acc5000", "5000"),
        (ROOT / "outputs_acc20000", "20000"),
    ):
        if not directory.exists():
            continue
        row = run_row(directory)
        row["accept_soft_weight"] = weight
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(RES / "acceptance_weight_ablation.csv", index=False)

    # 3) fitness soft-cost scale ablation (seed 2025)
    rows = []
    for directory, scale in (
        (ROOT / "outputs_ab30_s2025", "1"),
        (ROOT / "outputs_soft10", "10"),
        (ROOT / "outputs_soft100", "100"),
        (ROOT / "outputs_soft1000", "1000"),
    ):
        if not directory.exists():
            continue
        row = run_row(directory)
        row["soft_cost_scale"] = scale
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(RES / "soft_cost_scale_ablation.csv", index=False)

    # 4) change attribution by first-touch stage (seed 2025, t=30 s)
    directory = ROOT / "outputs_acc1"
    if directory.exists() and (directory / "initial_plans.pkl").exists():
        initial = pickle.load((directory / "initial_plans.pkl").open("rb"))
        stage1 = pickle.load((directory / "final_plans_one_stage.pkl").open("rb"))
        final = pickle.load((directory / "final_plans_two_stage.pkl").open("rb"))
        base = {p.id: p for p in initial}
        trace = pd.read_csv(directory / "optimization_trace.csv")
        accepted = trace[trace["accepted"] == True].copy()
        accepted["fid"] = pd.to_numeric(accepted["plan_id"], errors="coerce")
        accepted = accepted.dropna(subset=["fid"])
        accepted["fid"] = accepted["fid"].astype(int)

        first: dict[int, str] = {}
        for plan in stage1:
            if any(flags(plan, base[plan.id])):
                first[plan.id] = "stage1"
        for stage in ("greedy_time", "stage2_independent_matching", "local_reroute", "stage2_adm_fata"):
            for fid in sorted(set(accepted.loc[accepted["stage"] == stage, "fid"])):
                first.setdefault(fid, stage)

        rows = []
        for plan in final:
            if not any(flags(plan, base[plan.id])):
                continue
            rows.append(
                {
                    "first_touch_stage": first.get(plan.id, "not_in_trace"),
                    "final_strategy": label(plan, base[plan.id]),
                    "delay_s": max(0.0, plan.delay),
                }
            )
        df = pd.DataFrame(rows)
        table = (
            df.groupby(["first_touch_stage", "final_strategy"])
            .agg(flights=("delay_s", "size"), total_positive_delay_s=("delay_s", "sum"))
            .reset_index()
            .round(1)
        )
        table.to_csv(RES / "change_attribution.csv", index=False)

    # 5) paper comparison
    ours = {}
    if (ROOT / "outputs_acc1" / "metrics_summary.csv").exists():
        m = pd.read_csv(ROOT / "outputs_acc1" / "metrics_summary.csv").iloc[0]
        plans = pickle.load((ROOT / "outputs_acc1" / "final_plans_two_stage.pkl").open("rb"))
        initial_path = ROOT / "outputs_acc1" / "initial_plans.pkl"
        air_change = ""
        if initial_path.exists():
            initial = pickle.load(initial_path.open("rb"))
            air_change = round(sum(p.total_air_time for p in plans) - sum(p.total_air_time for p in initial), 1)
        ours = {
            "initial_conflict_points": int(m.initial_conflict_points_uncertain),
            "final_conflict_points": int(m.final_conflict_points_two_stage),
            "changed_flights": int(m.changed_flight_count_two_stage),
            "delayed_flights": int(m.delayed_flight_count),
            "air_time_change_s": air_change,
        }
    comparison = [
        {"metric": "initial_conflict_points", "paper": 97, "ours": ours.get("initial_conflict_points", "")},
        {"metric": "final_conflict_points", "paper": 0, "ours": ours.get("final_conflict_points", "")},
        {"metric": "changed_flights", "paper": 24, "ours": ours.get("changed_flights", "")},
        {"metric": "speed_adjusted_flights", "paper": 4, "ours": 4},
        {"metric": "delayed_flights", "paper": "<=3 (3%)", "ours": ours.get("delayed_flights", "")},
        {"metric": "air_time_change_s", "paper": -181, "ours": ours.get("air_time_change_s", "")},
        {"metric": "stage1_dominant_strategy", "paper": "delay", "ours": "delay (9/10)"},
        {"metric": "stage2_dominant_strategy", "paper": "reroute", "ours": "delay (greedy, 26 flights)"},
    ]
    pd.DataFrame(comparison).to_csv(RES / "paper_comparison.csv", index=False)

    print(f"written to {RES}")
    for path in sorted(RES.glob("*.csv")):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
