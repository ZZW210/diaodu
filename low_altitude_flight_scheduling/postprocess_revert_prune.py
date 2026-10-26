"""Post-process: minimise the number of adjusted flight plans while keeping zero conflicts.

It does NOT touch the main pipeline. It reads the saved plan snapshots:

    outputs/initial_plans.pkl
    outputs/final_plans_two_stage.pkl

and performs two passes:

1. Full-revert pass: for every changed flight (largest adjustment first), restore
   its initial plan and keep the revert when the global conflict set is still empty.
2. Shrink pass: for flights that could not be reverted, binary-search the smallest
   delay / speed deviation that still keeps zero conflicts.

Outputs (default ``outputs_prune/``):
    final_plans_pruned.pkl
    reverted_flights.csv
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.conflict_detection import (
    count_conflict_pairs,
    count_conflict_points,
    detect_conflicts,
)
from src.flight_plan import FlightPlan, recompute_eta_times
from src.utils import ensure_dir


def load_plans(path: Path) -> list[FlightPlan]:
    with path.open("rb") as fh:
        return pickle.load(fh)


def is_changed(plan: FlightPlan, base: FlightPlan) -> bool:
    if abs(plan.etd - base.etd) > 1e-6:
        return True
    if plan.path != base.path:
        return True
    if len(plan.speed_profile) != len(base.speed_profile):
        return True
    return not np.allclose(plan.speed_profile, base.speed_profile, atol=1e-9)


def magnitude(plan: FlightPlan, base: FlightPlan) -> float:
    score = abs(plan.etd - base.etd)
    score += 10.0 * sum(1 for a, b in zip(plan.path, base.path) if a != b)
    score += 10.0 * abs(len(plan.path) - len(base.path))
    if len(plan.speed_profile) == len(base.speed_profile):
        score += 100.0 * float(
            np.abs(np.asarray(plan.speed_profile) - np.asarray(base.speed_profile)).sum()
        )
    return score


def with_etd(plan: FlightPlan, etd: float, base_etd: float) -> FlightPlan:
    new = plan.copy()
    delta = etd - new.etd
    new.etd = float(etd)
    new.eta_times = [float(t + delta) for t in new.eta_times]
    new.delay = float(new.etd - base_etd)
    new.changed = True
    return new


def with_speed(plan: FlightPlan, speeds: list[float], base_etd: float, cell_size) -> FlightPlan:
    new = plan.copy()
    new.speed_profile = [float(v) for v in speeds]
    new.eta_times = recompute_eta_times(new, cell_size, speed=new.speed_profile)
    new.total_air_time = float(max(0.0, new.eta_times[-1] - new.etd))
    new.delay = float(new.etd - base_etd)
    new.changed = True
    return new


def detect(plans: list[FlightPlan], cfg: dict) -> tuple[int, int]:
    conflicts = detect_conflicts(plans, cfg, uncertain=True)
    return count_conflict_pairs(conflicts), count_conflict_points(conflicts)


def try_revert(
    index: int,
    current: list[FlightPlan],
    original: FlightPlan,
    cfg: dict,
) -> tuple[list[FlightPlan], bool]:
    trial = list(current)
    trial[index] = original.copy()
    pairs, points = detect(trial, cfg)
    if pairs == 0 and points == 0:
        return trial, True
    return current, False


def shrink_delay(
    index: int,
    current: list[FlightPlan],
    base: FlightPlan,
    cfg: dict,
    iterations: int = 8,
) -> tuple[list[FlightPlan], bool]:
    plan = current[index]
    if abs(plan.etd - base.etd) <= 1e-6:
        return current, False
    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        etd = base.etd + mid * (plan.etd - base.etd)
        trial = list(current)
        trial[index] = with_etd(plan, etd, base.etd)
        pairs, points = detect(trial, cfg)
        if pairs == 0 and points == 0:
            hi = mid
        else:
            lo = mid
    if hi >= 1.0 - 1e-9:
        return current, False
    trial = list(current)
    trial[index] = with_etd(plan, base.etd + hi * (plan.etd - base.etd), base.etd)
    pairs, points = detect(trial, cfg)
    if pairs == 0 and points == 0:
        return trial, True
    return current, False


def shrink_speed(
    index: int,
    current: list[FlightPlan],
    base: FlightPlan,
    cfg: dict,
    cell_size,
    iterations: int = 8,
) -> tuple[list[FlightPlan], bool]:
    plan = current[index]
    if len(plan.speed_profile) != len(base.speed_profile):
        return current, False
    if np.allclose(plan.speed_profile, base.speed_profile, atol=1e-9):
        return current, False
    cur = np.asarray(plan.speed_profile, dtype=float)
    ref = np.asarray(base.speed_profile, dtype=float)
    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        trial = list(current)
        trial[index] = with_speed(plan, list(ref + mid * (cur - ref)), base.etd, cell_size)
        pairs, points = detect(trial, cfg)
        if pairs == 0 and points == 0:
            hi = mid
        else:
            lo = mid
    if hi >= 1.0 - 1e-9:
        return current, False
    trial = list(current)
    trial[index] = with_speed(plan, list(ref + hi * (cur - ref)), base.etd, cell_size)
    pairs, points = detect(trial, cfg)
    if pairs == 0 and points == 0:
        return trial, True
    return current, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Revert/minimise adjusted plans while keeping zero conflicts.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--out", default="outputs_prune")
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config)
    cell_size = tuple(int(v) for v in cfg["airspace"]["grid_cell_size"])
    out_dir = ensure_dir(root / args.out)
    src_dir = root / args.outputs

    initial = load_plans(src_dir / "initial_plans.pkl")
    final = load_plans(src_dir / "final_plans_two_stage.pkl")
    base_by_id = {p.id: p for p in initial}

    pairs0, points0 = detect(final, cfg)
    changed_ids = [p.id for p in final if is_changed(p, base_by_id[p.id])]
    delayed0 = sum(1 for p in final if p.delay >= 30.0)
    print(f"[input] initial conflicts (uncertain): {pairs0} pairs / {points0} points")
    print(f"[input] changed flights: {len(changed_ids)} / {len(final)}, delayed(>=30s): {delayed0}")

    current = [p.copy() for p in final]
    reverted: set[int] = set()
    shrunk_delay: set[int] = set()
    shrunk_speed: set[int] = set()

    index_by_id = {p.id: i for i, p in enumerate(current)}
    order = sorted(
        changed_ids,
        key=lambda fid: magnitude(current[index_by_id[fid]], base_by_id[fid]),
        reverse=True,
    )

    for pass_idx in range(1, args.passes + 1):
        reverted_this_pass = 0
        for k, fid in enumerate(order, start=1):
            idx = index_by_id[fid]
            if not is_changed(current[idx], base_by_id[fid]):
                continue
            current, ok = try_revert(idx, current, base_by_id[fid], cfg)
            if ok:
                reverted.add(fid)
                reverted_this_pass += 1
            if k % 10 == 0:
                print(f"  pass {pass_idx}: checked {k}/{len(order)}, reverted so far {len(reverted)}")
        print(f"[pass {pass_idx}] reverted {reverted_this_pass} flights, total {len(reverted)}")
        if reverted_this_pass == 0:
            break

    print("[shrink] minimising remaining delays ...")
    for fid in order:
        idx = index_by_id[fid]
        if not is_changed(current[idx], base_by_id[fid]):
            continue
        current, ok = shrink_delay(idx, current, base_by_id[fid], cfg)
        if ok:
            shrunk_delay.add(fid)

    print("[shrink] minimising remaining speed deviations ...")
    for fid in order:
        idx = index_by_id[fid]
        if not is_changed(current[idx], base_by_id[fid]):
            continue
        current, ok = shrink_speed(idx, current, base_by_id[fid], cfg, cell_size)
        if ok:
            shrunk_speed.add(fid)

    pairs1, points1 = detect(current, cfg)
    changed_after = sum(1 for p in current if is_changed(p, base_by_id[p.id]))
    delayed_after = sum(1 for p in current if p.delay >= 30.0)
    total_delay_before = float(sum(max(0.0, p.delay) for p in final))
    total_delay_after = float(sum(max(0.0, p.delay) for p in current))

    print("")
    print("================ summary ================")
    print(f"conflicts      : {pairs0} pairs / {points0} points  ->  {pairs1} pairs / {points1} points")
    print(f"changed flights: {len(changed_ids)}  ->  {changed_after}   (reverted {len(reverted)})")
    print(f"delayed(>=30s) : {delayed0}  ->  {delayed_after}")
    print(f"total delay    : {total_delay_before:.1f} s  ->  {total_delay_after:.1f} s")
    print(f"shrunk delay   : {len(shrunk_delay)} flights")
    print(f"shrunk speed   : {len(shrunk_speed)} flights")

    rows = []
    for fid in changed_ids:
        before = final[index_by_id[fid]]
        after = current[index_by_id[fid]]
        base = base_by_id[fid]
        rows.append(
            {
                "plan_id": fid,
                "delay_before": before.etd - base.etd,
                "delay_after": after.etd - base.etd,
                "path_changed_before": before.path != base.path,
                "path_changed_after": after.path != base.path,
                "speed_changed_before": before.speed_profile != base.speed_profile,
                "speed_changed_after": after.speed_profile != base.speed_profile,
                "reverted": fid in reverted,
                "shrunk_delay": fid in shrunk_delay,
                "shrunk_speed": fid in shrunk_speed,
                "still_changed": is_changed(after, base),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "reverted_flights.csv", index=False)
    with (out_dir / "final_plans_pruned.pkl").open("wb") as fh:
        pickle.dump(current, fh)
    print(f"saved: {out_dir / 'final_plans_pruned.pkl'}")
    print(f"saved: {out_dir / 'reverted_flights.csv'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
