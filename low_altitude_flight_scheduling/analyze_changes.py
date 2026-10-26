"""Temporary analysis: when (which stage) were flights changed, with which strategy."""
from __future__ import annotations

import pickle
from collections import Counter

import numpy as np
import pandas as pd

from src.config import load_config
from src.conflict_detection import detect_conflicts, count_conflict_pairs, count_conflict_points

OUT = "outputs_acc1"
cfg = load_config("config.yaml")

initial = pickle.load(open(f"{OUT}/initial_plans.pkl", "rb"))
stage1 = pickle.load(open(f"{OUT}/final_plans_one_stage.pkl", "rb"))
final = pickle.load(open(f"{OUT}/final_plans_two_stage.pkl", "rb"))
base = {p.id: p for p in initial}


def flags(plan, ref):
    d = abs(plan.etd - ref.etd) > 1e-6
    path = plan.path != ref.path
    if len(plan.speed_profile) == len(ref.speed_profile):
        speed = not np.allclose(plan.speed_profile, ref.speed_profile, atol=1e-9)
    else:
        speed = False
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


changed1 = [p.id for p in stage1 if p.id in base and any(flags(p, base[p.id]))]
changed2 = [p.id for p in final if p.id in base and any(flags(p, base[p.id]))]

print("=== stage 1 ===")
print(f"changed key flights: {len(changed1)}")
c1 = Counter(label(p, base[p.id]) for p in stage1 if p.id in changed1)
print("strategy:", dict(c1))
for p in sorted(stage1, key=lambda x: x.id):
    if p.id not in changed1:
        continue
    ref = base[p.id]
    d, path, speed = flags(p, ref)
    extra = []
    if d:
        extra.append(f"dETD={p.etd - ref.etd:+.1f}s")
    if path:
        diff = [i for i, (a, b) in enumerate(zip(ref.path, p.path)) if a != b]
        extra.append(f"path_diff@{min(diff) if diff else -1}..{max(diff) if diff else -1}")
    if speed:
        idx = [i for i, (a, b) in enumerate(zip(ref.speed_profile, p.speed_profile)) if abs(a - b) > 1e-9]
        extra.append(f"speed_seg={len(idx)}")
    print(f"  flight {p.id:>3}: {label(p, ref):<12} " + " ".join(extra))

trace = pd.read_csv(f"{OUT}/optimization_trace.csv")
acc = trace[trace["accepted"] == True].copy()
acc["fid"] = pd.to_numeric(acc["plan_id"], errors="coerce")
acc_plan = acc.dropna(subset=["fid"]).copy()
acc_plan["fid"] = acc_plan["fid"].astype(int)

print("")
print("=== stage 2: first touch ===")
prev = set(changed1)
unattributed_order = ["greedy_time", "stage2_independent_matching", "local_reroute", "stage2_adm_fata"]
for stage in unattributed_order:
    rows = acc_plan[acc_plan["stage"] == stage]
    ids = set(rows["fid"])
    if not ids and stage not in acc_plan["stage"].unique():
        print(f"  {stage:<30} actions=0")
        continue
    new = ids - prev
    print(
        f"  {stage:<30} actions={len(rows):>3}  unique_plans={len(ids):>2}  "
        f"new_plans={len(new):>2}  {sorted(new) if new else ''}"
    )
    prev |= ids

print("")
print("=== stage 2: accepted action types ===")
print(acc.groupby(["stage", "action_type"]).size().to_string())
if "matched_strategy" in acc.columns:
    im = acc[acc["stage"] == "stage2_independent_matching"]
    print("IM matched_strategy:", dict(im["matched_strategy"].value_counts()))

print("")
print("=== final ===")
print(f"changed flights: {len(changed2)} / {len(final)}")
cf = Counter(label(p, base[p.id]) for p in final if p.id in changed2)
print("strategy mix:", dict(cf))

d1, p1, s1 = zip(*(flags(p, base[p.id]) for p in final if p.id in changed2))
print(
    f"  delay-flagged={sum(d1)}  path-changed={sum(p1)}  speed-flagged={sum(s1)}"
)

# speed acceleration / deceleration among speed-changed flights (same path or not)
accel = decel = 0
for p in final:
    if p.id not in changed2:
        continue
    ref = base[p.id]
    d, path, speed = flags(p, ref)
    if speed:
        ms_new = float(np.mean(p.speed_profile))
        ms_ref = float(np.mean(ref.speed_profile))
        if ms_new > ms_ref + 1e-9:
            accel += 1
        elif ms_new < ms_ref - 1e-9:
            decel += 1
print(f"  speed mean increased={accel}  decreased={decel}")

# path change location
print("")
print("=== path-change location ===")
for p in final:
    if p.id not in changed2:
        continue
    ref = base[p.id]
    d, path, speed = flags(p, ref)
    if not path:
        continue
    n = min(len(ref.path), len(p.path))
    diff = [i for i in range(n) if ref.path[i] != p.path[i]]
    if not diff:
        continue
    first, last = min(diff), max(diff)
    print(
        f"  flight {p.id:>3}: diff_idx={first}..{last} / {n} "
        f"({first / max(n - 1, 1):.0%}..{last / max(n - 1, 1):.0%}) len {len(ref.path)}->{len(p.path)}"
    )

# air time
air0 = sum(p.total_air_time for p in initial)
air1 = sum(p.total_air_time for p in final)
print("")
print("=== summary ===")
print(f"initial conflicts: {count_conflict_pairs(detect_conflicts(initial, cfg, uncertain=True))} pairs / "
      f"{count_conflict_points(detect_conflicts(initial, cfg, uncertain=True))} points")
print(f"final conflicts  : {count_conflict_pairs(detect_conflicts(final, cfg, uncertain=True))} pairs / "
      f"{count_conflict_points(detect_conflicts(final, cfg, uncertain=True))} points")
print(f"changed flights  : {len(changed2)}")
print(f"delayed (>=30s)  : {sum(1 for p in final if p.delay >= 30.0)}")
print(f"total delay      : {sum(max(0.0, p.delay) for p in final):.1f} s")
print(f"air time change  : {air1 - air0:+.1f} s")
print("")
print("=== paper reference ===")
print("changed flights  : 24")
print("speed strategy   : 4 (1 deceleration)")
print("stage 1 dominant : delay")
print("stage 2 dominant : reroute")
print("air time change  : -181 s")
print("delayed flights  : <= 3 (3%)")
