from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .flight_plan import FlightPlan
from .grid import GridPoint
from .utils import ensure_dir

_DETECT_CALLS = 0
_DETECT_TIME_SECONDS = 0.0


@dataclass
class Conflict:
    plan_a: int
    plan_b: int
    cell: GridPoint
    idx_a: int
    idx_b: int
    time_a: float
    time_b: float
    time_gap: float
    required_gap: float
    conflict_type: str

    @property
    def flight_a(self) -> int:
        return self.plan_a

    @property
    def flight_b(self) -> int:
        return self.plan_b

    @property
    def time_diff(self) -> float:
        return abs(self.time_gap)

    @property
    def required_sep(self) -> float:
        return self.required_gap

    @property
    def uncertain(self) -> bool:
        return self.conflict_type == "uncertain"


@dataclass
class ConflictSegment:
    plan_a: int
    plan_b: int
    conflicts: list[Conflict] = field(default_factory=list)

    @property
    def cells(self) -> list[GridPoint]:
        return [c.cell for c in self.conflicts]

    @property
    def start_idx_a(self) -> int:
        return min((c.idx_a for c in self.conflicts), default=-1)

    @property
    def end_idx_a(self) -> int:
        return max((c.idx_a for c in self.conflicts), default=-1)

    @property
    def start_idx_b(self) -> int:
        return min((c.idx_b for c in self.conflicts), default=-1)

    @property
    def end_idx_b(self) -> int:
        return max((c.idx_b for c in self.conflicts), default=-1)

    @property
    def required_shift(self) -> float:
        if not self.conflicts:
            return 0.0
        return max(max(0.0, c.required_gap - abs(c.time_gap)) for c in self.conflicts)


@dataclass
class OccupancyEvent:
    plan_id: int
    idx: int
    cell: GridPoint
    t_nominal: float
    t_early: float
    t_late: float


def reset_conflict_detection_stats() -> None:
    global _DETECT_CALLS, _DETECT_TIME_SECONDS
    _DETECT_CALLS = 0
    _DETECT_TIME_SECONDS = 0.0


def get_conflict_detection_stats() -> dict[str, float]:
    return {"number_of_detect_conflicts_calls": float(_DETECT_CALLS), "detect_conflicts_time": float(_DETECT_TIME_SECONDS)}


def sigma_t(elapsed_time: float, cfg: dict) -> float:
    c = cfg["conflict"]
    return float(c["sigma0"]) + float(c["sigma_rate"]) * max(0.0, float(elapsed_time))


def occupancy(plan: FlightPlan) -> dict[GridPoint, tuple[int, float]]:
    occ: dict[GridPoint, tuple[int, float]] = {}
    for idx, (cell, t) in enumerate(zip(plan.path, plan.eta_times)):
        old = occ.get(cell)
        if old is None or t < old[1]:
            occ[cell] = (idx, float(t))
    return occ


def build_occupancy_events(plans: list[FlightPlan], cfg: dict | None = None, uncertain: bool = True) -> dict[GridPoint, list[OccupancyEvent]]:
    occupancy_by_cell: dict[GridPoint, list[OccupancyEvent]] = {}
    for plan in plans:
        for idx, (cell, t) in enumerate(zip(plan.path, plan.eta_times)):
            sigma = sigma_t(float(t) - float(plan.etd), cfg) if cfg is not None and uncertain else 0.0
            event = OccupancyEvent(
                plan_id=plan.id,
                idx=idx,
                cell=cell,
                t_nominal=float(t),
                t_early=float(t - sigma),
                t_late=float(t + sigma),
            )
            occupancy_by_cell.setdefault(cell, []).append(event)
    return occupancy_by_cell


def detect_conflicts(
    plans: list[FlightPlan],
    cfg: dict,
    uncertain: bool = True,
    output_csv: str | Path | None = None,
    active_ids: set[int] | None = None,
) -> list[Conflict]:
    """Detect conflicts using a per-cell occupancy event index."""
    global _DETECT_CALLS, _DETECT_TIME_SECONDS
    started = time.perf_counter()
    _DETECT_CALLS += 1
    t_conflict = float(cfg["conflict"]["t_conflict"]) + float(cfg["conflict"].get("cell_occupancy_time", 0.0))
    alpha = float(cfg["conflict"]["alpha"])
    z_score = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    by_id = {plan.id: plan for plan in plans}
    occupancy_by_cell = build_occupancy_events(plans, cfg, uncertain=uncertain)
    if uncertain:
        max_sigma = max((sigma_t(event.t_nominal - by_id[event.plan_id].etd, cfg) for events in occupancy_by_cell.values() for event in events), default=0.0)
        time_window = t_conflict + z_score * 2.0 * max_sigma
    else:
        time_window = t_conflict
    conflicts: list[Conflict] = []
    seen: set[tuple[int, int, GridPoint, int, int, str]] = set()

    for cell, events in occupancy_by_cell.items():
        events.sort(key=lambda event: event.t_nominal)
        for i, event_a in enumerate(events):
            for event_b in events[i + 1 :]:
                if event_b.t_nominal - event_a.t_nominal > time_window:
                    break
                if event_a.plan_id == event_b.plan_id:
                    continue
                if active_ids is not None and event_a.plan_id not in active_ids and event_b.plan_id not in active_ids:
                    continue
                if event_a.plan_id <= event_b.plan_id:
                    a, b = event_a, event_b
                else:
                    a, b = event_b, event_a
                key = (a.plan_id, b.plan_id, cell, a.idx, b.idx, "uncertain" if uncertain else "deterministic")
                if key in seen:
                    continue
                seen.add(key)
                gap = a.t_nominal - b.t_nominal
                diff = abs(gap)
                if uncertain:
                    sa = sigma_t(a.t_nominal - by_id[a.plan_id].etd, cfg)
                    sb = sigma_t(b.t_nominal - by_id[b.plan_id].etd, cfg)
                    required = t_conflict + z_score * (sa + sb)
                else:
                    required = t_conflict
                if diff > required:
                    continue
                conflicts.append(
                    Conflict(
                        plan_a=a.plan_id,
                        plan_b=b.plan_id,
                        cell=cell,
                        idx_a=a.idx,
                        idx_b=b.idx,
                        time_a=a.t_nominal,
                        time_b=b.t_nominal,
                        time_gap=gap,
                        required_gap=required,
                        conflict_type="uncertain" if uncertain else "deterministic",
                    )
                )

    if output_csv is not None:
        write_conflicts_csv(conflicts, output_csv)
    _DETECT_TIME_SECONDS += time.perf_counter() - started
    return conflicts


def write_conflicts_csv(conflicts: list[Conflict], output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    ensure_dir(output_csv.parent)
    rows = [_conflict_row(conflict) for conflict in conflicts]
    columns = [
        "plan_a",
        "plan_b",
        "cell_x",
        "cell_y",
        "cell_z",
        "idx_a",
        "idx_b",
        "time_a",
        "time_b",
        "time_gap",
        "time_diff",
        "required_gap",
        "conflict_type",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(output_csv, index=False)


def _conflict_row(conflict: Conflict) -> dict[str, object]:
    return {
        "plan_a": conflict.plan_a,
        "plan_b": conflict.plan_b,
        "cell_x": conflict.cell[0],
        "cell_y": conflict.cell[1],
        "cell_z": conflict.cell[2],
        "idx_a": conflict.idx_a,
        "idx_b": conflict.idx_b,
        "time_a": conflict.time_a,
        "time_b": conflict.time_b,
        "time_gap": conflict.time_gap,
        "time_diff": conflict.time_diff,
        "required_gap": conflict.required_gap,
        "conflict_type": conflict.conflict_type,
    }


def conflict_cells(conflicts: list[Conflict]) -> list[GridPoint]:
    return [c.cell for c in conflicts]


def conflict_count_by_flight(conflicts: list[Conflict]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for c in conflicts:
        counts[c.flight_a] = counts.get(c.flight_a, 0) + 1
        counts[c.flight_b] = counts.get(c.flight_b, 0) + 1
    return counts


def count_conflict_points(conflicts: list[Conflict]) -> int:
    return len(conflicts)


def count_conflict_pairs(conflicts: list[Conflict]) -> int:
    return len({(min(c.plan_a, c.plan_b), max(c.plan_a, c.plan_b)) for c in conflicts})


def count_conflict_edges(conflicts: list[Conflict]) -> int:
    return count_conflict_pairs(conflicts)


def group_continuous_conflicts(conflicts: list[Conflict]) -> list[ConflictSegment]:
    by_pair: dict[tuple[int, int], list[Conflict]] = {}
    for conflict in conflicts:
        key = (min(conflict.plan_a, conflict.plan_b), max(conflict.plan_a, conflict.plan_b))
        by_pair.setdefault(key, []).append(conflict)

    segments: list[ConflictSegment] = []
    for (plan_a, plan_b), group in by_pair.items():
        ordered = sorted(group, key=lambda c: (c.idx_a, c.idx_b, c.time_a, c.cell))
        current: list[Conflict] = []
        previous: Conflict | None = None
        for conflict in ordered:
            if previous is None or _is_continuous(previous, conflict):
                current.append(conflict)
            else:
                segments.append(ConflictSegment(plan_a=plan_a, plan_b=plan_b, conflicts=current))
                current = [conflict]
            previous = conflict
        if current:
            segments.append(ConflictSegment(plan_a=plan_a, plan_b=plan_b, conflicts=current))
    return sorted(segments, key=lambda s: (-len(s.conflicts), s.plan_a, s.plan_b, s.start_idx_a))


def _is_continuous(a: Conflict, b: Conflict) -> bool:
    idx_adjacent = abs(a.idx_a - b.idx_a) <= 1 or abs(a.idx_b - b.idx_b) <= 1
    cell_adjacent = max(abs(a.cell[i] - b.cell[i]) for i in range(3)) <= 1
    return idx_adjacent or cell_adjacent


def write_conflict_diagnostics(
    conflicts: list[Conflict],
    output_dir: str | Path,
    repair_log: list[dict[str, object]] | None = None,
) -> None:
    out = ensure_dir(output_dir)
    repair_log = repair_log or []
    _write_unresolved_conflicts(out / "unresolved_conflicts.csv", conflicts, repair_log)
    _write_conflicts_by_pair(out / "final_conflicts_by_pair.csv", conflicts)
    _write_conflict_segments(out / "final_conflict_segments.csv", group_continuous_conflicts(conflicts))


def _write_unresolved_conflicts(path: Path, conflicts: list[Conflict], repair_log: list[dict[str, object]]) -> None:
    fields = [
        "plan_a",
        "plan_b",
        "cell_x",
        "cell_y",
        "cell_z",
        "time_a",
        "time_b",
        "time_gap",
        "required_gap",
        "tried_delay_a",
        "tried_delay_b",
        "tried_speed_a",
        "tried_speed_b",
        "tried_reroute",
        "reason_failed",
    ]
    attempts: dict[tuple[int, int], set[str]] = {}
    for item in repair_log:
        pair = item.get("pair")
        if not isinstance(pair, tuple) or len(pair) != 2:
            continue
        attempts.setdefault((min(int(pair[0]), int(pair[1])), max(int(pair[0]), int(pair[1]))), set()).add(str(item.get("action", "")))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for conflict in conflicts:
            key = (min(conflict.plan_a, conflict.plan_b), max(conflict.plan_a, conflict.plan_b))
            actions = attempts.get(key, set())
            writer.writerow(
                {
                    "plan_a": conflict.plan_a,
                    "plan_b": conflict.plan_b,
                    "cell_x": conflict.cell[0],
                    "cell_y": conflict.cell[1],
                    "cell_z": conflict.cell[2],
                    "time_a": conflict.time_a,
                    "time_b": conflict.time_b,
                    "time_gap": conflict.time_gap,
                    "required_gap": conflict.required_gap,
                    "tried_delay_a": any("delay_a" in action for action in actions),
                    "tried_delay_b": any("delay_b" in action for action in actions),
                    "tried_speed_a": any("speed_a" in action for action in actions),
                    "tried_speed_b": any("speed_b" in action for action in actions),
                    "tried_reroute": any("reroute" in action for action in actions),
                    "reason_failed": "remaining_after_repair" if conflicts else "",
                }
            )


def _write_conflicts_by_pair(path: Path, conflicts: list[Conflict]) -> None:
    rows = []
    for (a, b), group in _group_by_pair(conflicts).items():
        rows.append(
            {
                "plan_a": a,
                "plan_b": b,
                "conflict_points": len(group),
                "min_time_diff": min(c.time_diff for c in group),
                "max_required_gap": max(c.required_gap for c in group),
                "cells": ";".join(f"{c.cell[0]}:{c.cell[1]}:{c.cell[2]}" for c in group),
            }
        )
    pd.DataFrame(rows, columns=["plan_a", "plan_b", "conflict_points", "min_time_diff", "max_required_gap", "cells"]).to_csv(path, index=False)


def _write_conflict_segments(path: Path, segments: list[ConflictSegment]) -> None:
    rows = []
    for idx, segment in enumerate(segments):
        rows.append(
            {
                "segment_id": idx,
                "plan_a": segment.plan_a,
                "plan_b": segment.plan_b,
                "conflict_points": len(segment.conflicts),
                "start_idx_a": segment.start_idx_a,
                "end_idx_a": segment.end_idx_a,
                "start_idx_b": segment.start_idx_b,
                "end_idx_b": segment.end_idx_b,
                "required_shift": segment.required_shift,
                "cells": ";".join(f"{c.cell[0]}:{c.cell[1]}:{c.cell[2]}" for c in segment.conflicts),
            }
        )
    pd.DataFrame(
        rows,
        columns=[
            "segment_id",
            "plan_a",
            "plan_b",
            "conflict_points",
            "start_idx_a",
            "end_idx_a",
            "start_idx_b",
            "end_idx_b",
            "required_shift",
            "cells",
        ],
    ).to_csv(path, index=False)


def _group_by_pair(conflicts: list[Conflict]) -> dict[tuple[int, int], list[Conflict]]:
    by_pair: dict[tuple[int, int], list[Conflict]] = {}
    for conflict in conflicts:
        key = (min(conflict.plan_a, conflict.plan_b), max(conflict.plan_a, conflict.plan_b))
        by_pair.setdefault(key, []).append(conflict)
    return by_pair


def write_calibration_report(no_uncertain: int, uncertain: int, output_dir: str | Path) -> None:
    out = ensure_dir(output_dir)
    target_no = 53
    target_un = 97
    ok_no = abs(no_uncertain - target_no) <= max(15, target_no * 0.45)
    ok_un = abs(uncertain - target_un) <= max(20, target_un * 0.45)
    status = "within broad reproduction tolerance" if ok_no and ok_un else "outside broad reproduction tolerance"
    text = (
        "# Calibration Report\n\n"
        f"- no uncertainty conflicts: {no_uncertain} (paper reference about {target_no})\n"
        f"- uncertainty conflicts: {uncertain} (paper reference about {target_un})\n"
        f"- status: {status}\n\n"
        "The counts are generated from a configurable random city, A* routes, and the sigma_t model. "
        "They are not hard-coded. Tune `conflict.sigma0`, `conflict.sigma_rate`, and the traffic pulse settings "
        "to move the default scenario closer to the paper tables if desired.\n"
    )
    (out / "calibration_report.md").write_text(text, encoding="utf-8")


def summarize_conflicts(conflicts: list[Conflict]) -> dict[str, float]:
    if not conflicts:
        return {"points": 0, "pairs": 0, "edges": 0, "mean_time_diff": 0.0, "mean_required_sep": 0.0}
    return {
        "points": count_conflict_points(conflicts),
        "pairs": count_conflict_pairs(conflicts),
        "edges": count_conflict_edges(conflicts),
        "mean_time_diff": float(np.mean([c.time_diff for c in conflicts])),
        "mean_required_sep": float(np.mean([c.required_sep for c in conflicts])),
    }
