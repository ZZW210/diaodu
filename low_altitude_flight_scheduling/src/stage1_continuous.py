"""Continuous-mixed stage-1 decision space, ported from the desktop project.

Every key flight owns one block of genes:

- ``strategy_gene``: continuous ``[0, 3)``, floored to 0 / 1 / 2 meaning
  schedule (ATD shift), speed (per-segment speed profile) or reroute (local
  via-cell detour).  This is the mixed (integer) part of the search space.
- ``atd_gene``: continuous departure time inside the paper-style bounds
  ``[max(t_ATD_range[0], etd - max_advance), min(t_ATD_range[1], etd + t_delay_max)]``.
- ``speed_genes``: one continuous speed per path segment inside ``speed_range``.
- ``via_genes``: three continuous coordinates (x, y, z) per local reroute
  interval; they are rounded to integer cells before the A* detour is built.

The decoder mirrors the desktop ``paper_optimization`` stage-1 decode while the
objective stays this project's engineering fitness (conflict pairs / points plus
soft cost), so experiments can isolate the effect of the decision space.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .astar_3d import astar_path
from .conflict_detection import (
    Conflict,
    count_conflict_pairs,
    count_conflict_points,
    detect_conflicts,
)
from .flight_plan import FlightPlan
from .grid import AirspaceGrid, GridPoint
from .optimization_model import Evaluation, _soft_cost

INF_FITNESS = 1e30


class InfeasibleRoute(ValueError):
    """Raised when a candidate route violates endpoints, obstacles or moves."""


@dataclass
class Stage1Block:
    flight_id: int
    strategy_gene: int
    atd_gene: int
    speed_genes: slice
    segments: list[int]
    via_genes: list[slice]
    via_intervals: list[tuple[int, int]]
    incident_conflicts: list[int]


@dataclass
class Stage1Layout:
    blocks: list[Stage1Block]
    lower: np.ndarray
    upper: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.lower.size)


class RouteCache:
    """LRU cache for ``start -> via -> end`` A* detours.

    Via coordinates are rounded to cells, so the cache saturates quickly and the
    continuous search does not re-run A* for the same detour.
    """

    def __init__(self, maxsize: int = 32768, enabled: bool = True) -> None:
        self.maxsize = max(1, int(maxsize))
        self.enabled = bool(enabled)
        self.entries: OrderedDict[tuple, list[GridPoint] | None] = OrderedDict()
        self.environment: tuple | None = None

    def bind(self, grid: AirspaceGrid, risk_map: np.ndarray, cfg: dict) -> None:
        self.environment = (
            hashlib.sha256(np.ascontiguousarray(grid.obstacles).tobytes()).hexdigest(),
            hashlib.sha256(np.ascontiguousarray(risk_map, dtype=float).tobytes()).hexdigest(),
            tuple(grid.shape),
            tuple(grid.cell_size),
            float(cfg["risk"]["alpha_r"]),
            float(cfg["risk"]["alpha_L"]),
        )

    def route(self, start: GridPoint, via: GridPoint, end: GridPoint, grid: AirspaceGrid, risk_map: np.ndarray, cfg: dict) -> list[GridPoint] | None:
        key = (start, via, end, self.environment)
        if self.enabled and key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        options = {
            "alpha_r": float(cfg["risk"]["alpha_r"]),
            "alpha_l": float(cfg["risk"]["alpha_L"]),
            "max_expansions": int(cfg.get("paper_encoding", {}).get("astar_max_expansions", 45000)),
        }
        first = astar_path(grid, start, via, risk_map, **options)
        second = None
        if first:
            second = astar_path(grid, via, end, risk_map, **options)
        local = first + second[1:] if first and second else None
        if self.enabled:
            self.entries[key] = local
            if len(self.entries) > self.maxsize:
                self.entries.popitem(last=False)
        return local


def decode_primary_strategy(value: float) -> int:
    return int(np.clip(np.floor(float(value)), 0, 2))


def _atd_bounds(plan: FlightPlan, cfg: dict) -> tuple[float, float]:
    opt = cfg["optimization"]
    low, high = [float(v) for v in opt.get("stage1_atd_range", opt["t_ATD_range"])]
    if bool(opt.get("stage1_respect_delay_max", True)):
        low = max(low, float(plan.etd) - float(opt.get("stage1_max_advance_seconds", 600.0)))
        high = min(high, float(plan.etd) + float(opt["t_delay_max"]))
    if low > high:
        raise ValueError(f"Flight {plan.id} has no feasible ATD interval")
    return low, high


def _local_intervals(indices: list[int], path_length: int, window: int, merge_window: int) -> list[tuple[int, int]]:
    groups: list[list[int]] = []
    for idx in sorted(set(int(i) for i in indices)):
        if groups and idx - groups[-1][-1] <= merge_window:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    intervals = [(max(0, g[0] - window), min(path_length - 1, g[-1] + window)) for g in groups]
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def build_stage1_layout(
    plans: list[FlightPlan],
    key_ids: list[int],
    conflicts: list[Conflict],
    cfg: dict,
    grid: AirspaceGrid,
) -> Stage1Layout:
    """Build the continuous-mixed gene layout for the key flights."""

    by_id = {plan.id: plan for plan in plans}
    encoding = cfg.get("paper_encoding", {})
    window = max(1, int(encoding.get("local_window_segments", 4)))
    merge_window = int(encoding.get("reroute_merge_window", 5))
    margin = np.array(
        [float(encoding.get("local_margin_xy", 5))] * 2 + [float(encoding.get("local_margin_z", 1))]
    )
    speed_min, speed_max = [float(v) for v in cfg["optimization"]["speed_range"]]

    lower: list[float] = []
    upper: list[float] = []
    blocks: list[Stage1Block] = []
    for fid in key_ids:
        plan = by_id.get(fid)
        if plan is None:
            continue
        indices: list[int] = []
        conflict_indices: dict[int, int] = {}
        for ci, conflict in enumerate(conflicts):
            if fid not in (conflict.plan_a, conflict.plan_b):
                continue
            idx = conflict.idx_a if fid == conflict.plan_a else conflict.idx_b
            if not 0 <= idx < len(plan.path) or plan.path[idx] != conflict.cell:
                raise ValueError("Conflict indices must refer to the stage input path")
            indices.append(idx)
            conflict_indices[ci] = idx

        strategy_gene = len(lower)
        lower.append(0.0)
        upper.append(3.0)

        atd_gene = len(lower)
        atd_low, atd_high = _atd_bounds(plan, cfg)
        lower.append(atd_low)
        upper.append(atd_high)

        segments = list(range(max(0, len(plan.path) - 1)))
        speed_genes = slice(len(lower), len(lower) + len(segments))
        lower.extend([speed_min] * len(segments))
        upper.extend([speed_max] * len(segments))

        intervals = _local_intervals(indices, len(plan.path), window, merge_window)
        if not intervals and len(plan.path) > 1:
            intervals = _local_intervals([len(plan.path) // 2], len(plan.path), window, merge_window)
        via_genes: list[slice] = []
        for start, end in intervals:
            via_genes.append(slice(len(lower), len(lower) + 3))
            points = [plan.path[start], plan.path[end]] + [plan.path[idx] for idx in indices if start <= idx <= end]
            if len(points) == 2:
                points.append(plan.path[(start + end) // 2])
            points_arr = np.asarray(points, dtype=float)
            lower.extend(np.maximum(0, points_arr.min(axis=0) - margin).astype(float).tolist())
            upper.extend(np.minimum(np.asarray(grid.shape, dtype=float) - 1, points_arr.max(axis=0) + margin).astype(float).tolist())

        blocks.append(
            Stage1Block(
                flight_id=fid,
                strategy_gene=strategy_gene,
                atd_gene=atd_gene,
                speed_genes=speed_genes,
                segments=segments,
                via_genes=via_genes,
                via_intervals=intervals,
                incident_conflicts=list(conflict_indices),
            )
        )
    return Stage1Layout(blocks=blocks, lower=np.asarray(lower, dtype=float), upper=np.asarray(upper, dtype=float))


def _valid_path(path: list[GridPoint], plan: FlightPlan, grid: AirspaceGrid) -> bool:
    return (
        bool(path)
        and path[0] == plan.start
        and path[-1] == plan.goal
        and all(grid.is_free(c) for c in path)
        and all(max(abs(a[j] - b[j]) for j in range(3)) <= 1 and a != b for a, b in zip(path, path[1:]))
    )


def _recompute_timing(plan: FlightPlan, grid: AirspaceGrid, risk_map: np.ndarray) -> None:
    distances = np.linalg.norm(
        np.diff(np.asarray(plan.path, dtype=float), axis=0) * np.asarray(grid.cell_size, dtype=float), axis=1
    )
    speeds = np.asarray(plan.speed_profile, dtype=float)
    plan.eta_times = [float(plan.etd)] + (float(plan.etd) + np.cumsum(distances / speeds)).tolist()
    plan.total_air_time = float(max(0.0, plan.eta_times[-1] - plan.etd))
    plan.risk_sum = float(sum(float(risk_map[c]) for c in plan.path))


class _ViaBlock(Protocol):
    """Minimal interface shared by stage-1 and stage-2 gene blocks."""

    via_intervals: list[tuple[int, int]]
    via_genes: list[slice]


def _via_route(
    plan: FlightPlan,
    block: _ViaBlock,
    vector: np.ndarray,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    cfg: dict,
    route_cache: RouteCache,
    active_windows: set[int] | None = None,
) -> tuple[list[GridPoint], list[float]]:
    path: list[GridPoint] = []
    speeds: list[float] = []
    cursor = 0
    for wi, (interval, genes) in enumerate(zip(block.via_intervals, block.via_genes)):
        if active_windows is not None and wi not in active_windows:
            continue
        start, end = interval
        via = tuple(int(v) for v in np.rint(vector[genes]))
        if not (grid.in_bounds(via) and grid.is_free(via)):
            raise InfeasibleRoute("Via-cell intersects an obstacle")
        endpoints = plan.path[start], plan.path[end]
        original_local = plan.path[start : end + 1]
        local = route_cache.route(endpoints[0], via, endpoints[1], grid, risk_map, cfg)
        if not local:
            raise InfeasibleRoute("Via-cell is unreachable")
        path.extend(plan.path[cursor:start])
        speeds.extend(plan.speed_profile[cursor:start])
        path.extend(local[:-1])
        if local == original_local:
            speeds.extend(plan.speed_profile[start:end])
        else:
            old_dist = np.linalg.norm(
                np.diff(np.asarray(original_local, dtype=float), axis=0) * np.asarray(grid.cell_size, dtype=float), axis=1
            )
            new_dist = np.linalg.norm(
                np.diff(np.asarray(local, dtype=float), axis=0) * np.asarray(grid.cell_size, dtype=float), axis=1
            )
            old_edges = np.concatenate([[0.0], np.cumsum(old_dist)]) / max(float(old_dist.sum()), 1e-9)
            new_mid = (np.cumsum(new_dist) - 0.5 * new_dist) / max(float(new_dist.sum()), 1e-9)
            projected = np.searchsorted(old_edges[1:], new_mid, side="right")
            projected = np.clip(projected, 0, max(0, end - start - 1))
            speeds.extend([float(plan.speed_profile[start + int(i)]) for i in projected])
        cursor = end
    path.extend(plan.path[cursor:])
    speeds.extend(plan.speed_profile[cursor:])
    return path, speeds


def decode_stage1_solution(
    vector: np.ndarray,
    plans: list[FlightPlan],
    layout: Stage1Layout,
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    route_cache: RouteCache | None = None,
) -> list[FlightPlan]:
    """Decode one continuous-mixed stage-1 individual into flight plans."""

    vector = np.asarray(vector, dtype=float)
    if vector.shape != (layout.dim,):
        raise ValueError("Incorrect stage-1 decision vector dimension")
    original = {plan.id: plan for plan in plans}
    speed_min, speed_max = [float(v) for v in cfg["optimization"]["speed_range"]]
    cache = route_cache if route_cache is not None else RouteCache(int(cfg.get("paper_encoding", {}).get("astar_cache_size", 32768)))
    cache.bind(grid, risk_map, cfg)
    by_idx = {plan.id: idx for idx, plan in enumerate(plans)}
    out = list(plans)
    for block in layout.blocks:
        idx = by_idx[block.flight_id]
        base = plans[idx]
        plan = base.copy()
        plan.path = list(base.path)
        plan.speed_profile = list(base.speed_profile)
        strategy = decode_primary_strategy(vector[block.strategy_gene])
        if strategy == 0:
            target = float(np.clip(vector[block.atd_gene], layout.lower[block.atd_gene], layout.upper[block.atd_gene]))
            plan.etd = target
        elif strategy == 1:
            for segment, gene in zip(block.segments, range(block.speed_genes.start, block.speed_genes.stop)):
                plan.speed_profile[segment] = float(np.clip(vector[gene], speed_min, speed_max))
        else:
            path, speeds = _via_route(plan, block, vector, grid, risk_map, cfg, cache)
            plan.path = path
            plan.speed_profile = speeds
        if not _valid_path(plan.path, plan, grid):
            raise InfeasibleRoute("Route violates endpoints, obstacles or 26-neighborhood")
        _recompute_timing(plan, grid, risk_map)
        source = original[block.flight_id]
        plan.delay = float(plan.etd - source.etd)
        plan.rerouted = plan.path != source.path
        plan.changed = bool(
            plan.rerouted
            or abs(plan.delay) > 1e-6
            or not np.allclose(np.asarray(plan.speed_profile, dtype=float), np.asarray(source.speed_profile, dtype=float), atol=1e-6, rtol=0.0)
        )
        out[idx] = plan
    return out


def evaluate_stage1_solution(
    vector: np.ndarray,
    plans: list[FlightPlan],
    layout: Stage1Layout,
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    route_cache: RouteCache | None = None,
) -> Evaluation:
    """Engineering fitness of a continuous-mixed stage-1 candidate.

    The formula matches ``evaluate_delay_speed_solution`` so the discrete and
    continuous decision spaces stay directly comparable.
    """

    try:
        decoded = decode_stage1_solution(vector, plans, layout, cfg, grid, risk_map, route_cache)
    except InfeasibleRoute:
        return Evaluation(INF_FITNESS, 10**9, 0.0, 0.0, 0.0, 0, 0, list(plans))
    conflicts = detect_conflicts(decoded, cfg, uncertain=True)
    pairs = count_conflict_pairs(conflicts)
    points = count_conflict_points(conflicts)
    soft = _soft_cost(decoded, cfg)
    fitness = float(1_000_000.0 * pairs + 10_000.0 * points + soft)
    total_delay = float(sum(max(0.0, p.delay) for p in decoded))
    total_air = float(sum(p.total_air_time for p in decoded))
    total_risk = float(sum(p.risk_sum for p in decoded))
    delayed = int(sum(1 for p in decoded if p.delay >= float(cfg["optimization"].get("delay_count_threshold", 30.0))))
    battery = int(sum(1 for p in decoded if p.total_air_time > float(cfg["optimization"]["t_battery"])))
    return Evaluation(fitness, points, total_delay, total_air, total_risk, delayed, battery, decoded)


def make_stage1_objective(
    plans: list[FlightPlan],
    layout: Stage1Layout,
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """Return the FATA objective with a shared route cache across evaluations."""

    cache = RouteCache(int(cfg.get("paper_encoding", {}).get("astar_cache_size", 32768)))

    def objective(vector: np.ndarray) -> float:
        return evaluate_stage1_solution(vector, plans, layout, cfg, grid, risk_map, cache).fitness

    return objective
