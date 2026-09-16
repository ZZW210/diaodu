"""Stage-2 solver: ADM strategy sampling coupled with FATA, ported from the desktop line.

This mirrors the desktop project's ``adm_matching.adm_fata_optimize`` loop:

1. every conflict owns one continuous actor gene in ``[0, 2]``; ``< 1`` makes
   ``plan_a`` the actor, otherwise ``plan_b``;
2. every involved flight owns one ATD gene, per-segment speed genes for the
   local windows of its conflicts, and three via coordinates per local reroute
   window;
3. each FATA generation samples one strategy label (0 delay / 1 speed /
   2 reroute) per conflict for every individual from a probability matrix ``P``
   (uniform at the start), decodes the individual under that strategy context
   and evaluates it with this project's engineering fitness;
4. after the generation finished the probability matrix is updated from the
   strategy frequencies of the dominant ``dominant_fraction`` of the population
   with learning-rate smoothing, then the next generation starts.

The decoder keeps the desktop strategy semantics: the actor flight of a
conflict applies the strategy; a flight aggregates the requirements of all
conflicts it acts on and applies delay, then speed, then reroute.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .conflict_detection import (
    Conflict,
    count_conflict_pairs,
    count_conflict_points,
    detect_conflicts,
)
from .fata import FATAResult, fata_optimize
from .flight_plan import FlightPlan
from .grid import AirspaceGrid
from .optimization_model import Evaluation, _soft_cost
from .stage1_continuous import (
    INF_FITNESS,
    InfeasibleRoute,
    RouteCache,
    _atd_bounds,
    _local_intervals,
    _recompute_timing,
    _valid_path,
    _via_route,
)

STRATEGY_NAMES = ("delay", "speed", "reroute")


@dataclass
class Stage2Block:
    """Genes of one flight inside the stage-2 decision space."""

    flight_id: int
    atd_gene: int
    atd_bounds: tuple[float, float]
    speed_genes: dict[int, int]
    via_genes: list[slice]
    via_intervals: list[tuple[int, int]]
    conflict_segments: dict[int, list[int]]
    interval_of_conflict: dict[int, int]
    incident_conflicts: list[int]


@dataclass
class Stage2Layout:
    blocks: list[Stage2Block]
    actor_genes: list[int]
    conflict_refs: list[tuple[int, int, int, int]]
    lower: np.ndarray
    upper: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.lower.size)


@dataclass
class Stage2Result:
    plans: list[FlightPlan]
    conflicts: list[Conflict]
    pairs_before: int
    points_before: int
    pairs_after: int
    points_after: int
    generations: int
    dim: int
    blocks: int
    accepted: bool
    best_strategies: np.ndarray
    probability: np.ndarray
    convergence: list[float] = field(default_factory=list)
    log: list[dict[str, object]] = field(default_factory=list)


def build_stage2_layout(
    plans: list[FlightPlan],
    conflicts: list[Conflict],
    cfg: dict,
    grid: AirspaceGrid,
) -> Stage2Layout:
    """Build the stage-2 gene layout around the current conflict set."""

    by_id = {plan.id: plan for plan in plans}
    encoding = cfg.get("paper_encoding", {})
    window = max(1, int(encoding.get("local_window_segments", 4)))
    merge_window = int(encoding.get("reroute_merge_window", 5))
    margin = np.array(
        [float(encoding.get("local_margin_xy", 5))] * 2 + [float(encoding.get("local_margin_z", 1))]
    )
    speed_min, speed_max = [float(v) for v in cfg["optimization"]["speed_range"]]

    involved: dict[int, dict[int, int]] = {}
    conflict_refs: list[tuple[int, int, int, int]] = []
    for ci, conflict in enumerate(conflicts):
        conflict_refs.append((conflict.plan_a, conflict.plan_b, conflict.idx_a, conflict.idx_b))
        for fid, idx in ((conflict.plan_a, conflict.idx_a), (conflict.plan_b, conflict.idx_b)):
            plan = by_id.get(fid)
            if plan is None:
                raise ValueError(f"Conflict {ci} references unknown flight {fid}")
            if not 0 <= idx < len(plan.path) or plan.path[idx] != conflict.cell:
                raise ValueError("Conflict indices must refer to the stage input path")
            involved.setdefault(fid, {})[ci] = idx

    lower: list[float] = []
    upper: list[float] = []
    actor_genes: list[int] = []
    for _ in conflicts:
        actor_genes.append(len(lower))
        lower.append(0.0)
        upper.append(2.0)

    blocks: list[Stage2Block] = []
    for fid, conflicts_of_flight in involved.items():
        plan = by_id[fid]
        indices = [conflicts_of_flight[ci] for ci in sorted(conflicts_of_flight)]

        atd_gene = len(lower)
        atd_low, atd_high = _atd_bounds(plan, cfg)
        lower.append(atd_low)
        upper.append(atd_high)

        conflict_segments: dict[int, list[int]] = {}
        for ci in sorted(conflicts_of_flight):
            idx = conflicts_of_flight[ci]
            start = max(0, idx - window)
            end = min(len(plan.path) - 1, idx + window)
            conflict_segments[ci] = [segment for segment in range(start, end)]
        segments = sorted({segment for segs in conflict_segments.values() for segment in segs})
        speed_genes: dict[int, int] = {}
        for segment in segments:
            speed_genes[segment] = len(lower)
            lower.append(speed_min)
            upper.append(speed_max)

        intervals = _local_intervals(indices, len(plan.path), window, merge_window)
        via_genes: list[slice] = []
        for start, end in intervals:
            via_genes.append(slice(len(lower), len(lower) + 3))
            points = [plan.path[start], plan.path[end]] + [
                plan.path[idx] for idx in indices if start <= idx <= end
            ]
            if len(points) == 2:
                points.append(plan.path[(start + end) // 2])
            points_arr = np.asarray(points, dtype=float)
            lower.extend(np.maximum(0, points_arr.min(axis=0) - margin).astype(float).tolist())
            upper.extend(
                np.minimum(
                    np.asarray(grid.shape, dtype=float) - 1, points_arr.max(axis=0) + margin
                )
                .astype(float)
                .tolist()
            )
        interval_of_conflict: dict[int, int] = {}
        for ci in sorted(conflicts_of_flight):
            idx = conflicts_of_flight[ci]
            for wi, (start, end) in enumerate(intervals):
                if start <= idx <= end:
                    interval_of_conflict[ci] = wi
                    break

        blocks.append(
            Stage2Block(
                flight_id=fid,
                atd_gene=atd_gene,
                atd_bounds=(atd_low, atd_high),
                speed_genes=speed_genes,
                via_genes=via_genes,
                via_intervals=intervals,
                conflict_segments=conflict_segments,
                interval_of_conflict=interval_of_conflict,
                incident_conflicts=sorted(conflicts_of_flight),
            )
        )
    return Stage2Layout(
        blocks=blocks,
        actor_genes=actor_genes,
        conflict_refs=conflict_refs,
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
    )


def decode_stage2_solution(
    vector: np.ndarray,
    plans: list[FlightPlan],
    layout: Stage2Layout,
    strategies: np.ndarray,
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    route_cache: RouteCache | None = None,
) -> list[FlightPlan]:
    """Decode one stage-2 individual under a sampled strategy context."""

    vector = np.asarray(vector, dtype=float)
    if vector.shape != (layout.dim,):
        raise ValueError("Incorrect stage-2 decision vector dimension")
    strategies = np.asarray(strategies, dtype=int)
    if strategies.shape != (len(layout.actor_genes),):
        raise ValueError("Strategy context dimension mismatch")

    speed_min, speed_max = [float(v) for v in cfg["optimization"]["speed_range"]]
    cache = route_cache if route_cache is not None else RouteCache(
        int(cfg.get("paper_encoding", {}).get("astar_cache_size", 32768))
    )
    cache.bind(grid, risk_map, cfg)
    source = {plan.id: plan for plan in plans}
    by_idx = {plan.id: idx for idx, plan in enumerate(plans)}

    requirements: dict[int, dict[int, list[int]]] = {}
    for ci, (plan_a, plan_b, _idx_a, _idx_b) in enumerate(layout.conflict_refs):
        actor = plan_a if float(vector[layout.actor_genes[ci]]) < 1.0 else plan_b
        requirements.setdefault(actor, {}).setdefault(int(strategies[ci]), []).append(ci)

    out = list(plans)
    for block in layout.blocks:
        idx = by_idx.get(block.flight_id)
        if idx is None:
            continue
        base = plans[idx]
        needs = requirements.get(block.flight_id)
        if not needs:
            continue
        plan = base.copy()
        plan.path = list(base.path)
        plan.speed_profile = list(base.speed_profile)

        if 0 in needs:
            etd_low, etd_high = block.atd_bounds
            plan.etd = float(np.clip(vector[block.atd_gene], etd_low, etd_high))
        if 1 in needs:
            for ci in needs[1]:
                for segment in block.conflict_segments.get(ci, ()):
                    gene = block.speed_genes.get(segment)
                    if gene is not None:
                        plan.speed_profile[segment] = float(
                            np.clip(vector[gene], speed_min, speed_max)
                        )
        if 2 in needs:
            active = {
                block.interval_of_conflict[ci]
                for ci in needs[2]
                if ci in block.interval_of_conflict
            }
            if active:
                path, speeds = _via_route(
                    plan, block, vector, grid, risk_map, cfg, cache, active_windows=active
                )
                plan.path = path
                plan.speed_profile = speeds

        if not _valid_path(plan.path, plan, grid):
            raise InfeasibleRoute("Route violates endpoints, obstacles or 26-neighborhood")
        _recompute_timing(plan, grid, risk_map)
        src = source[block.flight_id]
        plan.delay = float(plan.etd - src.etd)
        plan.rerouted = plan.path != src.path
        plan.changed = bool(
            plan.rerouted
            or abs(plan.delay) > 1e-6
            or not np.allclose(
                np.asarray(plan.speed_profile, dtype=float),
                np.asarray(src.speed_profile, dtype=float),
                atol=1e-6,
                rtol=0.0,
            )
        )
        out[idx] = plan
    return out


def evaluate_stage2_solution(
    vector: np.ndarray,
    plans: list[FlightPlan],
    layout: Stage2Layout,
    strategies: np.ndarray,
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    route_cache: RouteCache | None = None,
) -> Evaluation:
    """Engineering fitness of a stage-2 candidate under its strategy context."""

    try:
        decoded = decode_stage2_solution(
            vector, plans, layout, strategies, cfg, grid, risk_map, route_cache
        )
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
    delayed = int(
        sum(
            1
            for p in decoded
            if p.delay >= float(cfg["optimization"].get("delay_count_threshold", 30.0))
        )
    )
    battery = int(sum(1 for p in decoded if p.total_air_time > float(cfg["optimization"]["t_battery"])))
    return Evaluation(fitness, points, total_delay, total_air, total_risk, delayed, battery, decoded)


def adm_fata_stage2(
    plans: list[FlightPlan],
    conflicts: list[Conflict],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    seed: int = 206,
) -> Stage2Result:
    """Run the coupled ADM sampling + FATA search on the current conflicts."""

    opt = cfg["optimization"]
    pairs_before = count_conflict_pairs(conflicts)
    points_before = count_conflict_points(conflicts)
    started = time.perf_counter()

    if not conflicts:
        return Stage2Result(
            plans=list(plans),
            conflicts=list(conflicts),
            pairs_before=0,
            points_before=0,
            pairs_after=0,
            points_after=0,
            generations=0,
            dim=0,
            blocks=0,
            accepted=False,
            best_strategies=np.zeros(0, dtype=int),
            probability=np.zeros((0, 3)),
        )

    layout = build_stage2_layout(plans, conflicts, cfg, grid)
    n_conflicts = len(layout.conflict_refs)
    probability = np.full((n_conflicts, 3), 1.0 / 3.0)
    dominant_fraction = float(np.clip(float(opt.get("stage2_fata_dominant_fraction", 0.2)), 0.02, 1.0))
    learning_rate = float(np.clip(float(opt.get("stage2_fata_learning_rate", 0.5)), 0.0, 1.0))
    generations = max(1, int(opt.get("stage2_fata_generations", cfg["fata"]["Ngen_max"])))
    cache = RouteCache(int(cfg.get("paper_encoding", {}).get("astar_cache_size", 32768)))

    def generation_context(generation: int, size: int, rng: np.random.Generator) -> np.ndarray:
        draws = rng.random((size, n_conflicts))
        cumulative = np.cumsum(probability, axis=1)[None, :, :]
        return np.sum(draws[:, :, None] >= cumulative, axis=2).clip(0, 2)

    def objective_with_context(vector: np.ndarray, generation: int, context: np.ndarray) -> float:
        return float(
            evaluate_stage2_solution(
                vector, plans, layout, context, cfg, grid, risk_map, cache
            ).fitness
        )

    def on_generation_evaluated(
        generation: int,
        positions: np.ndarray,
        fitness: np.ndarray,
        contexts: np.ndarray,
    ) -> None:
        nonlocal probability
        dominant = max(1, int(round(fitness.shape[0] * dominant_fraction)))
        ranked = np.argsort(fitness, kind="stable")[:dominant]
        winners = contexts[ranked]
        frequencies = np.stack([(winners == s).mean(axis=0) for s in range(3)], axis=1)
        probability = (1.0 - learning_rate) * probability + learning_rate * frequencies
        row_sums = probability.sum(axis=1, keepdims=True)
        probability = probability / np.where(row_sums <= 0.0, 1.0, row_sums)

    fallback_context = np.argmax(probability, axis=1).astype(int)

    def objective(vector: np.ndarray) -> float:
        return float(
            evaluate_stage2_solution(
                vector, plans, layout, fallback_context, cfg, grid, risk_map, cache
            ).fitness
        )

    result: FATAResult = fata_optimize(
        objective,
        layout.lower,
        layout.upper,
        dim=layout.dim,
        population=int(cfg["fata"]["NP"]),
        max_iter=generations,
        seed=seed,
        improved=True,
        parf=float(cfg["fata"]["Parf"]),
        generation_context=generation_context,
        objective_with_context=objective_with_context,
        on_generation_evaluated=on_generation_evaluated,
    )

    best_context = result.best_context
    if best_context is None:
        best_context = np.argmax(probability, axis=1).astype(int)
    best_context = np.asarray(best_context, dtype=int)

    best_plans = decode_stage2_solution(
        result.best_position, plans, layout, best_context, cfg, grid, risk_map, cache
    )
    remaining = detect_conflicts(best_plans, cfg, uncertain=True)
    pairs_after = count_conflict_pairs(remaining)
    points_after = count_conflict_points(remaining)
    accepted = pairs_after < pairs_before or (pairs_after == pairs_before and points_after < points_before)
    runtime_ms = (time.perf_counter() - started) * 1000.0
    if not accepted:
        best_plans = list(plans)
        remaining = list(conflicts)
        reason = "adm_fata_no_improvement"
    else:
        reason = f"adm_fata_k={generations}_blocks={len(layout.blocks)}_dim={layout.dim}"

    strategies = np.bincount(best_context, minlength=3)[:3]
    log = [
        {
            "stage": "stage2_adm_fata",
            "iter": 0,
            "action_type": "adm_fata",
            "plan_id": "",
            "old_pairs": pairs_before,
            "new_pairs": pairs_after,
            "old_points": points_before,
            "new_points": points_after,
            "accepted": accepted,
            "runtime_ms": runtime_ms,
            "rollback": not accepted,
            "reason": reason,
            "matched_strategy": "",
            "source": (
                f"delay={int(strategies[0])};speed={int(strategies[1])};"
                f"reroute={int(strategies[2])}"
            ),
        }
    ]
    return Stage2Result(
        plans=best_plans,
        conflicts=remaining,
        pairs_before=pairs_before,
        points_before=points_before,
        pairs_after=pairs_after,
        points_after=points_after,
        generations=generations,
        dim=layout.dim,
        blocks=len(layout.blocks),
        accepted=accepted,
        best_strategies=best_context,
        probability=probability,
        convergence=list(result.convergence),
        log=log,
    )
