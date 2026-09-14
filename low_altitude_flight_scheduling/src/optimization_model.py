from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .astar_3d import astar_path
from .conflict_detection import (
    Conflict,
    ConflictSegment,
    conflict_count_by_flight,
    count_conflict_pairs,
    count_conflict_points,
    detect_conflicts,
    group_continuous_conflicts,
)
from .flight_plan import FlightPlan, apply_speed_factor, recompute_eta_times, shift_plan_time, update_plan_timing
from .grid import AirspaceGrid, GridPoint
from .utils import path_distance_m

_ASTAR_CALLS = 0
_ADM_STRATEGIES = ("delay", "speed", "reroute")
_ADM_CLASSES = ("head_to_head", "cluster", "crossing")


def reset_optimization_stats() -> None:
    global _ASTAR_CALLS
    _ASTAR_CALLS = 0


def get_optimization_stats() -> dict[str, float]:
    return {"number_of_astar_calls": float(_ASTAR_CALLS)}


@dataclass
class Evaluation:
    fitness: float
    remaining_conflicts: int
    total_delay: float
    total_air_time: float
    total_risk: float
    delayed_count: int
    battery_violations: int
    plans: list[FlightPlan]


def clone_plans(plans: list[FlightPlan]) -> list[FlightPlan]:
    return [p.copy() for p in plans]


def recompute_plan(plan: FlightPlan, grid: AirspaceGrid, risk_map: np.ndarray, speed: float | None = None) -> FlightPlan:
    new = plan.copy()
    spd = float(speed if speed is not None else np.mean(plan.speed_profile or [10.0]))
    new.speed_profile = [spd for _ in range(max(0, len(new.path) - 1))]
    new.eta_times = recompute_eta_times(new, grid.cell_size, speed=spd)
    new.risk_sum = float(sum(risk_map[p] for p in new.path))
    new.total_air_time = max(0.0, new.eta_times[-1] - new.etd)
    return new


def apply_solution_vector(
    plans: list[FlightPlan],
    flight_ids: list[int],
    vector: np.ndarray,
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    route_options: dict[int, list[FlightPlan]] | None = None,
) -> list[FlightPlan]:
    out = list(plans)
    id_to_idx = {plan.id: idx for idx, plan in enumerate(out)}
    speed_min, speed_max = [float(v) for v in cfg["optimization"]["speed_range"]]
    delay_max = float(cfg["optimization"]["t_delay_max"])
    n = len(flight_ids)
    touched: set[int] = set()
    raw_delays = [float(np.clip(vector[j], 0.0, delay_max)) for j in range(n)]
    delay_threshold = float(cfg["optimization"].get("delay_count_threshold", 30.0))
    delay_cap = int(cfg["optimization"].get("delay_count_cap", max(1, int(np.floor(0.03 * len(plans))))))
    delayed_slots = [j for j, delay in enumerate(raw_delays) if delay >= delay_threshold]
    if len(delayed_slots) > delay_cap:
        keep = set(sorted(delayed_slots, key=lambda j: raw_delays[j], reverse=True)[:delay_cap])
        raw_delays = [delay if j in keep or delay < delay_threshold else 0.0 for j, delay in enumerate(raw_delays)]
    for j, fid in enumerate(flight_ids):
        if fid not in id_to_idx:
            continue
        delay = raw_delays[j]
        speed = float(np.clip(vector[n + j], speed_min, speed_max)) if len(vector) >= 2 * n else None
        idx = id_to_idx[fid]
        base_plan = out[idx]
        if route_options and fid in route_options and len(vector) >= 3 * n:
            options = route_options[fid]
            choice = int(np.clip(round(float(vector[2 * n + j])), 0, len(options) - 1))
            base_plan = options[choice].copy()
            base_plan.etd = out[idx].etd
            if base_plan.path != out[idx].path:
                base_plan.changed = True
                base_plan.rerouted = True
        out[idx] = update_plan_timing(base_plan, speed=speed, delay=delay, cell_size=grid.cell_size)
        if base_plan.rerouted:
            out[idx].rerouted = True
            out[idx].changed = True
        touched.add(idx)
    for idx in touched:
        plan = out[idx]
        out[idx] = recompute_plan(plan, grid, risk_map)
    return out


def evaluate_solution(
    vector: np.ndarray,
    plans: list[FlightPlan],
    flight_ids: list[int],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    n_gen: int = 1,
    n_gen_max: int | None = None,
    route_options: dict[int, list[FlightPlan]] | None = None,
) -> Evaluation:
    new_plans = apply_solution_vector(plans, flight_ids, vector, cfg, grid, risk_map, route_options=route_options)
    active_ids = set(flight_ids) if flight_ids else None
    conflicts = detect_conflicts(new_plans, cfg, uncertain=True, active_ids=active_ids)
    n_c = count_conflict_points(conflicts)
    total_delay = float(sum(max(0.0, p.delay) for p in new_plans))
    total_air = float(sum(p.total_air_time for p in new_plans))
    total_risk = float(sum(p.risk_sum for p in new_plans))
    delay_threshold = float(cfg["optimization"].get("delay_count_threshold", 30.0))
    delayed = int(sum(1 for p in new_plans if p.delay >= delay_threshold))
    battery = int(sum(1 for p in new_plans if p.total_air_time > float(cfg["optimization"]["t_battery"])))
    delay_cap = int(cfg["optimization"].get("delay_count_cap", max(1, int(np.floor(0.03 * len(plans))))))
    delay_cap_violations = max(0, delayed - max(0, delay_cap))
    changed = changed_count(new_plans)
    changed_cap = int(np.floor(float(cfg["optimization"].get("max_changed_flight_ratio", 1.0)) * len(plans)))
    changed_cap_violations = max(0, changed - max(1, changed_cap))

    base_air = max(1.0, float(sum(p.total_air_time for p in plans)))
    base_risk = max(1e-12, float(sum(p.risk_sum for p in plans)))
    normalized_delay = total_delay / max(1.0, len(plans) * max(1.0, float(cfg["optimization"]["t_delay_max"])))
    normalized_air = total_air / base_air
    normalized_risk = total_risk / base_risk
    soft_cost = (
        float(cfg["fata"]["omega_d"]) * normalized_delay
        + float(cfg["fata"]["omega_t"]) * normalized_air
        + float(cfg["fata"]["omega_r"]) * normalized_risk
        + 0.01 * changed / max(1, len(plans))
    )
    hard_constraint_violations = delay_cap_violations + changed_cap_violations
    fitness = float(
        float(cfg["optimization"].get("conflict_hard_penalty", 1_000_000.0)) * n_c
        + soft_cost
        + 10_000.0 * battery
        + 1_000.0 * hard_constraint_violations
    )
    return Evaluation(fitness, n_c, total_delay, total_air, total_risk, delayed, battery, new_plans)


def _delay_candidates(cfg: dict) -> list[float]:
    values = [float(v) for v in cfg["optimization"].get("delay_candidates", [-180, -120, -90, -60, -30, 0, 30, 60, 90, 120, 180, 300, 600])]
    if 0.0 not in values:
        values.insert(0, 0.0)
    return values


def _speed_factors(cfg: dict) -> list[float]:
    values = [float(v) for v in cfg["optimization"].get("speed_factors", [0.90, 0.95, 1.00, 1.05, 1.10, 1.15])]
    if 1.0 not in values:
        values.insert(0, 1.0)
    return values


def solution_bounds_delay_speed(flight_ids: list[int], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(flight_ids)
    delays = _delay_candidates(cfg)
    speeds = _speed_factors(cfg)
    lb = np.zeros(2 * n, dtype=float)
    ub = np.array([len(delays) - 1] * n + [len(speeds) - 1] * n, dtype=float)
    return lb, ub


def apply_delay_speed_vector(
    plans: list[FlightPlan],
    flight_ids: list[int],
    vector: np.ndarray,
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
) -> list[FlightPlan]:
    out = list(plans)
    id_to_idx = {plan.id: idx for idx, plan in enumerate(out)}
    delays = _delay_candidates(cfg)
    speeds = _speed_factors(cfg)
    speed_min, speed_max = [float(v) for v in cfg["optimization"]["speed_range"]]
    n = len(flight_ids)
    for j, fid in enumerate(flight_ids):
        if fid not in id_to_idx:
            continue
        delay_idx = int(np.clip(round(float(vector[j])), 0, len(delays) - 1))
        speed_idx = int(np.clip(round(float(vector[n + j])), 0, len(speeds) - 1)) if len(vector) >= 2 * n else speeds.index(1.0)
        plan = out[id_to_idx[fid]]
        updated = shift_plan_time(plan, delays[delay_idx]) if abs(delays[delay_idx]) > 1e-9 else plan.copy()
        updated = apply_speed_factor(updated, speeds[speed_idx], speed_min, speed_max, grid.cell_size)
        updated.risk_sum = float(sum(risk_map[p] for p in updated.path))
        out[id_to_idx[fid]] = updated
    return out


def evaluate_delay_speed_solution(
    vector: np.ndarray,
    plans: list[FlightPlan],
    flight_ids: list[int],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
) -> Evaluation:
    new_plans = apply_delay_speed_vector(plans, flight_ids, vector, cfg, grid, risk_map)
    active_ids = set(flight_ids) if flight_ids else None
    conflicts = detect_conflicts(new_plans, cfg, uncertain=True, active_ids=active_ids)
    pairs = count_conflict_pairs(conflicts)
    points = count_conflict_points(conflicts)
    total_delay = float(sum(max(0.0, p.delay) for p in new_plans))
    total_air = float(sum(p.total_air_time for p in new_plans))
    total_risk = float(sum(p.risk_sum for p in new_plans))
    soft = _soft_cost(new_plans, cfg)
    fitness = float(
        1_000_000.0 * pairs
        + 10_000.0 * points
        + soft
    )
    delayed = delayed_count(new_plans, float(cfg["optimization"].get("delay_count_threshold", 30.0)))
    battery = int(sum(1 for p in new_plans if p.total_air_time > float(cfg["optimization"]["t_battery"])))
    return Evaluation(fitness, points, total_delay, total_air, total_risk, delayed, battery, new_plans)


def _soft_cost(plans: list[FlightPlan], cfg: dict) -> float:
    delay = float(sum(max(0.0, p.delay) for p in plans))
    air = float(sum(p.total_air_time for p in plans))
    risk = float(sum(p.risk_sum for p in plans))
    return (
        0.001 * delay
        + 0.0001 * air
        + 0.0001 * risk
        + 10.0 * changed_count(plans)
        + 25.0 * delayed_count(plans, float(cfg["optimization"].get("delay_count_threshold", 30.0)))
    )


def try_apply_action_with_rollback(
    plans: list[FlightPlan],
    action: dict[str, object],
    base_conflicts: list[Conflict],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
) -> tuple[bool, list[FlightPlan], list[Conflict], dict[str, object]]:
    started = __import__("time").perf_counter()
    old_pairs = count_conflict_pairs(base_conflicts)
    old_points = count_conflict_points(base_conflicts)
    old_soft = _soft_cost(plans, cfg)
    trial = list(plans)
    id_to_idx = {plan.id: idx for idx, plan in enumerate(trial)}
    action_type = str(action.get("type", "unknown"))
    fid = int(action.get("plan_id", -1))
    if fid not in id_to_idx:
        log = _action_log(action_type, fid, old_pairs, old_pairs, old_points, old_points, False, started, "missing_plan")
        return False, plans, base_conflicts, log

    idx = id_to_idx[fid]
    if action_type == "delay":
        trial[idx] = shift_plan_time(trial[idx], float(action.get("delta", 0.0)))
    elif action_type == "speed":
        speed_min, speed_max = [float(v) for v in cfg["optimization"]["speed_range"]]
        trial[idx] = apply_speed_factor(trial[idx], float(action.get("factor", 1.0)), speed_min, speed_max, grid.cell_size)
    elif action_type in {"reroute", "cruise_level", "replace_plan"}:
        replacement = action.get("plan")
        if not isinstance(replacement, FlightPlan):
            log = _action_log(action_type, fid, old_pairs, old_pairs, old_points, old_points, False, started, "missing_replacement")
            return False, plans, base_conflicts, log
        trial[idx] = replacement.copy()
    else:
        log = _action_log(action_type, fid, old_pairs, old_pairs, old_points, old_points, False, started, "unknown_action")
        return False, plans, base_conflicts, log

    trial[idx] = recompute_plan(trial[idx], grid, risk_map)
    new_conflicts = detect_conflicts(trial, cfg, uncertain=True)
    new_pairs = count_conflict_pairs(new_conflicts)
    new_points = count_conflict_points(new_conflicts)
    new_soft = _soft_cost(trial, cfg)
    accepted = False
    reason = "rollback"
    if new_pairs < old_pairs:
        accepted = True
        reason = "pairs_decreased"
    elif new_pairs == old_pairs and new_points <= old_points and new_soft < old_soft - 1e-9:
        accepted = True
        reason = "soft_decreased"
    if accepted and new_pairs > old_pairs:
        raise AssertionError("Accepted action increased global conflict pairs.")
    log = _action_log(action_type, fid, old_pairs, new_pairs, old_points, new_points, accepted, started, reason)
    return (True, trial, new_conflicts, log) if accepted else (False, plans, base_conflicts, log)


def _action_log(
    action_type: str,
    fid: int,
    old_pairs: int,
    new_pairs: int,
    old_points: int,
    new_points: int,
    accepted: bool,
    started: float,
    reason: str,
) -> dict[str, object]:
    runtime_ms = (__import__("time").perf_counter() - started) * 1000.0
    return {
        "action_type": action_type,
        "plan_id": fid,
        "old_pairs": old_pairs,
        "new_pairs": new_pairs,
        "old_points": old_points,
        "new_points": new_points,
        "accepted": bool(accepted),
        "rollback": not bool(accepted),
        "runtime_ms": runtime_ms,
        "reason": reason,
    }


def _delay_within_limits(plan: FlightPlan, delta: float, cfg: dict) -> bool:
    """Check that a takeoff-time shift keeps the plan inside the paper bounds."""

    new_etd = plan.etd + float(delta)
    if new_etd <= 0.0:
        return False
    return plan.delay + float(delta) <= float(cfg["optimization"]["t_delay_max"]) + 1e-9


def greedy_time_deconfliction(
    plans: list[FlightPlan],
    conflicts: list[Conflict],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    max_rounds: int = 50,
    deadline: float | None = None,
    flights_per_round: int | None = None,
) -> tuple[list[FlightPlan], list[Conflict], list[dict[str, object]]]:
    """Greedy delay/speed repair over the remaining conflicts.

    Every round ranks flights by their global conflict contribution and then
    processes the top ``flights_per_round`` flights one after another. For each
    flight all discrete delay/speed candidates are evaluated against the current
    global state; the best strictly improving action is accepted immediately
    (first-improvement per flight) and the next flight sees the updated state.
    Flights without any improving action are marked as stuck and skipped in the
    following rounds. The loop stops when a full round accepts nothing, the
    round budget is exhausted or the deadline is reached.
    """

    import time

    opt_cfg = cfg.get("optimization", {})
    flights_per_round = int(
        flights_per_round
        if flights_per_round is not None
        else opt_cfg.get("greedy_flights_per_round", 2)
    )
    flights_per_round = max(1, flights_per_round)

    current = clone_plans(plans)
    current_conflicts = list(conflicts)
    log: list[dict[str, object]] = []
    stuck: set[int] = set()
    for round_idx in range(max_rounds):
        if not current_conflicts:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break
        scores = _conflict_contribution(current_conflicts)
        candidates = [fid for fid, _ in sorted(scores.items(), key=lambda item: (-item[1][0], -item[1][1], item[0])) if fid not in stuck]
        if not candidates:
            break
        accepted_this_round = 0
        for fid in candidates[:flights_per_round]:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            id_to_idx = {plan.id: idx for idx, plan in enumerate(current)}
            if fid not in id_to_idx:
                stuck.add(fid)
                continue
            best: tuple[list[FlightPlan], list[Conflict], tuple[int, int, int, float]] | None = None
            for action in _time_actions_for_plan(fid, cfg):
                if str(action.get("type", "")) == "delay" and not _delay_within_limits(current[id_to_idx[fid]], float(action.get("delta", 0.0)), cfg):
                    continue
                accepted, trial, trial_conflicts, action_log = try_apply_action_with_rollback(current, action, current_conflicts, cfg, grid, risk_map)
                action_log["stage"] = "greedy_time"
                action_log["iter"] = round_idx
                log.append(action_log)
                if not accepted:
                    continue
                rank = (
                    count_conflict_pairs(trial_conflicts),
                    count_conflict_points(trial_conflicts),
                    changed_count(trial),
                    float(sum(max(0.0, p.delay) for p in trial)),
                )
                if best is None or rank < best[2]:
                    best = (trial, trial_conflicts, rank)
            if best is None:
                stuck.add(fid)
                continue
            current, current_conflicts = best[0], best[1]
            accepted_this_round += 1
        if accepted_this_round == 0:
            break
        if count_conflict_pairs(current_conflicts) == 0:
            break
    return current, current_conflicts, log


def _conflict_contribution(conflicts: list[Conflict]) -> dict[int, tuple[int, int]]:
    pair_sets: dict[int, set[tuple[int, int]]] = {}
    point_counts: dict[int, int] = {}
    for conflict in conflicts:
        pair = (min(conflict.plan_a, conflict.plan_b), max(conflict.plan_a, conflict.plan_b))
        for fid in (conflict.plan_a, conflict.plan_b):
            pair_sets.setdefault(fid, set()).add(pair)
            point_counts[fid] = point_counts.get(fid, 0) + 1
    return {fid: (len(pair_sets.get(fid, set())), point_counts.get(fid, 0)) for fid in point_counts}


def _time_actions_for_plan(fid: int, cfg: dict) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    for delta in _delay_candidates(cfg):
        if abs(delta) > 1e-9:
            actions.append({"type": "delay", "plan_id": fid, "delta": float(delta)})
    for factor in _speed_factors(cfg):
        if abs(factor - 1.0) > 1e-9:
            actions.append({"type": "speed", "plan_id": fid, "factor": float(factor)})
    return actions


def independent_matching_deconfliction(
    plans: list[FlightPlan],
    conflicts: list[Conflict],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    max_rounds: int | None = None,
    deadline: float | None = None,
    seed: int | None = None,
) -> tuple[list[FlightPlan], list[Conflict], list[dict[str, object]]]:
    """Resolve remaining conflicts with a reference-[6]-style independent strategy match.

    Each continuous conflict segment is treated as a local unit. The unit is
    classified, matched to scheduling/speed/rerouting probabilities, and then
    only globally non-worsening actions are accepted.
    """
    import time

    opt_cfg = cfg.get("optimization", {})
    current = clone_plans(plans)
    current_conflicts = list(conflicts) if conflicts else detect_conflicts(current, cfg, uncertain=True)
    max_rounds = int(max_rounds if max_rounds is not None else opt_cfg.get("independent_matching_rounds", opt_cfg.get("stage2_repair_rounds", 20)))
    scan_limit = int(opt_cfg.get("independent_matching_segment_limit", opt_cfg.get("repair_segment_scan_limit", 8)))
    candidates_per_strategy = int(opt_cfg.get("independent_matching_candidates_per_strategy", 8))
    lrate = float(opt_cfg.get("independent_matching_lrate", 0.5))
    rng_seed = int(seed if seed is not None else int(cfg.get("flight", {}).get("random_seed", 2025)) + 607)
    rng = np.random.default_rng(rng_seed)
    probabilities = _initial_adm_probabilities(cfg)
    logs: list[dict[str, object]] = []

    for round_idx in range(max_rounds):
        if not current_conflicts:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break
        segments = group_continuous_conflicts(current_conflicts)
        if not segments:
            break

        accepted_this_round = False
        for seg_idx, segment in enumerate(segments[: max(1, scan_limit)]):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            conflict_class = _classify_adm_segment(current, current_conflicts, segment, cfg, grid)
            strategy_order = _sample_adm_strategy_order(probabilities[conflict_class], rng)
            pair = (min(segment.plan_a, segment.plan_b), max(segment.plan_a, segment.plan_b))
            best: tuple[list[FlightPlan], list[Conflict], dict[str, object], tuple[int, int, int, float], str] | None = None

            for strategy in strategy_order:
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                actions = _independent_matching_actions_for_strategy(
                    current,
                    segment,
                    strategy,
                    cfg,
                    grid,
                    risk_map,
                    candidates_per_strategy,
                )
                if not actions:
                    _log_adm_skip(
                        logs,
                        round_idx,
                        seg_idx,
                        segment,
                        conflict_class,
                        strategy,
                        probabilities[conflict_class],
                        "no_candidate",
                    )
                    continue
                for cand_idx, action in enumerate(actions[:candidates_per_strategy]):
                    accepted, trial, trial_conflicts, action_log = try_apply_action_with_rollback(
                        current,
                        action,
                        current_conflicts,
                        cfg,
                        grid,
                        risk_map,
                    )
                    _decorate_adm_log(
                        action_log,
                        round_idx,
                        seg_idx,
                        segment,
                        conflict_class,
                        strategy,
                        cand_idx,
                        probabilities[conflict_class],
                    )
                    logs.append(action_log)
                    if not accepted:
                        continue
                    rank = (
                        count_conflict_pairs(trial_conflicts),
                        count_conflict_points(trial_conflicts),
                        changed_count(trial),
                        float(sum(max(0.0, p.delay) for p in trial)),
                    )
                    if best is None or rank < best[3]:
                        best = (trial, trial_conflicts, action_log, rank, strategy)
                    if bool(opt_cfg.get("independent_matching_first_pair_resolution", True)) and not _pair_present(trial_conflicts, pair):
                        break
                if best is not None and bool(opt_cfg.get("independent_matching_first_strategy_success", True)):
                    break

            if best is None:
                continue
            current, current_conflicts = best[0], best[1]
            probabilities[conflict_class] = _update_adm_probability(probabilities[conflict_class], best[4], lrate)
            best[2]["prob_delay_after"] = probabilities[conflict_class]["delay"]
            best[2]["prob_speed_after"] = probabilities[conflict_class]["speed"]
            best[2]["prob_reroute_after"] = probabilities[conflict_class]["reroute"]
            accepted_this_round = True
            break

        if not accepted_this_round:
            logs.append(
                {
                    "stage": "stage2_independent_matching",
                    "iter": round_idx,
                    "action_type": "unresolved",
                    "plan_id": "",
                    "old_pairs": count_conflict_pairs(current_conflicts),
                    "new_pairs": count_conflict_pairs(current_conflicts),
                    "old_points": count_conflict_points(current_conflicts),
                    "new_points": count_conflict_points(current_conflicts),
                    "accepted": False,
                    "rollback": True,
                    "runtime_ms": 0.0,
                    "reason": "no_independent_match_reduced_global_conflicts",
                    "source": "ADM_ref6_independent_matching",
                }
            )
            break
    return current, current_conflicts, logs


def _initial_adm_probabilities(cfg: dict) -> dict[str, dict[str, float]]:
    defaults = {
        "head_to_head": {"delay": 0.15, "speed": 0.15, "reroute": 0.70},
        "cluster": {"delay": 0.50, "speed": 0.30, "reroute": 0.20},
        "crossing": {"delay": 0.25, "speed": 0.60, "reroute": 0.15},
    }
    raw = cfg.get("optimization", {}).get("independent_matching_strategy_priors", {})
    out: dict[str, dict[str, float]] = {}
    for conflict_class in _ADM_CLASSES:
        values = raw.get(conflict_class, defaults[conflict_class])
        if isinstance(values, dict):
            probs = {strategy: float(values.get(strategy, defaults[conflict_class][strategy])) for strategy in _ADM_STRATEGIES}
        else:
            sequence = list(values) if isinstance(values, (list, tuple)) else []
            probs = {
                strategy: float(sequence[idx]) if idx < len(sequence) else defaults[conflict_class][strategy]
                for idx, strategy in enumerate(_ADM_STRATEGIES)
            }
        out[conflict_class] = _normalize_adm_probability(probs)
    return out


def _normalize_adm_probability(probs: dict[str, float]) -> dict[str, float]:
    clean = {strategy: max(0.0, float(probs.get(strategy, 0.0))) for strategy in _ADM_STRATEGIES}
    total = sum(clean.values())
    if total <= 1e-12:
        return {strategy: 1.0 / len(_ADM_STRATEGIES) for strategy in _ADM_STRATEGIES}
    return {strategy: clean[strategy] / total for strategy in _ADM_STRATEGIES}


def _update_adm_probability(probs: dict[str, float], selected_strategy: str, lrate: float) -> dict[str, float]:
    eta = float(np.clip(lrate, 0.0, 1.0))
    dominant = {strategy: 1.0 if strategy == selected_strategy else 0.0 for strategy in _ADM_STRATEGIES}
    updated = {strategy: (1.0 - eta) * probs[strategy] + eta * dominant[strategy] for strategy in _ADM_STRATEGIES}
    return _normalize_adm_probability(updated)


def _sample_adm_strategy_order(probs: dict[str, float], rng: np.random.Generator) -> list[str]:
    """Sample a strategy order without replacement, tolerating zero priors.

    ``numpy.random.Generator.choice(..., replace=False, p=p)`` raises when the
    number of non-zero probabilities is smaller than the sample size, so the
    order is drawn sequentially instead: each step removes the already selected
    strategy from the candidate pool.
    """

    weights = np.array([max(0.0, float(probs.get(strategy, 0.0))) for strategy in _ADM_STRATEGIES], dtype=float)
    if float(weights.sum()) <= 1e-12:
        weights = np.ones(len(_ADM_STRATEGIES), dtype=float)
    weights = weights / float(weights.sum())
    remaining = list(range(len(_ADM_STRATEGIES)))
    order: list[str] = []
    for _ in range(len(_ADM_STRATEGIES)):
        local = weights[remaining]
        total = float(local.sum())
        if total <= 1e-12:
            pick = int(rng.integers(0, len(remaining)))
        else:
            pick = int(rng.choice(len(remaining), p=local / total))
        order.append(str(_ADM_STRATEGIES[remaining.pop(pick)]))
    return order


def _classify_adm_segment(
    plans: list[FlightPlan],
    conflicts: list[Conflict],
    segment: ConflictSegment,
    cfg: dict,
    grid: AirspaceGrid,
) -> str:
    id_to_plan = {p.id: p for p in plans}
    plan_a = id_to_plan.get(segment.plan_a)
    plan_b = id_to_plan.get(segment.plan_b)
    if plan_a is not None and plan_b is not None and segment.conflicts:
        conflict = segment.conflicts[len(segment.conflicts) // 2]
        va = _local_path_direction(plan_a.path, conflict.idx_a)
        vb = _local_path_direction(plan_b.path, conflict.idx_b)
        if va is not None and vb is not None:
            dot = float(np.dot(va, vb))
            threshold = float(cfg.get("optimization", {}).get("head_to_head_dot_threshold", -0.35))
            if dot <= threshold:
                return "head_to_head"

    cluster_threshold = int(cfg.get("optimization", {}).get("stage2_cluster_points_threshold", 3))
    if len(segment.conflicts) >= cluster_threshold:
        return "cluster"
    radius = int(cfg.get("optimization", {}).get("stage2_cluster_neighbor_radius", 2))
    segment_cells = segment.cells
    density = 0
    for conflict in conflicts:
        if any(_cell_chebyshev_xy(conflict.cell, cell) <= radius for cell in segment_cells):
            density += 1
            if density >= cluster_threshold:
                return "cluster"
    return "crossing"


def _local_path_direction(path: list[GridPoint], idx: int) -> np.ndarray | None:
    if len(path) < 2:
        return None
    lo = max(0, int(idx) - 1)
    hi = min(len(path) - 1, int(idx) + 1)
    if lo == hi:
        return None
    vec = np.array(path[hi], dtype=float) - np.array(path[lo], dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return None
    return vec / norm


def _cell_chebyshev_xy(a: GridPoint, b: GridPoint) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _independent_matching_actions_for_strategy(
    plans: list[FlightPlan],
    segment: ConflictSegment,
    strategy: str,
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    limit: int,
) -> list[dict[str, object]]:
    if strategy == "delay":
        return _adm_delay_actions(plans, segment, cfg, limit)
    if strategy == "speed":
        return _adm_speed_actions(plans, segment, cfg, limit)
    if strategy == "reroute":
        return _adm_reroute_actions(plans, segment, cfg, grid, risk_map, limit)
    return []


def _adm_delay_actions(plans: list[FlightPlan], segment: ConflictSegment, cfg: dict, limit: int) -> list[dict[str, object]]:
    id_to_plan = {p.id: p for p in plans}
    if not segment.conflicts:
        return []
    conflict = max(segment.conflicts, key=lambda c: max(0.0, c.required_gap - abs(c.time_gap)))
    safety_margin = float(cfg.get("optimization", {}).get("safety_margin_seconds", 10.0))
    req = float(max(segment.required_shift, conflict.required_gap - abs(conflict.time_gap)) + safety_margin)
    raw: list[tuple[int, float]] = [
        (conflict.plan_a, req - conflict.time_gap),
        (conflict.plan_a, -req - conflict.time_gap),
        (conflict.plan_b, conflict.time_gap + req),
        (conflict.plan_b, conflict.time_gap - req),
    ]
    for fid in (segment.plan_a, segment.plan_b):
        for delta in _delay_candidates(cfg):
            if abs(delta) > 1e-9:
                raw.append((fid, float(delta)))

    delay_threshold = float(cfg["optimization"].get("delay_count_threshold", 30.0))
    delay_cap = int(cfg["optimization"].get("delay_count_cap", max(1, int(np.floor(0.03 * len(plans))))))
    current_delayed = delayed_count(plans, delay_threshold)
    seen: set[tuple[int, int]] = set()
    actions: list[dict[str, object]] = []
    for fid, delta in raw:
        plan = id_to_plan.get(fid)
        if plan is None or abs(delta) < 1.0:
            continue
        if delta > 0 and plan.delay < delay_threshold and current_delayed >= delay_cap:
            continue
        if plan.etd + delta <= 0.0 or plan.delay + delta > float(cfg["optimization"]["t_delay_max"]):
            continue
        key = (fid, int(round(delta)))
        if key in seen:
            continue
        seen.add(key)
        actions.append({"type": "delay", "plan_id": fid, "delta": float(delta)})
        if len(actions) >= limit:
            break
    return actions


def _adm_speed_actions(plans: list[FlightPlan], segment: ConflictSegment, cfg: dict, limit: int) -> list[dict[str, object]]:
    id_to_plan = {p.id: p for p in plans}
    if not segment.conflicts:
        return []
    conflict = min(segment.conflicts, key=lambda c: abs(c.time_gap))
    factors = [float(v) for v in cfg.get("optimization", {}).get("speed_factors", [0.90, 0.95, 1.05, 1.10, 1.15])]
    faster = sorted([f for f in factors if f > 1.0], reverse=True)
    slower = sorted([f for f in factors if 0.0 < f < 1.0])
    ordered: list[tuple[int, float]] = []
    if conflict.time_gap <= 0.0:
        ordered.extend((conflict.plan_a, factor) for factor in faster)
        ordered.extend((conflict.plan_b, factor) for factor in slower)
        ordered.extend((conflict.plan_a, factor) for factor in slower)
        ordered.extend((conflict.plan_b, factor) for factor in faster)
    else:
        ordered.extend((conflict.plan_b, factor) for factor in faster)
        ordered.extend((conflict.plan_a, factor) for factor in slower)
        ordered.extend((conflict.plan_b, factor) for factor in slower)
        ordered.extend((conflict.plan_a, factor) for factor in faster)

    speed_min, speed_max = [float(v) for v in cfg["optimization"]["speed_range"]]
    seen: set[tuple[int, int]] = set()
    actions: list[dict[str, object]] = []
    for fid, factor in ordered:
        plan = id_to_plan.get(fid)
        if plan is None:
            continue
        base_speed = float(np.mean(plan.speed_profile or [cfg["flight"]["default_speed"]]))
        new_speed = float(np.clip(base_speed * factor, speed_min, speed_max))
        if abs(new_speed - base_speed) <= 1e-6:
            continue
        key = (fid, int(round(new_speed * 100)))
        if key in seen:
            continue
        seen.add(key)
        actions.append({"type": "speed", "plan_id": fid, "factor": float(factor)})
        if len(actions) >= limit:
            break
    return actions


def _adm_reroute_actions(
    plans: list[FlightPlan],
    segment: ConflictSegment,
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    limit: int,
) -> list[dict[str, object]]:
    opt_cfg = cfg.get("optimization", {})
    if not bool(opt_cfg.get("independent_matching_allow_reroute", True)):
        return []
    id_to_plan = {p.id: p for p in plans}
    cells = list(dict.fromkeys(segment.cells))
    windows = [int(v) for v in opt_cfg.get("local_reroute_windows", [8, 12])]
    radii = [int(v) for v in opt_cfg.get("local_reroute_radii", [1, 2])]
    min_layer = int(cfg["flight"].get("min_flight_layer", 0))
    actions: list[dict[str, object]] = []
    seen: set[tuple[int, tuple[GridPoint, ...]]] = set()

    roles = [segment.plan_a, segment.plan_b]
    for fid in roles:
        plan = id_to_plan.get(fid)
        if plan is None:
            continue
        for level in _nearby_levels(segment, grid):
            shifted = try_altitude_shift(plan, grid, risk_map, level, min_layer=min_layer)
            if shifted is None:
                continue
            key = (fid, tuple(shifted.path))
            if key in seen:
                continue
            seen.add(key)
            actions.append({"type": "cruise_level", "plan_id": fid, "plan": shifted})
            if len(actions) >= limit:
                return actions

    for fid in roles:
        plan = id_to_plan.get(fid)
        if plan is None:
            continue
        conflict_indices = [c.idx_a if fid == c.plan_a else c.idx_b for c in segment.conflicts if fid in {c.plan_a, c.plan_b}]
        rerouted = try_local_reroute(
            plan,
            grid,
            risk_map,
            cfg,
            cells,
            conflict_indices=conflict_indices,
            windows=windows,
            radii=radii,
            allow_global_fallback=False,
        )
        if rerouted is None:
            continue
        key = (fid, tuple(rerouted.path))
        if key in seen:
            continue
        seen.add(key)
        actions.append({"type": "reroute", "plan_id": fid, "plan": rerouted})
        if len(actions) >= limit:
            return actions
    return actions


def _decorate_adm_log(
    action_log: dict[str, object],
    round_idx: int,
    segment_idx: int,
    segment: ConflictSegment,
    conflict_class: str,
    strategy: str,
    candidate_idx: int,
    probs: dict[str, float],
) -> None:
    action_log["stage"] = "stage2_independent_matching"
    action_log["iter"] = round_idx
    action_log["segment_idx"] = segment_idx
    action_log["pair"] = f"{min(segment.plan_a, segment.plan_b)}-{max(segment.plan_a, segment.plan_b)}"
    action_log["segment_points"] = len(segment.conflicts)
    action_log["conflict_class"] = conflict_class
    action_log["matched_strategy"] = strategy
    action_log["candidate_rank"] = candidate_idx
    action_log["prob_delay"] = probs["delay"]
    action_log["prob_speed"] = probs["speed"]
    action_log["prob_reroute"] = probs["reroute"]
    action_log["source"] = "ADM_ref6_independent_matching"


def _log_adm_skip(
    logs: list[dict[str, object]],
    round_idx: int,
    segment_idx: int,
    segment: ConflictSegment,
    conflict_class: str,
    strategy: str,
    probs: dict[str, float],
    reason: str,
) -> None:
    item = {
        "stage": "stage2_independent_matching",
        "iter": round_idx,
        "segment_idx": segment_idx,
        "action_type": strategy,
        "plan_id": "",
        "old_pairs": "",
        "new_pairs": "",
        "old_points": "",
        "new_points": "",
        "accepted": False,
        "rollback": True,
        "runtime_ms": 0.0,
        "reason": reason,
    }
    _decorate_adm_log(item, round_idx, segment_idx, segment, conflict_class, strategy, -1, probs)
    logs.append(item)


def limited_local_reroute(
    plans: list[FlightPlan],
    conflicts: list[Conflict],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    max_attempts: int = 20,
    deadline: float | None = None,
) -> tuple[list[FlightPlan], list[Conflict], list[dict[str, object]]]:
    import time

    if count_conflict_pairs(conflicts) > 20:
        return clone_plans(plans), list(conflicts), []
    current = clone_plans(plans)
    current_conflicts = list(conflicts)
    log: list[dict[str, object]] = []
    attempts = 0
    for segment in group_continuous_conflicts(current_conflicts)[:10]:
        if attempts >= max_attempts or not current_conflicts:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break
        for conflict in segment.conflicts[:1]:
            for fid, idx in ((conflict.plan_a, conflict.idx_a), (conflict.plan_b, conflict.idx_b)):
                if attempts >= max_attempts:
                    break
                id_to_idx = {p.id: i for i, p in enumerate(current)}
                if fid not in id_to_idx:
                    continue
                for radius in (1, 2):
                    for window in (8, 12):
                        if attempts >= max_attempts:
                            break
                        attempts += 1
                        rerouted = try_local_reroute(
                            current[id_to_idx[fid]],
                            grid,
                            risk_map,
                            cfg,
                            [conflict.cell],
                            conflict_indices=[idx],
                            windows=[window],
                            radii=[radius],
                            allow_global_fallback=False,
                        )
                        if rerouted is None:
                            continue
                        accepted, current, current_conflicts, action_log = try_apply_action_with_rollback(
                            current,
                            {"type": "reroute", "plan_id": fid, "plan": rerouted},
                            current_conflicts,
                            cfg,
                            grid,
                            risk_map,
                        )
                        action_log["stage"] = "local_reroute"
                        action_log["iter"] = attempts
                        log.append(action_log)
                        if accepted:
                            break
                    if attempts >= max_attempts:
                        break
    return current, current_conflicts, log


def try_local_reroute(
    plan: FlightPlan,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    cfg: dict,
    forbidden_cells: list[GridPoint],
    radius: int = 1,
    conflict_indices: list[int] | None = None,
    windows: list[int] | None = None,
    radii: list[int] | None = None,
    allow_global_fallback: bool = True,
) -> FlightPlan | None:
    global _ASTAR_CALLS
    opt_cfg = cfg.get("optimization", {})
    windows = windows or [int(v) for v in opt_cfg.get("local_reroute_windows", [5, 10, 15, 20])]
    radii = radii if radii is not None else [int(radius)]
    cells = list(dict.fromkeys(forbidden_cells))
    if cells:
        for rr in radii:
            for window in windows:
                rerouted_path = _try_spliced_reroute(plan, grid, risk_map, cfg, cells, rr, window, conflict_indices)
                if rerouted_path:
                    new = plan.copy()
                    new.path = rerouted_path
                    new.rerouted = True
                    new.changed = True
                    return recompute_plan(new, grid, risk_map)
    if not allow_global_fallback:
        return None

    forbidden = _xy_forbidden(grid, cells, max(radii) if radii else radius)
    min_layer = int(cfg["flight"].get("min_flight_layer", 0))
    forbidden.update((x, y, z) for x in range(grid.shape[0]) for y in range(grid.shape[1]) for z in range(min_layer))
    forbidden.discard(plan.start)
    forbidden.discard(plan.goal)
    _ASTAR_CALLS += 1
    path = astar_path(
        grid,
        plan.start,
        plan.goal,
        risk_map,
        alpha_r=float(cfg["risk"]["alpha_r"]),
        alpha_l=float(cfg["risk"]["alpha_L"]),
        forbidden=forbidden,
        max_expansions=45000,
    )
    if not path:
        return None
    new = plan.copy()
    new.path = path
    new.rerouted = True
    new.changed = True
    return recompute_plan(new, grid, risk_map)


def _try_spliced_reroute(
    plan: FlightPlan,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    cfg: dict,
    forbidden_cells: list[GridPoint],
    radius: int,
    window: int,
    conflict_indices: list[int] | None,
) -> list[GridPoint] | None:
    global _ASTAR_CALLS
    if len(plan.path) < 4:
        return None
    indices = list(conflict_indices or [])
    if not indices:
        for cell in forbidden_cells:
            indices.extend(idx for idx, p in enumerate(plan.path) if p == cell)
    indices = sorted(set(idx for idx in indices if 0 < idx < len(plan.path) - 1))
    if not indices:
        return None

    min_layer = int(cfg["flight"].get("min_flight_layer", 0))
    forbidden = _xy_forbidden(grid, forbidden_cells, radius)
    forbidden.update((x, y, z) for x in range(grid.shape[0]) for y in range(grid.shape[1]) for z in range(min_layer))
    for idx in indices:
        lo = max(0, idx - int(window))
        hi = min(len(plan.path) - 1, idx + int(window))
        if hi - lo < 2:
            continue
        start = plan.path[lo]
        goal = plan.path[hi]
        local_forbidden = set(forbidden)
        local_forbidden.discard(start)
        local_forbidden.discard(goal)
        _ASTAR_CALLS += 1
        sub = astar_path(
            grid,
            start,
            goal,
            risk_map,
            alpha_r=float(cfg["risk"]["alpha_r"]) * 1.15,
            alpha_l=float(cfg["risk"]["alpha_L"]),
            forbidden=local_forbidden,
            max_expansions=30000,
            vertical_move_penalty=float(cfg.get("astar", {}).get("vertical_move_penalty", 1.8)),
        )
        if sub:
            candidate = plan.path[:lo] + sub + plan.path[hi + 1 :]
            if _path_is_valid(candidate, grid):
                return candidate
    return None


def _xy_forbidden(grid: AirspaceGrid, cells: list[GridPoint], radius: int) -> set[GridPoint]:
    forbidden: set[GridPoint] = set()
    for x, y, z in cells:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                p = (x + dx, y + dy, z)
                if grid.in_bounds(p):
                    forbidden.add(p)
    return forbidden


def _path_is_valid(path: list[GridPoint], grid: AirspaceGrid) -> bool:
    if not path:
        return False
    for p in path:
        if not grid.is_free(p):
            return False
    return all(max(abs(a[i] - b[i]) for i in range(3)) <= 1 for a, b in zip(path[:-1], path[1:]))


def build_route_option_bank(
    plans: list[FlightPlan],
    flight_ids: list[int],
    conflicts: list[Conflict],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    max_options_per_flight: int = 8,
) -> dict[int, list[FlightPlan]]:
    id_to_plan = {p.id: p for p in plans}
    min_layer = int(cfg["flight"].get("min_flight_layer", 0))
    by_flight: dict[int, list[GridPoint]] = {}
    for conflict in conflicts:
        by_flight.setdefault(conflict.flight_a, []).append(conflict.cell)
        by_flight.setdefault(conflict.flight_b, []).append(conflict.cell)

    bank: dict[int, list[FlightPlan]] = {}
    for fid in flight_ids:
        if fid not in id_to_plan:
            continue
        base = recompute_plan(id_to_plan[fid], grid, risk_map)
        options = [base]
        seen = {tuple(base.path)}

        current_z = int(round(np.median([p[2] for p in base.path])))
        layers = sorted(range(min_layer, grid.shape[2]), key=lambda z: (abs(z - current_z), z))
        for z in layers:
            shifted = try_altitude_shift(base, grid, risk_map, z, min_layer=min_layer)
            if shifted is None:
                continue
            key = tuple(shifted.path)
            if key not in seen:
                options.append(shifted)
                seen.add(key)
            if len(options) >= max_options_per_flight:
                break

        related = by_flight.get(fid, [])
        if related and len(options) < max_options_per_flight:
            related_unique = list(dict.fromkeys(related))
            subsets = [related_unique[:4], related_unique[:8], related_unique[:16], related_unique]
            for radius in (0, 1, 2):
                for subset in subsets:
                    if not subset:
                        continue
                    rerouted = try_local_reroute(base, grid, risk_map, cfg, subset, radius=radius)
                    if rerouted is None:
                        continue
                    key = tuple(rerouted.path)
                    if key in seen:
                        continue
                    options.append(rerouted)
                    seen.add(key)
                    if len(options) >= max_options_per_flight:
                        break
                if len(options) >= max_options_per_flight:
                    break

        bank[fid] = options
    return bank


def try_altitude_shift(plan: FlightPlan, grid: AirspaceGrid, risk_map: np.ndarray, target_z: int, min_layer: int = 0) -> FlightPlan | None:
    target_z = int(target_z)
    if target_z < min_layer or target_z >= grid.shape[2]:
        return None
    if len(plan.path) < 3:
        return None
    start_z = int(plan.path[0][2])
    goal_z = int(plan.path[-1][2])
    climb_steps = abs(target_z - start_z)
    descent_steps = abs(target_z - goal_z)
    if len(plan.path) <= climb_steps + descent_steps + 1:
        return None
    shifted: list[GridPoint] = []
    last_idx = len(plan.path) - 1
    for idx, (x, y, _) in enumerate(plan.path):
        if idx == 0:
            shifted.append(plan.start)
        elif idx == last_idx:
            shifted.append(plan.goal)
        elif idx <= climb_steps:
            step = 1 if target_z >= start_z else -1
            shifted.append((x, y, start_z + step * idx))
        elif last_idx - idx <= descent_steps:
            step = 1 if target_z >= goal_z else -1
            shifted.append((x, y, goal_z + step * (last_idx - idx)))
        else:
            shifted.append((x, y, target_z))
    if any(not grid.is_free(p) for p in shifted):
        return None
    if not _path_is_valid(shifted, grid):
        return None
    new = plan.copy()
    new.path = shifted
    new.start = plan.start
    new.goal = plan.goal
    new.rerouted = True
    new.changed = True
    return recompute_plan(new, grid, risk_map)


def greedy_resolve_conflicts(
    plans: list[FlightPlan],
    conflicts: list[Conflict],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    preferred_ids: set[int] | None = None,
    max_rounds: int = 60,
) -> tuple[list[FlightPlan], list[Conflict]]:
    current = clone_plans(plans)
    preferred_ids = preferred_ids or set()
    current_conflicts = list(conflicts)
    speed_min, speed_max = [float(v) for v in cfg["optimization"]["speed_range"]]
    delay_threshold = float(cfg["optimization"].get("delay_count_threshold", 30.0))
    delay_cap = int(cfg["optimization"].get("delay_count_cap", max(1, int(np.floor(0.03 * len(current))))))

    for _ in range(max_rounds):
        if not current_conflicts:
            break
        counts = conflict_count_by_flight(current_conflicts)
        candidates = sorted(counts, key=lambda f: ((f not in preferred_ids), -counts[f]))
        improved = False
        id_to_idx = {p.id: idx for idx, p in enumerate(current)}
        candidate_limit = int(cfg["optimization"].get("greedy_candidate_limit", 4))
        for fid in candidates[: min(candidate_limit, len(candidates))]:
            idx = id_to_idx[fid]
            related_cells = [c.cell for c in current_conflicts if c.flight_a == fid or c.flight_b == fid]
            original_count = len(current_conflicts)

            options: list[FlightPlan] = []
            current_z = int(round(np.median([p[2] for p in current[idx].path])))
            min_layer = int(cfg["flight"].get("min_flight_layer", 0))
            layers = sorted(range(min_layer, grid.shape[2]), key=lambda z: (abs(z - current_z), z))
            for z in layers:
                shifted = try_altitude_shift(current[idx], grid, risk_map, z, min_layer=min_layer)
                if shifted is not None:
                    options.append(shifted)

            related_unique = list(dict.fromkeys(related_cells))
            reroute_attempts = int(cfg["optimization"].get("greedy_reroute_attempts", 2))
            reroute_cases = [(1, related_unique[:8]), (0, related_unique[:4]), (2, related_unique[:12]), (1, related_unique)]
            for radius, subset in reroute_cases[:reroute_attempts]:
                if not subset:
                    continue
                rerouted = try_local_reroute(current[idx], grid, risk_map, cfg, subset, radius=radius)
                if rerouted is not None and rerouted.total_air_time <= float(cfg["optimization"]["t_battery"]) * 1.20:
                    options.append(rerouted)

            for factor in (0.62, 0.78, 0.90, 1.10, 1.28, 1.55, 2.00):
                base_speed = float(np.mean(current[idx].speed_profile or [cfg["flight"]["default_speed"]]))
                speed = float(np.clip(base_speed * factor, speed_min, speed_max))
                sp = update_plan_timing(current[idx], speed=speed, delay=0.0, cell_size=grid.cell_size)
                sp.changed = True
                options.append(recompute_plan(sp, grid, risk_map, speed=speed))

            if delayed_count(current, delay_threshold) < delay_cap:
                for delay in (60.0, 90.0, 150.0, 240.0, 420.0, 720.0, 1050.0, 1500.0):
                    options.append(update_plan_timing(current[idx], delay=delay, cell_size=grid.cell_size))

            best_plans = None
            best_conflicts = current_conflicts
            current_global_risk = float(sum(p.risk_sum for p in current))
            best_score = len(current_conflicts) * 1000.0 + current_global_risk
            unaffected_conflicts = [c for c in current_conflicts if c.flight_a != fid and c.flight_b != fid]
            for option in options:
                trial = list(current)
                trial[idx] = option
                active_conflicts = detect_conflicts(trial, cfg, uncertain=True, active_ids={fid})
                trial_conflicts = unaffected_conflicts + active_conflicts
                trial_risk = current_global_risk - current[idx].risk_sum + option.risk_sum
                trial_score = len(trial_conflicts) * 1000.0 + 0.05 * trial_risk + 0.25 * changed_count(trial)
                if len(trial_conflicts) < len(best_conflicts) or trial_score < best_score:
                    best_conflicts = trial_conflicts
                    best_plans = trial
                    best_score = trial_score
            if best_plans is not None and len(best_conflicts) < original_count:
                current = best_plans
                current_conflicts = best_conflicts
                improved = True
                break
        if not improved:
            if len(current_conflicts) <= 35 and _force_time_separation(current, current_conflicts, cfg, grid, risk_map):
                current_conflicts = detect_conflicts(current, cfg, uncertain=True)
                continue
            break
    return current, current_conflicts


def repair_conflicts(
    plans: list[FlightPlan],
    conflicts: list[Conflict],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    max_rounds: int = 50,
) -> tuple[list[FlightPlan], list[dict[str, object]]]:
    current = clone_plans(plans)
    current_conflicts = list(conflicts) if conflicts else detect_conflicts(current, cfg, uncertain=True)
    repair_log: list[dict[str, object]] = []
    for round_idx in range(max_rounds):
        if not current_conflicts:
            break
        result = _choose_best_segment_action(current, current_conflicts, cfg, grid, risk_map)
        if result is None:
            repair_log.append(
                {
                    "round": round_idx,
                    "action": "unresolved",
                    "before_conflicts": len(current_conflicts),
                    "after_conflicts": len(current_conflicts),
                    "reason": "no_action_reduced_global_conflicts",
                }
            )
            break
        current, current_conflicts, log_item = result
        log_item["round"] = round_idx
        repair_log.append(log_item)
    return current, repair_log


def resolve_conflict_segments(
    plans: list[FlightPlan],
    conflicts: list[Conflict],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    max_rounds: int | None = None,
) -> tuple[list[FlightPlan], list[Conflict], list[dict[str, object]]]:
    current = clone_plans(plans)
    current_conflicts = list(conflicts) if conflicts else detect_conflicts(current, cfg, uncertain=True)
    repair_log: list[dict[str, object]] = []
    max_rounds = int(max_rounds if max_rounds is not None else max(1, len(group_continuous_conflicts(current_conflicts)) * 2))

    for round_idx in range(max_rounds):
        if not current_conflicts:
            break
        result = _choose_best_segment_action(current, current_conflicts, cfg, grid, risk_map)
        if result is None:
            repair_log.append(
                {
                    "round": round_idx,
                    "action": "segment_unresolved",
                    "before_conflicts": len(current_conflicts),
                    "after_conflicts": len(current_conflicts),
                    "reason": "no_segment_action_reduced_global_conflicts",
                }
            )
            break
        current, current_conflicts, log_item = result
        log_item["round"] = round_idx
        repair_log.append(log_item)
    return current, current_conflicts, repair_log


def force_resolve_small_conflicts(
    plans: list[FlightPlan],
    conflicts: list[Conflict],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    max_rounds: int = 20,
) -> tuple[list[FlightPlan], list[Conflict], list[dict[str, object]]]:
    current = clone_plans(plans)
    current_conflicts = list(conflicts) if conflicts else detect_conflicts(current, cfg, uncertain=True)
    logs: list[dict[str, object]] = []
    for round_idx in range(max_rounds):
        if not current_conflicts:
            break
        if len(current_conflicts) > int(cfg["optimization"].get("force_resolve_conflict_limit", 8)):
            break
        result = _choose_exact_time_shift(current, current_conflicts, cfg, grid, risk_map)
        if result is None:
            break
        current, current_conflicts, log_item = result
        log_item["round"] = round_idx
        logs.append(log_item)
    return current, current_conflicts, logs


def _choose_exact_time_shift(
    plans: list[FlightPlan],
    conflicts: list[Conflict],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
) -> tuple[list[FlightPlan], list[Conflict], dict[str, object]] | None:
    id_to_idx = {p.id: idx for idx, p in enumerate(plans)}
    current_count = count_conflict_points(conflicts)
    margin = float(cfg["optimization"].get("safety_margin_seconds", 10.0))
    delay_threshold = float(cfg["optimization"].get("delay_count_threshold", 30.0))
    delay_cap = int(cfg["optimization"].get("delay_count_cap", max(1, int(np.floor(0.03 * len(plans))))))
    current_delayed = delayed_count(plans, delay_threshold)
    allow_extra_delay = current_count <= int(cfg["optimization"].get("force_allow_extra_delay_conflicts", 2))
    best_state: list[FlightPlan] | None = None
    best_conflicts: list[Conflict] | None = None
    best_log: dict[str, object] | None = None
    best_score = _repair_score(plans, conflicts, cfg)
    equal_pair_solved: tuple[list[FlightPlan], list[Conflict], dict[str, object], float] | None = None

    for conflict in conflicts:
        pair = (min(conflict.plan_a, conflict.plan_b), max(conflict.plan_a, conflict.plan_b))
        req = float(conflict.required_gap + margin)
        exact = [
            (conflict.plan_a, req - conflict.time_gap, "exact_delay_a"),
            (conflict.plan_a, -req - conflict.time_gap, "exact_advance_a"),
            (conflict.plan_b, conflict.time_gap + req, "exact_delay_b"),
            (conflict.plan_b, conflict.time_gap - req, "exact_advance_b"),
        ]
        for fid in (conflict.plan_a, conflict.plan_b):
            for delta in cfg["optimization"].get("force_shift_candidates", [-300, -240, -180, -120, -90, -60, -30, 30, 60, 90, 120, 180, 240, 300, 420, 600]):
                exact.append((fid, float(delta), "force_grid_shift"))
        for fid, delta, action_name in exact:
            if fid not in id_to_idx or abs(delta) < 1.0:
                continue
            plan = plans[id_to_idx[fid]]
            if delta > 0 and plan.delay < delay_threshold and current_delayed >= delay_cap and not allow_extra_delay:
                continue
            option = _shift_plan_atd(plan, delta, cfg, grid, risk_map)
            if option is None:
                continue
            trial = list(plans)
            trial[id_to_idx[fid]] = option
            trial_conflicts = detect_conflicts(trial, cfg, uncertain=True)
            trial_count = count_conflict_points(trial_conflicts)
            score = _repair_score(trial, trial_conflicts, cfg)
            log_item = {
                "pair": pair,
                "flight_id": fid,
                "action": action_name,
                "before_conflicts": current_count,
                "after_conflicts": trial_count,
                "delta": float(delta),
            }
            if trial_count < current_count and score < best_score:
                best_state = trial
                best_conflicts = trial_conflicts
                best_log = log_item
                best_score = score
                if trial_count == 0:
                    return best_state, best_conflicts, best_log
            elif trial_count == current_count and not _pair_present(trial_conflicts, pair):
                if equal_pair_solved is None or score < equal_pair_solved[3]:
                    equal_pair_solved = (trial, trial_conflicts, log_item, score)

    if best_state is not None and best_conflicts is not None and best_log is not None:
        return best_state, best_conflicts, best_log
    if equal_pair_solved is not None:
        state, next_conflicts, log_item, _ = equal_pair_solved
        return state, next_conflicts, log_item
    return None


def _pair_present(conflicts: list[Conflict], pair: tuple[int, int]) -> bool:
    return any((min(c.plan_a, c.plan_b), max(c.plan_a, c.plan_b)) == pair for c in conflicts)


def _choose_best_segment_action(
    plans: list[FlightPlan],
    conflicts: list[Conflict],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
) -> tuple[list[FlightPlan], list[Conflict], dict[str, object]] | None:
    current_count = count_conflict_points(conflicts)
    if current_count == 0:
        return None
    segments = group_continuous_conflicts(conflicts)
    if not segments:
        return None

    best_state: list[FlightPlan] | None = None
    best_conflicts: list[Conflict] | None = None
    best_log: dict[str, object] | None = None
    best_score = _repair_score(plans, conflicts, cfg)

    scan_limit = int(cfg["optimization"].get("repair_segment_scan_limit", 4))
    action_limit = int(cfg["optimization"].get("repair_action_limit", 40))
    first_improvement = bool(cfg["optimization"].get("repair_first_improvement", True))
    id_to_idx = {p.id: idx for idx, p in enumerate(plans)}
    for segment in segments[: min(len(segments), scan_limit)]:
        for include_reroute in (False, True):
            actions = _generate_segment_actions(plans, segment, cfg, grid, risk_map, include_reroute=include_reroute)
            for action_name, fid, option in actions[:action_limit]:
                if fid not in id_to_idx:
                    continue
                trial = list(plans)
                trial[id_to_idx[fid]] = option
                trial_conflicts = detect_conflicts(trial, cfg, uncertain=True)
                trial_count = count_conflict_points(trial_conflicts)
                if trial_count >= current_count:
                    continue
                trial_score = _repair_score(trial, trial_conflicts, cfg)
                if trial_score < best_score or best_conflicts is None:
                    best_score = trial_score
                    best_state = trial
                    best_conflicts = trial_conflicts
                    best_log = {
                        "pair": (segment.plan_a, segment.plan_b),
                        "flight_id": fid,
                        "action": action_name,
                        "before_conflicts": current_count,
                        "after_conflicts": trial_count,
                        "segment_points": len(segment.conflicts),
                    }
                    if first_improvement or trial_count == 0:
                        return best_state, best_conflicts, best_log
            if best_state is not None and first_improvement:
                return best_state, best_conflicts or [], best_log or {}

    if best_state is None or best_conflicts is None or best_log is None:
        return None
    return best_state, best_conflicts, best_log


def _generate_segment_actions(
    plans: list[FlightPlan],
    segment: ConflictSegment,
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    include_reroute: bool = True,
) -> list[tuple[str, int, FlightPlan]]:
    id_to_plan = {p.id: p for p in plans}
    actions: list[tuple[str, int, FlightPlan]] = []
    safety_margin = float(cfg["optimization"].get("safety_margin_seconds", 10.0))
    required_shift = float(segment.required_shift + safety_margin)
    delay_candidates = _ordered_unique(
        [
            -required_shift,
            -(required_shift + 30.0),
            required_shift,
            required_shift + 30.0,
            required_shift + 60.0,
        ]
        + [float(v) for v in cfg["optimization"].get("delay_candidates", [30, 60, 90, 120, 180, 300, 600])]
    )
    speed_factors = [float(v) for v in cfg["optimization"].get("speed_factors", [0.85, 0.90, 0.95, 1.05, 1.10, 1.15])]
    cells = list(dict.fromkeys(segment.cells))
    radii = [int(v) for v in cfg["optimization"].get("local_reroute_radii", [1, 2, 3])]
    windows = [int(v) for v in cfg["optimization"].get("local_reroute_windows", [5, 10, 15, 20])]
    roles = sorted(
        [(segment.plan_a, "a"), (segment.plan_b, "b")],
        key=lambda item: not bool(id_to_plan.get(item[0]) and (id_to_plan[item[0]].changed or id_to_plan[item[0]].rerouted or abs(id_to_plan[item[0]].delay) > 1e-6)),
    )
    delay_threshold = float(cfg["optimization"].get("delay_count_threshold", 30.0))
    delay_cap = int(cfg["optimization"].get("delay_count_cap", max(1, int(np.floor(0.03 * len(plans))))))
    current_delayed = delayed_count(plans, delay_threshold)

    for fid, role in roles:
        plan = id_to_plan.get(fid)
        if plan is None:
            continue
        for factor in speed_factors:
            adjusted = _scale_plan_speed(plan, factor, cfg, grid, risk_map)
            if adjusted is not None:
                actions.append((f"speed_{role}_{factor:.2f}", fid, adjusted))
        for level in _nearby_levels(segment, grid):
            shifted_level = try_altitude_shift(plan, grid, risk_map, level, min_layer=int(cfg["flight"].get("min_flight_layer", 0)))
            if shifted_level is not None:
                actions.append((f"altitude_{role}_{level}", fid, shifted_level))
        if include_reroute:
            conflict_indices = [c.idx_a if fid == c.plan_a else c.idx_b for c in segment.conflicts if fid in {c.plan_a, c.plan_b}]
            rerouted = try_local_reroute(plan, grid, risk_map, cfg, cells, conflict_indices=conflict_indices, windows=windows, radii=radii)
            if rerouted is not None:
                actions.append((f"reroute_{role}", fid, rerouted))
        for delta in delay_candidates:
            if delta > 0 and plan.delay < delay_threshold and current_delayed >= delay_cap:
                continue
            shifted = _shift_plan_atd(plan, delta, cfg, grid, risk_map)
            if shifted is not None:
                actions.append((f"delay_{role}_{int(round(delta))}", fid, shifted))

    return _dedupe_actions(actions)


def _shift_plan_atd(
    plan: FlightPlan,
    delta: float,
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
) -> FlightPlan | None:
    new_etd = float(plan.etd + delta)
    if new_etd <= 0.0:
        return None
    if plan.delay + delta > float(cfg["optimization"]["t_delay_max"]):
        return None
    shifted = update_plan_timing(plan, delay=delta, cell_size=grid.cell_size)
    shifted.changed = True
    return recompute_plan(shifted, grid, risk_map)


def _scale_plan_speed(
    plan: FlightPlan,
    factor: float,
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
) -> FlightPlan | None:
    speed_min, speed_max = [float(v) for v in cfg["optimization"]["speed_range"]]
    base_speed = float(np.mean(plan.speed_profile or [cfg["flight"]["default_speed"]]))
    new_speed = float(np.clip(base_speed * factor, speed_min, speed_max))
    if abs(new_speed - base_speed) <= 1e-6:
        return None
    adjusted = update_plan_timing(plan, speed=new_speed, delay=0.0, cell_size=grid.cell_size)
    adjusted.changed = True
    return recompute_plan(adjusted, grid, risk_map, speed=new_speed)


def _nearby_levels(segment: ConflictSegment, grid: AirspaceGrid) -> list[int]:
    if not segment.conflicts:
        return []
    z_values = [c.cell[2] for c in segment.conflicts]
    center_z = int(round(float(np.median(z_values))))
    preferred = [center_z + 1] if center_z <= 1 else [center_z - 1]
    preferred.extend([center_z - 1, center_z + 1])
    preferred.extend(range(grid.shape[2]))
    return [int(z) for z in _ordered_unique(preferred) if 0 <= int(z) < grid.shape[2]]


def _ordered_unique(values: list[float] | list[int]) -> list[float]:
    seen: set[float] = set()
    out: list[float] = []
    for value in values:
        rounded = float(round(float(value), 6))
        if rounded in seen:
            continue
        seen.add(rounded)
        out.append(float(value))
    return out


def _dedupe_actions(actions: list[tuple[str, int, FlightPlan]]) -> list[tuple[str, int, FlightPlan]]:
    seen: set[tuple[int, tuple[GridPoint, ...], int, int]] = set()
    out: list[tuple[str, int, FlightPlan]] = []
    for name, fid, plan in actions:
        sig = (fid, tuple(plan.path), int(round(plan.etd)), int(round(float(np.mean(plan.speed_profile or [0.0])) * 100)))
        if sig in seen:
            continue
        seen.add(sig)
        out.append((name, fid, plan))
    return out


def _repair_score(plans: list[FlightPlan], conflicts: list[Conflict], cfg: dict) -> float:
    delay_threshold = float(cfg["optimization"].get("delay_count_threshold", 30.0))
    delay_cap = int(cfg["optimization"].get("delay_count_cap", max(1, int(np.floor(0.03 * len(plans))))))
    delayed_over = max(0, delayed_count(plans, delay_threshold) - delay_cap)
    changed_cap = max(1, int(np.floor(float(cfg["optimization"].get("max_changed_flight_ratio", 1.0)) * len(plans))))
    changed_over = max(0, changed_count(plans) - changed_cap)
    total_delay = float(sum(max(0.0, p.delay) for p in plans))
    total_air = float(sum(p.total_air_time for p in plans))
    total_risk = float(sum(p.risk_sum for p in plans))
    return (
        count_conflict_points(conflicts) * float(cfg["optimization"].get("conflict_hard_penalty", 1_000_000.0))
        + delayed_over * 100_000.0
        + changed_over * 50_000.0
        + changed_count(plans) * 100.0
        + total_delay * 0.01
        + total_air * 0.001
        + total_risk * 0.0001
    )


def polish_final_conflicts(
    plans: list[FlightPlan],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    max_steps: int = 5,
    beam_width: int = 24,
) -> tuple[list[FlightPlan], list[Conflict]]:
    delay_threshold = float(cfg["optimization"].get("delay_count_threshold", 30.0))
    delay_cap = int(cfg["optimization"].get("delay_count_cap", max(1, int(np.floor(0.03 * len(plans))))))
    current = list(plans)
    current_conflicts = detect_conflicts(current, cfg, uncertain=True)
    if not current_conflicts:
        return current, current_conflicts

    def simple_score(state: list[FlightPlan], conflicts: list[Conflict] | None = None) -> float:
        conflicts = conflicts if conflicts is not None else detect_conflicts(state, cfg, uncertain=True)
        cap_violation = max(0, delayed_count(state, delay_threshold) - delay_cap)
        total_delay = float(sum(max(0.0, p.delay) for p in state))
        return len(conflicts) * 1_000_000.0 + cap_violation * 200_000.0 + changed_count(state) * 20.0 + total_delay * 0.01

    for _ in range(max_steps):
        if not current_conflicts:
            return current, current_conflicts
        best_state = None
        best_conflicts = current_conflicts
        best_score = simple_score(current, current_conflicts)
        ids = set(conflict_count_by_flight(current_conflicts))
        id_to_idx = {p.id: idx for idx, p in enumerate(current)}
        for fid in ids:
            idx = id_to_idx[fid]
            related = [c.cell for c in current_conflicts if c.flight_a == fid or c.flight_b == fid]
            unaffected_conflicts = [c for c in current_conflicts if c.flight_a != fid and c.flight_b != fid]
            for option in _polish_options_for_plan(current[idx], related, cfg, grid, risk_map, include_delay=False):
                trial = list(current)
                trial[idx] = option
                if delayed_count(trial, delay_threshold) > delay_cap:
                    continue
                trial_conflicts = unaffected_conflicts + detect_conflicts(trial, cfg, uncertain=True, active_ids={fid})
                score = simple_score(trial, trial_conflicts)
                if len(trial_conflicts) < len(best_conflicts) or score < best_score:
                    best_state = trial
                    best_conflicts = trial_conflicts
                    best_score = score
        if best_state is None or len(best_conflicts) >= len(current_conflicts):
            break
        current = best_state
        current_conflicts = best_conflicts

    if not current_conflicts:
        return current, current_conflicts

    delayed_ids = [p.id for p in current if p.delay >= delay_threshold]
    conflict_ids = set(conflict_count_by_flight(current_conflicts))
    id_to_idx = {p.id: idx for idx, p in enumerate(current)}
    best_state = current
    best_conflicts = current_conflicts
    best_score = simple_score(current, current_conflicts)

    for free_fid in delayed_ids:
        free_idx = id_to_idx[free_fid]
        reset_seed = _reset_delay(current[free_idx])
        free_options = _polish_options_for_plan(reset_seed, [], cfg, grid, risk_map, include_delay=False)
        free_options = [opt for opt in free_options if opt.delay < delay_threshold]
        for free_option in free_options[:24]:
            freed = list(current)
            freed[free_idx] = free_option
            for conflict_fid in conflict_ids:
                conflict_idx = id_to_idx[conflict_fid]
                delay_options = [
                    recompute_plan(update_plan_timing(freed[conflict_idx], delay=delay, cell_size=grid.cell_size), grid, risk_map)
                    for delay in (60.0, 120.0, 240.0, 420.0, 720.0, 900.0, 1050.0, 1200.0, 1500.0, 1800.0)
                ]
                for delayed_option in delay_options:
                    trial = list(freed)
                    trial[conflict_idx] = delayed_option
                    if delayed_count(trial, delay_threshold) > delay_cap:
                        continue
                    trial_conflicts = detect_conflicts(trial, cfg, uncertain=True, active_ids={free_fid, conflict_fid})
                    unaffected = [
                        c
                        for c in current_conflicts
                        if c.flight_a not in {free_fid, conflict_fid} and c.flight_b not in {free_fid, conflict_fid}
                    ]
                    trial_conflicts = unaffected + trial_conflicts
                    score = simple_score(trial, trial_conflicts)
                    if len(trial_conflicts) < len(best_conflicts) or score < best_score:
                        best_state = trial
                        best_conflicts = trial_conflicts
                        best_score = score
                    if not trial_conflicts:
                        return trial, trial_conflicts

    return best_state, best_conflicts


def _reset_delay(plan: FlightPlan) -> FlightPlan:
    if plan.delay <= 0:
        return plan.copy()
    new = plan.copy()
    delay = float(new.delay)
    new.delay = 0.0
    new.etd -= delay
    new.eta_times = [t - delay for t in new.eta_times]
    new.changed = bool(new.rerouted or abs(float(np.mean(new.speed_profile or [10.0])) - 10.0) > 1e-6)
    return new


def _polish_options_for_plan(
    plan: FlightPlan,
    related_cells: list[GridPoint],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    include_delay: bool = True,
) -> list[FlightPlan]:
    speed_min, speed_max = [float(v) for v in cfg["optimization"]["speed_range"]]
    min_layer = int(cfg["flight"].get("min_flight_layer", 0))
    options: list[FlightPlan] = []
    seeds = [plan]
    if plan.delay > 0:
        seeds.append(_reset_delay(plan))

    for seed in seeds:
        options.append(seed.copy())
        for speed in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20):
            speed = float(np.clip(speed, speed_min, speed_max))
            sp = update_plan_timing(seed, speed=speed, delay=0.0, cell_size=grid.cell_size)
            sp.changed = True
            options.append(recompute_plan(sp, grid, risk_map, speed=speed))
        for z in range(min_layer, grid.shape[2]):
            shifted = try_altitude_shift(seed, grid, risk_map, z, min_layer=min_layer)
            if shifted is not None:
                options.append(shifted)

    if related_cells:
        unique_cells = list(dict.fromkeys(related_cells))
        for radius in (0, 1, 2, 3, 4):
            rerouted = try_local_reroute(plan, grid, risk_map, cfg, unique_cells, radius=radius)
            if rerouted is not None:
                options.append(rerouted)

    if include_delay:
        for delay in (60.0, 120.0, 240.0, 420.0, 720.0, 900.0, 1050.0, 1200.0, 1500.0, 1800.0):
            delayed = update_plan_timing(plan, delay=delay, cell_size=grid.cell_size)
            options.append(recompute_plan(delayed, grid, risk_map))

    deduped: list[FlightPlan] = []
    seen: set[tuple[tuple[GridPoint, ...], int, int]] = set()
    for option in options:
        if option.total_air_time > float(cfg["optimization"]["t_battery"]) * 1.25:
            continue
        sig = (tuple(option.path), int(round(float(np.mean(option.speed_profile or [0.0])) * 10)), int(round(option.delay)))
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(option)
    return deduped


def _force_time_separation(
    plans: list[FlightPlan],
    conflicts: list[Conflict],
    cfg: dict,
    grid: AirspaceGrid,
    risk_map: np.ndarray,
) -> bool:
    delay_threshold = float(cfg["optimization"].get("delay_count_threshold", 30.0))
    delay_cap = int(cfg["optimization"].get("delay_count_cap", max(1, int(np.floor(0.03 * len(plans))))))
    if delayed_count(plans, delay_threshold) >= delay_cap:
        return False
    counts = conflict_count_by_flight(conflicts)
    if not counts:
        return False
    fid = max(counts, key=counts.get)
    id_to_idx = {p.id: idx for idx, p in enumerate(plans)}
    idx = id_to_idx[fid]
    base = plans[idx]
    best_plan: FlightPlan | None = None
    best_count = len(conflicts)
    for delay in (60.0, 120.0, 240.0, 420.0, 720.0, 1050.0, 1500.0, 1800.0):
        shifted = update_plan_timing(base, delay=delay, cell_size=grid.cell_size)
        shifted.changed = True
        trial = clone_plans(plans)
        trial[idx] = recompute_plan(shifted, grid, risk_map)
        trial_conflicts = detect_conflicts(trial, cfg, uncertain=True)
        if len(trial_conflicts) < best_count:
            best_count = len(trial_conflicts)
            best_plan = trial[idx]
    if best_plan is None:
        return False
    plans[idx] = best_plan
    return True


def solution_bounds(
    flight_ids: list[int],
    cfg: dict,
    route_options: dict[int, list[FlightPlan]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(flight_ids)
    speed_min, speed_max = [float(v) for v in cfg["optimization"]["speed_range"]]
    delay_max = float(cfg["optimization"]["t_delay_max"])
    lb = np.array([0.0] * n + [speed_min] * n, dtype=float)
    ub = np.array([delay_max] * n + [speed_max] * n, dtype=float)
    if route_options is not None:
        lb = np.concatenate([lb, np.zeros(n, dtype=float)])
        ub = np.concatenate(
            [
                ub,
                np.array([max(0, len(route_options.get(fid, [])) - 1) for fid in flight_ids], dtype=float),
            ]
        )
    return lb, ub


def changed_count(plans: list[FlightPlan]) -> int:
    return int(sum(1 for p in plans if p.changed or p.rerouted or p.delay > 1e-6))


def delayed_count(plans: list[FlightPlan], threshold: float = 30.0) -> int:
    return int(sum(1 for p in plans if p.delay >= threshold))


def risk_increase_ratio(initial: list[FlightPlan], final: list[FlightPlan]) -> float:
    exposure = max(1.0, float(sum(path_distance_m(p.path) for p in initial)) / 10.0)
    base = max(exposure, float(sum(p.risk_sum for p in initial)))
    new = float(sum(p.risk_sum for p in final))
    old = float(sum(p.risk_sum for p in initial))
    return (new - old) / base
