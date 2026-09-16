from __future__ import annotations

import copy
import csv
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .astar_3d import astar_distance_unit_m, astar_path
from .grid import AirspaceGrid, GridPoint
from .utils import ensure_dir, path_distance_m, safe_tqdm


@dataclass
class FlightPlan:
    id: int
    start: GridPoint
    goal: GridPoint
    path: list[GridPoint]
    etd: float
    eta_times: list[float]
    speed_profile: list[float]
    risk_sum: float
    total_air_time: float
    delay: float = 0.0
    changed: bool = False
    rerouted: bool = False
    generation_mode: str = ""
    waypoint: GridPoint | None = None
    start_ground: GridPoint | None = None
    goal_ground: GridPoint | None = None
    start_level: int | None = None
    goal_level: int | None = None
    cruise_level: int | None = None
    altitude_profile: dict[str, int] | None = None

    def copy(self) -> "FlightPlan":
        return copy.deepcopy(self)


def compute_eta_times(path: list[GridPoint], etd: float, speed: float | list[float], cell_size: tuple[int, int, int]) -> list[float]:
    times = [float(etd)]
    current = float(etd)
    speeds = _segment_speeds(path, speed)
    for seg_idx, (a, b) in enumerate(zip(path[:-1], path[1:])):
        dist = path_distance_m([a, b], cell_size)
        current += dist / max(float(speeds[seg_idx]), 1e-6)
        times.append(current)
    return times


def _segment_speeds(path: list[GridPoint], speed: float | list[float]) -> list[float]:
    n_segments = max(0, len(path) - 1)
    if n_segments == 0:
        return []
    if isinstance(speed, (int, float, np.floating)):
        return [float(speed)] * n_segments
    values = [float(v) for v in speed]
    if not values:
        return [10.0] * n_segments
    if len(values) == n_segments:
        return values
    if len(values) == len(path):
        return values[:-1]
    if len(values) < n_segments:
        return values + [values[-1]] * (n_segments - len(values))
    return values[:n_segments]


def recompute_eta_times(plan: FlightPlan, cell_size: tuple[int, int, int] = (100, 100, 30), speed: float | list[float] | None = None) -> list[float]:
    return compute_eta_times(plan.path, plan.etd, speed if speed is not None else plan.speed_profile, cell_size)


def shift_plan_time(plan: FlightPlan, delta_t: float) -> FlightPlan:
    new = plan.copy()
    new.etd = float(new.etd + delta_t)
    new.eta_times = [float(t + delta_t) for t in new.eta_times]
    new.delay = float(new.delay + delta_t)
    new.changed = new.changed or abs(delta_t) > 1e-6
    if hasattr(new, "atd"):
        setattr(new, "atd", new.etd)
    return new


def apply_speed_factor(
    plan: FlightPlan,
    factor: float,
    speed_min: float,
    speed_max: float,
    cell_size: tuple[int, int, int] = (100, 100, 30),
) -> FlightPlan:
    new = plan.copy()
    base_speed = float(np.mean(_segment_speeds(plan.path, plan.speed_profile or [10.0]) or [10.0]))
    speed = float(np.clip(base_speed * float(factor), speed_min, speed_max))
    new.speed_profile = [speed for _ in range(max(0, len(new.path) - 1))]
    new.eta_times = recompute_eta_times(new, cell_size, speed=speed)
    new.total_air_time = max(0.0, new.eta_times[-1] - new.etd)
    new.changed = new.changed or abs(speed - base_speed) > 1e-6
    return new


def update_plan_timing(plan: FlightPlan, speed: float | None = None, delay: float = 0.0, cell_size: tuple[int, int, int] = (100, 100, 30)) -> FlightPlan:
    new = plan.copy()
    old_speed = float(np.mean(_segment_speeds(plan.path, plan.speed_profile or [10.0]) or [10.0]))
    base_speed = float(speed if speed is not None else old_speed)
    new.delay = float(plan.delay + delay)
    new.etd = float(plan.etd + delay)
    if speed is None:
        new.eta_times = [float(t + delay) for t in plan.eta_times]
        new.speed_profile = list(plan.speed_profile)
    else:
        new.eta_times = compute_eta_times(new.path, new.etd, base_speed, cell_size)
        new.speed_profile = [base_speed for _ in range(max(0, len(new.path) - 1))]
    new.total_air_time = max(0.0, new.eta_times[-1] - new.etd)
    new.changed = bool(delay) or speed is not None and abs(base_speed - old_speed) > 1e-6 or plan.changed
    return new


def _normalized_probabilities(values: list[float] | np.ndarray, n: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if n is not None:
        if len(arr) < n:
            arr = np.pad(arr, (0, n - len(arr)), constant_values=0.0)
        elif len(arr) > n:
            arr = arr[:n]
    arr = np.where(np.isfinite(arr) & (arr > 0.0), arr, 0.0)
    if arr.size == 0 or float(arr.sum()) <= 0.0:
        arr = np.ones(n if n is not None else 1, dtype=float)
    return arr / float(arr.sum())


def _altitude_cfg(config: dict) -> dict:
    return config.get("flight_generation", config).get("altitude", {})


def sample_altitude_profile(rng: np.random.Generator, config: dict) -> dict[str, int]:
    alt_cfg = _altitude_cfg(config)
    fallback = alt_cfg.get("level_probs", [0.05, 0.35, 0.45, 0.15])
    n_levels = max(len(alt_cfg.get("cruise_level_probs", fallback)), len(fallback))
    ground_level = int(alt_cfg.get("ground_level", 0))
    cruise_probs = _normalized_probabilities(alt_cfg.get("cruise_level_probs", fallback), n_levels)
    cruise_level = int(rng.choice(np.arange(n_levels), p=cruise_probs))
    return {"start_level": ground_level, "goal_level": ground_level, "cruise_level": cruise_level}


def _clip_altitude_profile(profile: dict[str, int], grid: AirspaceGrid, cfg: dict) -> dict[str, int]:
    alt_cfg = cfg.get("flight_generation", {}).get("altitude", {})
    max_level = grid.shape[2] - 1
    ground_level = int(np.clip(alt_cfg.get("ground_level", profile.get("start_level", 0)), 0, max_level))
    return {
        "start_level": ground_level,
        "goal_level": ground_level,
        "cruise_level": int(np.clip(profile["cruise_level"], 0, max_level)),
    }


def _sample_altitude_layer(rng: np.random.Generator, grid: AirspaceGrid, fg_cfg: dict, kind: str = "start_goal") -> int:
    alt_cfg = fg_cfg.get("altitude", {})
    if kind == "cruise":
        probs = _normalized_probabilities(alt_cfg.get("cruise_level_probs", alt_cfg.get("level_probs", [1.0] * grid.shape[2])), grid.shape[2])
        return int(rng.choice(np.arange(grid.shape[2]), p=probs))
    return int(np.clip(alt_cfg.get("ground_level", 0), 0, grid.shape[2] - 1))


def _sample_truncated_normal(rng: np.random.Generator, mean: float, std: float, lo: float, hi: float) -> float:
    for _ in range(80):
        value = float(rng.normal(mean, std))
        if lo <= value <= hi:
            return value
    return float(np.clip(rng.normal(mean, std), lo, hi))


def _cell_distance_m(a: GridPoint, b: GridPoint, grid: AirspaceGrid) -> float:
    return float(np.linalg.norm((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) * np.asarray(grid.cell_size, dtype=float)))


def _build_heterogeneous_hotspots(grid: AirspaceGrid, od_cfg: dict, rng: np.random.Generator) -> np.ndarray:
    nx, ny, _ = grid.shape
    count = max(4, int(od_cfg.get("hotspot_count", 10)))
    std = float(od_cfg.get("hotspot_std_cells", 6.5))
    center_bias = float(np.clip(od_cfg.get("center_bias_strength", 0.25), 0.0, 1.0))
    center_count = min(3, max(1, int(round(count * center_bias))))
    edge_count = min(count - center_count, max(4, int(round(count * 0.40))))
    center_low = np.array([nx * 22.0 / 60.0, ny * 22.0 / 60.0], dtype=float)
    center_high = np.array([nx * 38.0 / 60.0, ny * 38.0 / 60.0], dtype=float)

    hotspots: list[np.ndarray] = []
    for _ in range(center_count):
        xy = rng.uniform(center_low, center_high)
        xy += rng.normal(0.0, max(1.0, std * 0.25), size=2)
        hotspots.append(np.clip(xy, center_low, center_high))

    edge_anchors = np.array(
        [
            [0.10 * nx, 0.14 * ny],
            [0.12 * nx, 0.84 * ny],
            [0.88 * nx, 0.12 * ny],
            [0.90 * nx, 0.86 * ny],
            [0.08 * nx, 0.50 * ny],
            [0.92 * nx, 0.50 * ny],
            [0.50 * nx, 0.08 * ny],
            [0.50 * nx, 0.92 * ny],
        ],
        dtype=float,
    )
    anchor_probs = _normalized_probabilities(np.array([1.0, 0.7, 1.25, 0.85, 1.1, 0.9, 0.75, 1.0]))
    while len(hotspots) < center_count + edge_count:
        anchor = edge_anchors[int(rng.choice(np.arange(len(edge_anchors)), p=anchor_probs))]
        xy = anchor + rng.normal(0.0, max(1.0, std * 0.9), size=2)
        hotspots.append(np.clip(xy, [0.5, 0.5], [nx - 1.5, ny - 1.5]))
    while len(hotspots) < count:
        if rng.random() < center_bias * 0.5:
            xy = rng.normal([nx / 2.0, ny / 2.0], max(1.0, std * 1.3), size=2)
        else:
            xy = rng.uniform([0.5, 0.5], [nx - 1.5, ny - 1.5])
        hotspots.append(np.clip(xy, [0.5, 0.5], [nx - 1.5, ny - 1.5]))
    return np.vstack(hotspots)


def _sample_free_near_xy(
    grid: AirspaceGrid,
    xy: np.ndarray,
    z: int,
    std_cells: float,
    rng: np.random.Generator,
    attempts: int = 180,
) -> GridPoint | None:
    nx, ny, _ = grid.shape
    for _ in range(attempts):
        x = int(np.clip(round(float(rng.normal(xy[0], std_cells))), 0, nx - 1))
        y = int(np.clip(round(float(rng.normal(xy[1], std_cells))), 0, ny - 1))
        p = (x, y, int(z))
        if grid.is_free(p):
            return p
    return None


def _sample_random_free_cell(grid: AirspaceGrid, rng: np.random.Generator, fg_cfg: dict, z: int | None = None, attempts: int = 500) -> GridPoint | None:
    nx, ny, _ = grid.shape
    for _ in range(attempts):
        p = (int(rng.integers(0, nx)), int(rng.integers(0, ny)), int(z if z is not None else _sample_altitude_layer(rng, grid, fg_cfg)))
        if grid.is_free(p):
            return p
    return None


def _sample_edge_free_cell(grid: AirspaceGrid, side: str, rng: np.random.Generator, fg_cfg: dict, z: int | None = None, attempts: int = 400) -> GridPoint | None:
    nx, ny, _ = grid.shape
    level = int(z if z is not None else _sample_altitude_layer(rng, grid, fg_cfg))
    margin = max(2, int(round(min(nx, ny) * 0.08)))
    center_bias = float(fg_cfg.get("od_pattern", {}).get("center_bias_strength", 0.25))
    for _ in range(attempts):
        offset = int(np.clip(abs(rng.normal(1.0, max(1.0, margin / 2.0))), 0, margin))
        if side == "west":
            x = offset
            y = int(np.clip(round(rng.normal(ny / 2.0, ny * 0.20)) if rng.random() < center_bias else rng.integers(0, ny), 0, ny - 1))
        elif side == "east":
            x = nx - 1 - offset
            y = int(np.clip(round(rng.normal(ny / 2.0, ny * 0.20)) if rng.random() < center_bias else rng.integers(0, ny), 0, ny - 1))
        elif side == "south":
            x = int(np.clip(round(rng.normal(nx / 2.0, nx * 0.20)) if rng.random() < center_bias else rng.integers(0, nx), 0, nx - 1))
            y = offset
        else:
            x = int(np.clip(round(rng.normal(nx / 2.0, nx * 0.20)) if rng.random() < center_bias else rng.integers(0, nx), 0, nx - 1))
            y = ny - 1 - offset
        p = (x, y, level)
        if grid.is_free(p):
            return p
    return None


def _sample_hub_pair(
    grid: AirspaceGrid,
    hotspots: np.ndarray,
    fg_cfg: dict,
    rng: np.random.Generator,
    target_m: float,
    tolerance_m: float,
    altitude_profile: dict[str, int],
) -> tuple[GridPoint, GridPoint] | None:
    od_cfg = fg_cfg.get("od_pattern", {})
    hotspot_std = float(od_cfg.get("hotspot_std_cells", 5.0))
    n_hotspots = len(hotspots)
    if n_hotspots < 2:
        return None
    a_idx = int(rng.integers(0, n_hotspots))
    xy_scale = np.asarray(grid.cell_size[:2], dtype=float)
    distances = np.linalg.norm((hotspots - hotspots[a_idx]) * xy_scale, axis=1)
    weights = np.exp(-0.5 * ((distances - target_m) / max(tolerance_m, 100.0)) ** 2)
    weights[a_idx] = 0.0
    if float(weights.sum()) <= 1e-12:
        weights = np.ones(n_hotspots, dtype=float)
        weights[a_idx] = 0.0
    weights = weights / float(weights.sum())
    b_idx = int(rng.choice(np.arange(n_hotspots), p=weights))
    start = _sample_free_near_xy(grid, hotspots[a_idx], altitude_profile["start_level"], hotspot_std, rng)
    goal = _sample_free_near_xy(grid, hotspots[b_idx], altitude_profile["goal_level"], hotspot_std, rng)
    if start is None or goal is None:
        return None
    return start, goal


def _sample_edge_pair(grid: AirspaceGrid, fg_cfg: dict, rng: np.random.Generator, altitude_profile: dict[str, int], force_opposite: bool = False) -> tuple[GridPoint, GridPoint] | None:
    side_pairs = [("west", "east"), ("east", "west"), ("south", "north"), ("north", "south")]
    if force_opposite:
        side_a, side_b = side_pairs[int(rng.integers(0, len(side_pairs)))]
    else:
        sides = ["west", "east", "south", "north"]
        side_a = sides[int(rng.integers(0, len(sides)))]
        candidates = [side for side in sides if side != side_a]
        side_b = candidates[int(rng.integers(0, len(candidates)))]
    start = _sample_edge_free_cell(grid, side_a, rng, fg_cfg, z=altitude_profile["start_level"])
    goal = _sample_edge_free_cell(grid, side_b, rng, fg_cfg, z=altitude_profile["goal_level"])
    if start is None or goal is None:
        return None
    return start, goal


def _sample_center_crossing_pair(grid: AirspaceGrid, fg_cfg: dict, rng: np.random.Generator, altitude_profile: dict[str, int]) -> tuple[GridPoint, GridPoint] | None:
    return _sample_edge_pair(grid, fg_cfg, rng, altitude_profile, force_opposite=True)


def _sample_random_pair(grid: AirspaceGrid, fg_cfg: dict, rng: np.random.Generator, altitude_profile: dict[str, int]) -> tuple[GridPoint, GridPoint] | None:
    start = _sample_random_free_cell(grid, rng, fg_cfg, z=altitude_profile["start_level"])
    goal = _sample_random_free_cell(grid, rng, fg_cfg, z=altitude_profile["goal_level"])
    if start is None or goal is None:
        return None
    return start, goal


def _sample_od_pair(
    mode: str,
    grid: AirspaceGrid,
    hotspots: np.ndarray,
    fg_cfg: dict,
    rng: np.random.Generator,
    target_m: float,
    tolerance_m: float,
    altitude_profile: dict[str, int],
) -> tuple[GridPoint, GridPoint] | None:
    if mode == "hub_to_hub":
        return _sample_hub_pair(grid, hotspots, fg_cfg, rng, target_m, tolerance_m, altitude_profile)
    if mode == "edge_to_edge":
        return _sample_edge_pair(grid, fg_cfg, rng, altitude_profile, force_opposite=False)
    if mode == "center_crossing":
        return _sample_center_crossing_pair(grid, fg_cfg, rng, altitude_profile)
    return _sample_random_pair(grid, fg_cfg, rng, altitude_profile)


def _sample_takeoff_time(rng: np.random.Generator, fg_cfg: dict) -> float:
    time_cfg = fg_cfg.get("takeoff_time", {})
    t0, t1 = [float(v) for v in time_cfg.get("window", [0, 1800])]
    if rng.random() < float(time_cfg.get("uniform_noise_ratio", 0.15)):
        return float(rng.uniform(t0, t1))
    centers = np.asarray(time_cfg.get("peak_centers", [300, 750, 1200]), dtype=float)
    stds = np.asarray(time_cfg.get("peak_stds", [120, 180, 160]), dtype=float)
    weights = _normalized_probabilities(time_cfg.get("peak_weights", [0.25, 0.50, 0.25]), len(centers))
    peak = int(rng.choice(np.arange(len(centers)), p=weights))
    return float(np.clip(rng.normal(centers[peak], max(stds[peak], 1.0)), t0, t1))


def _sample_initial_speed(rng: np.random.Generator, fg_cfg: dict) -> float:
    speed_cfg = fg_cfg.get("speed", {})
    return _sample_truncated_normal(
        rng,
        float(speed_cfg.get("mean", 10.0)),
        float(speed_cfg.get("std", 1.5)),
        float(speed_cfg.get("min", 7.0)),
        float(speed_cfg.get("max", 14.0)),
    )


def _scaled_corridor_regions(grid: AirspaceGrid, count: int) -> list[tuple[float, float, float, float]]:
    nx, ny, _ = grid.shape
    base = [
        (20, 30, 25, 35),
        (35, 45, 20, 30),
        (15, 25, 35, 45),
        (40, 50, 35, 45),
    ]
    regions: list[tuple[float, float, float, float]] = []
    for x0, x1, y0, y1 in base[: max(1, min(count, len(base)))]:
        regions.append((x0 * nx / 60.0, x1 * nx / 60.0, y0 * ny / 60.0, y1 * ny / 60.0))
    return regions


def _sample_corridor_waypoint(grid: AirspaceGrid, fg_cfg: dict, rng: np.random.Generator, cruise_level: int) -> GridPoint | None:
    nx, ny, _ = grid.shape
    overlap_cfg = fg_cfg.get("route_overlap", {})
    width = max(1.0, float(overlap_cfg.get("corridor_width_cells", 6)))
    corridor_count = int(overlap_cfg.get("corridor_count", 4))
    regions = _scaled_corridor_regions(grid, corridor_count) if overlap_cfg.get("use_multiple_corridors", True) else [(25 * nx / 60.0, 35 * nx / 60.0, 25 * ny / 60.0, 35 * ny / 60.0)]
    for _ in range(50):
        x0, x1, y0, y1 = regions[int(rng.integers(0, len(regions)))]
        xy = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)], dtype=float)
        p = _sample_free_near_xy(grid, xy, cruise_level, width * 0.5, rng, attempts=40)
        if p is not None:
            return p
    return None


def _astar_route(
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    cfg: dict,
    start: GridPoint,
    goal: GridPoint,
    waypoint: GridPoint | None = None,
    altitude_profile: dict[str, int] | None = None,
) -> list[GridPoint]:
    astar_cfg = cfg.get("astar", {})
    alpha_r = float(astar_cfg.get("risk_weight", cfg["risk"]["alpha_r"]))
    alpha_l = float(astar_cfg.get("distance_weight", cfg["risk"]["alpha_L"]))
    cruise_level = None if altitude_profile is None else int(altitude_profile["cruise_level"])
    ground_level = int(cfg.get("flight_generation", {}).get("altitude", {}).get("ground_level", 0))
    astar_kwargs = {
        "alpha_r": alpha_r,
        "alpha_l": alpha_l,
        "cruise_level": cruise_level,
        "altitude_preference_enabled": bool(astar_cfg.get("altitude_preference_enabled", True)),
        "cruise_altitude_penalty_weight": float(astar_cfg.get("cruise_altitude_penalty_weight", 0.35)),
        "ground_hugging_penalty_weight": float(astar_cfg.get("ground_hugging_penalty_weight", 0.8)),
        "ground_level": ground_level,
        "climb_descent_cells": int(cfg.get("flight_generation", {}).get("altitude", {}).get("climb_descent_cells", 3)),
        "vertical_move_penalty": float(astar_cfg.get("vertical_move_penalty", 1.8)),
        "risk_weight": alpha_r,
        "distance_weight": alpha_l,
    }
    if waypoint is None:
        return astar_path(grid, start, goal, risk_map, **astar_kwargs)
    first = astar_path(grid, start, waypoint, risk_map, **astar_kwargs)
    if not first:
        return []
    second = astar_path(grid, waypoint, goal, risk_map, **astar_kwargs)
    if not second:
        return []
    return first + second[1:]


def _cruise_altitude_fraction(path: list[GridPoint], altitude_profile: dict[str, int], fg_cfg: dict) -> float:
    if not path:
        return 0.0
    climb = max(0, int(fg_cfg.get("altitude", {}).get("climb_descent_cells", 3)))
    middle = path[climb : len(path) - climb] if len(path) > 2 * climb else path
    if not middle:
        middle = path
    cruise = int(altitude_profile["cruise_level"])
    return float(np.mean([abs(p[2] - cruise) <= 1 for p in middle]))


def _plan_route_with_optional_waypoint(
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    cfg: dict,
    fg_cfg: dict,
    start: GridPoint,
    goal: GridPoint,
    altitude_profile: dict[str, int],
    use_corridor: bool,
    rng: np.random.Generator,
) -> tuple[list[GridPoint], GridPoint | None]:
    if use_corridor and fg_cfg.get("route_overlap", {}).get("encourage_overlap", True):
        for _ in range(18):
            waypoint = _sample_corridor_waypoint(grid, fg_cfg, rng, altitude_profile["cruise_level"])
            if waypoint is None:
                continue
            path = _astar_route(grid, risk_map, cfg, start, goal, waypoint, altitude_profile=altitude_profile)
            if path:
                return path, waypoint
    path = _astar_route(grid, risk_map, cfg, start, goal, altitude_profile=altitude_profile)
    return path, None


def _route_density_xy(plans: list[FlightPlan], grid: AirspaceGrid) -> np.ndarray:
    density = np.zeros(grid.shape[:2], dtype=float)
    for plan in plans:
        seen = {(int(x), int(y)) for x, y, _ in plan.path}
        for x, y in seen:
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                density[x, y] += 1.0
    return density


def _gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    arr = np.sort(np.clip(arr, 0.0, None))
    total = float(arr.sum())
    if total <= 0.0:
        return 0.0
    n = arr.size
    cumulative = np.cumsum(arr)
    return float((n + 1.0 - 2.0 * float(cumulative.sum()) / total) / n)


def route_density_gini(plans: list[FlightPlan], grid: AirspaceGrid) -> float:
    return _gini(_route_density_xy(plans, grid))


def _ground_validation_counts(plans: list[FlightPlan], ground_level: int = 0) -> dict[str, int]:
    return {
        "start_goal_above_ground_count": sum(
            1 for plan in plans if plan.start[2] != ground_level or plan.goal[2] != ground_level
        ),
        "path_start_not_ground_count": sum(
            1 for plan in plans if not plan.path or plan.path[0][2] != ground_level
        ),
        "path_goal_not_ground_count": sum(
            1 for plan in plans if not plan.path or plan.path[-1][2] != ground_level
        ),
    }


def _validate_grounded_flight_plans(plans: list[FlightPlan], ground_level: int = 0) -> None:
    counts = _ground_validation_counts(plans, ground_level)
    if counts["start_goal_above_ground_count"] != 0:
        raise ValueError("Invalid flight generation: some start/goal points are above ground.")
    if counts["path_start_not_ground_count"] != 0 or counts["path_goal_not_ground_count"] != 0:
        raise ValueError("Invalid flight generation: some paths do not start and end at ground level.")
    for plan in plans:
        assert plan.start[2] == ground_level
        assert plan.goal[2] == ground_level
        assert plan.path[0][2] == ground_level
        assert plan.path[-1][2] == ground_level


def _plot_od_points(grid: AirspaceGrid, plans: list[FlightPlan], hotspots: np.ndarray, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    starts = np.array([[p.start[0] + 0.5, p.start[1] + 0.5] for p in plans], dtype=float)
    goals = np.array([[p.goal[0] + 0.5, p.goal[1] + 0.5] for p in plans], dtype=float)
    if len(starts):
        ax.scatter(starts[:, 0], starts[:, 1], c="#2f6fdd", s=28, alpha=0.78, label="start")
        ax.scatter(goals[:, 0], goals[:, 1], c="#d63f3f", s=28, alpha=0.78, label="goal")
    if len(hotspots):
        ax.scatter(hotspots[:, 0], hotspots[:, 1], c="#111111", marker="*", s=150, label="OD hotspot")
    ax.set_xlim(0, grid.shape[0])
    ax.set_ylim(0, grid.shape[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x cell")
    ax.set_ylabel("y cell")
    ax.set_title("Heterogeneous OD points and hotspots")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_takeoff_time_hist(plans: list[FlightPlan], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist([p.etd for p in plans], bins=24, color="#4c78a8", alpha=0.86, edgecolor="white")
    ax.set_xlabel("takeoff time (s)")
    ax.set_ylabel("flight count")
    ax.set_title("Takeoff-time mixture peaks")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_route_density_heatmap(grid: AirspaceGrid, plans: list[FlightPlan], path: str | Path) -> None:
    density = _route_density_xy(plans, grid)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(density.T, origin="lower", cmap="magma", extent=[0, grid.shape[0], 0, grid.shape[1]], aspect="equal")
    ax.set_xlabel("x cell")
    ax.set_ylabel("y cell")
    ax.set_title("2D route-density heatmap")
    fig.colorbar(im, ax=ax, shrink=0.82, label="route count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_route_length_hist(grid: AirspaceGrid, plans: list[FlightPlan], path: str | Path) -> None:
    lengths = [path_distance_m(plan.path, grid.cell_size) for plan in plans]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(lengths, bins=22, color="#54a24b", alpha=0.86, edgecolor="white")
    ax.set_xlabel("route length (m)")
    ax.set_ylabel("flight count")
    ax.set_title("Route-length distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _altitude_level_counts(plans: list[FlightPlan], grid: AirspaceGrid) -> np.ndarray:
    counts = np.zeros(grid.shape[2], dtype=float)
    for plan in plans:
        for _, _, z in plan.path:
            if 0 <= z < grid.shape[2]:
                counts[z] += 1.0
    return counts


def _plot_altitude_level_hist(grid: AirspaceGrid, plans: list[FlightPlan], path: str | Path) -> None:
    counts = _altitude_level_counts(plans, grid)
    levels = np.arange(grid.shape[2])
    heights = (levels + 0.5) * float(grid.cell_size[2])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(levels, counts, color=["#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3"][: len(levels)])
    ax.set_xticks(levels)
    ax.set_xticklabels([f"L{level}\n{height:.0f}m" for level, height in zip(levels, heights)])
    ax.set_xlabel("altitude level")
    ax.set_ylabel("path-point count")
    ax.set_title("Altitude-level distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_route_altitude_profile_examples(grid: AirspaceGrid, plans: list[FlightPlan], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if plans:
        rng = np.random.default_rng(3107)
        sample_size = min(10, len(plans))
        sample_idx = rng.choice(np.arange(len(plans)), size=sample_size, replace=False)
        for idx in sample_idx:
            plan = plans[int(idx)]
            heights = [(z + 0.5) * grid.cell_size[2] for _, _, z in plan.path]
            ax.plot(range(len(heights)), heights, linewidth=1.4, alpha=0.82, label=f"F{plan.id}")
    ax.set_xlabel("path-point index")
    ax.set_ylabel("altitude (m)")
    ax.set_title("Route altitude-profile examples")
    ax.grid(True, alpha=0.24)
    if plans:
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_generation_diagnostics(
    output_dir: Path,
    grid: AirspaceGrid,
    plans: list[FlightPlan],
    hotspots: np.ndarray,
    warnings: list[str],
) -> None:
    _plot_od_points(grid, plans, hotspots, output_dir / "od_points.png")
    _plot_takeoff_time_hist(plans, output_dir / "takeoff_time_hist.png")
    _plot_route_density_heatmap(grid, plans, output_dir / "route_density_heatmap.png")
    _plot_route_length_hist(grid, plans, output_dir / "route_length_hist.png")
    _plot_altitude_level_hist(grid, plans, output_dir / "altitude_level_hist.png")
    _plot_route_altitude_profile_examples(grid, plans, output_dir / "route_altitude_profile_examples.png")
    with (output_dir / "flight_generation_hotspots.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["hotspot_id", "x_cell", "y_cell"])
        writer.writeheader()
        for idx, (x, y) in enumerate(hotspots):
            writer.writerow({"hotspot_id": idx, "x_cell": float(x), "y_cell": float(y)})
    warning_path = output_dir / "flight_generation_warnings.log"
    warning_path.write_text(("\n".join(warnings) + "\n") if warnings else "", encoding="utf-8")


def build_flight_generation_report(
    plans: list[FlightPlan],
    grid: AirspaceGrid,
    initial_conflicts_without_uncertainty: int,
    initial_conflicts_with_uncertainty: int,
    conflicts: list[object] | None = None,
) -> dict[str, float]:
    if not plans:
        report = {
            "start_x_mean": 0.0,
            "start_y_mean": 0.0,
            "goal_x_mean": 0.0,
            "goal_y_mean": 0.0,
            "route_length_mean": 0.0,
            "route_length_std": 0.0,
            "route_length_min": 0.0,
            "route_length_max": 0.0,
            "takeoff_time_mean": 0.0,
            "takeoff_time_std": 0.0,
            "speed_mean": 0.0,
            "speed_std": 0.0,
            "route_density_gini": 0.0,
            "center_crossing_ratio_actual": 0.0,
            "center_area_route_ratio": 0.0,
            "center_area_conflict_ratio": 0.0,
            "ground_hugging_route_ratio": 0.0,
            "start_goal_above_ground_count": 0.0,
            "path_start_not_ground_count": 0.0,
            "path_goal_not_ground_count": 0.0,
            "initial_conflicts_without_uncertainty": float(initial_conflicts_without_uncertainty),
            "initial_conflicts_with_uncertainty": float(initial_conflicts_with_uncertainty),
        }
        for level in range(grid.shape[2]):
            report[f"altitude_level_{level}_ratio"] = 0.0
            report[f"cruise_level_{level}_count"] = 0.0
        return report
    starts = np.asarray([[p.start[0], p.start[1]] for p in plans], dtype=float)
    goals = np.asarray([[p.goal[0], p.goal[1]] for p in plans], dtype=float)
    lengths = np.asarray([path_distance_m(p.path, grid.cell_size) for p in plans], dtype=float)
    takeoffs = np.asarray([p.etd for p in plans], dtype=float)
    speeds = np.asarray([float(np.mean(p.speed_profile or [0.0])) for p in plans], dtype=float)
    lo_x, hi_x = int(round(grid.shape[0] * 25.0 / 60.0)), int(round(grid.shape[0] * 35.0 / 60.0))
    lo_y, hi_y = int(round(grid.shape[1] * 25.0 / 60.0)), int(round(grid.shape[1] * 35.0 / 60.0))
    center_lo_x, center_hi_x = int(round(grid.shape[0] * 20.0 / 60.0)), int(round(grid.shape[0] * 40.0 / 60.0))
    center_lo_y, center_hi_y = int(round(grid.shape[1] * 20.0 / 60.0)), int(round(grid.shape[1] * 40.0 / 60.0))
    center_crossing = [
        any(lo_x <= x <= hi_x and lo_y <= y <= hi_y for x, y, _ in plan.path)
        for plan in plans
    ]
    all_points = [cell for plan in plans for cell in plan.path]
    altitude_counts = _altitude_level_counts(plans, grid)
    altitude_total = float(altitude_counts.sum())
    route_center_count = sum(1 for x, y, _ in all_points if center_lo_x <= x <= center_hi_x and center_lo_y <= y <= center_hi_y)
    ground_hugging = [
        float(np.mean([z == 0 for _, _, z in plan.path])) > 0.50
        for plan in plans
    ]
    ground_counts = _ground_validation_counts(plans, ground_level=0)
    cruise_counts = np.zeros(grid.shape[2], dtype=float)
    for plan in plans:
        if plan.cruise_level is not None and 0 <= int(plan.cruise_level) < grid.shape[2]:
            cruise_counts[int(plan.cruise_level)] += 1.0
    conflict_cells = [getattr(conflict, "cell", None) for conflict in (conflicts or [])]
    conflict_cells = [cell for cell in conflict_cells if cell is not None]
    conflict_center_count = sum(1 for x, y, _ in conflict_cells if center_lo_x <= x <= center_hi_x and center_lo_y <= y <= center_hi_y)
    report = {
        "start_x_mean": float(starts[:, 0].mean()),
        "start_y_mean": float(starts[:, 1].mean()),
        "goal_x_mean": float(goals[:, 0].mean()),
        "goal_y_mean": float(goals[:, 1].mean()),
        "route_length_mean": float(lengths.mean()),
        "route_length_std": float(lengths.std(ddof=0)),
        "route_length_min": float(lengths.min()),
        "route_length_max": float(lengths.max()),
        "takeoff_time_mean": float(takeoffs.mean()),
        "takeoff_time_std": float(takeoffs.std(ddof=0)),
        "speed_mean": float(speeds.mean()),
        "speed_std": float(speeds.std(ddof=0)),
        "route_density_gini": float(route_density_gini(plans, grid)),
        "center_crossing_ratio_actual": float(np.mean(center_crossing)),
        "center_area_route_ratio": float(route_center_count / max(1, len(all_points))),
        "center_area_conflict_ratio": float(conflict_center_count / max(1, len(conflict_cells))),
        "ground_hugging_route_ratio": float(np.mean(ground_hugging)),
        "start_goal_above_ground_count": float(ground_counts["start_goal_above_ground_count"]),
        "path_start_not_ground_count": float(ground_counts["path_start_not_ground_count"]),
        "path_goal_not_ground_count": float(ground_counts["path_goal_not_ground_count"]),
        "initial_conflicts_without_uncertainty": float(initial_conflicts_without_uncertainty),
        "initial_conflicts_with_uncertainty": float(initial_conflicts_with_uncertainty),
    }
    for level in range(grid.shape[2]):
        report[f"altitude_level_{level}_ratio"] = float(altitude_counts[level] / max(1.0, altitude_total))
        report[f"cruise_level_{level}_count"] = float(cruise_counts[level])
    if report["start_goal_above_ground_count"] != 0:
        raise ValueError("Invalid flight generation: some start/goal points are above ground.")
    return report


def write_flight_generation_report(
    output_dir: str | Path,
    plans: list[FlightPlan],
    grid: AirspaceGrid,
    initial_conflicts_without_uncertainty: int,
    initial_conflicts_with_uncertainty: int,
    conflicts: list[object] | None = None,
) -> dict[str, float]:
    out = ensure_dir(output_dir)
    report = build_flight_generation_report(plans, grid, initial_conflicts_without_uncertainty, initial_conflicts_with_uncertainty, conflicts=conflicts)
    with (out / "flight_generation_report.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(report.keys()))
        writer.writeheader()
        writer.writerow(report)
    if report["route_density_gini"] < 0.25:
        warning = "Warning: generated routes are too spatially balanced; consider increasing center_bias_strength or hotspot_std_cells."
        print(warning)
        logging.warning(warning)
        with (out / "flight_generation_warnings.log").open("a", encoding="utf-8") as fh:
            fh.write(warning + "\n")
    warnings = []
    if report.get("ground_hugging_route_ratio", 0.0) > 0.20:
        warnings.append("Warning: too many routes are ground-hugging; increase ground_hugging_penalty_weight or cruise_level_probs.")
    if report.get("center_area_route_ratio", 0.0) > 0.45:
        warnings.append("Warning: routes are over-concentrated in the center; reduce center_bias_strength or shared_corridor_ratio.")
    if report.get("altitude_level_0_ratio", 0.0) > 0.35:
        warnings.append("Warning: too many path points are in the lowest altitude layer.")
    if warnings:
        with (out / "flight_generation_warnings.log").open("a", encoding="utf-8") as fh:
            for warning in warnings:
                print(warning)
                logging.warning(warning)
                fh.write(warning + "\n")
    return report


def generate_heterogeneous_flight_tasks(
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    cfg: dict,
    output_dir: str | Path | None = None,
    n_flights: int | None = None,
    seed: int | None = None,
) -> list[FlightPlan]:
    fg_cfg = cfg.get("flight_generation", {})
    flight_cfg = cfg["flight"]
    n = int(n_flights if n_flights is not None else fg_cfg.get("n_flights", flight_cfg["n_flights"]))
    seed = int(seed if seed is not None else flight_cfg["random_seed"])
    rng = np.random.default_rng(seed + 101)
    target_m = float(fg_cfg.get("distance_target_m", flight_cfg.get("route_distance_target", 6000)))
    tolerance_m = float(fg_cfg.get("distance_tolerance_m", 1500))
    relaxed_tolerance_m = max(1800.0, tolerance_m)
    max_attempts = max(200, int(fg_cfg.get("max_sampling_attempts", 30000)))
    od_cfg = fg_cfg.get("od_pattern", {})
    mode_names = np.asarray(["hub_to_hub", "edge_to_edge", "center_crossing", "pure_random"], dtype=object)
    mode_probs = _normalized_probabilities(
        [
            float(od_cfg.get("hub_to_hub_ratio", 0.45)),
            float(od_cfg.get("edge_to_edge_ratio", 0.25)),
            float(od_cfg.get("center_crossing_ratio", 0.15)),
            float(od_cfg.get("pure_random_ratio", 0.15)),
        ],
        len(mode_names),
    )
    overlap_cfg = fg_cfg.get("route_overlap", {})
    corridor_ratio = float(overlap_cfg.get("shared_corridor_ratio", 0.12)) if overlap_cfg.get("encourage_overlap", True) else 0.0
    hotspots = _build_heterogeneous_hotspots(grid, od_cfg, rng)
    warnings: list[str] = []

    plans: list[FlightPlan] = []
    for i in safe_tqdm(range(n), desc="planning flights", leave=False):
        mode = str(rng.choice(mode_names, p=mode_probs))
        use_corridor = bool(rng.random() < corridor_ratio)
        path: list[GridPoint] = []
        waypoint: GridPoint | None = None
        start: GridPoint | None = None
        goal: GridPoint | None = None
        altitude_profile = _clip_altitude_profile(sample_altitude_profile(rng, cfg), grid, cfg)
        relaxed_logged = False
        relax_after = max(1, int(max_attempts * 0.65))
        for attempt in range(max_attempts):
            active_tolerance = tolerance_m if attempt < relax_after else relaxed_tolerance_m
            if attempt == relax_after and not relaxed_logged:
                message = (
                    f"Flight {i}: OD sampling did not satisfy {target_m:.0f}m +/- {tolerance_m:.0f}m "
                    f"within {relax_after} attempts; relaxing tolerance to +/- {relaxed_tolerance_m:.0f}m."
                )
                logging.warning(message)
                warnings.append(message)
                relaxed_logged = True
            pair = _sample_od_pair(mode, grid, hotspots, fg_cfg, rng, target_m, active_tolerance, altitude_profile)
            if pair is None:
                continue
            candidate_start, candidate_goal = pair
            if candidate_start == candidate_goal:
                continue
            if abs(_cell_distance_m(candidate_start, candidate_goal, grid) - target_m) > active_tolerance:
                continue
            candidate_path, candidate_waypoint = _plan_route_with_optional_waypoint(
                grid,
                risk_map,
                cfg,
                fg_cfg,
                candidate_start,
                candidate_goal,
                altitude_profile,
                use_corridor,
                rng,
            )
            if candidate_path and (
                not fg_cfg.get("altitude", {}).get("force_cruise_altitude", True)
                or _cruise_altitude_fraction(candidate_path, altitude_profile, fg_cfg) >= float(fg_cfg.get("altitude", {}).get("min_cruise_fraction", 0.65))
            ):
                start, goal = candidate_start, candidate_goal
                path, waypoint = candidate_path, candidate_waypoint
                break
        if not path or start is None or goal is None:
            raise RuntimeError(f"Could not generate heterogeneous A* path for flight {i} in mode {mode}")
        speed = _sample_initial_speed(rng, fg_cfg)
        etd = _sample_takeoff_time(rng, fg_cfg)
        eta = compute_eta_times(path, etd, speed, grid.cell_size)
        risk_sum = float(sum(risk_map[p] for p in path))
        plans.append(
            FlightPlan(
                id=i,
                start=start,
                goal=goal,
                path=path,
                etd=etd,
                eta_times=eta,
                speed_profile=[speed for _ in range(max(0, len(path) - 1))],
                risk_sum=risk_sum,
                total_air_time=max(0.0, eta[-1] - etd),
                generation_mode=mode,
                waypoint=waypoint,
                start_ground=start,
                goal_ground=goal,
                start_level=altitude_profile["start_level"],
                goal_level=altitude_profile["goal_level"],
                cruise_level=altitude_profile["cruise_level"],
                altitude_profile=dict(altitude_profile),
            )
        )

    ground_level = int(fg_cfg.get("altitude", {}).get("ground_level", 0))
    _validate_grounded_flight_plans(plans, ground_level=ground_level)
    if output_dir is not None:
        out = ensure_dir(output_dir)
        with (out / "initial_plans.pkl").open("wb") as fh:
            pickle.dump(plans, fh)
        plot_routes_3d(grid, plans, out / "initial_routes.png", title="Initial 4D flight routes")
        _write_generation_diagnostics(out, grid, plans, hotspots, warnings)
    return plans


def generate_initial_flight_plans(seed: int = 2025, n_flights: int | None = None, output_dir: str | Path | None = None) -> list[FlightPlan]:
    from .config import load_config
    from .risk_map import generate_risk_map

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.yaml")
    cfg["flight"]["random_seed"] = int(seed)
    if n_flights is not None:
        cfg["flight"]["n_flights"] = int(n_flights)
        cfg.setdefault("flight_generation", {})["n_flights"] = int(n_flights)
    grid = AirspaceGrid.from_config(cfg, seed=seed)
    risk_map = generate_risk_map(grid, cfg, output_dir=None)
    return generate_flight_plans(grid, risk_map, cfg, output_dir=output_dir, n_flights=n_flights, seed=seed)


@dataclass(frozen=True)
class PaperTrafficTask:
    id: int
    start: GridPoint
    goal: GridPoint
    etd: float
    speed: float


def sample_paper_random_traffic(
    grid: AirspaceGrid,
    cfg: dict,
    n_flights: int | None = None,
    seed: int | None = None,
) -> list[PaperTrafficTask]:
    """Freeze OD/ETD before A*: risk and distance scale never consume traffic RNG."""

    scene = cfg.get("paper_scene", {})
    fg = cfg["flight_generation"]
    n = int(n_flights if n_flights is not None else fg.get("n_flights", 100))
    traffic_seed = int(seed if seed is not None else cfg.get("traffic_seed", cfg["flight"]["random_seed"]))
    rng = np.random.default_rng(traffic_seed + 101)
    target = float(fg.get("distance_target_m", 6000.0))
    tolerance = float(fg.get("distance_tolerance_m", 1200.0))
    takeoff = scene.get("takeoff", {"min": 0.0, "max": 1800.0})
    speed = float(scene.get("initial_speed", {}).get("value", 10.0))
    tasks: list[PaperTrafficTask] = []
    for plan_id in range(n):
        start: GridPoint = (0, 0, 0)
        goal: GridPoint = (0, 0, 0)
        for _ in range(int(fg.get("max_sampling_attempts", 30000))):
            start = (int(rng.integers(grid.shape[0])), int(rng.integers(grid.shape[1])), 0)
            goal = (int(rng.integers(grid.shape[0])), int(rng.integers(grid.shape[1])), 0)
            if grid.is_free(start) and grid.is_free(goal) and abs(_cell_distance_m(start, goal, grid) - target) <= tolerance:
                break
        else:
            raise RuntimeError(f"Paper random OD sampling exhausted for flight {plan_id}; tolerance unchanged")
        tasks.append(
            PaperTrafficTask(
                plan_id,
                start,
                goal,
                float(rng.uniform(float(takeoff["min"]), float(takeoff["max"]))),
                speed,
            )
        )
    return tasks


def plan_paper_random_traffic(grid: AirspaceGrid, risk_map: np.ndarray, cfg: dict, tasks: list[PaperTrafficTask]) -> list[FlightPlan]:
    """Replan a fixed task table; a failed path aborts instead of resampling OD."""

    weights = cfg.get("paper_scene", {}).get("astar", {})
    unit = astar_distance_unit_m(str(cfg.get("astar_distance_scale_mode", "meter")), grid.cell_size[0])
    plans: list[FlightPlan] = []
    for task in tasks:
        path = astar_path(
            grid,
            task.start,
            task.goal,
            risk_map,
            alpha_r=float(weights.get("risk_weight", 0.8)),
            alpha_l=float(weights.get("distance_weight", 0.2)),
            distance_unit_m=unit,
        )
        if not path:
            raise RuntimeError(f"Fixed OD has no A* path for flight {task.id}; traffic and tolerance unchanged")
        eta = compute_eta_times(path, task.etd, task.speed, grid.cell_size)
        plans.append(
            FlightPlan(
                id=task.id,
                start=task.start,
                goal=task.goal,
                path=path,
                etd=task.etd,
                eta_times=eta,
                speed_profile=[task.speed] * max(0, len(path) - 1),
                risk_sum=float(sum(risk_map[p] for p in path)),
                total_air_time=max(0.0, eta[-1] - task.etd),
                generation_mode="paper_random",
                start_ground=task.start,
                goal_ground=task.goal,
                start_level=0,
                goal_level=0,
                cruise_level=max(p[2] for p in path),
            )
        )
    return plans


def generate_paper_random_flight_tasks(
    grid: AirspaceGrid,
    risk_map: np.ndarray,
    cfg: dict,
    output_dir: str | Path | None = None,
    n_flights: int | None = None,
    seed: int | None = None,
) -> list[FlightPlan]:
    tasks = sample_paper_random_traffic(grid, cfg, n_flights=n_flights, seed=seed)
    plans = plan_paper_random_traffic(grid, risk_map, cfg, tasks)
    if output_dir is not None:
        out = ensure_dir(output_dir)
        with (out / "initial_plans.pkl").open("wb") as fh:
            pickle.dump(plans, fh)
        plot_routes_3d(grid, plans, out / "initial_routes.png", title="Initial 4D flight routes")
        _write_generation_diagnostics(out, grid, plans, np.empty((0, 2)), [])
    return plans


def generate_flight_plans(grid: AirspaceGrid, risk_map: np.ndarray, cfg: dict, output_dir: str | Path | None = None, n_flights: int | None = None, seed: int | None = None) -> list[FlightPlan]:
    if cfg.get("flight_generation", {}).get("mode") == "paper_random":
        return generate_paper_random_flight_tasks(grid, risk_map, cfg, output_dir, n_flights, seed)
    if cfg.get("flight_generation", {}).get("mode") == "heterogeneous_random":
        return generate_heterogeneous_flight_tasks(grid, risk_map, cfg, output_dir, n_flights=n_flights, seed=seed)

    flight_cfg = cfg["flight"]
    n = int(n_flights if n_flights is not None else flight_cfg["n_flights"])
    seed = int(seed if seed is not None else flight_cfg["random_seed"])
    rng = np.random.default_rng(seed + 101)
    speed = float(flight_cfg["default_speed"])
    target = float(flight_cfg["route_distance_target"])
    alpha_r = float(cfg["risk"]["alpha_r"])
    alpha_l = float(cfg["risk"]["alpha_L"])
    t0, t1 = [float(v) for v in flight_cfg["takeoff_time_window"]]
    pulse_count = max(1, int(flight_cfg.get("traffic_pulse_count", 1)))
    pulse_jitter = float(flight_cfg.get("traffic_pulse_jitter", 0.0))
    pulses = np.linspace(t0 + 80, t1 - 80, pulse_count)
    min_layer = int(flight_cfg.get("min_flight_layer", 0))
    forbidden_low_layers = {(x, y, z) for x in range(grid.shape[0]) for y in range(grid.shape[1]) for z in range(min_layer)}
    allowed_layers = list(range(min_layer, grid.shape[2]))
    weights = np.asarray(flight_cfg.get("altitude_preference_weights", [1.0] * len(allowed_layers)), dtype=float)
    if len(weights) != len(allowed_layers) or weights.sum() <= 0:
        weights = np.ones(len(allowed_layers), dtype=float)
    weights = weights / weights.sum()
    altitude_strength = float(flight_cfg.get("altitude_preference_strength", 0.0))
    route_bias_strength = float(flight_cfg.get("route_random_bias_strength", 0.0))

    plans: list[FlightPlan] = []
    for i in safe_tqdm(range(n), desc="planning flights", leave=False):
        path: list[GridPoint] = []
        start: GridPoint
        goal: GridPoint
        preferred_layer = int(rng.choice(allowed_layers, p=weights))
        route_bias = rng.random(grid.shape)
        for _ in range(80):
            start, goal = grid.sample_terminal_pair(target, z_preference=preferred_layer)
            path = astar_path(
                grid,
                start,
                goal,
                risk_map,
                alpha_r=alpha_r,
                alpha_l=alpha_l,
                forbidden=forbidden_low_layers,
                altitude_preference=preferred_layer,
                altitude_weight=altitude_strength,
                cost_bias_map=route_bias,
                cost_bias_weight=route_bias_strength,
            )
            if path:
                break
        if not path:
            raise RuntimeError(f"Could not find A* path for flight {i}")
        wave = pulses[i % len(pulses)]
        etd = float(np.clip(wave + rng.normal(0, pulse_jitter), t0, t1))
        eta = compute_eta_times(path, etd, speed, grid.cell_size)
        risk_sum = float(sum(risk_map[p] for p in path))
        plans.append(
            FlightPlan(
                id=i,
                start=start,
                goal=goal,
                path=path,
                etd=etd,
                eta_times=eta,
                speed_profile=[speed for _ in range(max(0, len(path) - 1))],
                risk_sum=risk_sum,
                total_air_time=max(0.0, eta[-1] - etd),
            )
        )

    if output_dir is not None:
        out = ensure_dir(output_dir)
        with (out / "initial_plans.pkl").open("wb") as fh:
            pickle.dump(plans, fh)
        plot_routes_3d(grid, plans, out / "initial_routes.png", title="Initial 4D flight routes")
    return plans


def plot_routes_3d(grid: AirspaceGrid, plans: list[FlightPlan], path: str | Path, title: str = "Routes") -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(plans), 20)))
    for plan in plans:
        pts = (np.array(plan.path, dtype=float) + 0.5) * np.array(grid.cell_size, dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[plan.id % len(colors)], linewidth=0.8, alpha=0.72)
        if plan.id < 10:
            start_ground = np.array([(plan.start[0] + 0.5) * grid.cell_size[0], (plan.start[1] + 0.5) * grid.cell_size[1], 0.0])
            goal_ground = np.array([(plan.goal[0] + 0.5) * grid.cell_size[0], (plan.goal[1] + 0.5) * grid.cell_size[1], 0.0])
            ax.scatter([start_ground[0]], [start_ground[1]], [start_ground[2]], color=colors[plan.id % len(colors)], s=18, alpha=0.85)
            ax.scatter([goal_ground[0]], [goal_ground[1]], [goal_ground[2]], color="#111111", marker="x", s=20, alpha=0.85)
            ax.plot([start_ground[0], pts[0, 0]], [start_ground[1], pts[0, 1]], [start_ground[2], pts[0, 2]], color=colors[plan.id % len(colors)], linewidth=0.55, alpha=0.55)
            ax.plot([goal_ground[0], pts[-1, 0]], [goal_ground[1], pts[-1, 1]], [goal_ground[2], pts[-1, 2]], color="#111111", linewidth=0.55, alpha=0.45)
    ox, oy, oz = np.where(grid.obstacles)
    sample = np.arange(len(ox))
    if len(sample) > 1200:
        sample = np.random.default_rng(9).choice(sample, 1200, replace=False)
    obs = (np.column_stack([ox[sample], oy[sample], oz[sample]]).astype(float) + 0.5) * np.array(grid.cell_size, dtype=float)
    ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], c="#444444", s=8, alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("altitude (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
