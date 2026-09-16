from __future__ import annotations

import heapq
from typing import Optional

import numpy as np

from .grid import AirspaceGrid, GridPoint
from .utils import euclidean


def astar_distance_unit_m(mode: str, horizontal_cell_size_m: float = 100.0) -> float:
    """Divisor applied consistently to step length and heuristic, not risk."""

    units = {"meter": 1.0, "grid": float(horizontal_cell_size_m), "kilometer": 1000.0}
    if mode not in units:
        raise ValueError(f"Unknown astar_distance_scale_mode: {mode}")
    return units[mode]


def astar_path(
    grid: AirspaceGrid,
    start: GridPoint,
    goal: GridPoint,
    risk_map: np.ndarray | None = None,
    alpha_r: float = 0.8,
    alpha_l: float = 0.2,
    forbidden: Optional[set[GridPoint]] = None,
    altitude_preference: int | None = None,
    altitude_weight: float = 0.0,
    cruise_level: int | None = None,
    altitude_preference_enabled: bool = False,
    cruise_altitude_penalty_weight: float = 0.0,
    ground_hugging_penalty_weight: float = 0.0,
    ground_level: int = 0,
    climb_descent_cells: int = 0,
    vertical_move_penalty: float = 1.0,
    risk_weight: float | None = None,
    distance_weight: float | None = None,
    cost_bias_map: np.ndarray | None = None,
    cost_bias_weight: float = 0.0,
    max_expansions: int = 80000,
    distance_unit_m: float = 100.0,
) -> list[GridPoint]:
    """Improved 3D A* with risk-length cost and distance-dependent heuristic weight."""
    if not grid.is_free(start, forbidden) or not grid.is_free(goal, forbidden):
        return []
    risk = risk_map if risk_map is not None else np.zeros(grid.shape, dtype=float)
    start_goal_dist = max(euclidean(start, goal, grid.cell_size), 1.0)
    risk_w = float(alpha_r if risk_weight is None else risk_weight)
    dist_w = float(alpha_l if distance_weight is None else distance_weight)
    preferred_level = cruise_level if cruise_level is not None else altitude_preference
    preference_enabled = bool(altitude_preference_enabled or altitude_preference is not None or cruise_level is not None)
    altitude_w = float(cruise_altitude_penalty_weight if cruise_altitude_penalty_weight > 0 else altitude_weight)
    terminal_radius = max(0, int(climb_descent_cells))

    open_heap: list[tuple[float, int, GridPoint]] = []
    heapq.heappush(open_heap, (0.0, 0, start))
    came_from: dict[GridPoint, GridPoint] = {}
    g_score: dict[GridPoint, float] = {start: 0.0}
    visited: set[GridPoint] = set()
    counter = 0

    while open_heap and len(visited) < max_expansions:
        _, _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        if current == goal:
            return _reconstruct(came_from, current)
        visited.add(current)
        for neighbor in grid.neighbors_26(current, forbidden):
            if neighbor in visited:
                continue
            move = euclidean(current, neighbor, grid.cell_size) / distance_unit_m
            if neighbor[2] != current[2]:
                move *= max(1.0, float(vertical_move_penalty))
            risk_cost = float(risk[neighbor])
            altitude_cost = 0.0
            if preference_enabled and preferred_level is not None:
                altitude_cost += altitude_w * abs(neighbor[2] - int(preferred_level))
            if neighbor[2] == ground_level and not _near_terminal(neighbor, start, goal, terminal_radius):
                altitude_cost += float(ground_hugging_penalty_weight)
            bias_cost = float(cost_bias_map[neighbor]) * cost_bias_weight if cost_bias_map is not None else 0.0
            step_cost = dist_w * move + risk_w * risk_cost + altitude_cost + bias_cost
            tentative = g_score[current] + step_cost
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                h = euclidean(neighbor, goal, grid.cell_size) / distance_unit_m
                w = 1.0 + euclidean(neighbor, goal, grid.cell_size) / start_goal_dist
                counter += 1
                heapq.heappush(open_heap, (tentative + w * h, counter, neighbor))
    return []


def _near_terminal(p: GridPoint, start: GridPoint, goal: GridPoint, radius: int) -> bool:
    if radius <= 0:
        return False
    start_d = abs(p[0] - start[0]) + abs(p[1] - start[1])
    goal_d = abs(p[0] - goal[0]) + abs(p[1] - goal[1])
    return start_d <= radius or goal_d <= radius


def _reconstruct(came_from: dict[GridPoint, GridPoint], current: GridPoint) -> list[GridPoint]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
