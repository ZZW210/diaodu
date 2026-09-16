from __future__ import annotations

import numpy as np

from src.conflict_detection import detect_conflicts
from src.flight_plan import FlightPlan
from src.grid import AirspaceGrid
from src.stage1_continuous import (
    InfeasibleRoute,
    build_stage1_layout,
    decode_stage1_solution,
    evaluate_stage1_solution,
)


def _cfg() -> dict:
    return {
        "flight": {"default_speed": 10.0, "min_flight_layer": 0, "random_seed": 2025},
        "conflict": {
            "t_conflict": 30.0,
            "cell_occupancy_time": 0.0,
            "alpha": 0.05,
            "sigma0": 0.0,
            "sigma_rate": 0.0,
        },
        "optimization": {
            "speed_range": [5.0, 20.0],
            "t_ATD_range": [1.0, 3600.0],
            "stage1_atd_range": [1.0, 3600.0],
            "t_delay_max": 1800.0,
            "t_battery": 1200.0,
            "stage1_respect_delay_max": True,
            "stage1_max_advance_seconds": 600.0,
            "delay_count_threshold": 30.0,
            "delay_count_cap": 3,
        },
        "paper_encoding": {
            "local_window_segments": 2,
            "reroute_merge_window": 1,
            "local_margin_xy": 1,
            "local_margin_z": 0,
            "astar_cache_size": 256,
            "astar_max_expansions": 2000,
        },
        "risk": {"alpha_r": 0.8, "alpha_L": 0.2},
    }


def _plan(fid: int, etd: float, path: list[tuple[int, int, int]]) -> FlightPlan:
    eta = [etd + 10.0 * i for i in range(len(path))]
    return FlightPlan(fid, path[0], path[-1], list(path), etd, eta, [10.0] * (len(path) - 1), 0.0, eta[-1] - etd)


def _plans() -> list[FlightPlan]:
    straight = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
    crossing = [(0, 2, 0), (1, 1, 0), (2, 0, 0), (3, 0, 0)]
    return [_plan(0, 100.0, straight), _plan(1, 100.0, crossing)]


def _grid() -> AirspaceGrid:
    return AirspaceGrid(
        shape=(5, 5, 2),
        cell_size=(100, 100, 30),
        obstacle_ratio=0.0,
        seed=1,
        obstacles=np.zeros((5, 5, 2), dtype=bool),
    )


def test_layout_and_fitness_are_finite() -> None:
    cfg, grid = _cfg(), _grid()
    plans = _plans()
    conflicts = detect_conflicts(plans, cfg, uncertain=True)
    layout = build_stage1_layout(plans, [0, 1], conflicts, cfg, grid)

    assert len(layout.blocks) == 2
    expected_min = 2 * (1 + 1 + (len(plans[0].path) - 1))  # strategy + ATD + per-segment speeds
    assert layout.dim >= expected_min
    assert layout.lower.shape == layout.upper.shape == (layout.dim,)
    assert np.all(layout.upper >= layout.lower)

    value = evaluate_stage1_solution(layout.lower, plans, layout, cfg, grid, np.zeros(grid.shape, dtype=float))
    assert np.isfinite(value.fitness)
    # Decoding must never mutate the input plans.
    assert plans[0].etd == 100.0 and plans[1].etd == 100.0


def test_strategy_gene_selects_schedule_speed_or_reroute() -> None:
    cfg, grid = _cfg(), _grid()
    plans = _plans()
    conflicts = detect_conflicts(plans, cfg, uncertain=True)
    layout = build_stage1_layout(plans, [0, 1], conflicts, cfg, grid)
    risk = np.zeros(grid.shape, dtype=float)
    block = layout.blocks[0]

    scheduled = layout.lower.copy()
    scheduled[block.strategy_gene] = 0.5
    scheduled[block.atd_gene] = 160.0
    decoded = decode_stage1_solution(scheduled, plans, layout, cfg, grid, risk)
    shifted = next(p for p in decoded if p.id == block.flight_id)
    assert abs(shifted.etd - 160.0) < 1e-9
    assert abs(shifted.delay - 60.0) < 1e-9

    speed_vector = layout.lower.copy()
    speed_vector[block.strategy_gene] = 1.5
    speed_vector[block.speed_genes.start] = 12.0
    decoded = decode_stage1_solution(speed_vector, plans, layout, cfg, grid, risk)
    adjusted = next(p for p in decoded if p.id == block.flight_id)
    assert abs(adjusted.speed_profile[block.segments[0]] - 12.0) < 1e-9
    assert abs(adjusted.etd - 100.0) < 1e-9

    rerouted_vector = layout.lower.copy()
    rerouted_vector[block.strategy_gene] = 2.5
    decoded = decode_stage1_solution(rerouted_vector, plans, layout, cfg, grid, risk)
    detoured = next(p for p in decoded if p.id == block.flight_id)
    assert detoured.path[0] == detoured.start and detoured.path[-1] == detoured.goal
    assert all(grid.is_free(c) for c in detoured.path)


def test_obstacle_via_is_rejected() -> None:
    cfg, grid = _cfg(), _grid()
    plans = _plans()
    conflicts = detect_conflicts(plans, cfg, uncertain=True)
    layout = build_stage1_layout(plans, [0, 1], conflicts, cfg, grid)
    risk = np.zeros(grid.shape, dtype=float)
    block = layout.blocks[0]
    assert block.via_genes, "Key flights must own at least one reroute interval"

    blocked = layout.lower.copy()
    blocked[block.strategy_gene] = 2.5
    genes = block.via_genes[0]
    blocked[genes.start : genes.stop] = [3.0, 4.0, 0.0]
    grid.obstacles[3, 4, 0] = True
    try:
        decode_stage1_solution(blocked, plans, layout, cfg, grid, risk)
    except InfeasibleRoute:
        pass
    else:  # pragma: no cover - guard against silently ignoring blocked vias
        raise AssertionError("A via cell inside an obstacle must be rejected")
