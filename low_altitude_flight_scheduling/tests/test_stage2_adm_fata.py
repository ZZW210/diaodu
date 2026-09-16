from __future__ import annotations

import numpy as np

from src.conflict_detection import count_conflict_pairs, detect_conflicts
from src.flight_plan import FlightPlan
from src.grid import AirspaceGrid
from src.stage2_adm_fata import (
    adm_fata_stage2,
    build_stage2_layout,
    decode_stage2_solution,
    evaluate_stage2_solution,
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
            "stage2_fata_generations": 4,
            "stage2_fata_dominant_fraction": 0.25,
            "stage2_fata_learning_rate": 0.5,
        },
        "fata": {"NP": 8, "Ngen_max": 4, "Parf": 0.2},
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


def test_stage2_layout_uses_conflict_actors() -> None:
    cfg, grid = _cfg(), _grid()
    plans = _plans()
    conflicts = detect_conflicts(plans, cfg, uncertain=True)
    assert conflicts
    layout = build_stage2_layout(plans, conflicts, cfg, grid)

    assert len(layout.actor_genes) == len(conflicts)
    assert len(layout.conflict_refs) == len(conflicts)
    assert layout.dim > len(conflicts)
    assert np.all(layout.upper >= layout.lower)
    for block in layout.blocks:
        assert block.incident_conflicts
        for ci in block.incident_conflicts:
            assert ci in block.conflict_segments
            assert block.conflict_segments[ci]


def test_stage2_decode_applies_actor_strategies() -> None:
    cfg, grid = _cfg(), _grid()
    plans = _plans()
    conflicts = detect_conflicts(plans, cfg, uncertain=True)
    layout = build_stage2_layout(plans, conflicts, cfg, grid)
    risk = np.zeros(grid.shape, dtype=float)
    plan_a = layout.conflict_refs[0][0]
    block = next(b for b in layout.blocks if b.flight_id == plan_a)

    # Strategy 0 (delay): actor gene < 1 keeps plan_a as the actor.
    vector = layout.lower.copy()
    vector[layout.actor_genes[0]] = 0.25
    vector[block.atd_gene] = float(block.atd_bounds[0]) + 60.0
    strategies = np.zeros(len(conflicts), dtype=int)
    decoded = decode_stage2_solution(vector, plans, layout, strategies, cfg, grid, risk)
    shifted = next(p for p in decoded if p.id == plan_a)
    assert abs(shifted.etd - (block.atd_bounds[0] + 60.0)) < 1e-9
    assert abs(shifted.delay - (block.atd_bounds[0] + 60.0 - 100.0)) < 1e-9

    # Strategy 1 (speed): the local-window segment gets the gene value.
    vector = layout.lower.copy()
    vector[layout.actor_genes[0]] = 0.25
    segment = block.conflict_segments[0][0]
    vector[block.speed_genes[segment]] = 12.0
    strategies = np.ones(len(conflicts), dtype=int)
    decoded = decode_stage2_solution(vector, plans, layout, strategies, cfg, grid, risk)
    adjusted = next(p for p in decoded if p.id == plan_a)
    assert abs(adjusted.speed_profile[segment] - 12.0) < 1e-9
    assert abs(adjusted.etd - 100.0) < 1e-9

    # Strategy 2 (reroute): the actor gene >= 1 moves the action to plan_b.
    vector = layout.lower.copy()
    vector[layout.actor_genes[0]] = 1.5
    strategies = np.full(len(conflicts), 2, dtype=int)
    decoded = decode_stage2_solution(vector, plans, layout, strategies, cfg, grid, risk)
    detoured = next(p for p in decoded if p.id == layout.conflict_refs[0][1])
    assert detoured.path[0] == detoured.start and detoured.path[-1] == detoured.goal
    assert all(grid.is_free(c) for c in detoured.path)
    # Decoding never mutates the input plans.
    assert plans[0].etd == 100.0 and plans[1].etd == 100.0


def test_stage2_adm_fata_is_deterministic_and_improves() -> None:
    cfg, grid = _cfg(), _grid()
    risk = np.zeros(grid.shape, dtype=float)
    plans = _plans()
    conflicts = detect_conflicts(plans, cfg, uncertain=True)
    pairs_before = count_conflict_pairs(conflicts)

    first = adm_fata_stage2(plans, conflicts, cfg, grid, risk, seed=7)
    second = adm_fata_stage2(plans, conflicts, cfg, grid, risk, seed=7)

    assert first.pairs_after <= pairs_before
    assert first.pairs_after == second.pairs_after
    assert np.array_equal(first.best_strategies, second.best_strategies)
    assert len(first.convergence) == cfg["optimization"]["stage2_fata_generations"]
    assert first.probability.shape == (len(conflicts), 3)
    assert np.allclose(first.probability.sum(axis=1), 1.0)
    assert first.log and first.log[0]["stage"] == "stage2_adm_fata"

    remaining = detect_conflicts(first.plans, cfg, uncertain=True)
    assert count_conflict_pairs(remaining) == first.pairs_after


def test_stage2_evaluation_matches_decoded_conflicts() -> None:
    cfg, grid = _cfg(), _grid()
    plans = _plans()
    conflicts = detect_conflicts(plans, cfg, uncertain=True)
    layout = build_stage2_layout(plans, conflicts, cfg, grid)
    risk = np.zeros(grid.shape, dtype=float)
    strategies = np.zeros(len(conflicts), dtype=int)

    evaluation = evaluate_stage2_solution(
        layout.lower, plans, layout, strategies, cfg, grid, risk
    )
    assert np.isfinite(evaluation.fitness)
    assert evaluation.plans


def test_stage2_adm_fata_without_conflicts_is_a_noop() -> None:
    cfg, grid = _cfg(), _grid()
    plans = _plans()
    result = adm_fata_stage2(plans, [], cfg, grid, np.zeros(grid.shape, dtype=float), seed=7)
    assert result.generations == 0
    assert result.accepted is False
    assert result.plans == plans
