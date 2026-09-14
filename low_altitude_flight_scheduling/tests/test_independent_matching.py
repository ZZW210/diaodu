from __future__ import annotations

import numpy as np

from src.conflict_detection import count_conflict_pairs, detect_conflicts
from src.flight_plan import FlightPlan
from src.grid import AirspaceGrid
from src.optimization_model import independent_matching_deconfliction


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
            "t_delay_max": 1800.0,
            "t_battery": 1200.0,
            "delay_candidates": [-90.0, -60.0, 60.0, 90.0, 120.0],
            "speed_factors": [0.90, 0.95, 1.05, 1.10],
            "delay_count_threshold": 30.0,
            "delay_count_cap": 3,
            "max_changed_flight_ratio": 1.0,
            "safety_margin_seconds": 10.0,
            "independent_matching_rounds": 4,
            "independent_matching_lrate": 0.5,
            "independent_matching_segment_limit": 3,
            "independent_matching_candidates_per_strategy": 6,
            "independent_matching_first_pair_resolution": True,
            "independent_matching_first_strategy_success": True,
            "independent_matching_allow_reroute": False,
            "head_to_head_dot_threshold": -0.35,
            "stage2_cluster_points_threshold": 10,
            "stage2_cluster_neighbor_radius": 1,
            "independent_matching_strategy_priors": {
                "head_to_head": [1.0, 0.0, 0.0],
                "cluster": [1.0, 0.0, 0.0],
                "crossing": [1.0, 0.0, 0.0],
            },
        },
        "risk": {"alpha_r": 0.8, "alpha_L": 0.2},
    }


def _plan(fid: int, etd: float) -> FlightPlan:
    path = [(0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0)]
    eta = [etd, etd + 10.0, etd + 20.0, etd + 30.0]
    return FlightPlan(fid, path[0], path[-1], path, etd, eta, [10.0, 10.0, 10.0], 0.0, 30.0)


def test_independent_matching_does_not_increase_conflict_pairs() -> None:
    cfg = _cfg()
    grid = AirspaceGrid(shape=(5, 5, 2), cell_size=(100, 100, 30), obstacle_ratio=0.0, seed=1, obstacles=np.zeros((5, 5, 2), dtype=bool))
    risk = np.zeros(grid.shape, dtype=float)
    plans = [_plan(0, 100.0), _plan(1, 100.0)]
    conflicts = detect_conflicts(plans, cfg, uncertain=True)

    _, new_conflicts, logs = independent_matching_deconfliction(plans, conflicts, cfg, grid, risk, max_rounds=4, seed=7)

    assert count_conflict_pairs(new_conflicts) <= count_conflict_pairs(conflicts)
    assert any(row.get("stage") == "stage2_independent_matching" for row in logs)
    assert any(row.get("accepted") is True and row.get("matched_strategy") == "delay" for row in logs)
