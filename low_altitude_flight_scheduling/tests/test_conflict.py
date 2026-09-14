from __future__ import annotations

from src.conflict_detection import count_conflict_pairs, detect_conflicts
from src.flight_plan import FlightPlan, apply_speed_factor, shift_plan_time, update_plan_timing


def _plan(fid: int, etd: float) -> FlightPlan:
    path = [(0, 0, 1), (1, 1, 1), (2, 2, 1)]
    eta = [etd, etd + 10, etd + 20]
    return FlightPlan(fid, path[0], path[-1], path, etd, eta, [10, 10, 10], 0.0, 20.0)


def test_conflict_detection_same_cell_near_time() -> None:
    cfg = {"conflict": {"t_conflict": 30.0, "alpha": 0.05, "sigma0": 1.0, "sigma_rate": 0.0}}
    conflicts = detect_conflicts([_plan(0, 0), _plan(1, 5)], cfg, uncertain=False)
    assert len(conflicts) == 3
    assert count_conflict_pairs(conflicts) == 1
    assert conflicts[0].cell in {(0, 0, 1), (1, 1, 1), (2, 2, 1)}


def test_speed_adjustment_recomputes_eta() -> None:
    plan = _plan(0, 0)
    baseline = update_plan_timing(plan, speed=10.0, cell_size=(100, 100, 30))
    faster = apply_speed_factor(plan, factor=2.0, speed_min=5.0, speed_max=20.0, cell_size=(100, 100, 30))
    assert faster.total_air_time < baseline.total_air_time * 0.6


def test_takeoff_shift_moves_all_eta_times() -> None:
    plan = _plan(0, 0)
    shifted = shift_plan_time(plan, 120.0)
    assert shifted.etd == 120.0
    assert shifted.eta_times == [t + 120.0 for t in plan.eta_times]
