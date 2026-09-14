from __future__ import annotations

from src.flight_plan import generate_initial_flight_plans


def test_start_goal_are_ground() -> None:
    plans = generate_initial_flight_plans(seed=2025, n_flights=20)
    for plan in plans:
        assert plan.start[2] == 0
        assert plan.goal[2] == 0
        assert plan.path[0][2] == 0
        assert plan.path[-1][2] == 0


def test_cruise_level_can_be_above_ground() -> None:
    plans = generate_initial_flight_plans(seed=2025, n_flights=50)
    assert any(plan.cruise_level is not None and plan.cruise_level > 0 for plan in plans)
