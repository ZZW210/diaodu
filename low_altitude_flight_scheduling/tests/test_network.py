from __future__ import annotations

from src.conflict_detection import Conflict
from src.conflict_network import build_conflict_network, collective_influence
from src.flight_plan import FlightPlan


def _fp(fid: int) -> FlightPlan:
    return FlightPlan(fid, (0, 0, 1), (1, 1, 1), [(0, 0, 1), (1, 1, 1)], 0.0, [0.0, 10.0], [10, 10], 0.0, 10.0)


def test_network_edges_and_ci() -> None:
    plans = [_fp(i) for i in range(5)]
    conflicts = [
        Conflict(0, 1, (0, 0, 1), 0, 0, 0, 0, 0, 30, "uncertain"),
        Conflict(0, 2, (0, 0, 1), 0, 0, 0, 0, 0, 30, "uncertain"),
        Conflict(0, 3, (0, 0, 1), 0, 0, 0, 0, 0, 30, "uncertain"),
        Conflict(3, 4, (0, 0, 1), 0, 0, 0, 0, 0, 30, "uncertain"),
    ]
    graph = build_conflict_network(plans, conflicts)
    assert graph.number_of_edges() == 4
    ci = collective_influence(graph, l=1)
    assert ci[0] >= ci[1]
