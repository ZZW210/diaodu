from __future__ import annotations

import numpy as np

from src.astar_3d import astar_path
from src.grid import AirspaceGrid


def test_astar_avoids_obstacle_and_reaches_goal() -> None:
    obstacles = np.zeros((6, 6, 3), dtype=bool)
    obstacles[2, 1:5, 1] = True
    grid = AirspaceGrid(shape=(6, 6, 3), cell_size=(1, 1, 1), obstacles=obstacles)
    path = astar_path(grid, (0, 0, 1), (5, 5, 1), np.zeros(grid.shape))
    assert path[0] == (0, 0, 1)
    assert path[-1] == (5, 5, 1)
    assert all(not grid.obstacles[p] for p in path)
