from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

GridPoint = tuple[int, int, int]


@dataclass
class AirspaceGrid:
    shape: tuple[int, int, int] = (60, 60, 4)
    cell_size: tuple[int, int, int] = (100, 100, 30)
    obstacle_ratio: float = 0.10
    seed: int = 2025
    obstacles: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        if self.obstacles is None:
            self.obstacles = self._generate_city_obstacles()
        else:
            self.obstacles = np.asarray(self.obstacles, dtype=bool)

    def _generate_city_obstacles(self) -> np.ndarray:
        nx, ny, nz = self.shape
        obs = np.zeros(self.shape, dtype=bool)
        target_cells = int(nx * ny * nz * self.obstacle_ratio)
        occupied = 0
        attempts = 0
        while occupied < target_cells and attempts < nx * ny * 20:
            attempts += 1
            x = int(self.rng.integers(2, nx - 2))
            y = int(self.rng.integers(2, ny - 2))
            footprint = int(self.rng.choice([1, 1, 1, 2]))
            height = int(self.rng.choice([1, 2, 2, 3, 4]))
            for dx in range(footprint):
                for dy in range(footprint):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < nx and 0 <= yy < ny:
                        before = int(obs[xx, yy, :height].sum())
                        obs[xx, yy, :height] = True
                        occupied += int(obs[xx, yy, :height].sum()) - before
        return obs

    @classmethod
    def from_config(cls, cfg: dict, seed: int | None = None) -> "AirspaceGrid":
        air = cfg["airspace"]
        return cls(
            shape=tuple(int(v) for v in air["grid_shape"]),
            cell_size=tuple(int(v) for v in air["grid_cell_size"]),
            obstacle_ratio=float(air["obstacle_ratio"]),
            seed=int(seed if seed is not None else cfg["flight"]["random_seed"]),
        )

    def in_bounds(self, p: GridPoint) -> bool:
        return all(0 <= p[i] < self.shape[i] for i in range(3))

    def is_free(self, p: GridPoint, forbidden: set[GridPoint] | None = None) -> bool:
        if not self.in_bounds(p):
            return False
        if forbidden and p in forbidden:
            return False
        return not bool(self.obstacles[p])

    def neighbors_26(self, p: GridPoint, forbidden: set[GridPoint] | None = None) -> list[GridPoint]:
        out: list[GridPoint] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    q = (p[0] + dx, p[1] + dy, p[2] + dz)
                    if self.is_free(q, forbidden):
                        out.append(q)
        return out

    def world_to_grid(self, xyz: tuple[float, float, float]) -> GridPoint:
        idx = tuple(int(np.clip(np.floor(xyz[i] / self.cell_size[i]), 0, self.shape[i] - 1)) for i in range(3))
        return idx  # type: ignore[return-value]

    def grid_to_world(self, p: GridPoint) -> tuple[float, float, float]:
        return tuple((p[i] + 0.5) * self.cell_size[i] for i in range(3))  # type: ignore[return-value]

    def sample_free_cell(self, z_preference: int | None = None, border: str | None = None) -> GridPoint:
        nx, ny, nz = self.shape
        for _ in range(5000):
            z = int(np.clip(z_preference if z_preference is not None else self.rng.integers(1, nz), 0, nz - 1))
            if border == "west":
                p = (0, int(self.rng.integers(0, ny)), z)
            elif border == "east":
                p = (nx - 1, int(self.rng.integers(0, ny)), z)
            elif border == "south":
                p = (int(self.rng.integers(0, nx)), 0, z)
            elif border == "north":
                p = (int(self.rng.integers(0, nx)), ny - 1, z)
            else:
                p = (int(self.rng.integers(0, nx)), int(self.rng.integers(0, ny)), z)
            if self.is_free(p):
                return p
        raise RuntimeError("Could not sample a free grid cell")

    def _sample_cruise_layer(self, z_preference: int | None = None) -> int:
        nz = self.shape[2]
        if nz <= 1:
            return 0
        if z_preference is None:
            return int(self.rng.choice([z for z in (1, 2, 2, 3) if z < nz]))
        base = int(np.clip(z_preference, 1, nz - 1))
        candidates = [base, base + 1, base - 1]
        weights = [0.62, 0.22, 0.16]
        valid = [(z, w) for z, w in zip(candidates, weights) if 1 <= z < nz]
        layers = np.asarray([z for z, _ in valid], dtype=int)
        probs = np.asarray([w for _, w in valid], dtype=float)
        probs = probs / probs.sum()
        return int(self.rng.choice(layers, p=probs))

    def sample_terminal_pair(self, target_distance_m: float = 6000.0, z_preference: int | None = None) -> tuple[GridPoint, GridPoint]:
        pairs = [("west", "east"), ("east", "west"), ("south", "north"), ("north", "south")]
        border_a, border_b = pairs[int(self.rng.integers(0, len(pairs)))]
        z_a = self._sample_cruise_layer(z_preference)
        z_b = self._sample_cruise_layer(z_preference)
        best: tuple[GridPoint, GridPoint] | None = None
        best_err = float("inf")
        for _ in range(120):
            a = self.sample_free_cell(z_preference=z_a, border=border_a)
            b = self.sample_free_cell(z_preference=z_b, border=border_b)
            dist = float(np.linalg.norm((np.array(a) - np.array(b)) * np.array(self.cell_size)))
            err = abs(dist - target_distance_m)
            if err < best_err:
                best = (a, b)
                best_err = err
        assert best is not None
        return best

    def expand_forbidden(self, cells: Iterable[GridPoint], radius: int = 1) -> set[GridPoint]:
        forbidden: set[GridPoint] = set()
        for x, y, z in cells:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        p = (x + dx, y + dy, z + dz)
                        if self.in_bounds(p):
                            forbidden.add(p)
        return forbidden
