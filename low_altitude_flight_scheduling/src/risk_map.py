from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .grid import AirspaceGrid
from .utils import ensure_dir, minmax_normalize


def generate_population_density(grid: AirspaceGrid, cfg: dict) -> np.ndarray:
    nx, ny, _ = grid.shape
    rng = np.random.default_rng(grid.seed + 17)
    base = float(cfg["risk"]["population_density_base"])
    x = np.linspace(0, 1, nx)[:, None]
    y = np.linspace(0, 1, ny)[None, :]
    density = np.full((nx, ny), base, dtype=float)
    for _ in range(5):
        cx, cy = rng.uniform(0.15, 0.85, size=2)
        amp = rng.uniform(0.5, 1.7) * base
        width = rng.uniform(0.045, 0.13)
        density += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * width**2))
    building_shadow = grid.obstacles.any(axis=2).astype(float)
    density *= 1.0 + 0.18 * building_shadow
    return density / 1_000_000.0


def generate_risk_map(grid: AirspaceGrid, cfg: dict, output_dir: str | Path | None = None) -> np.ndarray:
    risk_cfg = cfg["risk"]
    density = generate_population_density(grid, cfg)
    nx, ny, nz = grid.shape
    heights = (np.arange(nz) + 0.5) * grid.cell_size[2]
    risk = np.zeros(grid.shape, dtype=float)

    p_fail = float(risk_cfg["failure_probability"])
    r_uav = float(risk_cfg["r_UAV"])
    r_buf = float(risk_cfg["r_buf"])
    m = float(risk_cfg["m"])
    shield = float(risk_cfg["S"])
    cd_b = float(risk_cfg["C_D_b"])
    cd_p = float(risk_cfg["C_D_p"])
    area_parachute = float(risk_cfg["A_p"])
    wind = float(risk_cfg["v_wind"])
    g = 9.81
    air_density = 1.225
    lambda_ = 1e6
    mu = 100.0

    impact_area = math.pi * (r_uav + r_buf) ** 2
    for k, h in enumerate(heights):
        terminal = math.sqrt(max(1e-9, 2 * m * g / (air_density * cd_b)))
        v_bal = terminal * math.sqrt(max(0.0, 1.0 - math.exp(-h / max(1.0, terminal))))
        e_bal = 0.5 * m * v_bal**2
        sev_bal = 1.0 / (1.0 + math.sqrt(lambda_ / mu) * (mu / max(e_bal, 1e-6)) ** (1.0 / (4.0 * shield)))

        t_par = math.sqrt(2 * m * max(h, 1.0) / (max(area_parachute * cd_p, 1e-6) * g))
        drift_scale = 1.0 + min(0.8, wind * t_par / 6000.0)
        e_par = 0.5 * m * (max(2.0, h / max(t_par, 1e-6))) ** 2
        sev_par = 1.0 / (1.0 + math.sqrt(lambda_ / mu) * (mu / max(e_par, 1e-6)) ** (1.0 / (4.0 * shield)))

        p_impact_bal = impact_area * density
        p_impact_par = impact_area * drift_scale * density
        risk[:, :, k] = p_fail * (p_impact_bal * sev_bal + p_impact_par * sev_par)

    risk = minmax_normalize(risk)
    risk[grid.obstacles] = 1.0

    if output_dir is not None:
        out = ensure_dir(output_dir)
        np.save(out / "risk_map.npy", risk)
        plot_risk_map(grid, risk, out / "risk_map_3d.png")
    return risk


def plot_risk_map(grid: AirspaceGrid, risk: np.ndarray, path: str | Path) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    xs, ys, zs = np.where(~grid.obstacles)
    sample = np.arange(len(xs))
    if len(sample) > 3500:
        rng = np.random.default_rng(42)
        sample = rng.choice(sample, size=3500, replace=False)
    sc = ax.scatter(xs[sample], ys[sample], zs[sample], c=risk[xs[sample], ys[sample], zs[sample]], cmap="viridis", s=8, alpha=0.65)
    ox, oy, oz = np.where(grid.obstacles)
    ax.scatter(ox, oy, oz, c="#333333", s=9, alpha=0.35, label="obstacle")
    ax.set_title("3D normalized ground-risk map")
    ax.set_xlabel("x grid")
    ax.set_ylabel("y grid")
    ax.set_zlabel("z grid")
    fig.colorbar(sc, ax=ax, shrink=0.65, label="normalized risk")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
