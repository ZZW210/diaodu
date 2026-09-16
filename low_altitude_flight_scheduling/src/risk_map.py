from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from .grid import AirspaceGrid
from .utils import ensure_dir, minmax_normalize


RHO_STATIC_KM2 = 3500.0
DIFFUSION_RADIUS_M = 1000.0
POPULATION_ASSUMPTIONS = """## Population implementation assumptions

Paper scene population follows reference [28]: population centers are detected
from the simulated building footprint (10x10-cell windows, local maxima, 1000 m
minimum spacing, at most four centers). Only the nearest center contributes:
``rho_center * exp(1 - r_km^2)`` inside 1 km, otherwise the static 3500
persons/km^2 baseline. The legacy scene keeps the original Gaussian population
generator. Risk keeps the existing ballistic + parachute severity structure.
"""


@dataclass(frozen=True)
class PopulationCenter:
    center_id: int
    grid_x: float
    grid_y: float
    physical_x_m: float
    physical_y_m: float
    building_density: float
    center_population_density: float


def population_density_per_m2(density_km2: np.ndarray) -> np.ndarray:
    """Explicit unit boundary: 3500 persons/km^2 = 0.0035 persons/m^2."""

    density = np.asarray(density_km2, dtype=float)
    if not np.all(np.isfinite(density)) or np.any(density < 0):
        raise ValueError("Population density must be finite and nonnegative")
    return density / 1_000_000.0


def detect_population_centers_from_buildings(
    grid: AirspaceGrid,
    window_size: int = 10,
    min_center_spacing_m: float = 1000.0,
    n_population_centers: int = 4,
    beta: float = 2.0,
) -> list[PopulationCenter]:
    """Deterministic local maxima plus non-maximum suppression on the footprint."""

    if window_size < 1 or n_population_centers < 1 or min_center_spacing_m < 0:
        raise ValueError("Require positive window/count and nonnegative center spacing")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    footprint = grid.obstacles.any(axis=2)
    nx, ny = footprint.shape
    windows = [(x, y) for x in range(0, nx, window_size) for y in range(0, ny, window_size)]
    densities = np.array(
        [footprint[x : x + window_size, y : y + window_size].mean() for x, y in windows]
    ).reshape(math.ceil(nx / window_size), math.ceil(ny / window_size))
    max_density = float(densities.max()) if densities.size else 0.0
    if max_density == 0:
        return []
    candidates = []
    for ix in range(densities.shape[0]):
        for iy in range(densities.shape[1]):
            value = float(densities[ix, iy])
            neighborhood = densities[max(0, ix - 1) : ix + 2, max(0, iy - 1) : iy + 2]
            if value > 0 and value >= neighborhood.max():
                x, y = ix * window_size, iy * window_size
                gx = (x + min(nx, x + window_size) - 1) / 2
                gy = (y + min(ny, y + window_size) - 1) / 2
                candidates.append((value, gx, gy))
    selected: list[PopulationCenter] = []
    for value, gx, gy in sorted(candidates, key=lambda c: (-c[0], c[1], c[2])):
        wx, wy = (gx + 0.5) * grid.cell_size[0], (gy + 0.5) * grid.cell_size[1]
        if any(math.hypot(wx - c.physical_x_m, wy - c.physical_y_m) < min_center_spacing_m for c in selected):
            continue
        selected.append(
            PopulationCenter(len(selected) + 1, gx, gy, wx, wy, value, RHO_STATIC_KM2 * (1 + beta * value / max_density))
        )
        if len(selected) == n_population_centers:
            break
    return selected


def gravity_population_density(grid: AirspaceGrid, centers: list[PopulationCenter]) -> np.ndarray:
    """Nearest-center piecewise diffusion, output in persons/km^2."""

    nx, ny, _ = grid.shape
    density = np.full((nx, ny), RHO_STATIC_KM2)
    if not centers:
        return density
    x = (np.arange(nx)[:, None] + 0.5) * grid.cell_size[0]
    y = (np.arange(ny)[None, :] + 0.5) * grid.cell_size[1]
    distances = np.stack([np.hypot(x - c.physical_x_m, y - c.physical_y_m) for c in centers])
    nearest = distances.argmin(axis=0)
    r_km = distances.min(axis=0) / 1000.0
    strength = np.array([c.center_population_density for c in centers])[nearest]
    inside = r_km < 1.0
    density[inside] = strength[inside] * np.exp(1.0 - r_km[inside] ** 2)
    return density


def _old_gaussian_population_density(grid: AirspaceGrid, cfg: dict) -> np.ndarray:
    """Original engineering generator, returning persons/km^2."""

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
    return density


def _population_layer(grid: AirspaceGrid, cfg: dict) -> tuple[np.ndarray, list[PopulationCenter]]:
    model = cfg.get("population_model", "old_gaussian")
    if model == "old_gaussian":
        return _old_gaussian_population_density(grid, cfg), []
    if model != "reference28_gravity":
        raise ValueError(f"Unknown population_model: {model}")
    settings = cfg.get("population", {})
    centers = detect_population_centers_from_buildings(
        grid,
        int(settings.get("window_size", 10)),
        float(settings.get("min_center_spacing_m", 1000.0)),
        int(settings.get("n_population_centers", 4)),
        float(settings.get("beta", 2.0)),
    )
    return gravity_population_density(grid, centers), centers


def generate_population_density(grid: AirspaceGrid, cfg: dict) -> np.ndarray:
    """Static spatial population snapshot in persons/km^2."""

    return _population_layer(grid, cfg)[0]


def calculate_raw_risk(grid: AirspaceGrid, cfg: dict, population_km2: np.ndarray) -> np.ndarray:
    """Unnormalized P_failure * (P_impact * P_severity) for ballistic and parachute descent."""

    risk_cfg = cfg["risk"]
    density = population_density_per_m2(population_km2)
    if density.shape != grid.shape[:2]:
        raise ValueError("Population layer must match the horizontal grid")
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

    return risk


def generate_risk_map(grid: AirspaceGrid, cfg: dict, output_dir: str | Path | None = None) -> np.ndarray:
    """Normalized A* risk map; the population model follows the scene mode."""

    population, centers = _population_layer(grid, cfg)
    raw_risk = calculate_raw_risk(grid, cfg, population)
    risk = minmax_normalize(raw_risk)
    if cfg.get("population_model", "old_gaussian") == "old_gaussian":
        risk[grid.obstacles] = 1.0
    if output_dir is not None:
        out = ensure_dir(output_dir)
        np.save(out / "risk_map.npy", risk)
        np.save(out / "risk_map_raw.npy", raw_risk)
        write_population_risk_diagnostics(grid, cfg, population, centers, risk, out)
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


def write_population_risk_diagnostics(
    grid: AirspaceGrid,
    cfg: dict,
    population: np.ndarray,
    centers: list[PopulationCenter],
    risk: np.ndarray,
    out: str | Path,
) -> None:
    """Population map, detected centers, per-layer risk statistics and figures."""

    out = ensure_dir(out)
    run_id = str(cfg.get("run", {}).get("run_id", "unarchived"))
    np.save(out / "population_map.npy", np.asarray(population, dtype=float))
    (out / "population_layer_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "population_model": cfg.get("population_model", "old_gaussian"),
                "population_units": "persons/km^2",
                "risk_density_units": "persons/m^2",
                "environment_seed": grid.seed,
                "traffic_seed": cfg.get("traffic_seed"),
                "requested_centers": cfg.get("population", {}).get("n_population_centers", 4),
                "actual_gravity_centers": len(centers),
                "diffusion_radius_m": DIFFUSION_RADIUS_M,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    columns = [
        "run_id",
        "center_id",
        "grid_x",
        "grid_y",
        "physical_x_m",
        "physical_y_m",
        "building_density",
        "center_population_density",
    ]
    pd.DataFrame([dict(run_id=run_id, **asdict(center)) for center in centers], columns=columns).to_csv(
        out / "population_centers.csv", index=False
    )
    extent = (0, grid.shape[0] * grid.cell_size[0], 0, grid.shape[1] * grid.cell_size[1])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(population.T, origin="lower", extent=extent, cmap="YlOrRd")
    footprint = grid.obstacles.any(axis=2)
    ax.imshow(
        np.ma.masked_where(~footprint.T, footprint.T),
        origin="lower",
        extent=extent,
        cmap="Greys",
        vmin=0,
        vmax=1,
        alpha=0.25,
    )
    for center in centers:
        ax.plot(center.physical_x_m, center.physical_y_m, "b+", markersize=10)
        ax.text(center.physical_x_m, center.physical_y_m, str(center.center_id), color="blue")
        ax.add_patch(
            Circle(
                (center.physical_x_m, center.physical_y_m),
                DIFFUSION_RADIUS_M,
                fill=False,
                color="blue",
                linestyle="--",
                linewidth=0.8,
            )
        )
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Static population; buildings shaded, centers + 1km radius")
    fig.colorbar(im, ax=ax, label="persons/km^2")
    fig.tight_layout()
    fig.savefig(out / "population_density_map.png", dpi=140)
    plt.close(fig)
    rows = []
    for k in range(grid.shape[2]):
        values = risk[:, :, k]
        rows.append(
            {
                "run_id": run_id,
                "level": f"z{k + 1}",
                "height_m": (k + 0.5) * grid.cell_size[2],
                "min": float(values.min()),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "max": float(values.max()),
                "std": float(values.std()),
                **{f"p{p}": float(np.percentile(values, p)) for p in (90, 95, 99)},
            }
        )
    pd.DataFrame(rows).to_csv(out / "risk_map_stats.csv", index=False)
    (out / "population_implementation_assumptions.md").write_text(POPULATION_ASSUMPTIONS, encoding="utf-8")
