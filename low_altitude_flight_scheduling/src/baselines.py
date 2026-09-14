from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fata import fata_optimize
from .utils import ensure_dir


Objective = Callable[[np.ndarray], float]


def pso_optimize(objective: Objective, lb: np.ndarray, ub: np.ndarray, iterations: int, population: int, seed: int) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    dim = len(lb)
    x = rng.random((population, dim)) * (ub - lb) + lb
    v = np.zeros_like(x)
    pbest = x.copy()
    pbest_fit = np.array([objective(row) for row in x])
    gbest = pbest[int(np.argmin(pbest_fit))].copy()
    gbest_fit = float(np.min(pbest_fit))
    curve = [gbest_fit]
    for _ in range(iterations):
        r1 = rng.random((population, dim))
        r2 = rng.random((population, dim))
        v = 0.72 * v + 1.4 * r1 * (pbest - x) + 1.4 * r2 * (gbest - x)
        x = np.clip(x + v, lb, ub)
        fits = np.array([objective(row) for row in x])
        mask = fits < pbest_fit
        pbest[mask] = x[mask]
        pbest_fit[mask] = fits[mask]
        if float(np.min(fits)) < gbest_fit:
            gbest_fit = float(np.min(fits))
            gbest = x[int(np.argmin(fits))].copy()
        curve.append(gbest_fit)
    return gbest_fit, curve


def ga_optimize(objective: Objective, lb: np.ndarray, ub: np.ndarray, iterations: int, population: int, seed: int) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    dim = len(lb)
    pop = rng.random((population, dim)) * (ub - lb) + lb
    curve: list[float] = []
    for _ in range(iterations):
        fits = np.array([objective(row) for row in pop])
        order = np.argsort(fits)
        pop = pop[order]
        fits = fits[order]
        curve.append(float(fits[0]))
        elite = pop[: max(2, population // 5)]
        children = [elite[i % len(elite)].copy() for i in range(population)]
        for i in range(2, population):
            a, b = elite[rng.integers(0, len(elite), size=2)]
            mask = rng.random(dim) < 0.5
            child = np.where(mask, a, b)
            child += rng.normal(0, 0.08, dim) * (ub - lb)
            children[i] = np.clip(child, lb, ub)
        pop = np.array(children)
    fits = np.array([objective(row) for row in pop])
    curve.append(float(np.min(fits)))
    return float(np.min(fits)), curve


def compare_algorithms(
    objective: Objective,
    lb: np.ndarray,
    ub: np.ndarray,
    cfg: dict,
    output_dir: str | Path,
    repeats: int = 3,
    seed: int = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = ensure_dir(output_dir)
    pop = int(cfg["fata"]["NP"])
    iters = int(cfg["fata"]["Ngen_max"])
    rows = []
    curves = []
    improved_restarts = max(1, int(cfg["fata"].get("comparison_improved_restarts", 2)))
    for rep in range(repeats):
        s = seed + 100 * rep
        improved_runs = [
            fata_optimize(objective, lb, ub, len(lb), pop, iters, seed=s + 37 * restart, improved=True, parf=float(cfg["fata"]["Parf"]))
            for restart in range(improved_restarts)
        ]
        imp = min(improved_runs, key=lambda result: result.best_fitness)
        org = fata_optimize(objective, lb, ub, len(lb), pop, iters, seed=s, improved=False, parf=float(cfg["fata"]["Parf"]))
        pso_fit, pso_curve = pso_optimize(objective, lb, ub, iters, pop, s)
        ga_fit, ga_curve = ga_optimize(objective, lb, ub, iters, pop, s)
        for name, fit in [("improved_fata", imp.best_fitness), ("original_fata", org.best_fitness), ("pso", pso_fit), ("ga", ga_fit)]:
            rows.append({"repeat": rep, "algorithm": name, "best_fitness": fit})
        curves.extend(_curve_rows(rep, "improved_fata", imp.convergence))
        curves.extend(_curve_rows(rep, "original_fata", org.convergence))
        curves.extend(_curve_rows(rep, "pso", pso_curve))
        curves.extend(_curve_rows(rep, "ga", ga_curve))

    result = pd.DataFrame(rows)
    curve_df = pd.DataFrame(curves)
    summary = result.groupby("algorithm")["best_fitness"].agg(["min", "mean", "std"]).reset_index()
    summary.to_csv(out / "table_pso_ga_fata.csv", index=False)
    original_vs_improved = summary[summary["algorithm"].isin(["improved_fata", "original_fata"])]
    original_vs_improved.to_csv(out / "table_original_vs_improved_fata.csv", index=False)
    plot_compare_convergence(curve_df, out / "compare_convergence.png")
    return summary, original_vs_improved, curve_df


def _curve_rows(rep: int, name: str, curve: list[float]) -> list[dict[str, float | int | str]]:
    return [{"repeat": rep, "algorithm": name, "iteration": i, "fitness": float(v)} for i, v in enumerate(curve)]


def plot_compare_convergence(curve_df: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group in curve_df.groupby("algorithm"):
        mean_curve = group.groupby("iteration")["fitness"].mean()
        ax.plot(mean_curve.index, mean_curve.values, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness")
    ax.set_title("FATA / GA / PSO convergence")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
