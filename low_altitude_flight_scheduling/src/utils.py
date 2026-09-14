from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

import numpy as np

T = TypeVar("T")


def set_random_seed(seed: int) -> np.random.Generator:
    """Seed Python and NumPy and return a local generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def safe_tqdm(iterable: Iterable[T], **kwargs: object) -> Iterator[T]:
    """Use tqdm when installed; otherwise yield the iterable unchanged."""
    try:
        from tqdm import tqdm

        return iter(tqdm(iterable, **kwargs))
    except Exception:
        return iter(iterable)


def euclidean(a: tuple[int, int, int], b: tuple[int, int, int], cell_size: tuple[int, int, int] = (100, 100, 30)) -> float:
    va = np.array(a, dtype=float) * np.array(cell_size, dtype=float)
    vb = np.array(b, dtype=float) * np.array(cell_size, dtype=float)
    return float(np.linalg.norm(va - vb))


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def path_distance_m(path: list[tuple[int, int, int]], cell_size: tuple[int, int, int] = (100, 100, 30)) -> float:
    if len(path) < 2:
        return 0.0
    return sum(euclidean(a, b, cell_size) for a, b in zip(path[:-1], path[1:]))


def deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def rolling_mean(values: list[float], window: int = 5) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(values[lo : i + 1])))
    return out
