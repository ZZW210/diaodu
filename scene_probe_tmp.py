"""Temporary scene probe: generate the initial scene only, no scheduling.

Usage (run with cwd set to a project folder):
    python <path to this file> [default|legacy_engineering|paper_strict ...]
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())

from src.config import load_config  # noqa: E402
from src.conflict_detection import detect_conflicts  # noqa: E402
from src.flight_plan import generate_flight_plans  # noqa: E402
from src.grid import AirspaceGrid  # noqa: E402
from src.risk_map import generate_risk_map  # noqa: E402

ROOT = Path(os.getcwd())
OUT = ROOT / "outputs" / "_scene_probe"
OUT.mkdir(parents=True, exist_ok=True)

requested = sys.argv[1:] or ["default"]
for raw_mode in requested:
    mode = None if raw_mode == "default" else raw_mode
    if mode in {"paper", "legacy"}:
        overrides = {"scene_mode": mode}
    elif mode is None:
        overrides = {}
    else:
        overrides = {"optimization": {"scheduler_mode": mode}}
    try:
        cfg = load_config(ROOT / "config.yaml", overrides)
        env_seed = int(cfg.get("environment_seed", cfg["flight"].get("random_seed", 2025)))
        traffic_seed = int(cfg.get("traffic_seed", cfg["flight"].get("random_seed", 2025)))
        cfg.setdefault("flight", {})["random_seed"] = traffic_seed
        grid = AirspaceGrid.from_config(cfg, seed=env_seed)
        risk = generate_risk_map(grid, cfg, OUT / str(raw_mode))
        plans = generate_flight_plans(grid, risk, cfg, None, seed=traffic_seed)
        c_no = detect_conflicts(plans, cfg, uncertain=False)
        c_unc = detect_conflicts(plans, cfg, uncertain=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[{raw_mode}] FAILED: {type(exc).__name__}: {exc}")
        continue
    lengths = []
    for plan in plans:
        arr = np.asarray(plan.path, dtype=float)
        if len(arr) > 1:
            lengths.append(float(np.linalg.norm(np.diff(arr, axis=0) * np.asarray(grid.cell_size), axis=1).sum()))
    conf = cfg.get("conflict", {})
    gen = cfg.get("flight_generation", {})
    astar = cfg.get("astar", {})
    digest = hashlib.md5()
    for plan in plans:
        digest.update(np.asarray(plan.path, dtype=np.int64).tobytes())
        digest.update(np.round(np.asarray(plan.eta_times, dtype=float), 3).tobytes())
        digest.update(np.asarray(plan.start, dtype=np.int64).tobytes())
        digest.update(np.asarray(plan.goal, dtype=np.int64).tobytes())
    risk_digest = hashlib.md5(np.asarray(risk, dtype=float).tobytes()).hexdigest()[:12]
    print(
        f"[{raw_mode}] flights={len(plans)} det={len(c_no)} unc={len(c_unc)} "
        f"t_conflict={conf.get('t_conflict')} sigma0={conf.get('sigma0')} sigma_rate={conf.get('sigma_rate')} "
        f"gen_mode={gen.get('mode')} mean_len_m={np.mean(lengths):.0f} "
        f"risk_w={astar.get('risk_weight')} dist_w={astar.get('distance_weight')} "
        f"plans_md5={digest.hexdigest()[:12]} risk_md5={risk_digest}"
    )
