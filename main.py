"""
main.py — Full pipeline with real 15-route CSV data
======================================================
Demonstrates the complete framework from Zhong et al. (2025):

  1. Load real UAV-style flight trajectories from 15_routes_full_trajectory.csv
     and convert to grid-based FlightPlan / Waypoint objects
       x = round((lat - lat_min) / LAT_STEP)   ← ~5 km/cell
       y = round((lon - lon_min) / LON_STEP)
       z = round((alt - alt_min) / ALT_STEP)    ← 1000 ft/cell
  2. Detect potential conflicts (Eq.21-24)
  3. Build conflict complex network, compute topology metrics (Eq.25-29)
  4. Identify key flight plans via Collective Influence ranking (Sec 2.2.3)
  5. Two-stage optimisation using improved FATA:
       Stage 1 — key plans: all 3 strategies (Δt + v + local rerouting)
       Stage 2 — remaining pairs: all 3 strategies, cumulative rerouting applied
  6. Full fitness S_fit (Eq.50-51)
  7. Matplotlib: conflict network · convergence · conflict reduction bar chart
  8. Interactive Plotly 3D: before / after scheduling (same style as schedule_fata.py)
"""

import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from scipy.interpolate import interp1d
from scipy.stats import norm as sp_norm

_t_total_start = time.perf_counter()

def _hms(seconds: float) -> str:
    """Format seconds as H h M m S.ss s."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h}h {m}m {s:.2f}s"
    if m:
        return f"{m}m {s:.2f}s"
    return f"{s:.2f}s"

import FATA as fata_mod
from FATA import FATA
from conflict_detection import (
    Waypoint, FlightPlan,
    detect_all_conflicts,
    apply_departure_and_speed, compute_orisk,
)
from complex_network import (
    build_conflict_network, identify_key_plans,
    compute_all_metrics, network_robustness,
)

# ══════════════════════════════════════════════════════════════════════════════
# 0.  Parameters  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)

T_CONFLICT  = 8.0     # s - aligned with MATLAB scheduling scripts
SIGMA_BASE  = 1.5     # s - constant ETA uncertainty, as in conflict_detect.m
ALPHA       = 0.05

MAX_DELAY   = 1800.0  # s — 30 min max departure shift
T_RAND_MAX  = 180.0   # s - MATLAB 40-route mixed scenario
MAX_DETOUR  = 15.0    # grid cells - matches MATLAB params.max_detour
V_DEFAULT   = 10.0
V_MIN       =  5.0
V_MAX       = 20.0
T_BATTERY   = 86400.0 # s — 24 h (effectively disabled for commercial flights)
T_DELAY_MAX = 1800.0

OMG_C       = 0.8
OMG_D       = 0.25
OMG_T       = 0.25
OMG_R       = 0.5
ORISK_SCALE = 1e6

N_POP_S1    = 40
N_POP_S2    = 20
FES_PER_DIM = 200

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load CSV and build grid-based FlightPlan objects
# ══════════════════════════════════════════════════════════════════════════════
_t1 = time.perf_counter()
#print("Loading 40_final_routes_trajectory.csv …")
#raw = pd.read_csv('40_final_routes_trajectory.csv')
# print("Loading 40_dec31_routes_trajectory.csv …")
# raw = pd.read_csv('40_dec31_routes_trajectory.csv')
#print("Loading 40_combined_routes_trajectory_new.csv …")
#raw = pd.read_csv('40_combined_routes_trajectory_new.csv')
print("Loading 40_complete_routes_trajectory.csv …")
raw = pd.read_csv('40_complete_routes_trajectory.csv')
#print("15_routes_full_trajectory.csv …")
#raw = pd.read_csv('15_routes_full_trajectory.csv')
raw['ts'] = pd.to_datetime(raw['timestamp'], utc=True)

flight_ids = sorted(raw['flight_id'].unique())
N_PLANS    = len(flight_ids)          # 15
print(f"  Flights loaded : {N_PLANS}")

v_per_flight = []
for fid in flight_ids:
    gs = raw.loc[raw['flight_id'] == fid, 'groundspeed']
    v_i = gs[gs > 50].mean() * 0.514444
    if pd.isna(v_i) or v_i < 50:
        v_i = 200.0
    v_per_flight.append(float(v_i))
V_MEAN = float(np.mean(v_per_flight))
V_DEFAULT = V_MEAN
V_MIN = V_MEAN * 0.80
V_MAX = V_MEAN * 1.20
rand_etd = np.sort(np.random.rand(max(N_PLANS - 15, 0)) * T_RAND_MAX)
print(f"  MATLAB-style speed bounds: V_MEAN={V_MEAN:.1f} m/s, "
      f"range=[{V_MIN:.1f}, {V_MAX:.1f}] m/s")
print(f"  MATLAB-style ETD: first 15 at 0 s, remaining {len(rand_etd)} in 0-{T_RAND_MAX:.0f} s")


class Route:
    """Real-world trajectory interpolated in relative time (for 3D visualisation)."""
    def __init__(self, fid, grp):
        grp = grp.sort_values('ts').dropna(subset=['latitude', 'longitude', 'altitude'])
        self.grp      = grp
        self.fid      = fid
        self.duration = (grp['ts'].iloc[-1] - grp['ts'].iloc[0]).total_seconds()
        t_rel = np.array(
            [(ts - grp['ts'].iloc[0]).total_seconds() for ts in grp['ts']],
            dtype=float,
        )
        lat = grp['latitude'].values.astype(float)
        lon = grp['longitude'].values.astype(float)
        alt = grp['altitude'].values.astype(float)
        kw = dict(bounds_error=False, fill_value=np.nan)
        self.f_lat = interp1d(t_rel, lat, **kw)
        self.f_lon = interp1d(t_rel, lon, **kw)
        self.f_alt = interp1d(t_rel, alt, **kw)


routes = [Route(fid, raw[raw['flight_id'] == fid]) for fid in flight_ids]
print("  Durations (min):", [f"{r.duration/60:.1f}" for r in routes])

# ── Compute geographic bounds ─────────────────────────────────────────────────
LAT_MIN, LAT_MAX = raw['latitude'].min(), raw['latitude'].max()
LON_MIN, LON_MAX = raw['longitude'].min(), raw['longitude'].max()
ALT_MIN_FT, ALT_MAX_FT = raw['altitude'].min(), raw['altitude'].max()
LAT_MID = (LAT_MIN + LAT_MAX) / 2.0

M_PER_DEG_LAT = 6371000.0 * np.pi / 180.0
M_PER_DEG_LON = M_PER_DEG_LAT * np.cos(np.deg2rad(LAT_MID))
CELL_M = 1000.0
CELL_Z_M = 305.0
ALT_MIN_M = ALT_MIN_FT * 0.3048
ALT_MAX_M = ALT_MAX_FT * 0.3048

# Grid cell size  (rerouting dx/dy/dz = ±1 cell)
LAT_STEP = 0.05   # degrees ≈ 5.5 km
LON_STEP = 0.05   # degrees ≈ 3.8 km at 47 °N
ALT_STEP = 1000.0 # feet
LAT_STEP = CELL_M / M_PER_DEG_LAT
LON_STEP = CELL_M / M_PER_DEG_LON
ALT_STEP = CELL_Z_M / 0.3048

GX = int(np.ceil((LON_MAX - LON_MIN) * M_PER_DEG_LON / CELL_M)) + 4
GY = int(np.ceil((LAT_MAX - LAT_MIN) * M_PER_DEG_LAT / CELL_M)) + 4
GZ = int(np.ceil((ALT_MAX_M - ALT_MIN_M) / CELL_Z_M)) + 2
print(f"  Grid dimensions: {GX} × {GY} × {GZ}  "
      f"(lat-step={LAT_STEP}°, lon-step={LON_STEP}°, alt-step={ALT_STEP:.0f}ft)")


def geo_to_grid(lat, lon, alt_ft):
    x = int(round(((lon - LON_MIN) * M_PER_DEG_LON) / CELL_M)) + 2
    y = int(round(((lat - LAT_MIN) * M_PER_DEG_LAT) / CELL_M)) + 2
    z = int(round(((alt_ft * 0.3048 - ALT_MIN_M) / CELL_Z_M))) + 2
    return (max(1, min(GX, x)),
            max(1, min(GY, y)),
            max(1, min(GZ, z)))


def grid_to_geo(x, y, z):
    return (LAT_MIN + (float(y) - 2.0) * CELL_M / M_PER_DEG_LAT,
            LON_MIN + (float(x) - 2.0) * CELL_M / M_PER_DEG_LON,
            ((float(z) - 2.0) * CELL_Z_M + ALT_MIN_M) / 0.3048)


def route_to_flight_plan(route, fid, etd=0.0, max_wpt=350):
    """Convert raw CSV rows using the same cruise-filtered timing as MATLAB."""
    rows = route.grp
    rows_cd = rows[rows['altitude'] >= 15000]
    if len(rows_cd) < 10:
        rows_cd = rows

    t_s = np.array(
        [(ts - rows['ts'].iloc[0]).total_seconds() for ts in rows_cd['ts']],
        dtype=float,
    ) + float(etd)
    lat = rows_cd['latitude'].to_numpy(dtype=float)
    lon = rows_cd['longitude'].to_numpy(dtype=float)
    alt = rows_cd['altitude'].to_numpy(dtype=float)

    waypoints = []
    prev_cell = None
    for ti, lati, loni, alti in zip(t_s, lat, lon, alt):
        cell = geo_to_grid(lati, loni, alti)
        if cell != prev_cell:
            waypoints.append(Waypoint(cell[0], cell[1], cell[2], float(ti)))
            prev_cell = cell

    if len(waypoints) < 2 and waypoints:
        wp = waypoints[-1]
        waypoints.append(Waypoint(min(GX, wp.x + 1), wp.y, wp.z,
                                  wp.t_eta + CELL_M / V_DEFAULT))

    if len(waypoints) > max_wpt:
        idx = np.round(np.linspace(0, len(waypoints) - 1, max_wpt)).astype(int)
        waypoints = [waypoints[i] for i in idx]

    return FlightPlan(fid, waypoints)


etd_by_flight = []
rand_idx = 0
for i in range(N_PLANS):
    if i < 15:
        etd_by_flight.append(0.0)
    else:
        etd_by_flight.append(float(rand_etd[rand_idx]))
        rand_idx += 1

plans = [
    route_to_flight_plan(r, fid, etd)
    for r, fid, etd in zip(routes, flight_ids, etd_by_flight)
]
plan_ids = [p.plan_id for p in plans]
print(f"Total flight plans : {N_PLANS}")
print(f"Avg waypoints/plan : {np.mean([len(p.waypoints) for p in plans]):.0f}")
print(f"  [时间] 数据加载+格点转换: {_hms(time.perf_counter() - _t1)}")


# ── Risk map (uniform low risk — no casualty-probability data for these routes) ─
risk_map = np.full((GX, GY, GZ), 5e-8)

orisk_0  = compute_orisk(plans, risk_map)
t_air_0  = sum(p.t_total for p in plans)
print(f"Baseline T_air     : {t_air_0:.1f} s")
print(f"Baseline ORISK     : {orisk_0:.4e}")

# ══════════════════════════════════════════════════════════════════════════════
# FAST CONFLICT-DETECTION INFRASTRUCTURE
# Mirrors MATLAB's precompute_conflicts + conflict_detect pattern:
#   - Precompute spatially-overlapping pairs once (index arrays, not set ops)
#   - During FATA evaluation: pure numpy array comparisons, zero object creation
# ══════════════════════════════════════════════════════════════════════════════
_Z_ALPHA = float(sp_norm.ppf(1.0 - ALPHA / 2.0))   # ≈ 1.96, computed once

# Proximity thresholds — mirrors MATLAB precompute_conflicts.m
# H_SEP=2 cells × 5.5 km ≈ 11 km horizontal  (MATLAB uses 9 km ≈ 5 NM)
# V_SEP=1 layer  × 1000 ft vertical
H_SEP = 20
V_SEP = 3


def _build_shared_pairs(fp_list, H_sep=H_SEP, V_sep=V_SEP):
    """
    Precompute spatially-proximate plan pairs — mirrors MATLAB precompute_conflicts.m.
    Uses Euclidean proximity (dxy <= H_sep cells, dz <= V_sep layers) instead of
    exact-cell matching, so routes that pass within the safety envelope but in
    adjacent grid cells are no longer missed.
    Returns list of (i, j, ia, ib) where ia/ib are int32 index arrays into
    fp_list[i].waypoints and fp_list[j].waypoints of near-miss waypoint pairs.
    """
    n = len(fp_list)
    pairs = []
    for i in range(n):
        wps_i = fp_list[i].waypoints
        if not wps_i:
            continue
        xi = np.array([wp.x for wp in wps_i], dtype=np.float32)
        yi = np.array([wp.y for wp in wps_i], dtype=np.float32)
        zi = np.array([wp.z for wp in wps_i], dtype=np.float32)
        for j in range(i + 1, n):
            wps_j = fp_list[j].waypoints
            if not wps_j:
                continue
            xj = np.array([wp.x for wp in wps_j], dtype=np.float32)
            yj = np.array([wp.y for wp in wps_j], dtype=np.float32)
            zj = np.array([wp.z for wp in wps_j], dtype=np.float32)
            dxy = np.sqrt((xi[:, None] - xj[None, :]) ** 2 +
                          (yi[:, None] - yj[None, :]) ** 2)
            dz  = np.abs(zi[:, None] - zj[None, :])
            ia, ib = np.where((dxy <= H_sep) & (dz <= V_sep))
            if len(ia):
                pairs.append((i, j, ia.astype(np.int32), ib.astype(np.int32)))
    return pairs


# Base arrays (from original plans — timing never changes after rerouted_at)
_base_t_depart = np.array([p.t_depart for p in plans])
_base_intervals = [
    np.array([wp.t_eta - p.t_depart for wp in p.waypoints])
    for p in plans
]
_base_t_total = np.array([p.t_total for p in plans])
# Uniform risk map → orisk = constant regardless of rerouting
_orisk_const = float(risk_map.flat[0]) * sum(len(p.waypoints) for p in plans)

# Initial shared pairs (rebuilt after rerouting decisions are applied)
_sp = _build_shared_pairs(plans)
print(f"  Precomputed {len(_sp)} spatially-overlapping pairs  "
      f"[fast conflict engine ready]")


def _get_all_times(offsets, v_ratios):
    """Compute waypoint times for all plans — no object creation."""
    times = []
    for i in range(N_PLANS):
        t0 = _base_t_depart[i] + float(offsets[i])
        iv = _base_intervals[i]
        vr = float(v_ratios[i])
        times.append(t0 + (iv / vr if abs(vr - 1.0) > 1e-4 else iv))
    return times


def _n_conflicts(all_times, sp, excl=None):
    """
    Count pairwise conflicts using precomputed shared pairs.
    excl: optional dict {pidx: bool_keep_mask} — rows where mask is False
          are skipped (waypoints moved away by rerouting).
    Vectorised sigma computation matches conflict_detection.py Eq.22-24.
    """
    n_c = 0
    for pidx, (i, j, ia, ib) in enumerate(sp):
        if excl and pidx in excl:
            keep = excl[pidx]
            if not np.any(keep):
                continue
            ia = ia[keep]
            ib = ib[keep]
        ta = all_times[i][ia]
        tb = all_times[j][ib]
        t_req = _Z_ALPHA * (SIGMA_BASE + SIGMA_BASE) + T_CONFLICT
        if np.any(np.abs(ta - tb) <= t_req):
            n_c += 1
    return n_c


def _nearest_conflict_cell_for_target(fp_list, sp, idx_a, idx_b, target_idx, all_times):
    """
    Return one waypoint cell on target_idx for the nearest temporal near-miss
    between idx_a and idx_b, using the same H_SEP/V_SEP+t_req rule as detection.
    """
    t_req = _Z_ALPHA * (SIGMA_BASE + SIGMA_BASE) + T_CONFLICT
    for pi, pj, ia, ib in sp:
        if not ((pi == idx_a and pj == idx_b) or (pi == idx_b and pj == idx_a)):
            continue

        dt = np.abs(all_times[pi][ia] - all_times[pj][ib])
        candidate_rows = np.where(dt <= t_req)[0]
        if len(candidate_rows) == 0:
            return set()

        best_row = int(candidate_rows[np.argmin(dt[candidate_rows])])
        if target_idx == pi:
            wp_idx = int(ia[best_row])
        elif target_idx == pj:
            wp_idx = int(ib[best_row])
        else:
            return set()

        wp = fp_list[target_idx].waypoints[wp_idx]
        return {(wp.x, wp.y, wp.z)}

    return set()


def _sfit(offsets, v_ratios, n_c, delta):
    """Objective value S_fit (Eq.50-51) — pure numpy, no FlightPlan objects."""
    vr_safe = np.maximum(v_ratios, 1e-3)
    t_air   = float(np.sum(_base_t_total / vr_safe))
    t_delay = float(np.sum(np.abs(offsets)))
    n_delay = int(np.sum(np.abs(offsets) > 1.0))
    n_bat   = int(np.sum(_base_t_total / vr_safe > T_BATTERY))
    f = ((1 - OMG_C) * (OMG_D * t_delay + OMG_T * t_air
                         + OMG_R * _orisk_const * ORISK_SCALE)
         + OMG_C * delta * n_c * t_air)
    return f + 1000 * n_bat + 100 * n_delay

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Initial conflict detection
# ══════════════════════════════════════════════════════════════════════════════
_t2 = time.perf_counter()
initial_conflicts = detect_all_conflicts(plans, T_CONFLICT, SIGMA_BASE, ALPHA, H_SEP, V_SEP)
print(f"Initial conflicts  : {len(initial_conflicts)}  [{_hms(time.perf_counter()-_t2)}]")

if not initial_conflicts:
    print("No conflicts detected — nothing to optimise.")
    raise SystemExit(0)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Conflict complex network & key plan identification
# ══════════════════════════════════════════════════════════════════════════════
G_init = build_conflict_network(plan_ids, initial_conflicts)
metrics = compute_all_metrics(G_init)

key_plan_ids = identify_key_plans(G_init, top_k=max(1, N_PLANS // 10))
key_idx      = [i for i, p in enumerate(plans) if p.plan_id in key_plan_ids]

print(f"Key plans          : {key_plan_ids}  ({len(key_plan_ids)} plans)")
print(f"Network robustness : {network_robustness(G_init):.3f}")
print(f"Degree (key plans) : {[G_init.degree(k) for k in key_plan_ids]}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Stage 1 — optimise key plans (all 3 strategies)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Stage 1: global optimisation of key plans ──")
_t_s1 = time.perf_counter()

conflict_pair_set = {(min(a, b), max(a, b)) for a, b in initial_conflicts}
key_conflict_cells: dict[int, set] = {}
for global_i in key_idx:
    pid_i  = plans[global_i].plan_id
    cells: set = set()
    for j in range(N_PLANS):
        if j == global_i:
            continue
        pid_j = plans[j].plan_id
        pair  = (min(pid_i, pid_j), max(pid_i, pid_j))
        if pair in conflict_pair_set:
            cells |= plans[global_i].waypoint_set() & plans[j].waypoint_set()
    key_conflict_cells[global_i] = cells

# Precompute: for each shared pair that involves a key plan, which of the
# shared-waypoint rows lie inside that key plan's conflict cells.
# Used in fobj_stage1 to exclude rerouted cells without rebuilding FlightPlan.
# _s1_excl[(pidx, local_i)] = bool array, True = keep (NOT in cc)
_s1_excl: dict = {}
for local_i, global_i in enumerate(key_idx):
    cc = key_conflict_cells.get(global_i, frozenset())
    if not cc:
        continue
    wps_key = plans[global_i].waypoints
    for pidx, (pi, pj, ia, ib) in enumerate(_sp):
        if pi != global_i and pj != global_i:
            continue
        key_ia = ia if pi == global_i else ib
        keep = np.array(
            [(wps_key[int(k)].x, wps_key[int(k)].y, wps_key[int(k)].z) not in cc
             for k in key_ia],
            dtype=bool,
        )
        _s1_excl[(pidx, local_i)] = keep

dim_s1  = 5 * len(key_idx)
MaxFEs1 = FES_PER_DIM * dim_s1

lb_s1 = np.tile([0.0,       V_MIN / V_DEFAULT, 0.0, 0.0, 0.0], len(key_idx))
ub_s1 = np.tile([MAX_DELAY, V_MAX / V_DEFAULT, 0.0, 0.0, 0.0], len(key_idx))


def fobj_stage1(x_key: np.ndarray) -> float:
    delta    = fata_mod.FATA.delta
    offsets  = np.zeros(N_PLANS)
    v_ratios = np.ones(N_PLANS)
    # Decode decision variables; collect rerouting flags
    reroutes = []       # (local_i, global_i, do_rr)
    for local_i, global_i in enumerate(key_idx):
        offsets[global_i]  = x_key[5 * local_i]
        v_ratios[global_i] = x_key[5 * local_i + 1]
        dx = int(np.round(np.clip(x_key[5 * local_i + 2], -1.0, 1.0)))
        dy = int(np.round(np.clip(x_key[5 * local_i + 3], -1.0, 1.0)))
        dz = int(np.round(np.clip(x_key[5 * local_i + 4], -1.0, 1.0)))
        reroutes.append((local_i, global_i, bool(dx or dy or dz)))

    # Compute waypoint times — pure numpy, zero FlightPlan object creation
    times = _get_all_times(offsets, v_ratios)

    # Build exclusion dict: rerouted cc cells are removed from conflict check
    excl: dict = {}
    for local_i, global_i, do_rr in reroutes:
        if not do_rr:
            continue
        for pidx in range(len(_sp)):
            key = (pidx, local_i)
            if key not in _s1_excl:
                continue
            keep = _s1_excl[key]
            # AND-combine if multiple key plans affect the same pair
            excl[pidx] = keep if pidx not in excl else (excl[pidx] & keep)

    n_c = _n_conflicts(times, _sp, excl if excl else None)
    return _sfit(offsets, v_ratios, n_c, delta)


best_key, score_s1, cg_s1 = FATA(
    fobj_stage1, lb_s1, ub_s1,
    dim=dim_s1, N=N_POP_S1, MaxFEs=MaxFEs1,
    use_gps=True,
)

full_offsets_s1  = np.zeros(N_PLANS)
full_v_ratios_s1 = np.ones(N_PLANS)
reroute_decisions: list[tuple[int, set, int, int, int]] = []

for local_i, global_i in enumerate(key_idx):
    full_offsets_s1[global_i]  = best_key[5 * local_i]
    full_v_ratios_s1[global_i] = best_key[5 * local_i + 1]
    cells = key_conflict_cells.get(global_i, set())
    if cells:
        dx = int(np.round(np.clip(best_key[5 * local_i + 2], -1.0, 1.0)))
        dy = int(np.round(np.clip(best_key[5 * local_i + 3], -1.0, 1.0)))
        dz = int(np.round(np.clip(best_key[5 * local_i + 4], -1.0, 1.0)))
        if dx or dy or dz:
            reroute_decisions.append((global_i, cells, dx, dy, dz))

plans_after_s1 = apply_departure_and_speed(plans, full_offsets_s1, full_v_ratios_s1)
for plan_idx, cells, rdx, rdy, rdz in reroute_decisions:
    plans_after_s1[plan_idx] = plans_after_s1[plan_idx].rerouted_at(
        cells, rdx, rdy, rdz, grid_bounds=(GX, GY, GZ))

conflicts_after_s1 = detect_all_conflicts(plans_after_s1, T_CONFLICT, SIGMA_BASE, ALPHA, H_SEP, V_SEP)
print(f"Conflicts after S1 : {len(conflicts_after_s1)}  [Stage 1 耗时: {_hms(time.perf_counter()-_t_s1)}]")

# Rebuild shared pairs from the spatially-updated plans (rerouting changes cells).
# _base_t_depart / _base_intervals remain from original plans — rerouted_at
# preserves all ETAs, so timing computations stay valid.
_current_plans = list(plans_after_s1)
_sp = _build_shared_pairs(_current_plans)

for local_i, global_i in enumerate(key_idx):
    cells = key_conflict_cells.get(global_i, set())
    dx = int(np.round(np.clip(best_key[5*local_i+2], -1.0, 1.0))) if cells else 0
    dy = int(np.round(np.clip(best_key[5*local_i+3], -1.0, 1.0))) if cells else 0
    dz = int(np.round(np.clip(best_key[5*local_i+4], -1.0, 1.0))) if cells else 0
    rr = f"reroute=({dx},{dy},{dz})" if (dx or dy or dz) else "no reroute"
    print(f"  Key plan {plans[global_i].plan_id}: "
          f"Δt={full_offsets_s1[global_i]:+.1f}s  "
          f"v={full_v_ratios_s1[global_i]:.2f}  {rr}")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Stage 2 — resolve remaining conflicts
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Stage 2: local optimisation of remaining conflicts ──")
_t_s2 = time.perf_counter()

full_offsets_s2  = full_offsets_s1.copy()
full_v_ratios_s2 = full_v_ratios_s1.copy()
cg_s2_all: list[np.ndarray] = []

remaining = conflicts_after_s1.copy()
resolved  = 0

for pid_a, pid_b in remaining:
    _t_pair = time.perf_counter()
    idx_a = next(i for i, p in enumerate(plans) if p.plan_id == pid_a)
    idx_b = next(i for i, p in enumerate(plans) if p.plan_id == pid_b)

    dim_s2  = 7
    MaxFEs2 = FES_PER_DIM * dim_s2

    snap_off  = full_offsets_s2.copy()
    snap_vr   = full_v_ratios_s2.copy()
    fixed_rr  = list(reroute_decisions)
    snap_times = _get_all_times(snap_off, snap_vr)
    conflict_cells = _nearest_conflict_cell_for_target(
        _current_plans, _sp, idx_a, idx_b, idx_b, snap_times)

    # Precompute: which rows of the (idx_a, idx_b) shared pair lie inside
    # conflict_cells for plan idx_b.  When plan_b is rerouted those rows are
    # excluded from the fast conflict check without rebuilding any objects.
    _cc_pidx: int | None = None
    _cc_keep: np.ndarray | None = None
    if conflict_cells:
        wps_b = _current_plans[idx_b].waypoints
        for _pidx, (_pi, _pj, _ia, _ib) in enumerate(_sp):
            if not ((_pi == idx_a and _pj == idx_b) or
                    (_pi == idx_b and _pj == idx_a)):
                continue
            # _ib (or _ia) are indices into plan_b's waypoints
            b_indices = _ib if _pi == idx_a else _ia
            _cc_keep = np.array(
                [(wps_b[int(k)].x, wps_b[int(k)].y, wps_b[int(k)].z)
                 not in conflict_cells
                 for k in b_indices],
                dtype=bool,
            )
            _cc_pidx = _pidx
            break

    def fobj_stage2(
            x_pair: np.ndarray,
            _a=idx_a, _b=idx_b,
            _snap_off=snap_off, _snap_vr=snap_vr,
            _cc=conflict_cells,
            _cc_p=_cc_pidx, _cc_k=_cc_keep,
    ) -> float:
        delta    = fata_mod.FATA.delta
        offsets  = _snap_off.copy()
        v_ratios = _snap_vr.copy()
        offsets[_a]  = x_pair[0];  offsets[_b]  = x_pair[1]
        v_ratios[_a] = x_pair[2];  v_ratios[_b] = x_pair[3]
        dx = int(np.round(np.clip(x_pair[4], -MAX_DETOUR, MAX_DETOUR)))
        dy = int(np.round(np.clip(x_pair[5], -MAX_DETOUR, MAX_DETOUR)))
        dz = int(np.round(np.clip(x_pair[6], -MAX_DETOUR, MAX_DETOUR)))

        # Compute times — pure numpy, zero object creation
        times = _get_all_times(offsets, v_ratios)

        # When plan_b is rerouted, exclude its cc cells from the (a,b) pair check
        excl: dict = {}
        if _cc and (dx or dy or dz) and _cc_p is not None and _cc_k is not None:
            excl[_cc_p] = _cc_k

        n_c = _n_conflicts(times, _sp, excl if excl else None)

        t_delay   = float(np.abs(offsets[_a]) + np.abs(offsets[_b]))
        n_delay   = int(np.abs(offsets[_a]) > 1.0) + int(np.abs(offsets[_b]) > 1.0)
        vr_safe   = np.maximum(v_ratios, 1e-3)
        t_air     = float(np.sum(_base_t_total / vr_safe))
        n_battery = int(np.sum(_base_t_total / vr_safe > T_BATTERY))
        f_obj = (
            (1 - OMG_C) * (OMG_D * t_delay + OMG_T * t_air
                            + OMG_R * _orisk_const * ORISK_SCALE)
            + OMG_C * delta * n_c * t_air
        )
        return f_obj + 1000 * n_battery + 100 * n_delay

    lb_pair = np.array([0.0, 0.0,
                        V_MIN / V_DEFAULT, V_MIN / V_DEFAULT,
                        -MAX_DETOUR, -MAX_DETOUR, -MAX_DETOUR])
    ub_pair = np.array([MAX_DELAY, MAX_DELAY,
                        V_MAX / V_DEFAULT, V_MAX / V_DEFAULT,
                        MAX_DETOUR, MAX_DETOUR, MAX_DETOUR])

    best_pair, _, cg_pair = FATA(
        fobj_stage2, lb=lb_pair, ub=ub_pair,
        dim=dim_s2, N=N_POP_S2, MaxFEs=MaxFEs2,
        use_gps=True,
    )

    off_a, off_b = float(best_pair[0]), float(best_pair[1])
    vr_a,  vr_b  = float(best_pair[2]), float(best_pair[3])
    dx = int(np.round(np.clip(best_pair[4], -MAX_DETOUR, MAX_DETOUR)))
    dy = int(np.round(np.clip(best_pair[5], -MAX_DETOUR, MAX_DETOUR)))
    dz = int(np.round(np.clip(best_pair[6], -MAX_DETOUR, MAX_DETOUR)))

    trial_off = snap_off.copy();  trial_off[idx_a] = off_a;  trial_off[idx_b] = off_b
    trial_vr  = snap_vr.copy();   trial_vr[idx_a]  = vr_a;   trial_vr[idx_b]  = vr_b
    trial_mod = apply_departure_and_speed(plans, trial_off, trial_vr)
    for plan_idx, cells, rdx, rdy, rdz in reroute_decisions:
        trial_mod[plan_idx] = trial_mod[plan_idx].rerouted_at(
            cells, rdx, rdy, rdz, grid_bounds=(GX, GY, GZ))
    if conflict_cells and (dx or dy or dz):
        trial_mod[idx_b] = trial_mod[idx_b].rerouted_at(
            conflict_cells, dx, dy, dz, grid_bounds=(GX, GY, GZ))

    prev_mod = apply_departure_and_speed(plans, full_offsets_s2, full_v_ratios_s2)
    for plan_idx, cells, rdx, rdy, rdz in reroute_decisions:
        prev_mod[plan_idx] = prev_mod[plan_idx].rerouted_at(
            cells, rdx, rdy, rdz, grid_bounds=(GX, GY, GZ))
    prev_n = len(detect_all_conflicts(prev_mod, T_CONFLICT, SIGMA_BASE, ALPHA, H_SEP, V_SEP))

    after_trial = detect_all_conflicts(trial_mod, T_CONFLICT, SIGMA_BASE, ALPHA, H_SEP, V_SEP)
    if len(after_trial) <= prev_n:
        full_offsets_s2  = trial_off
        full_v_ratios_s2 = trial_vr
        if conflict_cells and (dx or dy or dz):
            reroute_decisions.append((idx_b, conflict_cells, dx, dy, dz))
            # Update spatial structure so next pair's fast engine is accurate
            _current_plans[idx_b] = _current_plans[idx_b].rerouted_at(
                conflict_cells, dx, dy, dz, grid_bounds=(GX, GY, GZ))
            _sp = _build_shared_pairs(_current_plans)
        resolved += 1

    cg_s2_all.append(cg_pair)
    rr = f"reroute=({dx},{dy},{dz})" if (dx or dy or dz) else "no reroute"
    print(f"  Conflict ({pid_a},{pid_b}) — "
          f"Δt=[{off_a:+.1f},{off_b:+.1f}]s  "
          f"v=[{vr_a:.2f},{vr_b:.2f}]  {rr}  [{_hms(time.perf_counter()-_t_pair)}]")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  Final evaluation
# ══════════════════════════════════════════════════════════════════════════════
plans_final = apply_departure_and_speed(plans, full_offsets_s2, full_v_ratios_s2)
for plan_idx, cells, rdx, rdy, rdz in reroute_decisions:
    plans_final[plan_idx] = plans_final[plan_idx].rerouted_at(
        cells, rdx, rdy, rdz, grid_bounds=(GX, GY, GZ))

final_conflicts = detect_all_conflicts(plans_final, T_CONFLICT, SIGMA_BASE, ALPHA, H_SEP, V_SEP)
t_air_final     = sum(p.t_total for p in plans_final)
orisk_final     = compute_orisk(plans_final, risk_map)

_t_s2_total = time.perf_counter() - _t_s2

print(f"\n{'─'*56}")
print(f"Initial conflicts  : {len(initial_conflicts)}")
print(f"After Stage 1      : {len(conflicts_after_s1)}")
print(f"After Stage 2      : {len(final_conflicts)}")
reduction_pct = 100 * (1 - len(final_conflicts) / max(len(initial_conflicts), 1))
print(f"Conflict reduction : {reduction_pct:.1f} %")
print(f"Total delay added  : {np.sum(np.abs(full_offsets_s2)):.1f} s")
delayed_pct = 100 * np.sum(np.abs(full_offsets_s2) > 1e-3) / N_PLANS
print(f"Plans with delay   : {delayed_pct:.1f} %")
s1_rr = sum(1 for pi,_,_,_,_ in reroute_decisions if pi in key_idx)
s2_rr = len(reroute_decisions) - s1_rr
print(f"Plans rerouted     : {len(reroute_decisions)}  (S1={s1_rr}, S2={s2_rr})")
t_air_delta_pct = 100 * (t_air_final - t_air_0) / max(t_air_0, 1)
orisk_ratio     = orisk_final / max(orisk_0, 1e-20)
print(f"T_air change       : {t_air_delta_pct:+.2f} %")
print(f"ORISK ratio        : {orisk_ratio:.4f}")
G_final = build_conflict_network(plan_ids, final_conflicts)
print(f"Final robustness   : {network_robustness(G_final):.3f}")
print(f"{'─'*56}")
print(f"[耗时汇总]")
print(f"  Stage 1 优化     : {_hms(time.perf_counter() - _t_s1 - _t_s2_total)}")
print(f"  Stage 2 优化     : {_hms(_t_s2_total)}  ({len(remaining)} 对冲突)")
print(f"{'─'*56}")

# Collect strategy info for each plan
rerouted_idx_set = {pi for pi, _, _, _, _ in reroute_decisions}

def _strat_zh(i):
    if i in rerouted_idx_set:                         return '局部改航'
    if abs(full_v_ratios_s2[i] - 1.0) > 0.05:        return '速度调整'
    if abs(full_offsets_s2[i]) > 30:                  return '延迟起飞'
    return '无需调整'

sids    = [str(f)[-5:] for f in flight_ids]
strat_zh = [_strat_zh(i) for i in range(N_PLANS)]

delays = np.maximum(full_offsets_s2, 0.0)
n_delayed = int(np.sum(delays > 1.0))
rerouted_flags = np.zeros(N_PLANS, dtype=bool)
for pi, _, _, _, _ in reroute_decisions:
    rerouted_flags[pi] = True
n_rerouted = int(np.sum(rerouted_flags))
n_speed_adj = int(np.sum(np.abs(full_v_ratios_s2 - 1.0) > 0.01))

# ══════════════════════════════════════════════════════════════════════════════
# 7.  Matplotlib 2D Visualisation
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
fig.suptitle(f"UAV Flight-Plan Scheduling — FATA Two-Stage Optimisation ({N_PLANS} Real Routes)",
             fontsize=13, fontweight='bold')

short_labels = {fid: str(fid)[-5:] for fid in flight_ids}

ax1 = fig.add_subplot(2, 3, 1)
pos = nx.spring_layout(G_init, seed=7)
node_color = ['#e74c3c' if v in key_plan_ids else '#3498db' for v in G_init.nodes()]
nx.draw_networkx(G_init, pos=pos, ax=ax1,
                 labels=short_labels,
                 node_color=node_color, node_size=350,
                 font_size=7, edge_color='#888', width=1.2, alpha=0.9)
ax1.set_title(f'Initial Conflict Network\n'
              f'({G_init.number_of_nodes()} nodes, '
              f'{G_init.number_of_edges()} edges)\nRed = key plans', fontsize=9)
ax1.axis('off')

ax2 = fig.add_subplot(2, 3, 2)
if G_final.number_of_edges() > 0:
    nx.draw_networkx(G_final, pos=pos, ax=ax2,
                     labels=short_labels,
                     node_color='#95a5a6', node_size=350,
                     font_size=7, edge_color='#e74c3c', width=1.5, alpha=0.9)
else:
    nx.draw_networkx_nodes(G_final, pos=pos, ax=ax2,
                           node_color='#95a5a6', node_size=350, alpha=0.9)
    nx.draw_networkx_labels(G_final, pos=pos, ax=ax2,
                            labels=short_labels, font_size=7)
ax2.set_title(f'Final Conflict Network\n'
              f'({G_final.number_of_edges()} remaining conflicts)', fontsize=9)
ax2.axis('off')

ax3 = fig.add_subplot(2, 3, 3)
degrees_init  = [G_init.degree(v)  for v in G_init.nodes()]
degrees_final = [G_final.degree(v) for v in G_final.nodes()]
bins = range(0, max(degrees_init + [1]) + 2)
ax3.hist(degrees_init,  bins=bins, alpha=0.6, color='#3498db', label='Initial',
         align='left', rwidth=0.4)
ax3.hist(degrees_final, bins=bins, alpha=0.6, color='#e74c3c', label='Final',
         align='mid',  rwidth=0.4)
ax3.set_title('Degree Distribution', fontsize=9)
ax3.set_xlabel('Degree');  ax3.set_ylabel('Count')
ax3.legend(fontsize=8);    ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(cg_s1, color='#e74c3c', linewidth=1.5, label='Stage 1 (key plans)')
ax4.axvline(len(cg_s1) // 2, color='gray', linestyle=':', linewidth=1, label='midpoint')
ax4.set_title('Stage 1 Convergence', fontsize=9)
ax4.set_xlabel('Iteration');  ax4.set_ylabel('Objective')
ax4.legend(fontsize=8);       ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(2, 3, 5)
colors_s2 = plt.cm.viridis(np.linspace(0, 0.85, max(len(cg_s2_all), 1)))
for k, cg in enumerate(cg_s2_all):
    ax5.plot(cg, color=colors_s2[k], linewidth=1.0, alpha=0.8,
             label=f'Conflict {k+1}' if len(cg_s2_all) <= 6 else None)
ax5.set_title(f'Stage 2 Convergence\n({len(cg_s2_all)} sub-problems)', fontsize=9)
ax5.set_xlabel('Iteration');  ax5.set_ylabel('Objective')
if 0 < len(cg_s2_all) <= 6:
    ax5.legend(fontsize=7)
ax5.grid(alpha=0.3)

ax6 = fig.add_subplot(2, 3, 6)
stages = ['Initial', 'After\nStage 1', 'After\nStage 2']
counts = [len(initial_conflicts), len(conflicts_after_s1), len(final_conflicts)]
bars = ax6.bar(stages, counts, color=['#3498db', '#f39c12', '#2ecc71'],
               width=0.5, edgecolor='white')
for bar, val in zip(bars, counts):
    ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
             str(val), ha='center', va='bottom', fontweight='bold', fontsize=10)
ax6.set_title('Conflict Count per Stage', fontsize=9)
ax6.set_ylabel('Number of Conflicts')
ax6.grid(axis='y', alpha=0.3)
ax6.set_ylim(0, max(counts) * 1.25 if max(counts) > 0 else 5)

plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to results.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 8.  Interactive Plotly 3D Visualisation
# ══════════════════════════════════════════════════════════════════════════════
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("\n[警告] plotly 未安装，跳过交互式3D图。请运行: pip install plotly")

if HAS_PLOTLY:
    print("\n生成交互式 3D 航线图 …")

    N_SAMP = 300

    def _sample(route):
        t   = np.linspace(0, route.duration, N_SAMP)
        lat = route.f_lat(t);  lon = route.f_lon(t);  alt = route.f_alt(t)
        ok  = ~(np.isnan(lat) | np.isnan(lon) | np.isnan(alt))
        return lat[ok], lon[ok], alt[ok]

    sampled = [_sample(r) for r in routes]

    CP = [
        '#C0392B', '#16A085', '#2471A3', '#D68910', '#7D3C98',
        '#148F77', '#CB4335', '#1E8449', '#E74C3C', '#117A65',
        '#6C3483', '#935116', '#1A5276', '#922B21', '#0E6655',
        '#F39C12', '#1ABC9C', '#2980B9', '#8E44AD', '#27AE60',
        '#E67E22', '#3498DB', '#9B59B6', '#2ECC71', '#E74C3C',
        '#F1C40F', '#1F618D', '#6D4C41', '#00838F', '#AD1457',
        '#558B2F', '#4527A0', '#D84315', '#00695C', '#283593',
        '#6A1520', '#004D40', '#4A148C', '#BF360C', '#1A237E',
    ]

    # MATLAB-style interactive strategy overview.
    idx_1based = np.arange(1, N_PLANS + 1)
    group_colors = ['#3366AA' if i < 15 else '#D96B27' for i in range(N_PLANS)]
    delay_hover = [
        f"航班: {flight_ids[i]}<br>"
        f"组别: {'前15架同时起飞' if i < 15 else '后25架随机起飞'}<br>"
        f"延误: {delays[i]:.1f} s ({delays[i]/60:.2f} min)<br>"
        f"策略: {strat_zh[i]}"
        for i in range(N_PLANS)
    ]
    speed_colors = np.where(
        full_v_ratios_s2 < 0.99, '#D9482A',
        np.where(full_v_ratios_s2 > 1.01, '#E3A21A', '#2FA84F'),
    )
    speed_hover = [
        f"航班: {flight_ids[i]}<br>"
        f"速度比: {full_v_ratios_s2[i]:.3f}<br>"
        f"策略: {strat_zh[i]}"
        for i in range(N_PLANS)
    ]
    reroute_hover = [
        f"航班: {flight_ids[i]}<br>"
        f"是否改航: {'是' if rerouted_flags[i] else '否'}<br>"
        f"策略: {strat_zh[i]}"
        for i in range(N_PLANS)
    ]

    fig_strategy = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}]],
        subplot_titles=(
            '3D ADS-B 航迹',
            f'策略一：错峰起飞（{n_delayed} 架延误，合计 {np.sum(delays)/60:.1f} 分钟）',
            f'策略二：调整速度（{n_speed_adj} 架调速）',
            f'策略三：局部改航（{n_rerouted} / {N_PLANS} 架）',
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.10,
    )

    for i, (lat, lon, alt) in enumerate(sampled):
        alt_km = alt * 0.3048 / 1000.0
        grp_name = '前15架同时起飞' if i < 15 else '后25架随机起飞'
        fig_strategy.add_trace(go.Scatter3d(
            x=lon, y=lat, z=alt_km,
            mode='lines',
            line=dict(color=group_colors[i], width=4),
            name=grp_name,
            legendgroup=grp_name,
            showlegend=(i == 0 or i == 15),
            text=[f"航班: {flight_ids[i]}"] * len(lon),
            hovertemplate='%{text}<br>经度: %{x:.3f}<br>纬度: %{y:.3f}<br>高度: %{z:.2f} km<extra></extra>',
        ), row=1, col=1)

    reroute_payload = []
    for i in range(N_PLANS):
        if not rerouted_flags[i]:
            reroute_payload.append(None)
            continue
        raw_lat, raw_lon, raw_alt = sampled[i]
        ox = raw_lon.astype(float).tolist()
        oy = raw_lat.astype(float).tolist()
        oz = (raw_alt * 0.3048 / 1000.0).astype(float).tolist()
        fx = raw_lon.astype(float).copy()
        fy = raw_lat.astype(float).copy()
        fz = (raw_alt * 0.3048 / 1000.0).astype(float).copy()
        mx, my, mz, mt = [], [], [], []
        for k, (wp0, wp1) in enumerate(zip(plans[i].waypoints, plans_final[i].waypoints), start=1):
            if (wp0.x, wp0.y, wp0.z) != (wp1.x, wp1.y, wp1.z):
                old_la, old_lo, old_al_ft = grid_to_geo(wp0.x, wp0.y, wp0.z)
                la, lo, al_ft = grid_to_geo(wp1.x, wp1.y, wp1.z)
                if len(fx):
                    dist = ((fx - old_lo) ** 2 +
                            (fy - old_la) ** 2 +
                            ((fz - old_al_ft * 0.3048 / 1000.0) / 20.0) ** 2)
                    center = int(np.argmin(dist))
                    width = max(3, min(10, len(fx) // 30))
                    lo0, la0, al0 = fx[center], fy[center], fz[center]
                    for jj in range(max(0, center - width), min(len(fx), center + width + 1)):
                        u = abs(jj - center) / max(width, 1)
                        w = 0.5 * (1.0 + np.cos(np.pi * u))
                        fx[jj] = (1.0 - w) * fx[jj] + w * lo
                        fy[jj] = (1.0 - w) * fy[jj] + w * la
                        fz[jj] = (1.0 - w) * fz[jj] + w * (al_ft * 0.3048 / 1000.0)
                    fx[center], fy[center], fz[center] = lo, la, al_ft * 0.3048 / 1000.0
                mx.append(lo)
                my.append(la)
                mz.append(al_ft * 0.3048 / 1000.0)
                mt.append(
                    f"航班: {flight_ids[i]}<br>"
                    f"改航点序号: {k}<br>"
                    f"原格点: ({wp0.x},{wp0.y},{wp0.z})<br>"
                    f"新格点: ({wp1.x},{wp1.y},{wp1.z})"
                )
        reroute_payload.append({
            'flight': str(flight_ids[i]),
            'sid': sids[i],
            'original': {'x': ox, 'y': oy, 'z': oz},
            'rerouted': {'x': fx.tolist(), 'y': fy.tolist(), 'z': fz.tolist()},
            'markers': {'x': mx, 'y': my, 'z': mz, 'text': mt},
        })

    comparison_trace_start = len(fig_strategy.data)
    for name, color in [
        ('原始真实航迹', '#2E86C1'),
        ('改航后航迹', '#D35400'),
    ]:
        fig_strategy.add_trace(go.Scatter3d(
            x=[], y=[], z=[],
            mode='lines',
            line=dict(color=color, width=6),
            name=name,
            legendgroup='改航对比',
            visible=False,
            hovertemplate=f'{name}<br>经度: %{{x:.3f}}<br>纬度: %{{y:.3f}}<br>高度: %{{z:.2f}} km<extra></extra>',
        ), row=1, col=1)
    fig_strategy.add_trace(go.Scatter3d(
        x=[], y=[], z=[],
        mode='markers+text',
        marker=dict(size=7, color='#E74C3C', symbol='diamond',
                    line=dict(width=1.5, color='#7B241C')),
        text=[],
        textposition='top center',
        name='改航点',
        legendgroup='改航对比',
        visible=False,
        hovertemplate='%{customdata}<extra></extra>',
        customdata=[],
    ), row=1, col=1)

    reroute_trace_index = len(fig_strategy.data) + 2

    fig_strategy.add_trace(go.Bar(
        x=idx_1based,
        y=delays / 60.0,
        marker_color=group_colors,
        text=[f'{v:.1f}' if v > 0.05 * max(np.max(delays / 60.0), 1.0) else ''
              for v in delays / 60.0],
        textposition='outside',
        hovertext=delay_hover,
        hoverinfo='text',
        name='起飞延误',
        showlegend=False,
    ), row=1, col=2)

    fig_strategy.add_trace(go.Bar(
        x=idx_1based,
        y=full_v_ratios_s2,
        marker_color=speed_colors,
        text=[f'{v:.2f}' for v in full_v_ratios_s2],
        textposition='outside',
        hovertext=speed_hover,
        hoverinfo='text',
        name='速度比',
        showlegend=False,
    ), row=2, col=1)

    rr_y = np.maximum(rerouted_flags.astype(float), 0.05)
    fig_strategy.add_trace(go.Bar(
        x=idx_1based,
        y=rr_y,
        marker_color=np.where(rerouted_flags, '#D96B27', '#BDBDBD'),
        marker_line=dict(color='black', width=0.5),
        text=['改' if flag else '-' for flag in rerouted_flags],
        textposition='outside',
        customdata=list(range(N_PLANS)),
        hovertext=reroute_hover,
        hoverinfo='text',
        name='改航状态',
        showlegend=False,
    ), row=2, col=2)

    delay_ymax = max(float(np.max(delays / 60.0)) * 1.25, 1.0)
    speed_ymin = min(0.75, float(np.min(full_v_ratios_s2)) - 0.05)
    speed_ymax = max(1.25, float(np.max(full_v_ratios_s2)) + 0.18)
    fig_strategy.add_trace(go.Scatter(
        x=[15.5, 15.5], y=[0, delay_ymax], mode='lines',
        line=dict(color='black', dash='dash', width=1),
        hoverinfo='skip', showlegend=False,
    ), row=1, col=2)
    fig_strategy.add_trace(go.Scatter(
        x=[0.5, N_PLANS + 0.5], y=[1.0, 1.0], mode='lines',
        line=dict(color='black', dash='dash', width=1),
        hoverinfo='skip', showlegend=False,
    ), row=2, col=1)
    fig_strategy.add_trace(go.Scatter(
        x=[15.5, 15.5], y=[speed_ymin, speed_ymax], mode='lines',
        line=dict(color='black', dash='dash', width=1),
        hoverinfo='skip', showlegend=False,
    ), row=2, col=1)
    fig_strategy.add_trace(go.Scatter(
        x=[15.5, 15.5], y=[-0.2, 1.45], mode='lines',
        line=dict(color='black', dash='dash', width=1),
        hoverinfo='skip', showlegend=False,
    ), row=2, col=2)

    fig_strategy.update_xaxes(title_text='航班序号', tickmode='linear', dtick=5, row=1, col=2)
    fig_strategy.update_xaxes(title_text='航班序号', tickmode='linear', dtick=5, row=2, col=1)
    fig_strategy.update_xaxes(title_text='航班序号', tickmode='linear', dtick=5, row=2, col=2)
    fig_strategy.update_yaxes(title_text='延误时间 (min)', row=1, col=2)
    fig_strategy.update_yaxes(title_text='速度比 v / v_ref', row=2, col=1)
    fig_strategy.update_yaxes(range=[speed_ymin, speed_ymax], row=2, col=1)
    fig_strategy.update_yaxes(title_text='是否改航', tickvals=[0, 1], ticktext=['否', '是'],
                              range=[-0.2, 1.45], row=2, col=2)
    fig_strategy.update_layout(
        title=dict(
            text=(f'改进FATA算法（三维调度） | 40架ADS-B航班 | 混合调度场景<br>'
                  f'<sup>网格 {GX}x{GY}x{GZ}（1km/格，305m/层） | '
                  f'冲突 {len(initial_conflicts)} → {len(final_conflicts)} | '
                  f'消解率 {reduction_pct:.1f}%</sup>'),
            x=0.5,
        ),
        scene=dict(
            xaxis_title='经度',
            yaxis_title='纬度',
            zaxis_title='高度 (km)',
            bgcolor='#F7F9FC',
        ),
        hovermode='closest',
        clickmode='event+select',
        paper_bgcolor='#F5F7FA',
        plot_bgcolor='#FFFFFF',
        font=dict(family='Microsoft YaHei, Arial, sans-serif', size=12),
        margin=dict(l=50, r=30, t=95, b=45),
        height=900,
        legend=dict(orientation='h', x=0.02, y=1.02),
    )
    reroute_payload_json = json.dumps(reroute_payload, ensure_ascii=False)
    overview_route_indices_json = json.dumps(list(range(N_PLANS)))
    post_script = f"""
    const reroutePayload = {reroute_payload_json};
    const overviewRouteIndices = {overview_route_indices_json};
    const comparisonTraceStart = {comparison_trace_start};
    const rerouteTraceIndex = {reroute_trace_index};
    const gd = document.getElementById('{{plot_id}}');

    function showRerouteComparison(flightIndex) {{
        const payload = reroutePayload[flightIndex];
        if (!payload) {{
            window.alert('该航班没有局部改航记录。');
            return;
        }}

        Plotly.restyle(gd, {{visible: false}}, overviewRouteIndices);
        Plotly.restyle(gd, {{
            x: [payload.original.x],
            y: [payload.original.y],
            z: [payload.original.z],
            visible: true,
            name: '原始航迹 ' + payload.sid
        }}, [comparisonTraceStart]);
        Plotly.restyle(gd, {{
            x: [payload.rerouted.x],
            y: [payload.rerouted.y],
            z: [payload.rerouted.z],
            visible: true,
            name: '改航后航迹 ' + payload.sid
        }}, [comparisonTraceStart + 1]);
        Plotly.restyle(gd, {{
            x: [payload.markers.x],
            y: [payload.markers.y],
            z: [payload.markers.z],
            text: [payload.markers.x.map(() => '改航点')],
            customdata: [payload.markers.text],
            visible: true
        }}, [comparisonTraceStart + 2]);
        Plotly.relayout(gd, {{
            'scene.xaxis.title.text': '经度',
            'scene.yaxis.title.text': '纬度',
            'scene.zaxis.title.text': '高度 (km)',
            'annotations[0].text': '改航对比：航班 ' + payload.sid + '<br><sup>蓝色线=原始ADS-B航迹，橙色线=平滑改航后航迹，红钻=改航点</sup>'
        }});
    }}

    gd.on('plotly_click', function(evt) {{
        if (!evt || !evt.points || !evt.points.length) return;
        const pt = evt.points[0];
        if (pt.curveNumber !== rerouteTraceIndex) return;
        const flightIndex = (pt.customdata !== undefined && pt.customdata !== null)
            ? Number(pt.customdata)
            : Number(pt.pointIndex);
        showRerouteComparison(flightIndex);
    }});
    """
    fig_strategy.write_html('strategy_overview.html', post_script=post_script)
    print("Saved: strategy_overview.html")

    plans_by_id = {p.plan_id: p for p in plans}
    plans_final_by_id = {p.plan_id: p for p in plans_final}
    fid_to_idx   = {fid: i for i, fid in enumerate(flight_ids)}

    def _get_conflict_3d(conflict_list, plan_lookup):
        """Locate conflict dots from the same near-miss rule used by conflict detection."""
        d_lon, d_lat, d_alt, d_txt = [], [], [], []
        pairs = []

        for pid_a, pid_b in conflict_list:
            plan_a = plan_lookup.get(pid_a)
            plan_b = plan_lookup.get(pid_b)
            if plan_a is None or plan_b is None:
                continue
            if not plan_a.waypoints or not plan_b.waypoints:
                continue
            i = fid_to_idx[pid_a];  j = fid_to_idx[pid_b]

            wa = plan_a.waypoints
            wb = plan_b.waypoints
            xa = np.array([wp.x for wp in wa], dtype=float)
            ya = np.array([wp.y for wp in wa], dtype=float)
            za = np.array([wp.z for wp in wa], dtype=float)
            ta = np.array([wp.t_eta for wp in wa], dtype=float)
            xb = np.array([wp.x for wp in wb], dtype=float)
            yb = np.array([wp.y for wp in wb], dtype=float)
            zb = np.array([wp.z for wp in wb], dtype=float)
            tb = np.array([wp.t_eta for wp in wb], dtype=float)

            dxy = np.sqrt((xa[:, None] - xb[None, :]) ** 2 +
                          (ya[:, None] - yb[None, :]) ** 2)
            dz = np.abs(za[:, None] - zb[None, :])
            dt = np.abs(ta[:, None] - tb[None, :])
            t_req = _Z_ALPHA * (SIGMA_BASE + SIGMA_BASE) + T_CONFLICT
            ia, ib = np.where((dxy <= H_SEP) & (dz <= V_SEP) & (dt <= t_req))
            if len(ia) == 0:
                ia, ib = np.where((dxy <= H_SEP) & (dz <= V_SEP))
            if len(ia) == 0:
                continue

            best = int(np.argmin(dt[ia, ib]))
            wp_a = wa[int(ia[best])]
            wp_b = wb[int(ib[best])]
            la1, lo1, al1 = grid_to_geo(wp_a.x, wp_a.y, wp_a.z)
            la2, lo2, al2 = grid_to_geo(wp_b.x, wp_b.y, wp_b.z)

            txt = (
                f"冲突对: {sids[i]} ↔ {sids[j]}"
                f"<br>格点A: ({wp_a.x},{wp_a.y},{wp_a.z})"
                f"<br>格点B: ({wp_b.x},{wp_b.y},{wp_b.z})"
                f"<br>时间差: {abs(wp_a.t_eta - wp_b.t_eta):.1f}s"
            )
            d_lon += [lo1, lo2]
            d_lat += [la1, la2]
            d_alt += [al1, al2]
            d_txt += [txt, txt]
            pairs.append(dict(
                lon1=lo1, lat1=la1, alt1=al1,
                lon2=lo2, lat2=la2, alt2=al2,
                si=sids[i], sj=sids[j], fi=i, fj=j,
            ))

        return (d_lon, d_lat, d_alt, d_txt), pairs

    c_dots, c_pairs_3d             = _get_conflict_3d(initial_conflicts, plans_by_id)
    c_dots_after, c_pairs_3d_after = _get_conflict_3d(final_conflicts, plans_final_by_id)
    print(f"  3D before conflict pairs drawn: {len(c_pairs_3d)} / {len(initial_conflicts)}")
    print(f"  3D after conflict pairs drawn : {len(c_pairs_3d_after)} / {len(final_conflicts)}")

    _scene = dict(
        xaxis=dict(title='<b>经度</b>', showgrid=True,
                   gridcolor='rgba(0,0,0,0.15)', color='#1A1A1A',
                   backgroundcolor='#F0F4F8'),
        yaxis=dict(title='<b>纬度</b>', showgrid=True,
                   gridcolor='rgba(0,0,0,0.15)', color='#1A1A1A',
                   backgroundcolor='#F0F4F8'),
        zaxis=dict(title='<b>高度 (ft)</b>', showgrid=True,
                   gridcolor='rgba(0,0,0,0.15)', color='#1A1A1A',
                   backgroundcolor='#F0F4F8'),
        bgcolor='#FFFFFF',
    )
    _layout_base = dict(
        paper_bgcolor='#F5F7FA',
        font=dict(color='#1A1A1A',
                  family='Microsoft YaHei, Arial, sans-serif', size=14),
        legend=dict(
            x=1.01, y=0.95,
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='rgba(0,0,0,0.25)', borderwidth=1,
            font=dict(size=12, color='#1A1A1A'),
        ),
        margin=dict(r=250, t=100, b=70, l=10),
    )

    def _dashed_3d(lon, lat, alt, seg=10, gap=5):
        xd, yd, zd = [], [], []
        n = len(lon)
        i = 0
        while i < n:
            e = min(i + seg, n)
            xd += list(lon[i:e]) + [None]
            yd += list(lat[i:e]) + [None]
            zd += list(alt[i:e]) + [None]
            i = e + gap
        return xd, yd, zd

    # ══════════════════════════════════════════════════════════════════════════
    # 图一：调度前（基线）
    # ══════════════════════════════════════════════════════════════════════════
    fig1 = go.Figure()

    for i in range(N_PLANS):
        lat, lon, alt = sampled[i]
        fig1.add_trace(go.Scatter3d(
            x=lon, y=lat, z=alt, mode='lines',
            line=dict(color=CP[i], width=6),
            name=f'航班 {sids[i]}', legendgroup=f'r{i}', showlegend=True,
            hovertemplate=(f'航班: {flight_ids[i]}<br>'
                           f'经度: %{{x:.3f}}<br>纬度: %{{y:.3f}}<br>'
                           f'高度: %{{z:.0f}} ft<extra></extra>'),
        ))

    for i in range(N_PLANS):
        lat, lon, alt = sampled[i]
        m = len(lat) // 2
        fig1.add_trace(go.Scatter3d(
            x=[lon[m]], y=[lat[m]], z=[alt[m] + 800],
            mode='text', text=[sids[i]],
            textfont=dict(size=16, color=CP[i]),
            name=f'编号{sids[i]}', showlegend=False, hoverinfo='skip',
        ))

    dl, dlt, da, dh = c_dots
    fig1.add_trace(go.Scatter3d(
        x=dl, y=dlt, z=da, mode='markers',
        marker=dict(size=3, color='#E74C3C', symbol='circle',
                    line=dict(width=1, color='#7B241C')),
        name='冲突点', text=dh,
        hovertemplate='%{text}<extra></extra>', showlegend=True,
    ))

    for cp in c_pairs_3d:
        n_seg   = 30
        seg_lon = np.linspace(cp['lon1'], cp['lon2'], n_seg)
        seg_lat = np.linspace(cp['lat1'], cp['lat2'], n_seg)
        seg_alt = np.linspace(cp['alt1'], cp['alt2'], n_seg)
        xd, yd, zd = _dashed_3d(seg_lon, seg_lat, seg_alt, seg=4, gap=3)
        fig1.add_trace(go.Scatter3d(
            x=xd, y=yd, z=zd, mode='lines',
            line=dict(color='#E74C3C', width=3),
            name=f"冲突连线 {cp['si']}↔{cp['sj']}",
            showlegend=False, legendgroup='cl',
            hovertemplate=f"冲突连线: {cp['si']} ↔ {cp['sj']}<extra></extra>",
        ))

    grp1_routes = list(range(N_PLANS))
    grp1_labels = list(range(N_PLANS, 2 * N_PLANS))
    grp1_dots   = [2 * N_PLANS]
    grp1_lines  = list(range(2 * N_PLANS + 1, 2 * N_PLANS + 1 + len(c_pairs_3d)))

    conflict_panel = '<b>冲突对列表</b><br>'
    for cp in c_pairs_3d:
        conflict_panel += (
            f'<span style="color:{CP[cp["fi"]]}">■ {cp["si"]}</span>'
            f' ─── '
            f'<span style="color:{CP[cp["fj"]]}">■ {cp["sj"]}</span><br>'
        )

    fig1.update_layout(
        **_layout_base, scene=_scene,
        title=dict(
            text=('<b>调度前航线分布（基线）</b><br>'
                  '<sup>所有航班同时出发 · 红点=格点冲突位置 · 虚线=冲突连线</sup>'),
            font=dict(size=18, color='#1A1A1A'),
        ),
        updatemenus=[dict(
            type='buttons', direction='down',
            x=1.17, y=0.88, xanchor='left', yanchor='top',
            pad=dict(r=8, t=8, b=8),
            bgcolor='rgba(240,244,248,0.95)',
            bordercolor='rgba(0,0,0,0.30)',
            font=dict(color='#1A1A1A', size=13),
            buttons=[
                dict(label='✓  航线轨迹', method='restyle',
                     args=[{'visible': True},  grp1_routes],
                     args2=[{'visible': False}, grp1_routes]),
                dict(label='✓  航班编号', method='restyle',
                     args=[{'visible': True},  grp1_labels],
                     args2=[{'visible': False}, grp1_labels]),
                dict(label='✓  冲突红点', method='restyle',
                     args=[{'visible': True},  grp1_dots],
                     args2=[{'visible': False}, grp1_dots]),
                dict(label='✓  冲突连线', method='restyle',
                     args=[{'visible': True},  grp1_lines],
                     args2=[{'visible': False}, grp1_lines]),
                dict(label='✓  冲突信息', method='relayout',
                     args=[{'annotations[0].opacity': 1.0}],
                     args2=[{'annotations[0].opacity': 0.0}]),
            ],
        )],
        annotations=[dict(
            text=conflict_panel,
            align='left', showarrow=False,
            xref='paper', yref='paper', x=1.01, y=0.02,
            xanchor='left', yanchor='bottom',
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='rgba(0,0,0,0.25)', borderwidth=1,
            font=dict(size=12, color='#1A1A1A'), opacity=1.0,
        )],
    )
    fig1.write_html('fata_3d_before.html')
    print("Saved: fata_3d_before.html")

    # ══════════════════════════════════════════════════════════════════════════
    # 图二：调度后（FATA 两阶段优化结果）
    # ══════════════════════════════════════════════════════════════════════════
    fig2 = go.Figure()

    ORIG_ALT_OFFSET = -2000

    # 原始轨迹虚线（下移 2000 ft 对比显示）
    for i in range(N_PLANS):
        lat, lon, alt = sampled[i]
        xd, yd, zd = _dashed_3d(lon, lat, alt + ORIG_ALT_OFFSET, seg=10, gap=5)
        fig2.add_trace(go.Scatter3d(
            x=xd, y=yd, z=zd, mode='lines',
            line=dict(color=CP[i], width=3),
            name=f'原始 {sids[i]}', legendgroup=f'orig{i}',
            showlegend=False,
            hovertemplate=(f'航班 {sids[i]} 原始轨迹（下移2000ft显示）<extra></extra>'),
        ))

    # 优化后航线实线
    for i in range(N_PLANS):
        lat, lon, alt = sampled[i]
        off_s  = full_offsets_s2[i]
        vr_str = f"{full_v_ratios_s2[i]:.2f}"
        fig2.add_trace(go.Scatter3d(
            x=lon, y=lat, z=alt, mode='lines',
            line=dict(color=CP[i], width=6),
            name=f'航班 {sids[i]}', legendgroup=f'opt{i}', showlegend=True,
            hovertemplate=(
                f'航班: {flight_ids[i]}<br>'
                f'调度策略: {strat_zh[i]}<br>'
                f'起飞偏移: {off_s:+.1f} s<br>'
                f'速度比: {vr_str}<br>'
                f'经度: %{{x:.3f}}<br>纬度: %{{y:.3f}}<br>'
                f'高度: %{{z:.0f}} ft<extra></extra>'
            ),
        ))

    # 航班编号标签
    for i in range(N_PLANS):
        lat, lon, alt = sampled[i]
        m = len(lat) // 2
        fig2.add_trace(go.Scatter3d(
            x=[lon[m]], y=[lat[m]], z=[alt[m] + 800],
            mode='text', text=[sids[i]],
            textfont=dict(size=16, color=CP[i]),
            name=f'编号{sids[i]}', showlegend=False, hoverinfo='skip',
        ))

    # 调度策略标签
    STRAT_COLOR = {
        '无需调整': '#27AE60',
        '速度调整': '#E67E22',
        '延迟起飞': '#E74C3C',
        '局部改航': '#8E44AD',
    }
    for i in range(N_PLANS):
        lat, lon, alt = sampled[i]
        q  = len(lat) * 3 // 4
        sc = STRAT_COLOR[strat_zh[i]]
        fig2.add_trace(go.Scatter3d(
            x=[lon[q]], y=[lat[q]], z=[alt[q] + 1600],
            mode='text', text=[f'[{strat_zh[i]}]'],
            textfont=dict(size=15, color=sc),
            name=f'策略{sids[i]}', showlegend=False,
            hovertemplate=(
                f'航班 {sids[i]}<br>'
                f'策略: {strat_zh[i]}<br>'
                f'偏移: {full_offsets_s2[i]:+.1f} s<extra></extra>'
            ),
        ))

    # 残余冲突点
    dl_a, dlt_a, da_a, dh_a = c_dots_after
    has_remain = len(dl_a) > 0
    fig2.add_trace(go.Scatter3d(
        x=dl_a, y=dlt_a, z=da_a, mode='markers',
        marker=dict(size=4, color='#FF6B35', symbol='diamond',
                    line=dict(width=1, color='#C0392B')),
        name='残余冲突点' if has_remain else '无残余冲突',
        text=dh_a, hovertemplate='%{text}<extra></extra>',
        showlegend=has_remain, visible=True,
    ))

    grp2_orig   = list(range(N_PLANS))
    grp2_opt    = list(range(N_PLANS,   2 * N_PLANS))
    grp2_labels = list(range(2 * N_PLANS, 3 * N_PLANS))
    grp2_strat  = list(range(3 * N_PLANS, 4 * N_PLANS))
    grp2_remain = [4 * N_PLANS]

    strat_legend = (
        '<b>调度策略说明</b><br>'
        f'<span style="color:{STRAT_COLOR["无需调整"]}">■</span>'
        ' 无需调整（偏移 &lt; 30 s）<br>'
        f'<span style="color:{STRAT_COLOR["速度调整"]}">■</span>'
        ' 速度调整（速度比 ≠ 1）<br>'
        f'<span style="color:{STRAT_COLOR["延迟起飞"]}">■</span>'
        ' 延迟起飞（偏移 ≥ 30 s）<br>'
        f'<span style="color:{STRAT_COLOR["局部改航"]}">■</span>'
        ' 局部改航（格点偏移 dx/dy/dz）<br>'
        '<br><b>优化结果</b><br>'
        f'冲突: {len(initial_conflicts)} → {len(final_conflicts)}<br>'
        f'减少: {reduction_pct:.1f} %'
    )

    fig2.update_layout(
        **_layout_base, scene=_scene,
        title=dict(
            text=('<b>调度后航线分布（FATA 两阶段优化）</b><br>'
                  '<sup>粗实线=优化结果 · 虚线=原始轨迹（下移2000ft对比）· 彩色标注=调度策略</sup>'),
            font=dict(size=18, color='#1A1A1A'),
        ),
        updatemenus=[dict(
            type='buttons', direction='down',
            x=1.17, y=0.92, xanchor='left', yanchor='top',
            pad=dict(r=8, t=8, b=8),
            bgcolor='rgba(240,244,248,0.95)',
            bordercolor='rgba(0,0,0,0.30)',
            font=dict(color='#1A1A1A', size=13),
            buttons=[
                dict(label='✓  原始轨迹', method='restyle',
                     args=[{'visible': True},  grp2_orig],
                     args2=[{'visible': False}, grp2_orig]),
                dict(label='✓  优化航线', method='restyle',
                     args=[{'visible': True},  grp2_opt],
                     args2=[{'visible': False}, grp2_opt]),
                dict(label='✓  航班编号', method='restyle',
                     args=[{'visible': True},  grp2_labels],
                     args2=[{'visible': False}, grp2_labels]),
                dict(label='✓  调度方式', method='restyle',
                     args=[{'visible': True},  grp2_strat],
                     args2=[{'visible': False}, grp2_strat]),
                dict(label='✓  残余冲突', method='restyle',
                     args=[{'visible': True},  grp2_remain],
                     args2=[{'visible': False}, grp2_remain]),
                dict(label='✓  策略说明', method='relayout',
                     args=[{'annotations[0].opacity': 1.0}],
                     args2=[{'annotations[0].opacity': 0.0}]),
            ],
        )],
        annotations=[dict(
            text=strat_legend,
            align='left', showarrow=False,
            xref='paper', yref='paper', x=1.01, y=0.02,
            xanchor='left', yanchor='bottom',
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='rgba(0,0,0,0.25)', borderwidth=1,
            font=dict(size=12, color='#1A1A1A'), opacity=1.0,
        )],
    )
    fig2.write_html('fata_3d_after.html')
    print("Saved: fata_3d_after.html")

    print("\n交互式3D图已生成，用浏览器打开查看：")
    print("  调度前: fata_3d_before.html")
    print("  调度后: fata_3d_after.html")

_t_total = time.perf_counter() - _t_total_start
print(f"\n{'═'*56}")
print(f"  总运行时间: {_hms(_t_total)}")
print(f"{'═'*56}")
print("\nAll done.")
