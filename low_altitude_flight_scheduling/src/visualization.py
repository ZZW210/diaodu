from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .conflict_detection import Conflict
from .flight_plan import FlightPlan, plot_routes_3d
from .grid import AirspaceGrid
from .utils import ensure_dir


def _plan_label(plan_id: int, one_based: bool = False) -> str:
    display_id = int(plan_id) + 1 if one_based else int(plan_id)
    return str(display_id)


def _world_points(grid: AirspaceGrid, cells: np.ndarray | list[tuple[int, int, int]]) -> np.ndarray:
    pts = np.asarray(cells, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 3)
    return (pts + 0.5) * np.asarray(grid.cell_size, dtype=float)


def _obstacle_cuboid_mesh(grid: AirspaceGrid) -> dict[str, list[float] | list[int]]:
    obstacles = np.asarray(grid.obstacles, dtype=bool)
    occupied = {tuple(int(v) for v in cell) for cell in np.argwhere(obstacles)}
    if not occupied:
        return {"x": [], "y": [], "z": [], "i": [], "j": [], "k": []}

    sx, sy, sz = (float(v) for v in grid.cell_size)
    faces = [
        ((-1, 0, 0), [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]),
        ((1, 0, 0), [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]),
        ((0, -1, 0), [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)]),
        ((0, 1, 0), [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]),
        ((0, 0, -1), [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]),
        ((0, 0, 1), [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]),
    ]
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    ii: list[int] = []
    jj: list[int] = []
    kk: list[int] = []

    for x, y, z in sorted(occupied):
        for delta, offsets in faces:
            neighbor = (x + delta[0], y + delta[1], z + delta[2])
            if neighbor in occupied:
                continue
            base = len(xs)
            for ox, oy, oz in offsets:
                xs.append((x + ox) * sx)
                ys.append((y + oy) * sy)
                zs.append((z + oz) * sz)
            ii.extend([base, base])
            jj.extend([base + 1, base + 2])
            kk.extend([base + 2, base + 3])

    return {"x": xs, "y": ys, "z": zs, "i": ii, "j": jj, "k": kk}


def _obstacle_edge_lines(grid: AirspaceGrid, max_edges: int = 20000) -> tuple[list[float | None], list[float | None], list[float | None]]:
    obstacles = np.asarray(grid.obstacles, dtype=bool)
    occupied = {tuple(int(v) for v in cell) for cell in np.argwhere(obstacles)}
    if not occupied:
        return [], [], []

    sx, sy, sz = (float(v) for v in grid.cell_size)
    faces = [
        ((-1, 0, 0), [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]),
        ((1, 0, 0), [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]),
        ((0, -1, 0), [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)]),
        ((0, 1, 0), [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]),
        ((0, 0, -1), [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]),
        ((0, 0, 1), [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]),
    ]
    face_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    edges: set[tuple[tuple[float, float, float], tuple[float, float, float]]] = set()

    for x, y, z in sorted(occupied):
        for delta, offsets in faces:
            neighbor = (x + delta[0], y + delta[1], z + delta[2])
            if neighbor in occupied:
                continue
            vertices = [((x + ox) * sx, (y + oy) * sy, (z + oz) * sz) for ox, oy, oz in offsets]
            for a_idx, b_idx in face_edges:
                a, b = vertices[a_idx], vertices[b_idx]
                edges.add((a, b) if a <= b else (b, a))

    edge_list = sorted(edges)
    if len(edge_list) > max_edges:
        rng = np.random.default_rng(57)
        keep = np.sort(rng.choice(np.arange(len(edge_list)), max_edges, replace=False))
        edge_list = [edge_list[int(idx)] for idx in keep]

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for a, b in edge_list:
        xs.extend([a[0], b[0], None])
        ys.extend([a[1], b[1], None])
        zs.extend([a[2], b[2], None])
    return xs, ys, zs


def _route_display_points(grid: AirspaceGrid, plan: FlightPlan, include_ground: bool = True, jitter: bool = True) -> np.ndarray:
    pts = _world_points(grid, plan.path).astype(float, copy=True)
    if jitter and len(pts) > 1:
        rng = np.random.default_rng(2029 + plan.id * 7919 + len(pts))
        cell_z = float(grid.cell_size[2])
        t = np.linspace(0.0, 1.0, len(pts))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        layer_offset = float(rng.uniform(-0.24, 0.24) * cell_z)
        wave = np.sin((2.0 + float(rng.uniform(-0.5, 1.2))) * np.pi * t + phase)
        fine = rng.normal(0.0, 0.05 * cell_z, len(pts))
        taper = 0.35 + 0.65 * np.sin(np.pi * t)
        pts[:, 2] += layer_offset + taper * (0.18 * cell_z * wave + fine)
        min_alt = max(1.0, 0.35 * cell_z)
        max_alt = grid.shape[2] * cell_z - 0.15 * cell_z
        pts[:, 2] = np.clip(pts[:, 2], min_alt, max_alt)
    if not include_ground or len(pts) == 0:
        return pts
    return pts


def _ground_endpoint_points(grid: AirspaceGrid, plan: FlightPlan, ground_z_m: float = 0.0) -> np.ndarray:
    return np.asarray(
        [
            [(plan.start[0] + 0.5) * grid.cell_size[0], (plan.start[1] + 0.5) * grid.cell_size[1], ground_z_m],
            [(plan.goal[0] + 0.5) * grid.cell_size[0], (plan.goal[1] + 0.5) * grid.cell_size[1], ground_z_m],
        ],
        dtype=float,
    )


def _endpoint_connector_points(ground_pts: np.ndarray, route_pts: np.ndarray) -> tuple[list[float | None], list[float | None], list[float | None]]:
    if len(route_pts) == 0:
        return [], [], []
    x = [ground_pts[0, 0], route_pts[0, 0], None, ground_pts[1, 0], route_pts[-1, 0]]
    y = [ground_pts[0, 1], route_pts[0, 1], None, ground_pts[1, 1], route_pts[-1, 1]]
    z = [ground_pts[0, 2], route_pts[0, 2], None, ground_pts[1, 2], route_pts[-1, 2]]
    return x, y, z


def plot_conflict_points(grid: AirspaceGrid, plans: list[FlightPlan], conflicts: list[Conflict], path: str | Path, title: str) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for plan in plans:
        pts = _route_display_points(grid, plan)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#7aa6c2", linewidth=0.5, alpha=0.25)
    if conflicts:
        pts = _world_points(grid, [c.cell for c in conflicts])
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="#e45756", s=36, alpha=0.88, label="conflict")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("altitude (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_key_flights(grid: AirspaceGrid, plans: list[FlightPlan], key_ids: list[int], path: str | Path) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    key = set(key_ids)
    for plan in plans:
        pts = _route_display_points(grid, plan)
        if plan.id in key:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2.2, alpha=0.95, label=_plan_label(plan.id) if len(key) <= 12 else None)
        else:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#aeb7c2", linewidth=0.45, alpha=0.22)
    ax.set_title("Key flight plans selected by CI")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("altitude (m)")
    if len(key) <= 12:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_before_after(grid: AirspaceGrid, before: list[FlightPlan], after: list[FlightPlan], path: str | Path, title: str) -> None:
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    changed = {p.id for p in after if p.changed or p.rerouted or p.delay > 1e-6}
    after_map = {p.id: p for p in after}
    for plan in before:
        pts = _route_display_points(grid, plan)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#9da9b5", linewidth=0.45, alpha=0.18)
        if plan.id in changed:
            pts2 = _route_display_points(grid, after_map[plan.id])
            ax.plot(pts2[:, 0], pts2[:, 1], pts2[:, 2], linewidth=1.6, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("altitude (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_fata_convergence(curve: list[float], path: str | Path, title: str = "Improved FATA convergence") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(range(1, len(curve) + 1), curve, color="#4c78a8", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sensitivity(df: pd.DataFrame, x: str, ys: list[str], path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for y in ys:
        if y in df:
            ax.plot(df[x], df[y], marker="o", label=y)
    ax.set_xlabel(x)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_fata_3d_html(
    grid: AirspaceGrid,
    plans: list[FlightPlan],
    conflicts: list[Conflict],
    output_path: str | Path,
    title: str,
    key_ids: list[int] | None = None,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    title = _clean_fata_title(title, output_path)
    try:
        import plotly.graph_objects as go
    except Exception:
        _write_html_fallback(output_path, title)
        return

    key = set(key_ids or [])
    plan_map = {plan.id: plan for plan in plans}
    no_ground_points = {plan.id: _route_display_points(grid, plan, include_ground=False) for plan in plans}
    cell_indices = {
        plan.id: {cell: idx for idx, cell in enumerate(plan.path)}
        for plan in plans
    }
    fig = go.Figure()
    obstacle_indices: list[int] = []
    obstacle_index = len(fig.data)
    obstacle_mesh = _obstacle_cuboid_mesh(grid)
    obstacle_indices.append(obstacle_index)
    fig.add_trace(
        go.Mesh3d(
            x=obstacle_mesh["x"],
            y=obstacle_mesh["y"],
            z=obstacle_mesh["z"],
            i=obstacle_mesh["i"],
            j=obstacle_mesh["j"],
            k=obstacle_mesh["k"],
            color="#FFFFFF",
            opacity=1.0,
            flatshading=True,
            lighting=dict(ambient=0.92, diffuse=0.28, roughness=0.96),
            name="障碍物/禁飞白膜",
            hoverinfo="skip",
            showlegend=True,
        )
    )
    palette = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#72b7b2", "#e45756", "#ff9da6", "#9d755d"]
    route_indices: list[int] = []
    endpoint_indices: list[int] = []
    route_trace_by_plan: dict[int, int] = {}
    endpoint_trace_by_plan: dict[int, list[int]] = {}
    shown_key_route_legend = False
    shown_regular_route_legend = False
    for plan in plans:
        display_pts = _route_display_points(grid, plan)
        label = _plan_label(plan.id)
        is_key = plan.id in key
        route_name = "关键计划" if is_key else "普通航线"
        show_route_legend = (is_key and not shown_key_route_legend) or ((not is_key) and not shown_regular_route_legend)
        shown_key_route_legend = shown_key_route_legend or is_key
        shown_regular_route_legend = shown_regular_route_legend or (not is_key)
        width = 5 if is_key else 3
        opacity = 0.95 if is_key else 0.58
        route_trace_index = len(fig.data)
        route_indices.append(route_trace_index)
        route_trace_by_plan[plan.id] = route_trace_index
        fig.add_trace(
            go.Scatter3d(
                x=display_pts[:, 0],
                y=display_pts[:, 1],
                z=display_pts[:, 2],
                mode="lines",
                line=dict(width=width, color=palette[plan.id % len(palette)]),
                opacity=opacity,
                name=route_name,
                legendgroup=route_name,
                hovertemplate="飞行计划 %{text}<br>X: %{x:.0f} m<br>Y: %{y:.0f} m<br>高度: %{z:.0f} m<extra></extra>",
                text=[label] * len(display_pts),
                customdata=[plan.id] * len(display_pts),
                showlegend=show_route_legend,
            )
        )
        if plan.id in key or plan.id < 8:
            ground_pts = _ground_endpoint_points(grid, plan, ground_z_m=0.0)
            endpoint_index = len(fig.data)
            endpoint_indices.append(endpoint_index)
            endpoint_trace_by_plan.setdefault(plan.id, []).append(endpoint_index)
            fig.add_trace(
                go.Scatter3d(
                    x=ground_pts[:, 0],
                    y=ground_pts[:, 1],
                    z=ground_pts[:, 2],
                    mode="markers",
                    marker=dict(size=6, color=[palette[plan.id % len(palette)], "#111111"], symbol=["circle", "x"]),
                    name=f"{label} 地面起终点",
                    hovertext=[f"{label} 地面起点", f"{label} 地面终点"],
                    hovertemplate="%{hovertext}<br>X: %{x:.0f} m<br>Y: %{y:.0f} m<br>高度: %{z:.0f} m<extra></extra>",
                    showlegend=False,
                )
            )
            connector_x, connector_y, connector_z = _endpoint_connector_points(ground_pts, display_pts)
            if connector_x:
                endpoint_index = len(fig.data)
                endpoint_indices.append(endpoint_index)
                endpoint_trace_by_plan.setdefault(plan.id, []).append(endpoint_index)
                fig.add_trace(
                    go.Scatter3d(
                        x=connector_x,
                        y=connector_y,
                        z=connector_z,
                        mode="lines",
                        line=dict(width=1.4, color=palette[plan.id % len(palette)]),
                        opacity=0.72,
                        name=f"{label} 起降连接",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
    conflict_point_indices: list[int] = []
    conflict_line_indices: list[int] = []
    conflict_line_indices_by_plan: dict[int, list[int]] = {}
    conflict_neighbors: dict[int, set[int]] = {plan.id: set() for plan in plans}
    all_conflict_point_payload: list[list[float | str]] = []
    conflict_points_by_plan: dict[int, list[list[float | str]]] = {plan.id: [] for plan in plans}
    conflict_point_trace_index: int | None = None
    if conflicts:
        pts = _world_points(grid, [c.cell for c in conflicts])
        for conflict, pt in zip(conflicts, pts):
            a_label = _plan_label(conflict.flight_a)
            b_label = _plan_label(conflict.flight_b)
            label = (
                f"冲突: {a_label} ↔ {b_label}"
                f"<br>时间差: {conflict.time_diff:.1f}s"
                f"<br>要求间隔: {conflict.required_sep:.1f}s"
            )
            record: list[float | str] = [float(pt[0]), float(pt[1]), float(pt[2]), label]
            all_conflict_point_payload.append(record)
            conflict_points_by_plan.setdefault(conflict.flight_a, []).append(record)
            conflict_points_by_plan.setdefault(conflict.flight_b, []).append(record)
            conflict_neighbors.setdefault(conflict.flight_a, set()).add(conflict.flight_b)
            conflict_neighbors.setdefault(conflict.flight_b, set()).add(conflict.flight_a)
        conflict_point_trace_index = len(fig.data)
        conflict_point_indices.append(conflict_point_trace_index)
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=2.6, color="#e45756", symbol="circle"),
                name="冲突点",
                customdata=[record[3] for record in all_conflict_point_payload],
                hovertemplate="%{customdata}<br>X: %{x:.0f} m<br>Y: %{y:.0f} m<br>高度: %{z:.0f} m<extra></extra>",
            )
        )
        for conflict in conflicts[:120]:
            a_idx = cell_indices.get(conflict.flight_a, {}).get(conflict.cell)
            b_idx = cell_indices.get(conflict.flight_b, {}).get(conflict.cell)
            if a_idx is None or b_idx is None:
                continue
            a_pts = no_ground_points.get(conflict.flight_a)
            b_pts = no_ground_points.get(conflict.flight_b)
            if a_pts is None or b_pts is None or a_idx >= len(a_pts) or b_idx >= len(b_pts):
                continue
            pa = a_pts[a_idx]
            pb = b_pts[b_idx]
            conflict_line_index = len(fig.data)
            conflict_line_indices.append(conflict_line_index)
            conflict_line_indices_by_plan.setdefault(conflict.flight_a, []).append(conflict_line_index)
            conflict_line_indices_by_plan.setdefault(conflict.flight_b, []).append(conflict_line_index)
            a_label = _plan_label(conflict.flight_a)
            b_label = _plan_label(conflict.flight_b)
            fig.add_trace(
                go.Scatter3d(
                    x=[pa[0], pb[0]],
                    y=[pa[1], pb[1]],
                    z=[pa[2], pb[2]],
                    mode="lines",
                    line=dict(width=1.5, color="#e45756"),
                    opacity=0.42,
                    name=f"冲突连线 {a_label}↔{b_label}",
                    hovertemplate=f"冲突连线: {a_label} ↔ {b_label}<extra></extra>",
                    showlegend=False,
                )
            )
    conflict_panel = "<b>冲突对列表</b><br>"
    if conflicts:
        for conflict in conflicts[:28]:
            a_label = _plan_label(conflict.flight_a)
            b_label = _plan_label(conflict.flight_b)
            conflict_panel += (
                f"<span style='color:#e45756'>◆</span> "
                f"{a_label} ↔ {b_label} "
                f"({conflict.time_diff:.1f}s / {conflict.required_sep:.1f}s)<br>"
            )
        if len(conflicts) > 28:
            conflict_panel += f"... 共 {len(conflicts)} 个冲突<br>"
    else:
        conflict_panel += "无残余冲突<br>"
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sup>红点=冲突位置 · 航线高度含层内随机起伏 · 起终点落在地面</sup>",
            font=dict(size=18, color="#1A1A1A"),
        ),
        autosize=True,
        paper_bgcolor="#F5F7FA",
        plot_bgcolor="white",
        scene=dict(
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
            xaxis_title="<b>X 坐标 (m)</b>",
            yaxis_title="<b>Y 坐标 (m)</b>",
            zaxis_title="<b>高度 (m)</b>",
            xaxis=dict(showbackground=True, backgroundcolor="rgb(248,250,252)", gridcolor="rgba(120,120,120,0.25)"),
            yaxis=dict(showbackground=True, backgroundcolor="rgb(248,250,252)", gridcolor="rgba(120,120,120,0.25)"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.35),
            zaxis=dict(range=[0, grid.shape[2] * grid.cell_size[2]]),
            camera=dict(eye=dict(x=1.45, y=-1.65, z=1.05)),
        ),
        margin=dict(l=0, r=0, t=55, b=0),
        legend=dict(
            x=0.012,
            y=0.985,
            xanchor="left",
            yanchor="top",
            itemsizing="constant",
            traceorder="grouped",
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor="rgba(0,0,0,0.18)",
            borderwidth=1,
            font=dict(size=12, color="#1A1A1A"),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="down",
                x=0.985,
                y=0.955,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(240,244,248,0.95)",
                bordercolor="rgba(0,0,0,0.28)",
                borderwidth=1,
                buttons=[
                    dict(label="✓ 障碍物白膜", method="restyle", args=[{"visible": True}, obstacle_indices], args2=[{"visible": False}, obstacle_indices]),
                    dict(label="✓ 航线轨迹", method="restyle", args=[{"visible": True}, route_indices], args2=[{"visible": False}, route_indices]),
                    dict(label="✓ 起终点", method="restyle", args=[{"visible": True}, endpoint_indices], args2=[{"visible": False}, endpoint_indices]),
                    dict(label="✓ 冲突红点", method="restyle", args=[{"visible": True}, conflict_point_indices], args2=[{"visible": False}, conflict_point_indices]),
                    dict(label="✓ 冲突连线", method="restyle", args=[{"visible": True}, conflict_line_indices], args2=[{"visible": False}, conflict_line_indices]),
                    dict(label="✓ 冲突信息", method="relayout", args=[{"annotations[0].opacity": 1.0}], args2=[{"annotations[0].opacity": 0.0}]),
                ],
            )
        ],
        annotations=[
            dict(
                text=conflict_panel,
                align="left",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.985,
                y=0.025,
                xanchor="right",
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(0,0,0,0.22)",
                borderwidth=1,
                font=dict(size=12, color="#1A1A1A"),
                opacity=1.0,
            )
        ],
    )
    route_trace_by_plan_payload = {str(k): v for k, v in route_trace_by_plan.items()}
    plan_by_route_trace_payload = {str(v): str(k) for k, v in route_trace_by_plan.items()}
    endpoint_trace_by_plan_payload = {str(k): v for k, v in endpoint_trace_by_plan.items()}
    conflict_neighbors_payload = {str(k): sorted(v) for k, v in conflict_neighbors.items() if v}
    conflict_line_indices_by_plan_payload = {
        str(k): sorted(set(v)) for k, v in conflict_line_indices_by_plan.items() if v
    }
    conflict_points_by_plan_payload = {
        str(k): v for k, v in conflict_points_by_plan.items() if v
    }
    conflict_point_trace_js = "null" if conflict_point_trace_index is None else str(conflict_point_trace_index)
    post_script = f"""
    const routeTraceByPlan = {json.dumps(route_trace_by_plan_payload, ensure_ascii=False)};
    const planByRouteTrace = {json.dumps(plan_by_route_trace_payload, ensure_ascii=False)};
    const endpointTraceByPlan = {json.dumps(endpoint_trace_by_plan_payload, ensure_ascii=False)};
    const routeIndices = {json.dumps(route_indices)};
    const endpointIndices = {json.dumps(endpoint_indices)};
    const conflictLineIndices = {json.dumps(conflict_line_indices)};
    const conflictNeighbors = {json.dumps(conflict_neighbors_payload, ensure_ascii=False)};
    const conflictLineIndicesByPlan = {json.dumps(conflict_line_indices_by_plan_payload, ensure_ascii=False)};
    const conflictPointTraceIndex = {conflict_point_trace_js};
    const allConflictPoints = {json.dumps(all_conflict_point_payload, ensure_ascii=False)};
    const conflictPointsByPlan = {json.dumps(conflict_points_by_plan_payload, ensure_ascii=False)};
    const gd = document.getElementById('{{plot_id}}');

    const controlledTraceIndices = uniqueTraceList([...routeIndices, ...endpointIndices, ...conflictLineIndices]);
    let pendingFilterFrame = null;
    let activePlanKey = null;

    function uniqueTraceList(indices) {{
        return Array.from(new Set((indices || []).filter((idx) => idx !== null && idx !== undefined)));
    }}

    function setControlledVisibility(visibleSet) {{
        if (!controlledTraceIndices.length) return Promise.resolve();
        const traceList = controlledTraceIndices;
        const visibleValues = traceList.map((idx) => visibleSet.has(idx));
        return Plotly.restyle(gd, {{visible: visibleValues}}, traceList);
    }}

    function setConflictPoints(points) {{
        if (conflictPointTraceIndex === null) return Promise.resolve();
        const data = points || [];
        return Plotly.restyle(gd, {{
            x: [data.map((p) => p[0])],
            y: [data.map((p) => p[1])],
            z: [data.map((p) => p[2])],
            customdata: [data.map((p) => p[3])],
            visible: data.length > 0
        }}, [conflictPointTraceIndex]);
    }}

    function endpointIndicesForPlans(planIds) {{
        const out = [];
        planIds.forEach((pid) => {{
            (endpointTraceByPlan[String(pid)] || []).forEach((idx) => out.push(idx));
        }});
        return out;
    }}

    function showRouteConflictCluster(planId) {{
        const key = String(planId);
        if (activePlanKey === key) return;
        activePlanKey = key;
        const keepPlans = new Set([key, ...(conflictNeighbors[key] || []).map(String)]);
        const visibleSet = new Set();
        Object.entries(routeTraceByPlan).forEach(([pid, idx]) => {{
            if (keepPlans.has(String(pid))) visibleSet.add(idx);
        }});
        endpointIndicesForPlans(Array.from(keepPlans)).forEach((idx) => visibleSet.add(idx));
        (conflictLineIndicesByPlan[key] || []).forEach((idx) => visibleSet.add(idx));
        Promise.all([
            setControlledVisibility(visibleSet),
            setConflictPoints(conflictPointsByPlan[key] || [])
        ]);
    }}

    function resetRouteConflictFilter() {{
        activePlanKey = null;
        Promise.all([
            setControlledVisibility(new Set(controlledTraceIndices)),
            setConflictPoints(allConflictPoints)
        ]);
    }}

    function scheduleRouteFilter(planId) {{
        if (pendingFilterFrame !== null) {{
            window.cancelAnimationFrame(pendingFilterFrame);
        }}
        pendingFilterFrame = window.requestAnimationFrame(function() {{
            pendingFilterFrame = null;
            showRouteConflictCluster(planId);
        }});
    }}

    gd.on('plotly_click', function(evt) {{
        if (!evt || !evt.points || !evt.points.length) return;
        const traceId = String(evt.points[0].curveNumber);
        const planId = planByRouteTrace[traceId];
        if (planId === undefined) return;
        scheduleRouteFilter(planId);
    }});

    gd.on('plotly_doubleclick', function() {{
        resetRouteConflictFilter();
    }});
    """
    _write_plotly_fullscreen_html(fig, output_path, post_script=post_script)


def _clean_fata_title(title: str, output_path: Path) -> str:
    text = str(title)
    if text.count("?") < 3:
        return text
    if "before" in output_path.stem:
        return "\u8c03\u5ea6\u524d\u822a\u7ebf\u5206\u5e03\uff08\u57fa\u7ebf\uff09"
    if "after" in output_path.stem:
        return "\u8c03\u5ea6\u540e\u822a\u7ebf\u5206\u5e03\uff08\u4e24\u9636\u6bb5\u4f18\u5316\uff09"
    return "\u4f4e\u7a7a\u822a\u7ebf\u4e09\u7ef4\u5206\u5e03"


def write_strategy_overview_html(
    output_path: str | Path,
    metrics: dict[str, float],
    key_flights: pd.DataFrame,
    attack_df: pd.DataFrame,
    convergence: list[float],
    grid: AirspaceGrid | None = None,
    initial: list[FlightPlan] | None = None,
    optimized: list[FlightPlan] | None = None,
    conflicts: list[Conflict] | None = None,
    key_ids: list[int] | None = None,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    try:
        import plotly.graph_objects as go
    except Exception:
        _write_html_fallback(output_path, "Strategy overview")
        return

    if grid is None or initial is None:
        _write_html_fallback(output_path, "Strategy overview")
        return

    initial_map = {p.id: p for p in initial}
    optimized_map = {p.id: p for p in (optimized or [])}
    key = set(key_ids or [int(v) for v in key_flights.head(10)["flight_id"].tolist()])
    selected = sorted(initial, key=lambda p: p.id)
    selected_ids = [p.id for p in selected]
    changed = {
        p.id
        for p in optimized_map.values()
        if p.id in initial_map and (p.changed or p.rerouted or p.delay > 1e-6 or p.path != initial_map[p.id].path)
    }

    def mean_speed(plan: FlightPlan) -> float:
        return float(np.mean(plan.speed_profile or [1.0]))

    fig = go.Figure()
    overview_route_indices: list[int] = []
    shown_key_legend = False
    shown_other_legend = False
    for idx, plan in enumerate(selected):
        display_pts = _route_display_points(grid, plan)
        is_key = plan.id in key
        name = "关键计划" if is_key else "其他航线"
        color = "#C0392B" if is_key else "#7A8793"
        line_width = 5 if is_key else 3
        opacity = 0.88 if is_key else 0.30
        showlegend = (is_key and not shown_key_legend) or ((not is_key) and not shown_other_legend)
        shown_key_legend = shown_key_legend or is_key
        shown_other_legend = shown_other_legend or (not is_key)
        overview_route_indices.append(len(fig.data))
        fig.add_trace(
            go.Scatter3d(
                x=display_pts[:, 0],
                y=display_pts[:, 1],
                z=display_pts[:, 2],
                mode="lines",
                line=dict(color=color, width=line_width),
                opacity=opacity,
                name=name,
                legendgroup=name,
                showlegend=showlegend,
                scene="scene",
                text=[f"飞行计划: {plan.id}<br>起飞: {plan.etd / 60:.1f} min"] * len(display_pts),
                hovertemplate="%{text}<br>X: %{x:.0f} m<br>Y: %{y:.0f} m<br>高度: %{z:.0f} m<extra></extra>",
            )
        )

    x = [p.id for p in selected]
    labels = [_plan_label(p.id) for p in selected]
    key_ticks = [(plan.id, _plan_label(plan.id)) for plan in selected if plan.id in key]
    key_tick_vals = [pos for pos, _ in key_ticks]
    key_tick_text = [label for _, label in key_ticks]
    delays = [float(optimized_map.get(p.id, p).delay) / 60.0 for p in selected]
    speed_ratios = [
        mean_speed(optimized_map.get(p.id, p)) / max(mean_speed(p), 1e-6)
        for p in selected
    ]
    rerouted = [
        1 if p.id in optimized_map and (optimized_map[p.id].rerouted or optimized_map[p.id].path != p.path) else 0
        for p in selected
    ]
    changed_flags = [p.id in changed for p in selected]
    bar_colors = ["#C0392B" if flag else "#3366AA" for flag in changed_flags]

    fig.add_trace(
        go.Bar(
            x=x,
            y=delays,
            name="起飞延误",
            marker_color=bar_colors,
            text=labels,
            xaxis="x",
            yaxis="y",
            showlegend=False,
            hovertemplate="%{text}<br>延误: %{y:.1f} min<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=speed_ratios,
            name="速度比",
            marker_color="#16A085",
            text=labels,
            xaxis="x2",
            yaxis="y2",
            showlegend=False,
            hovertemplate="%{text}<br>速度比: %{y:.2f}<extra></extra>",
        )
    )
    reroute_trace_index = len(fig.data)
    fig.add_trace(
        go.Bar(
            x=x,
            y=rerouted,
            name="改航状态",
            marker_color=["#D68910" if v else "#BDC3C7" for v in rerouted],
            text=labels,
            customdata=selected_ids,
            xaxis="x3",
            yaxis="y3",
            showlegend=False,
            hovertemplate="%{text}<br>是否改航: %{y}<extra></extra>",
        )
    )
    if x:
        fig.add_trace(go.Scatter(x=x, y=[0.0] * len(x), mode="lines", line=dict(color="#566573", width=1), xaxis="x", yaxis="y", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=[1.0] * len(x), mode="lines", line=dict(color="#C0392B", width=1, dash="dash"), xaxis="x2", yaxis="y2", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=[0.8] * len(x), mode="lines", line=dict(color="#95A5A6", width=1, dash="dot"), xaxis="x2", yaxis="y2", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=[1.2] * len(x), mode="lines", line=dict(color="#95A5A6", width=1, dash="dot"), xaxis="x2", yaxis="y2", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=[1.0] * len(x), mode="lines", line=dict(color="#566573", width=1, dash="dash"), xaxis="x3", yaxis="y3", showlegend=False, hoverinfo="skip"))

    reroute_payload: dict[str, dict[str, object]] = {}
    for plan in selected:
        if plan.id not in optimized_map:
            continue
        after = optimized_map[plan.id]
        before_pts = _route_display_points(grid, plan)
        after_pts = _route_display_points(grid, after)
        marker_pts = []
        marker_text = []
        raw_after = _route_display_points(grid, after, include_ground=False)
        max_len = max(len(plan.path), len(after.path))
        for k in range(max_len):
            old_cell = plan.path[k] if k < len(plan.path) else None
            new_cell = after.path[k] if k < len(after.path) else None
            if old_cell == new_cell:
                continue
            idx = min(k, len(raw_after) - 1)
            marker_pts.append(raw_after[idx].tolist())
            marker_text.append(f"{_plan_label(plan.id)}<br>改航点序号: {k + 1}<br>原格点: {old_cell}<br>新格点: {new_cell}")
            if len(marker_pts) >= 24:
                break
        reroute_payload[str(plan.id)] = {
            "flight": _plan_label(plan.id),
            "original": {"x": before_pts[:, 0].tolist(), "y": before_pts[:, 1].tolist(), "z": before_pts[:, 2].tolist()},
            "rerouted": {"x": after_pts[:, 0].tolist(), "y": after_pts[:, 1].tolist(), "z": after_pts[:, 2].tolist()},
            "markers": {
                "x": [p[0] for p in marker_pts],
                "y": [p[1] for p in marker_pts],
                "z": [p[2] for p in marker_pts],
                "text": marker_text,
            },
        }
    comparison_trace_start = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="lines",
            line=dict(color="#C0392B", width=6),
            name="原始航迹",
            legendgroup="点击改航对比",
            visible=False,
            scene="scene",
            hovertemplate="原始航迹<br>X: %{x:.0f} m<br>Y: %{y:.0f} m<br>高度: %{z:.0f} m<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="lines",
            line=dict(color="#2E7D32", width=6),
            name="改航后航迹",
            legendgroup="点击改航对比",
            visible=False,
            scene="scene",
            hovertemplate="改航后航迹<br>X: %{x:.0f} m<br>Y: %{y:.0f} m<br>高度: %{z:.0f} m<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="markers+text",
            marker=dict(size=7, color="#F39C12", symbol="diamond", line=dict(width=1, color="#7B241C")),
            text=[],
            textposition="top center",
            name="改航点",
            legendgroup="点击改航对比",
            customdata=[],
            visible=False,
            scene="scene",
            hovertemplate="%{customdata}<extra></extra>",
        )
    )

    initial_c = int(metrics.get("initial_conflicts_uncertain", 0))
    final_c = int(metrics.get("final_conflicts_two_stage", 0))
    resolved = 100.0 * (initial_c - final_c) / max(1, initial_c)
    changed_count = int(sum(changed_flags))
    delayed_count = int(sum(delay > 1e-6 for delay in delays))
    tuned_count = int(sum(abs(v - 1.0) > 1e-3 for v in speed_ratios))
    rerouted_count = int(sum(rerouted))
    grid_desc = f"{grid.shape[0]}x{grid.shape[1]}x{grid.shape[2]}（{grid.cell_size[0]}m/格，{grid.cell_size[2]}m/层）"
    title = (
        f"改进FATA算法（三维调度） | {len(selected)}架低空飞行计划 | 混合调度场景"
        f"<br><sup>网格 {grid_desc} | 冲突 {initial_c} → {final_c} | 消解率 {resolved:.1f}% | 变化航班 {changed_count}</sup>"
    )
    key_axis_ticks = (
        dict(tickmode="array", tickvals=key_tick_vals, ticktext=key_tick_text, tickangle=-45, tickfont=dict(color="#C0392B", size=10))
        if key_tick_vals
        else dict(tickmode="linear", dtick=5)
    )
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=900,
        paper_bgcolor="#F5F7FA",
        plot_bgcolor="#FFFFFF",
        hovermode="closest",
        clickmode="event+select",
        font=dict(family="Microsoft YaHei, SimHei, Arial", size=12, color="#1A1A1A"),
        legend=dict(orientation="h", x=0.02, y=1.02),
        margin=dict(l=50, r=30, t=95, b=45),
        scene=dict(
            domain=dict(x=[0.0, 0.46], y=[0.55, 1.0]),
            xaxis=dict(title=dict(text="X 坐标 (m)"), showgrid=True, gridcolor="rgba(0,0,0,0.15)", color="#1A1A1A", backgroundcolor="#F0F4F8"),
            yaxis=dict(title=dict(text="Y 坐标 (m)"), showgrid=True, gridcolor="rgba(0,0,0,0.15)", color="#1A1A1A", backgroundcolor="#F0F4F8"),
            zaxis=dict(title=dict(text="高度 (m)"), range=[0, grid.shape[2] * grid.cell_size[2]], showgrid=True, gridcolor="rgba(0,0,0,0.15)", color="#1A1A1A", backgroundcolor="#F0F4F8"),
            bgcolor="#FFFFFF",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.35),
            camera=dict(eye=dict(x=1.45, y=-1.65, z=1.05)),
        ),
        xaxis=dict(anchor="y", domain=[0.54, 1.0], title=dict(text="航班编号（标注关键计划）"), **key_axis_ticks),
        yaxis=dict(anchor="x", domain=[0.55, 1.0], title=dict(text="延误时间 (min)")),
        xaxis2=dict(anchor="y2", domain=[0.0, 0.46], title=dict(text="航班编号（标注关键计划）"), **key_axis_ticks),
        yaxis2=dict(anchor="x2", domain=[0.0, 0.45], title=dict(text="速度比 v / v_ref"), range=[0.75, 1.38]),
        xaxis3=dict(anchor="y3", domain=[0.54, 1.0], title=dict(text="航班编号（标注关键计划）"), **key_axis_ticks),
        yaxis3=dict(anchor="x3", domain=[0.0, 0.45], title=dict(text="是否改航"), tickvals=[0, 1], ticktext=["否", "是"], range=[-0.2, 1.45]),
        annotations=[
            dict(text="3D 低空航迹", x=0.23, y=1.0, xref="paper", yref="paper", xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=16)),
            dict(text=f"策略一：错峰起飞（{delayed_count} 架延误）", x=0.77, y=1.0, xref="paper", yref="paper", xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=16)),
            dict(text=f"策略二：调整速度（{tuned_count} 架调速）", x=0.23, y=0.45, xref="paper", yref="paper", xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=16)),
            dict(text=f"策略三：局部改航（{rerouted_count} / {len(selected)} 架）", x=0.77, y=0.45, xref="paper", yref="paper", xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=16)),
        ],
    )
    post_script = f"""
    const reroutePayload = {json.dumps(reroute_payload, ensure_ascii=False)};
    const overviewRouteIndices = {json.dumps(overview_route_indices)};
    const rerouteTraceIndex = {reroute_trace_index};
    const comparisonTraceStart = {comparison_trace_start};
    const gd = document.getElementById('{{plot_id}}');

    function showRerouteComparison(flightId) {{
        const payload = reroutePayload[String(flightId)];
        if (!payload) {{
            window.alert('该飞行计划没有优化结果。');
            return;
        }}
        Plotly.restyle(gd, {{visible: false}}, overviewRouteIndices);
        Plotly.restyle(gd, {{
            x: [payload.original.x],
            y: [payload.original.y],
            z: [payload.original.z],
            visible: true,
            name: '原始航迹 ' + payload.flight
        }}, [comparisonTraceStart]);
        Plotly.restyle(gd, {{
            x: [payload.rerouted.x],
            y: [payload.rerouted.y],
            z: [payload.rerouted.z],
            visible: true,
            name: '改航后航迹 ' + payload.flight
        }}, [comparisonTraceStart + 1]);
        Plotly.restyle(gd, {{
            x: [payload.markers.x],
            y: [payload.markers.y],
            z: [payload.markers.z],
            text: [payload.markers.x.map(() => '改航点')],
            customdata: [payload.markers.text],
            visible: payload.markers.x.length > 0
        }}, [comparisonTraceStart + 2]);
        Plotly.relayout(gd, {{
            'annotations[0].text': '改航对比：' + payload.flight + '<br><sup>红色线=原始航迹，绿色线=优化后航迹，黄钻=改航点</sup>'
        }});
    }}

    function resetOverview() {{
        Plotly.restyle(gd, {{visible: true}}, overviewRouteIndices);
        Plotly.restyle(gd, {{visible: false}}, [comparisonTraceStart, comparisonTraceStart + 1, comparisonTraceStart + 2]);
        Plotly.relayout(gd, {{
            'annotations[0].text': '3D 低空航迹'
        }});
    }}

    gd.on('plotly_click', function(evt) {{
        if (!evt || !evt.points || !evt.points.length) return;
        const pt = evt.points[0];
        if (pt.curveNumber !== rerouteTraceIndex) return;
        const flightId = pt.customdata !== undefined ? pt.customdata : null;
        if (flightId === null) return;
        showRerouteComparison(flightId);
    }});

    gd.on('plotly_doubleclick', function() {{
        resetOverview();
    }});
    """
    fig.write_html(output_path, include_plotlyjs=True, full_html=True, post_script=post_script)


def _write_plotly_fullscreen_html(fig: object, path: Path, post_script: str | None = None) -> None:
    div = fig.to_html(
        include_plotlyjs=True,
        full_html=False,
        default_width="100%",
        default_height="100%",
        config={"responsive": True, "displaylogo": False},
        post_script=post_script,
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    html, body {{
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden;
      background: #F5F7FA;
    }}
    body > div,
    .plotly-graph-div,
    .plot-container,
    .svg-container {{
      width: 100vw !important;
      height: 100vh !important;
      max-width: none !important;
      max-height: none !important;
    }}
  </style>
</head>
<body>
{div}
<script>
  function resizePlot() {{
    const graph = document.querySelector('.plotly-graph-div');
    if (graph && window.Plotly) {{
      graph.style.width = window.innerWidth + 'px';
      graph.style.height = window.innerHeight + 'px';
      Plotly.Plots.resize(graph);
    }}
  }}
  window.addEventListener('load', resizePlot);
  window.addEventListener('resize', resizePlot);
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _write_html_fallback(path: Path, title: str) -> None:
    path.write_text(f"<html><head><meta charset='utf-8'></head><body><h1>{title}</h1><p>Plotly is not available.</p></body></html>", encoding="utf-8")


def write_all_route_visuals(
    output_dir: str | Path,
    grid: AirspaceGrid,
    initial: list[FlightPlan],
    two_stage: list[FlightPlan],
    one_stage: list[FlightPlan],
    conflicts_uncertain: list[Conflict],
    conflicts_no: list[Conflict],
    final_conflicts: list[Conflict],
    key_ids: list[int],
) -> None:
    out = ensure_dir(output_dir)
    plot_routes_3d(grid, initial, out / "initial_routes.png", "Initial routes")
    plot_conflict_points(grid, initial, conflicts_uncertain, out / "conflict_points_uncertain.png", "Conflict points with uncertainty")
    plot_conflict_points(grid, initial, conflicts_no, out / "conflict_points_no_uncertain.png", "Conflict points without uncertainty")
    plot_key_flights(grid, initial, key_ids, out / "key_flights_3d.png")
    plot_before_after(grid, initial, two_stage, out / "two_stage_routes_before_after.png", "Two-stage route changes")
    plot_before_after(grid, initial, one_stage, out / "one_stage_routes_before_after.png", "One-stage route changes")
    write_fata_3d_html(grid, initial, conflicts_uncertain, out / "fata_3d_before.html", "调度前航线分布（基线）", key_ids)
    write_fata_3d_html(grid, two_stage, final_conflicts, out / "fata_3d_after.html", "调度后航线分布（两阶段优化）", key_ids)
