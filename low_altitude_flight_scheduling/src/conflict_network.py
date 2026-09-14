from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .conflict_detection import Conflict
from .flight_plan import FlightPlan
from .utils import ensure_dir


def build_conflict_network(plans: list[FlightPlan], conflicts: list[Conflict]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(plan.id for plan in plans)
    for c in conflicts:
        if graph.has_edge(c.flight_a, c.flight_b):
            graph[c.flight_a][c.flight_b]["count"] += 1
        else:
            graph.add_edge(c.flight_a, c.flight_b, count=1, cell=c.cell, time_diff=c.time_diff)
    return graph


def collective_influence(graph: nx.Graph, l: int = 2) -> dict[int, float]:
    scores: dict[int, float] = {}
    degrees = dict(graph.degree())
    for node in graph.nodes:
        if degrees[node] <= 1:
            scores[node] = 0.0
            continue
        boundary = [n for n, dist in nx.single_source_shortest_path_length(graph, node, cutoff=l).items() if dist == l]
        scores[node] = float((degrees[node] - 1) * sum(max(0, degrees[v] - 1) for v in boundary))
    return scores


def network_metrics(graph: nx.Graph, ci_l: int = 2) -> pd.DataFrame:
    degree = dict(graph.degree())
    deg_cent = nx.degree_centrality(graph)
    close = nx.closeness_centrality(graph) if graph.number_of_edges() else {n: 0.0 for n in graph.nodes}
    between = nx.betweenness_centrality(graph, normalized=True) if graph.number_of_edges() else {n: 0.0 for n in graph.nodes}
    pagerank = nx.pagerank(graph) if graph.number_of_edges() else {n: 0.0 for n in graph.nodes}
    ci = collective_influence(graph, ci_l)
    rows = []
    for node in graph.nodes:
        rows.append(
            {
                "flight_id": node,
                "degree": degree[node],
                "degree_centrality": deg_cent[node],
                "closeness": close[node],
                "betweenness": between[node],
                "pagerank": pagerank[node],
                "collective_influence": ci[node],
            }
        )
    return pd.DataFrame(rows).sort_values(["collective_influence", "degree", "pagerank"], ascending=False)


def select_key_flights(metrics: pd.DataFrame, important_ratio: float, n_flights: int) -> list[int]:
    k = max(1, int(round(float(important_ratio) * n_flights)))
    return [int(v) for v in metrics.head(k)["flight_id"].tolist()]


def attack_experiment(graph: nx.Graph, metrics: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    ordered = [int(v) for v in metrics.sort_values(metric_name, ascending=False)["flight_id"].tolist()]
    g = graph.copy()
    initial_edges = max(1, graph.number_of_edges())
    initial_nodes = max(1, graph.number_of_nodes())
    rows = [{"removed": 0, "remaining_edges": g.number_of_edges(), "edge_ratio": g.number_of_edges() / initial_edges, "lcc_ratio": _lcc_ratio(g, initial_nodes), "metric": metric_name}]
    for i, node in enumerate(ordered, start=1):
        if g.has_node(node):
            g.remove_node(node)
        rows.append(
            {
                "removed": i,
                "remaining_edges": g.number_of_edges(),
                "edge_ratio": g.number_of_edges() / initial_edges,
                "lcc_ratio": _lcc_ratio(g, initial_nodes),
                "metric": metric_name,
            }
        )
    return pd.DataFrame(rows)


def run_attack_suite(graph: nx.Graph, metrics: pd.DataFrame) -> pd.DataFrame:
    names = ["collective_influence", "degree", "pagerank", "betweenness", "closeness"]
    return pd.concat([attack_experiment(graph, metrics, name) for name in names], ignore_index=True)


def _lcc_ratio(graph: nx.Graph, initial_nodes: int) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    largest = max((len(c) for c in nx.connected_components(graph)), default=0)
    return largest / max(1, initial_nodes)


def write_network_outputs(graph: nx.Graph, metrics: pd.DataFrame, attack_df: pd.DataFrame, output_dir: str | Path) -> None:
    out = ensure_dir(output_dir)
    metrics.to_csv(out / "key_flights.csv", index=False)
    plot_network(graph, metrics, out / "conflict_network.png")
    plot_attack_curves(attack_df, out / "network_attack_edges.png", y="edge_ratio", ylabel="Remaining edge ratio")
    plot_attack_curves(attack_df, out / "network_attack_lcc.png", y="lcc_ratio", ylabel="Largest component ratio")


def plot_network(graph: nx.Graph, metrics: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    if graph.number_of_nodes() == 0:
        fig.savefig(path)
        plt.close(fig)
        return
    pos = nx.spring_layout(graph, seed=7, k=0.35)
    metric_map = metrics.set_index("flight_id")["collective_influence"].to_dict()
    sizes = [80 + 24 * graph.degree(n) for n in graph.nodes]
    colors = [metric_map.get(n, 0.0) for n in graph.nodes]
    nx.draw_networkx_edges(graph, pos, alpha=0.18, width=0.8, ax=ax)
    nodes = nx.draw_networkx_nodes(graph, pos, node_size=sizes, node_color=colors, cmap="plasma", alpha=0.9, ax=ax)
    ax.set_title("Conflict complex network")
    ax.axis("off")
    fig.colorbar(nodes, ax=ax, shrink=0.75, label="Collective influence")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_attack_curves(attack_df: pd.DataFrame, path: str | Path, y: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for metric, group in attack_df.groupby("metric"):
        ax.plot(group["removed"], group[y], label=metric, linewidth=1.8)
    ax.set_xlabel("Removed nodes")
    ax.set_ylabel(ylabel)
    ax.set_title("Sequential attack experiment")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
