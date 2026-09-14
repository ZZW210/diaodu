from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import apply_quick_overrides, load_config
from src.conflict_detection import (
    count_conflict_edges,
    count_conflict_pairs,
    count_conflict_points,
    detect_conflicts,
    get_conflict_detection_stats,
    reset_conflict_detection_stats,
    write_calibration_report,
    write_conflict_diagnostics,
)
from src.conflict_network import build_conflict_network, network_metrics, run_attack_suite, select_key_flights, write_network_outputs
from src.fata import fata_optimize
from src.flight_plan import generate_flight_plans, write_flight_generation_report
from src.grid import AirspaceGrid
from src.optimization_model import (
    apply_delay_speed_vector,
    changed_count,
    delayed_count,
    evaluate_delay_speed_solution,
    get_optimization_stats,
    greedy_time_deconfliction,
    independent_matching_deconfliction,
    limited_local_reroute,
    reset_optimization_stats,
    risk_increase_ratio,
    solution_bounds_delay_speed,
)
from src.risk_map import generate_risk_map
from src.utils import ensure_dir, set_random_seed
from src.visualization import plot_fata_convergence, write_all_route_visuals, write_strategy_overview_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the low-altitude scheduling reproduction.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-flights", type=int, default=None)
    parser.add_argument("--outputs", default="outputs")
    return parser.parse_args()


def _counts(conflicts: list[object]) -> dict[str, int]:
    return {
        "points": count_conflict_points(conflicts),
        "pairs": count_conflict_pairs(conflicts),
        "edges": count_conflict_edges(conflicts),
    }


def _print_counts(stage: str, conflicts: list[object]) -> None:
    counts = _counts(conflicts)
    print(f"{stage}: {counts['points']} points, {counts['pairs']} pairs")


def _append_trace(trace: list[dict[str, object]], stage: str, logs: list[dict[str, object]]) -> None:
    for idx, item in enumerate(logs):
        row = {
            "stage": item.get("stage", stage),
            "iter": item.get("iter", idx),
            "action_type": item.get("action_type", item.get("action", "")),
            "plan_id": item.get("plan_id", item.get("flight_id", "")),
            "old_pairs": item.get("old_pairs", item.get("before_pairs", "")),
            "new_pairs": item.get("new_pairs", item.get("after_pairs", "")),
            "old_points": item.get("old_points", item.get("before_conflicts", "")),
            "new_points": item.get("new_points", item.get("after_conflicts", "")),
            "accepted": bool(item.get("accepted", item.get("accepted", False))),
            "runtime_ms": float(item.get("runtime_ms", 0.0)),
            "rollback": bool(item.get("rollback", not bool(item.get("accepted", False)))),
            "reason": item.get("reason", ""),
            "matched_strategy": item.get("matched_strategy", ""),
            "conflict_class": item.get("conflict_class", ""),
            "segment_idx": item.get("segment_idx", ""),
            "segment_points": item.get("segment_points", ""),
            "pair": item.get("pair", ""),
            "candidate_rank": item.get("candidate_rank", ""),
            "prob_delay": item.get("prob_delay", ""),
            "prob_speed": item.get("prob_speed", ""),
            "prob_reroute": item.get("prob_reroute", ""),
            "prob_delay_after": item.get("prob_delay_after", ""),
            "prob_speed_after": item.get("prob_speed_after", ""),
            "prob_reroute_after": item.get("prob_reroute_after", ""),
            "source": item.get("source", ""),
        }
        if row["accepted"] and row["old_pairs"] != "" and row["new_pairs"] != "" and int(row["new_pairs"]) > int(row["old_pairs"]):
            raise AssertionError("Trace contains accepted action with increased conflict pairs.")
        trace.append(row)


def _write_trace(out: Path, trace: list[dict[str, object]]) -> None:
    columns = [
        "stage",
        "iter",
        "action_type",
        "plan_id",
        "old_pairs",
        "new_pairs",
        "old_points",
        "new_points",
        "accepted",
        "runtime_ms",
        "rollback",
        "reason",
        "matched_strategy",
        "conflict_class",
        "segment_idx",
        "segment_points",
        "pair",
        "candidate_rank",
        "prob_delay",
        "prob_speed",
        "prob_reroute",
        "prob_delay_after",
        "prob_speed_after",
        "prob_reroute_after",
        "source",
    ]
    pd.DataFrame(trace, columns=columns).to_csv(out / "optimization_trace.csv", index=False)


def _runtime_exceeded(deadline: float) -> bool:
    return time.perf_counter() >= deadline


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config)
    if args.quick:
        cfg = apply_quick_overrides(cfg)
    if args.n_flights is not None:
        cfg["flight"]["n_flights"] = int(args.n_flights)
        cfg.setdefault("flight_generation", {})["n_flights"] = int(args.n_flights)
    cfg["flight"]["random_seed"] = int(args.seed)
    set_random_seed(args.seed)
    out = ensure_dir(root / args.outputs)
    reset_conflict_detection_stats()
    reset_optimization_stats()
    start_time = time.perf_counter()
    max_runtime = float(
        cfg["optimization"].get(
            "quick_max_runtime_seconds" if args.quick else "max_runtime_seconds",
            60 if args.quick else 180,
        )
    )
    deadline = start_time + max_runtime
    trace: list[dict[str, object]] = []

    grid = AirspaceGrid.from_config(cfg, seed=args.seed)
    risk = generate_risk_map(grid, cfg, out)
    plans = generate_flight_plans(grid, risk, cfg, out, seed=args.seed)

    conflicts_no = detect_conflicts(plans, cfg, uncertain=False, output_csv=out / "conflicts_no_uncertain.csv")
    conflicts0 = detect_conflicts(plans, cfg, uncertain=True, output_csv=out / "conflicts_uncertain.csv")
    write_flight_generation_report(out, plans, grid, count_conflict_points(conflicts_no), count_conflict_points(conflicts0), conflicts=conflicts0)
    write_calibration_report(count_conflict_points(conflicts_no), count_conflict_points(conflicts0), out)
    _print_counts("Initial N_c", conflicts0)

    graph = build_conflict_network(plans, conflicts0)
    metrics_df = network_metrics(graph, ci_l=int(cfg["network"]["ci_l"]))
    key_ratio = float(cfg["optimization"].get("stage1_key_ratio", cfg["optimization"]["important_ratio"]))
    key_ids = select_key_flights(metrics_df, key_ratio, len(plans))
    attack_df = run_attack_suite(graph, metrics_df)
    write_network_outputs(graph, metrics_df, attack_df, out)

    stage1_fata_time = 0.0
    if key_ids and not _runtime_exceeded(deadline):
        lb, ub = solution_bounds_delay_speed(key_ids, cfg)

        def objective(vec: np.ndarray) -> float:
            return evaluate_delay_speed_solution(vec, plans, key_ids, cfg, grid, risk).fitness

        fata_started = time.perf_counter()
        fata_res = fata_optimize(
            objective,
            lb,
            ub,
            dim=len(lb),
            population=int(cfg["fata"]["NP"]),
            max_iter=int(cfg["fata"]["Ngen_max"]),
            seed=args.seed,
            improved=True,
            parf=float(cfg["fata"]["Parf"]),
        )
        stage1_fata_time = time.perf_counter() - fata_started
        stage1_plans = apply_delay_speed_vector(plans, key_ids, fata_res.best_position, cfg, grid, risk)
    else:
        fata_res = fata_optimize(lambda x: float(np.sum(x * x)), np.array([0.0]), np.array([1.0]), 1, 4, 4, seed=args.seed)
        stage1_plans = [p.copy() for p in plans]

    conflicts1 = detect_conflicts(stage1_plans, cfg, uncertain=True)
    if count_conflict_pairs(conflicts1) > count_conflict_pairs(conflicts0):
        stage1_plans = [p.copy() for p in plans]
        conflicts1 = list(conflicts0)
        trace.append(
            {
                "stage": "stage1_fata",
                "iter": 0,
                "action_type": "rollback_stage1",
                "plan_id": "",
                "old_pairs": count_conflict_pairs(conflicts0),
                "new_pairs": count_conflict_pairs(conflicts1),
                "old_points": count_conflict_points(conflicts0),
                "new_points": count_conflict_points(conflicts1),
                "accepted": False,
                "runtime_ms": 0.0,
                "rollback": True,
                "reason": "stage1_increased_pairs",
            }
        )
    _print_counts("Stage 1 FATA N_c", conflicts1)

    two_stage_plans = [p.copy() for p in stage1_plans]
    final_conflicts = list(conflicts1)
    stage2_log: list[dict[str, object]] = []
    greedy_repair_time = 0.0
    local_reroute_time = 0.0
    stage2_independent_matching_time = 0.0
    reroute_attempts = int(cfg["optimization"].get("max_local_reroute_attempts", 20))
    reroute_enabled = not (args.quick and bool(cfg["optimization"].get("quick_disable_reroute", True)))

    # 2.1 贪心修复：按冲突贡献排名逐架尝试错峰起飞 / 速度调整
    greedy_iter_offset = 0
    if not _runtime_exceeded(deadline):
        started = time.perf_counter()
        two_stage_plans, final_conflicts, greedy_log = greedy_time_deconfliction(
            two_stage_plans,
            final_conflicts,
            cfg,
            grid,
            risk,
            max_rounds=int(cfg["optimization"].get("stage1_greedy_rounds", 50)),
            deadline=deadline,
        )
        greedy_repair_time += time.perf_counter() - started
        stage2_log.extend(greedy_log)
        greedy_iter_offset = max((int(row.get("iter", 0)) for row in greedy_log), default=-1) + 1
        _print_counts("Stage 2 greedy N_c", final_conflicts)

    # 2.2 独立匹配策略（参考文献 [6]）：逐冲突段匹配错峰 / 速度 / 局部改航
    if not _runtime_exceeded(deadline):
        started = time.perf_counter()
        two_stage_plans, final_conflicts, adm_log = independent_matching_deconfliction(
            two_stage_plans,
            final_conflicts,
            cfg,
            grid,
            risk,
            max_rounds=int(cfg["optimization"].get("independent_matching_rounds", cfg["optimization"].get("stage2_repair_rounds", 20))),
            deadline=deadline,
            seed=args.seed + 206,
        )
        stage2_independent_matching_time += time.perf_counter() - started
        stage2_log.extend(adm_log)
        _print_counts("Stage 2 independent matching N_c", final_conflicts)

    # 2.3 受限局部改航：仅当剩余冲突对很少时触发
    if reroute_enabled and not _runtime_exceeded(deadline):
        started = time.perf_counter()
        two_stage_plans, final_conflicts, reroute_log = limited_local_reroute(
            two_stage_plans,
            final_conflicts,
            cfg,
            grid,
            risk,
            max_attempts=reroute_attempts,
            deadline=deadline,
        )
        local_reroute_time += time.perf_counter() - started
        stage2_log.extend(reroute_log)
        _print_counts("Stage 2 local reroute N_c", final_conflicts)

    # 2.4 收尾贪心修复：清理上一阶段残留的少量冲突
    if not _runtime_exceeded(deadline):
        started = time.perf_counter()
        tail_plans, tail_conflicts, tail_log = greedy_time_deconfliction(
            two_stage_plans,
            final_conflicts,
            cfg,
            grid,
            risk,
            max_rounds=int(cfg["optimization"].get("final_greedy_rounds", 30)),
            deadline=deadline,
        )
        greedy_repair_time += time.perf_counter() - started
        for row in tail_log:
            row["iter"] = int(row.get("iter", 0)) + greedy_iter_offset
        stage2_log.extend(tail_log)
        two_stage_plans, final_conflicts = tail_plans, tail_conflicts
        _print_counts("Stage 2 tail greedy N_c", final_conflicts)

    if reroute_enabled and not _runtime_exceeded(deadline):
        started = time.perf_counter()
        two_stage_plans, final_conflicts, tail_reroute_log = limited_local_reroute(
            two_stage_plans,
            final_conflicts,
            cfg,
            grid,
            risk,
            max_attempts=reroute_attempts,
            deadline=deadline,
        )
        local_reroute_time += time.perf_counter() - started
        stage2_log.extend(tail_reroute_log)
        _print_counts("Stage 2 tail reroute N_c", final_conflicts)

    _append_trace(trace, "stage2_independent_matching", stage2_log)
    _print_counts("Stage 2 N_c", final_conflicts)

    runtime = time.perf_counter() - start_time
    detect_stats = get_conflict_detection_stats()
    opt_stats = get_optimization_stats()
    accepted_actions = int(sum(1 for row in trace if row.get("accepted") is True))
    rejected_actions = int(sum(1 for row in trace if row.get("accepted") is False))
    rollback_count = int(sum(1 for row in trace if row.get("rollback") is True))
    stage2_strategy_counts = {
        strategy: int(
            sum(
                1
                for row in trace
                if row.get("stage") == "stage2_independent_matching"
                and row.get("accepted") is True
                and row.get("matched_strategy") == strategy
            )
        )
        for strategy in ("delay", "speed", "reroute")
    }
    one_stage_final_conflicts = conflicts1
    final_eval = evaluate_delay_speed_solution(np.array([]), two_stage_plans, [], cfg, grid, risk)

    summary = {
        "initial_conflicts_uncertain": count_conflict_points(conflicts0),
        "initial_conflicts_without_uncertainty": count_conflict_points(conflicts_no),
        "initial_conflict_points_uncertain": count_conflict_points(conflicts0),
        "initial_conflict_pairs_uncertain": count_conflict_pairs(conflicts0),
        "initial_conflict_edges_uncertain": count_conflict_edges(conflicts0),
        "initial_conflict_points_no_uncertain": count_conflict_points(conflicts_no),
        "initial_conflict_pairs_no_uncertain": count_conflict_pairs(conflicts_no),
        "initial_conflict_edges_no_uncertain": count_conflict_edges(conflicts_no),
        "final_conflicts_two_stage": count_conflict_points(final_conflicts),
        "final_conflicts_one_stage": count_conflict_points(one_stage_final_conflicts),
        "final_conflict_points_two_stage": count_conflict_points(final_conflicts),
        "final_conflict_pairs_two_stage": count_conflict_pairs(final_conflicts),
        "final_conflict_edges_two_stage": count_conflict_edges(final_conflicts),
        "final_conflict_points_one_stage": count_conflict_points(one_stage_final_conflicts),
        "final_conflict_pairs_one_stage": count_conflict_pairs(one_stage_final_conflicts),
        "final_conflict_edges_one_stage": count_conflict_edges(one_stage_final_conflicts),
        "changed_flight_count_two_stage": changed_count(two_stage_plans),
        "changed_flight_count_one_stage": changed_count(stage1_plans),
        "delayed_flight_count": delayed_count(two_stage_plans, float(cfg["optimization"].get("delay_count_threshold", 30.0))),
        "runtime_seconds": runtime,
        "total_runtime": runtime,
        "final_fitness": final_eval.fitness,
        "risk_increase_ratio": risk_increase_ratio(plans, two_stage_plans),
        "detect_conflicts_time": detect_stats["detect_conflicts_time"],
        "stage1_fata_time": stage1_fata_time,
        "stage2_independent_matching_time": stage2_independent_matching_time,
        "greedy_repair_time": greedy_repair_time,
        "local_reroute_time": local_reroute_time,
        "number_of_detect_conflicts_calls": int(detect_stats["number_of_detect_conflicts_calls"]),
        "number_of_astar_calls": int(opt_stats["number_of_astar_calls"]),
        "accepted_actions": accepted_actions,
        "rejected_actions": rejected_actions,
        "rollback_count": rollback_count,
        "stage2_delay_matches": stage2_strategy_counts["delay"],
        "stage2_speed_matches": stage2_strategy_counts["speed"],
        "stage2_reroute_matches": stage2_strategy_counts["reroute"],
    }
    pd.DataFrame([summary]).to_csv(out / "metrics_summary.csv", index=False)
    _write_trace(out, trace)
    with (out / "final_plans_two_stage.pkl").open("wb") as fh:
        pickle.dump(two_stage_plans, fh)
    with (out / "final_plans_one_stage.pkl").open("wb") as fh:
        pickle.dump(stage1_plans, fh)
    write_conflict_diagnostics(final_conflicts, out, repair_log=trace)
    pd.DataFrame(
        [
            {
                "method": "two_stage_fata_independent_matching",
                "final_conflicts": count_conflict_points(final_conflicts),
                "final_conflict_pairs": count_conflict_pairs(final_conflicts),
                "changed_flights": changed_count(two_stage_plans),
                "fitness": final_eval.fitness,
            },
            {
                "method": "stage1_delay_speed_only",
                "final_conflicts": count_conflict_points(one_stage_final_conflicts),
                "final_conflict_pairs": count_conflict_pairs(one_stage_final_conflicts),
                "changed_flights": changed_count(stage1_plans),
                "fitness": evaluate_delay_speed_solution(np.array([]), stage1_plans, [], cfg, grid, risk).fitness,
            },
        ]
    ).to_csv(out / "table_two_stage_vs_one_stage.csv", index=False)

    if time.perf_counter() < deadline + 30.0:
        plot_fata_convergence(fata_res.convergence, out / "fata_convergence.png")
        write_all_route_visuals(out, grid, plans, two_stage_plans, stage1_plans, conflicts0, conflicts_no, final_conflicts, key_ids)
        write_strategy_overview_html(
            out / "strategy_overview.html",
            summary,
            metrics_df,
            attack_df,
            fata_res.convergence,
            grid=grid,
            initial=plans,
            optimized=two_stage_plans,
            conflicts=final_conflicts,
            key_ids=key_ids,
        )

    print("Main metrics summary")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    if count_conflict_points(final_conflicts) > 0:
        print("Optimization did not fully resolve conflicts; check outputs/unresolved_conflicts.csv")
    print(f"outputs: {out}")


if __name__ == "__main__":
    main()
