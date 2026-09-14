# 基于复杂网络的城市低空飞行计划优化调度：算法级复现

本项目复现论文《基于复杂网络的城市低空飞行计划优化调度》的主要算法流程：三维栅格空域、城市低空第三方风险地图、改进 A* 四维飞行计划生成、考虑过点时间不确定性的冲突探测、冲突复杂网络关键飞行计划识别、多策略两阶段优化，以及原始/改进 FATA、PSO、GA 对比。

这是算法级复现，不是作者私有随机数据的逐点复刻。论文未公开完整仿真数据、随机种子、障碍物生成、人口密度细节和过点时间误差函数，因此本项目使用可配置随机城市与交通脉冲生成器，并在 `outputs/calibration_report.md` 中记录默认冲突数与论文现象的差异。

## 安装与运行

```bash
pip install -r requirements.txt
python run_main.py --seed 2025
python run_main.py --quick --seed 2025
python run_sensitivity.py --quick --seed 2025
pytest
```

`--quick` uses a smaller FATA population and generation budget (`NP <= 20`,
`Ngen_max <= 50`). It is intended for smoke tests and fast diagnostics; it can
significantly reduce conflicts, but it does not guarantee that every conflict
will be eliminated. Use `python run_main.py --seed 2025` for the normal
two-stage run.

The default scheduler now runs in safe mode: every accepted action is checked
against global conflict pairs, and actions that increase global conflict pairs
are rolled back. Quick mode disables local A* rerouting and only tests takeoff
time shifts plus speed factors. Normal mode first uses the same delay/speed
logic and only tries a capped number of local reroutes after the remaining
conflict-pair count is small.

如果环境暂时缺少 `pyyaml` 或 `tqdm`，代码会使用内置 fallback；`pytest` 仍建议安装后运行。

## 模块说明

- `src/grid.py`：60 x 60 x 4 三维空域栅格，柱状城市障碍物，26 邻域。
- `src/risk_map.py`：按论文式 (1)-(10) 的结构实现地面风险近似，输出归一化风险图。
- `src/astar_3d.py`：改进 A*，`f(n)=g(n)+w(n)h(n)`，其中 `g` 融合风险和长度。
- `src/flight_plan.py`：生成无人机起终点、A* 路径、ETD、ETAP、速度剖面和初始计划。
- `src/conflict_detection.py`：不考虑/考虑不确定性两种冲突探测，正态分位数由 `alpha` 控制。
- `src/conflict_network.py`：冲突网络、度/接近/介数/PageRank/CI 指标和顺序攻击实验。
- `src/optimization_model.py`：地面等待、速度调整、局部改航的统一解码、评估和贪心修复。
- `src/fata.py`：原始 FATA 与佳点集初始化的改进 FATA。
- `src/baselines.py`：PSO、GA、原始 FATA 与改进 FATA 的快速对比实验。
- `src/visualization.py`：PNG 与 Plotly 离线 HTML 可视化，HTML 风格对齐参考文件。

## 公式对应与近似

风险地图保留论文的综合风险结构 `P_total = P_bal + P_par`、撞击影响概率、严重程度概率、弹道下坠和降落伞漂移项。由于论文没有给出完整城市人口场，本项目用基础人口密度、热点和建筑影响生成空间人口密度，并进行 min-max normalization 供 A* 使用。

冲突探测中，不考虑不确定性使用同一栅格过点时间差 `|t_a - t_b| <= t_conflict`。考虑不确定性时，`t_ETA ~ Normal(mu_t, sigma_t^2)`，默认 `sigma_t = sigma0 + sigma_rate * elapsed_time`，并用正态置信区间扩展安全时间间隔。

默认采用地面起降与空域巡航分离机制：飞行计划的起点和终点固定在 `z=0` 地面层，巡航层按 `flight_generation.altitude.cruise_level_probs` 随机采样。可视化中航迹线使用栅格中心高度，`z=0/1/2/3` 分别显示为空域格点中心约 15/45/75/105 m；起终点 marker 单独画在 0 m 地面高度。

适应度中的延误、空中时间和风险均按初始计划尺度归一化，冲突项使用剩余冲突数量和总空中时间归一化，避免风险和时间量纲压倒冲突项。`risk_increase_ratio` 使用归一化风险增量除以按 10 m 采样的总飞行距离暴露量，表示平均暴露风险增量。电池约束和延误计划数作为罚项加入。

## 输出

主流程输出位于 `outputs/`：

- `metrics_summary.csv`
- `conflicts_uncertain.csv`
- `conflicts_no_uncertain.csv`
- `unresolved_conflicts.csv`
- `final_conflicts_by_pair.csv`
- `final_conflict_segments.csv`
- `key_flights.csv`
- `table_two_stage_vs_one_stage.csv`
- `table_pso_ga_fata.csv`
- `table_original_vs_improved_fata.csv`
- 多张 PNG 图，包括风险、航迹、网络、攻击曲线、收敛曲线和敏感性图
- Plotly 离线 HTML：`strategy_overview.html`、`fata_3d_before.html`、`fata_3d_after.html`

结果不是硬编码论文表格，而是由当前随机种子、配置和算法运行计算得到。

## 非均匀初始飞行计划生成

论文没有公开原始 OD 生成代码，因此本项目采用“非均匀随机热点 OD 模型”来模拟城市低空交通中的物流站点、服务热点和中心穿越流量。该模型不使用完全均匀随机采样、网格分层采样或 Latin Hypercube，也不人为保证各区域飞行计划数量均衡。

默认配置 `flight_generation.mode=heterogeneous_random` 会生成热点 OD、时间高峰、异质速度和中心走廊重叠航迹：约 55% 航班在热点之间飞行，约 25% 从边缘穿越中心区域，约 20% 保留背景随机流量；起飞时间由多个高斯峰和少量均匀噪声组成。该方法比均匀随机更接近论文仿真图中表现出的冲突不均匀、中心区域航迹密集和连续重叠航段现象。

相关诊断输出包括 `outputs/od_points.png`、`outputs/takeoff_time_hist.png`、`outputs/route_density_heatmap.png`、`outputs/route_length_hist.png` 和 `outputs/flight_generation_report.csv`。若 `route_density_gini < 0.25`，程序会提示航迹仍过于均衡，可提高 `center_bias_strength` 或 `hotspot_std_cells`。
