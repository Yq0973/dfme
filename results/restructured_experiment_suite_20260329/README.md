# 重构实验结果索引（2026-03-29）

本目录按 `docs/实验清单.md` 的四个实验维度重构。

## 01_core_performance
- main_results_noR_full.csv：去除 R 后主线在 case14/118 的全量结果。
- asr_vs_query_budget_noR.csv：去除 R 后预算-成功率曲线数据。
- main_results_with_baselines.csv：与 deterministic 与既有 PG-ZOO 结果的汇总对照。

## 02_ablation
- ablation_summary_from_existing.csv：沿用既有 full ablation 汇总（含 no-shrink/hard-support/no-semantic）。
- zoo_vs_mainline_compare_m256.csv：新增 m=256 的 ZOO 基线与主线补充对照。

## 03_physical_compliance
- physical_compliance_noR.csv：BDD 一致性与扰动隐蔽性统计。

## 04_cross_model_generalization
- 当前仓库暂无 GNN 目标模型与对应全流程评测脚本，待补充。

## 说明
- 本次新增的补充对照（zoo_vs_mainline）使用 `max_samples=256`，用于快速验证趋势。
- 主线 noR 结果使用全量样本（max_samples=0）。
