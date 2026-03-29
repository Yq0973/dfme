# key_results 说明

本目录只保留当前论文写作、答辩和组会汇报真正需要的关键结果。

## 1. 正式主线

当前正式在线主线为 `deterministic_mainline_v1`：

- `case14`：固定二维局部区域上的确定性 compass 搜索
- `case118`：固定 top-64 support 上的确定性 basis 搜索

## 2. 文件对应关系

- `main_results_summary.csv`
  正式主线在全测试集上的核心结果汇总，包含 `ASR`、`avg_queries`、`median`、`p90`、`p95`、`max` 以及预算成功率。

- `figures/mainline_overview.png`
  跨系统总览图，展示正式主线在不同系统上的 `ASR`、平均查询数和 probe/search 查询构成。

- `query_cap_sweep_summary.csv`
  每样本查询上限扫描汇总表，包含不同上限下的成功率和触顶比例。

- `figures/query_cap_effects.png`
  每样本查询上限对 ASR 与触顶比例影响的可视化图。

- `mainline_comparison.csv`
  `scale_adaptive_mainline_v1`、`layered_active_mainline_v1`、`one_shot_mainline_v1` 与 `deterministic_mainline_v1` 的对比总表。

- `single_shot_full_comparison.csv`
  一次性注入假设下，不同 one-shot 变体之间的对比结果。

- `single_shot_support_sweep_s256.csv`
  固定 support 大小 sweep，主要用于解释 `case118` 中 `k=64` 的选择。

- `case14/main/attack_summary.json`
  `case14` 正式主线全测试集结果。

- `case14/main/query_distribution_summary.csv`
  `case14` 的查询分布统计。

- `case14/main/query_budget_curve.csv`
  `case14` 的预算成功率曲线。

- `case14/main/query_distribution_profile.png`
  `case14` 的查询分位点可视化图。

- `case14/main/query_budget_curve_plot.png`
  `case14` 的预算成功率可视化曲线。

- `case118/main/attack_summary.json`
  `case118` 正式主线全测试集结果。

- `case118/main/query_distribution_summary.csv`
  `case118` 的查询分布统计。

- `case118/main/query_budget_curve.csv`
  `case118` 的预算成功率曲线。

- `case118/main/query_distribution_profile.png`
  `case118` 的查询分位点可视化图。

- `case118/main/query_budget_curve_plot.png`
  `case118` 的预算成功率可视化曲线。

## 3. 补充验证

### case14

- `case14/one_shot_validation/attack_summary_deterministic_compass.json`
  当前确定性 compass 主线的归档副本。

- `case14/one_shot_validation/attack_summary_region2_c1.json`
  旧随机单区域版本，对照“去运气化”前后的差异。

- `case14/one_shot_validation/attack_summary_region2_c2.json`
  两候选局部区域版本。

- `case14/one_shot_validation/attack_summary_region2.json`
  早期 `region_size=2, region_candidates=3` 版本。

- `case14/one_shot_validation/attack_summary_region3.json`
  早期 `region_size=3, region_candidates=3` 版本。

- `case14/one_shot_validation/attack_summary_k2.json`
  固定 support 的失败对照版本。

### case118

- `case118/one_shot_validation/attack_summary_k64.json`
  当前固定 top-64 support 主线的归档副本。

## 4. layered 参考线

- `case14/layered_reference/attack_summary.json`
- `case14/layered_reference/stage_breakdown_smoke128.csv`
- `case118/layered_reference/attack_summary.json`
- `case118/layered_reference/stage_breakdown_smoke128.csv`

这部分只用于说明 support 的可压缩性和分层分析结果，不再作为正式在线主线。

## 5. 使用建议

- 正文主表优先使用 `main_results_summary.csv`。
- 查询效率分析不要只看 `avg_queries`，还要同时查看各系统主线目录下的 `query_distribution_summary.csv` 与 `query_budget_curve.csv`。
- 如果要强调“从随机搜索改为确定性搜索”的意义，优先对比 `case14/main/attack_summary.json` 与 `case14/one_shot_validation/attack_summary_region2_c1.json`。
- 如果要解释为什么不再把 layered 作为正式在线流程，优先引用 `single_shot_full_comparison.csv` 和 layered 参考线结果。
