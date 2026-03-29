# results 目录说明

当前 `results/` 只保留论文写作和组会汇报需要的关键结果，正式结果集中放在 `key_results/` 下。

## 当前正式主线

正式在线主线为 `deterministic_mainline_v1`：

- `case14`：固定二维局部区域上的确定性 compass 搜索
- `case118`：固定 top-64 support 上的确定性 basis 搜索

这条主线强调“一次性确定搜索子空间，再在该子空间内完成全部黑盒查询”。

## 主要文件

- `key_results/main_results_summary.csv`
- `key_results/mainline_comparison.csv`
- `key_results/single_shot_full_comparison.csv`
- `key_results/single_shot_support_sweep_s256.csv`
- `key_results/query_cap_sweep_summary.csv`
- `key_results/figures/mainline_overview.png`
- `key_results/figures/query_cap_effects.png`
- `key_results/case14/main/attack_summary.json`
- `key_results/case14/main/query_distribution_summary.csv`
- `key_results/case14/main/query_budget_curve.csv`
- `key_results/case14/main/query_distribution_profile.png`
- `key_results/case14/main/query_budget_curve_plot.png`
- `key_results/case118/main/attack_summary.json`
- `key_results/case118/main/query_distribution_summary.csv`
- `key_results/case118/main/query_budget_curve.csv`
- `key_results/case118/main/query_distribution_profile.png`
- `key_results/case118/main/query_budget_curve_plot.png`

其中：

- `main_results_summary.csv` 汇总正式主线在全测试集上的核心结果；
- `query_distribution_summary.csv` 用于分析查询分布是否存在长尾；
- `query_budget_curve.csv` 用于分析固定预算下的成功率；
- `mainline_overview.png` 用于汇总展示不同系统上的 ASR、平均查询数和 probe/search 查询构成；
- `query_cap_sweep_summary.csv` 汇总“每样本查询上限”下的成功率与触顶比例；
- `query_cap_effects.png` 可视化展示查询上限对 ASR 与触顶比例的影响；
- `query_distribution_profile.png` 用于展示不同分位点下的查询长尾情况；
- `query_budget_curve_plot.png` 用于展示预算成功率曲线；
- `mainline_comparison.csv` 用于比较旧主线与当前主线；
- `single_shot_support_sweep_s256.csv` 主要用于解释 `case118` 为什么取固定 top-64 support。

## 参考线

项目中仍保留少量参考线结果，用于支撑论文中的对照分析：

- `key_results/case14/one_shot_validation`
- `key_results/case118/one_shot_validation`
- `key_results/case14/layered_reference`
- `key_results/case118/layered_reference`

其中 `layered_reference` 只作为离线分析线，不再视为正式在线攻击流程。
