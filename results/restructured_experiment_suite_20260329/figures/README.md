# 可视化图表索引

- 01_main_results_overview.png：主结果线图（攻击成功率 + 平均查询）
  - 线条设置：确定性主线、PG-ZOO（历史主线）、PG-ZOO（noR，全量）、PG-ZOO（full117）、PG-ZOO（support64）
- 02_query_tail_overview.png：查询尾部线图（Median + P95）
  - 线条设置同 01 图，展示不同设置下的查询长尾
- 03_budget_curves_noR.png：noR 主线预算曲线线图
  - 红线：预算内成功率（ASR）；棕线虚线：预算内完成率
- 03b_asr_vs_perturbation_cap.png：不同扰动等级（2/5/10/15/20%）下的 ASR 线图
  - 线条设置：最小成功准则、固定预算准则、边界推进准则
- 03c_asr_vs_perturbation_cap_aligned.png：在 03b 基础上叠加主结果 noR ASR 水平虚线
  - 线条设置：最小成功准则、固定预算准则、边界推进准则、主结果ASR基线（noR）
- 04_ablation_m256.png：m=256 消融线图
  - 线条设置：Vanilla ZOO（无降维）、主线 PG-ZOO、仅稀疏策略、稀疏混合策略
- 05_physical_compliance.png：物理一致性与隐蔽性线图
  - 左图：清洁/对抗样本在 Chi-square 与 LNRT 下的未告警比例
  - 右图：扰动均值/中位数与支撑外能量比
- 06_marginal_asr_gain.png：扰动预算边际收益曲线（minimum-success口径）
  - 线条设置：蓝线为每+5%扰动上限带来的ASR增量，棕色虚线为零增益参考
