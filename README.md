# 拓扑约束低维黑盒攻击主线

当前目录只保留一条正式维护的方案：`scale_adaptive_mainline_v1`。

这条方案的定位是：

- 在状态子空间而不是全测量空间中做黑盒搜索
- 用拓扑约束生成候选区域
- 用少量 probe 查询筛掉低价值区域
- 将最终扰动投影回 `col(H)`，并保持 FDIA 主干约束
- 按系统尺度自适应选择子空间规模
  - `case14` 使用增强局部区域搜索
  - `case118` 使用全状态子空间搜索

历史上的测量显著图、物理门控、反馈闭环、PG-ZOO、预算门控等实验分支已经从正式维护范围中移除，只保留核心框架代码与关键结果，避免后续维护继续发散。

## 运行方式

推荐命令：

```bash
conda run -n dfme python topo_latent_blackbox/scripts/run_topology_latent_attack.py ^
  --systems 14 118 ^
  --seed 42 ^
  --max_samples 0 ^
  --attack_preset scale_adaptive_mainline_v1 ^
  --topology_mode auto
```

如果需要手工覆盖参数，可使用：

```bash
conda run -n dfme python topo_latent_blackbox/scripts/run_topology_latent_attack.py ^
  --systems 14 ^
  --attack_preset manual ^
  --region_size 8 ^
  --region_candidates 4 ^
  --probe_directions 2 ^
  --population 4 ^
  --rounds 10
```

## 当前保留结果

正式结果只保留在 `results/key_results/` 下，并统一使用测试集中全部可攻击样本。

这里的“全部”指：

- 测试集中所有攻击前被 oracle 正确判为 `FDIA` 的样本
- 这是黑盒攻击评测的标准攻击集合

| system  |    ASR | avg_queries | mean perturbation ratio |
| ------- | -----: | ----------: | ----------------------: |
| case14  | 94.82% |      131.63 |                  0.1030 |
| case118 | 87.91% |       86.45 |                  0.1887 |

对应文件：

- `results/key_results/case14/main/attack_summary.json`
- `results/key_results/case118/main/attack_summary.json`
- `results/key_results/main_results_summary.csv`

更详细的保留范围、参数和结果目录说明见：

- `docs/主线方案说明.md`
