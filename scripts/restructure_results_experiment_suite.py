#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "restructured_experiment_suite_20260329"

NO_R_RUNS = {
    14: RESULTS
    / "misc"
    / "misc"
    / "validate_noR_physics_subspace_20260329"
    / "runs"
    / "ieee_case14_validate_noR_physics_subspace_20260329"
    / "attack_summary.json",
    118: RESULTS
    / "misc"
    / "misc"
    / "validate_noR_physics_subspace_20260329"
    / "runs"
    / "ieee_case118_validate_noR_physics_subspace_20260329"
    / "attack_summary.json",
}

DETERMINISTIC_MAIN = RESULTS / "key_results" / "main_results_summary.csv"
PGZOO_KEY = RESULTS / "key_results" / "pgzoo_results_summary.csv"
ABLATION_V2 = (
    RESULTS
    / "paper"
    / "physics_subspace_submission_v1"
    / "summary"
    / "ablation_v2"
    / "ablation_summary.csv"
)
COMPARE_CASE14 = RESULTS / "misc" / "misc" / "m14" / "summary" / "zoo_sparse_compare_summary.csv"
COMPARE_CASE118 = RESULTS / "misc" / "misc" / "mc118r" / "summary" / "zoo_sparse_compare_summary.csv"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dirs() -> None:
    for name in [
        "01_core_performance",
        "02_ablation",
        "03_physical_compliance",
        "04_cross_model_generalization",
    ]:
        (OUT / name).mkdir(parents=True, exist_ok=True)


def build_core_tables() -> None:
    rows = []
    budget_rows = []
    for sid, path in NO_R_RUNS.items():
        s = _load_json(path)
        qdist = s["query_distribution"]["total_all"]
        rows.append(
            {
                "system_id": sid,
                "variant": "physics_subspace_zoo_v1_noR",
                "subset_size": int(s["subset_size"]),
                "attack_success_rate": float(s["attack_success_rate"]),
                "avg_queries": float(s["avg_queries"]),
                "median_queries": float(qdist["median"]),
                "p90_queries": float(qdist["p90"]),
                "p95_queries": float(qdist["p95"]),
                "max_queries": float(qdist["max"]),
                "avg_probe_queries": float(s.get("avg_probe_queries", 0.0)),
                "avg_search_queries": float(s.get("avg_search_queries", 0.0)),
                "noise_model": str(s.get("noise_model", "unknown")),
                "result_dir": str(s.get("result_dir", "")),
            }
        )
        for item in s.get("query_budget_curve", []):
            budget_rows.append(
                {
                    "system_id": sid,
                    "budget": int(item["budget"]),
                    "finished_rate": float(item["finished_rate"]),
                    "success_rate": float(item["success_rate"]),
                    "conditional_success_rate": float(item["conditional_success_rate"]),
                }
            )

    core_df = pd.DataFrame(rows).sort_values("system_id")
    core_df.to_csv(OUT / "01_core_performance" / "main_results_noR_full.csv", index=False)

    budget_df = pd.DataFrame(budget_rows).sort_values(["system_id", "budget"])
    budget_df.to_csv(OUT / "01_core_performance" / "asr_vs_query_budget_noR.csv", index=False)

    det = pd.read_csv(DETERMINISTIC_MAIN)
    det = det[[
        "system_id",
        "method",
        "subset_size",
        "attack_success_rate",
        "avg_queries",
        "median_queries",
        "p95_queries",
    ]].copy()
    det["variant"] = "deterministic_mainline_v1"

    pg = pd.read_csv(PGZOO_KEY)
    pg = pg[[
        "system_id",
        "variant",
        "subset_size",
        "attack_success_rate",
        "avg_queries",
        "median_queries",
        "p95_queries",
    ]].copy()

    combined = pd.concat(
        [
            det[[
                "system_id",
                "variant",
                "subset_size",
                "attack_success_rate",
                "avg_queries",
                "median_queries",
                "p95_queries",
            ]],
            pg,
            core_df[[
                "system_id",
                "variant",
                "subset_size",
                "attack_success_rate",
                "avg_queries",
                "median_queries",
                "p95_queries",
            ]],
        ],
        ignore_index=True,
    )
    combined = combined.sort_values(["system_id", "variant"])
    combined.to_csv(OUT / "01_core_performance" / "main_results_with_baselines.csv", index=False)


def build_ablation_tables() -> None:
    abl = pd.read_csv(ABLATION_V2)
    abl.to_csv(OUT / "02_ablation" / "ablation_summary_from_existing.csv", index=False)

    c14 = pd.read_csv(COMPARE_CASE14)
    c118 = pd.read_csv(COMPARE_CASE118)
    compare = pd.concat([c14, c118], ignore_index=True)
    compare = compare[[
        "system_id",
        "method",
        "subset_size",
        "attack_success_rate",
        "avg_queries",
        "avg_probe_queries",
        "avg_search_queries",
        "delta_over_clean_l2_ratio_mean",
        "state_support_retention_ratio_mean",
        "state_offsupport_energy_ratio_mean",
    ]]
    compare.to_csv(OUT / "02_ablation" / "zoo_vs_mainline_compare_m256.csv", index=False)


def build_physical_tables() -> None:
    rows = []
    for sid, path in NO_R_RUNS.items():
        s = _load_json(path)
        bdd_clean = s.get("clean_bdd", {})
        bdd_adv = s.get("adv_bdd", {})
        eff = s.get("fdia_effect_summary", {})
        rows.append(
            {
                "system_id": sid,
                "subset_size": int(s["subset_size"]),
                "clean_chi2_not_flagged_ratio": float(bdd_clean.get("chi2_not_flagged_ratio", 0.0)),
                "adv_chi2_not_flagged_ratio": float(bdd_adv.get("chi2_not_flagged_ratio", 0.0)),
                "clean_lnrt_not_flagged_ratio": float(bdd_clean.get("lnrt_not_flagged_ratio", 0.0)),
                "adv_lnrt_not_flagged_ratio": float(bdd_adv.get("lnrt_not_flagged_ratio", 0.0)),
                "delta_over_clean_l2_ratio_mean": float(s.get("delta_over_clean_l2_ratio_mean", 0.0)),
                "delta_over_clean_l2_ratio_median": float(s.get("delta_over_clean_l2_ratio_median", 0.0)),
                "state_projection_ratio_mean": float(eff.get("state_projection_ratio_mean", 0.0)),
                "measurement_projection_ratio_mean": float(eff.get("measurement_projection_ratio_mean", 0.0)),
                "state_offsupport_energy_ratio_mean": float(eff.get("state_offsupport_energy_ratio_mean", 0.0)),
            }
        )

    pd.DataFrame(rows).sort_values("system_id").to_csv(
        OUT / "03_physical_compliance" / "physical_compliance_noR.csv", index=False
    )


def write_index() -> None:
    lines = [
        "# 重构实验结果索引（2026-03-29）",
        "",
        "本目录按 `docs/实验清单.md` 的四个实验维度重构。",
        "",
        "## 01_core_performance",
        "- main_results_noR_full.csv：去除 R 后主线在 case14/118 的全量结果。",
        "- asr_vs_query_budget_noR.csv：去除 R 后预算-成功率曲线数据。",
        "- main_results_with_baselines.csv：与 deterministic 与既有 PG-ZOO 结果的汇总对照。",
        "",
        "## 02_ablation",
        "- ablation_summary_from_existing.csv：沿用既有 full ablation 汇总（含 no-shrink/hard-support/no-semantic）。",
        "- zoo_vs_mainline_compare_m256.csv：新增 m=256 的 ZOO 基线与主线补充对照。",
        "",
        "## 03_physical_compliance",
        "- physical_compliance_noR.csv：BDD 一致性与扰动隐蔽性统计。",
        "",
        "## 04_cross_model_generalization",
        "- 当前仓库暂无 GNN 目标模型与对应全流程评测脚本，待补充。",
        "",
        "## 说明",
        "- 本次新增的补充对照（zoo_vs_mainline）使用 `max_samples=256`，用于快速验证趋势。",
        "- 主线 noR 结果使用全量样本（max_samples=0）。",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "04_cross_model_generalization" / "README.md").write_text(
        "当前代码库未包含 GNN oracle 训练与攻击评测流水线，暂未完成实验清单第4部分。\n",
        encoding="utf-8",
    )


def main() -> None:
    _ensure_dirs()
    build_core_tables()
    build_ablation_tables()
    build_physical_tables()
    write_index()
    print(f"Wrote restructured suite to: {OUT}")


if __name__ == "__main__":
    main()
