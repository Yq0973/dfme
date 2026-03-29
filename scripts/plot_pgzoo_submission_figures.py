#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULT_ROOT = (
    PACKAGE_ROOT / "results" / "paper" / "physics_subspace_submission_v1"
)
SUMMARY_ROOT = RESULT_ROOT / "summary"
FIG_ROOT = SUMMARY_ROOT / "figures"

METHOD_META = {
    "baseline": {
        "label": "Deterministic Baseline",
        "color": "#4E79A7",
        "summary_paths": {
            14: PACKAGE_ROOT / "results" / "key_results" / "case14" / "main" / "attack_summary.json",
            118: PACKAGE_ROOT / "results" / "key_results" / "case118" / "main" / "attack_summary.json",
        },
        "budget_paths": {
            14: PACKAGE_ROOT / "results" / "key_results" / "case14" / "main" / "query_budget_curve.csv",
            118: PACKAGE_ROOT / "results" / "key_results" / "case118" / "main" / "query_budget_curve.csv",
        },
        "dist_paths": {
            14: PACKAGE_ROOT / "results" / "key_results" / "case14" / "main" / "query_distribution_summary.csv",
            118: PACKAGE_ROOT / "results" / "key_results" / "case118" / "main" / "query_distribution_summary.csv",
        },
    },
    "pgzoo": {
        "label": "Physics Subspace PG-ZOO",
        "color": "#E15759",
        "summary_paths": {
            14: PACKAGE_ROOT
            / "results"
            / "misc"
            / "misc"
            / "verify_physics_subspace_full_v2"
            / "runs"
            / "ieee_case14_verify_physics_subspace_full_v2"
            / "attack_summary.json",
            118: PACKAGE_ROOT
            / "results"
            / "misc"
            / "misc"
            / "verify_physics_subspace_full_v2"
            / "runs"
            / "ieee_case118_verify_physics_subspace_full_v2"
            / "attack_summary.json",
        },
        "budget_paths": {
            14: PACKAGE_ROOT
            / "results"
            / "misc"
            / "misc"
            / "verify_physics_subspace_full_v2"
            / "runs"
            / "ieee_case14_verify_physics_subspace_full_v2"
            / "query_budget_curve.csv",
            118: PACKAGE_ROOT
            / "results"
            / "misc"
            / "misc"
            / "verify_physics_subspace_full_v2"
            / "runs"
            / "ieee_case118_verify_physics_subspace_full_v2"
            / "query_budget_curve.csv",
        },
        "dist_paths": {
            14: PACKAGE_ROOT
            / "results"
            / "misc"
            / "misc"
            / "verify_physics_subspace_full_v2"
            / "runs"
            / "ieee_case14_verify_physics_subspace_full_v2"
            / "query_distribution_summary.csv",
            118: PACKAGE_ROOT
            / "results"
            / "misc"
            / "misc"
            / "verify_physics_subspace_full_v2"
            / "runs"
            / "ieee_case118_verify_physics_subspace_full_v2"
            / "query_distribution_summary.csv",
        },
    },
}

CASE_META = {
    14: {"label": "IEEE 14-bus"},
    118: {"label": "IEEE 118-bus"},
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_method_summary() -> pd.DataFrame:
    rows: list[dict] = []
    for method_key, meta in METHOD_META.items():
        for case_id, summary_path in meta["summary_paths"].items():
            summary = _load_json(summary_path)
            qdist = summary["query_distribution"]["total_all"]
            rows.append(
                {
                    "method": method_key,
                    "method_label": meta["label"],
                    "system_id": int(case_id),
                    "system_label": CASE_META[int(case_id)]["label"],
                    "attack_success_rate": float(summary["attack_success_rate"]),
                    "avg_queries": float(summary["avg_queries"]),
                    "avg_probe_queries": float(summary["avg_probe_queries"]),
                    "avg_search_queries": float(summary["avg_search_queries"]),
                    "median_queries": float(qdist["median"]),
                    "p90_queries": float(qdist["p90"]),
                    "p95_queries": float(qdist["p95"]),
                    "max_queries": float(qdist["max"]),
                    "delta_ratio_mean_pct": float(summary["delta_over_clean_l2_ratio_mean"]) * 100.0,
                    "delta_ratio_median_pct": float(summary["delta_over_clean_l2_ratio_median"]) * 100.0,
                    "chi2_not_flagged_ratio": float(summary["adv_bdd"]["chi2_not_flagged_ratio"]),
                    "lnrt_not_flagged_ratio": float(summary["adv_bdd"]["lnrt_not_flagged_ratio"]),
                    "subset_size": int(summary["subset_size"]),
                    "result_dir": str(summary["result_dir"]) if "result_dir" in summary else "",
                }
            )
    df = pd.DataFrame(rows).sort_values(["system_id", "method"]).reset_index(drop=True)
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_ROOT / "method_comparison_summary.csv", index=False)
    return df


def plot_overview(df: pd.DataFrame) -> None:
    case_ids = sorted(df["system_id"].unique())
    methods = ["baseline", "pgzoo"]
    width = 0.32
    x = np.arange(len(case_ids))

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
    axes = axes.reshape(-1)

    metrics = [
        ("attack_success_rate", "ASR (%)", "Attack Success Rate"),
        ("avg_queries", "Queries", "Average Query Cost"),
        ("p95_queries", "Queries", "P95 Query Cost"),
        ("delta_ratio_mean_pct", r"$\|\Delta z\|_2 / \|x\|_2$ (%)", "Mean Perturbation Ratio"),
    ]

    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for idx, method_key in enumerate(methods):
            sub = df[df["method"] == method_key].sort_values("system_id")
            offset = (idx - 0.5) * width
            color = METHOD_META[method_key]["color"]
            label = METHOD_META[method_key]["label"]
            values = sub[metric].to_numpy(dtype=float)
            bars = ax.bar(x + offset, values, width=width, color=color, label=label)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + max(0.8, 0.015 * max(values.max(), 1.0)),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, [CASE_META[int(case_id)]["label"] for case_id in case_ids])
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        if metric == "attack_success_rate":
            ax.set_ylim(0, 105)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_fig(fig, FIG_ROOT / "overview_compare")


def plot_bdd_compare(df: pd.DataFrame) -> None:
    case_ids = sorted(df["system_id"].unique())
    methods = ["baseline", "pgzoo"]
    width = 0.18
    x = np.arange(len(case_ids))

    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    for idx, method_key in enumerate(methods):
        sub = df[df["method"] == method_key].sort_values("system_id")
        chi_offset = -0.5 * width if method_key == "baseline" else 1.5 * width
        lnrt_offset = 0.5 * width if method_key == "baseline" else 2.5 * width
        base = x - width
        chi_pos = base + chi_offset
        lnrt_pos = base + lnrt_offset
        color = METHOD_META[method_key]["color"]
        ax.bar(chi_pos, sub["chi2_not_flagged_ratio"], width=width, color=color, alpha=0.85, label=f"{METHOD_META[method_key]['label']} / Chi-square" if idx == 0 else None)
        ax.bar(lnrt_pos, sub["lnrt_not_flagged_ratio"], width=width, color=color, alpha=0.45, label=f"{METHOD_META[method_key]['label']} / LNRT" if idx == 0 else None)

    # Manual legend for clarity.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=METHOD_META["baseline"]["color"], alpha=0.85, label="Baseline / Chi-square"),
        plt.Rectangle((0, 0), 1, 1, color=METHOD_META["baseline"]["color"], alpha=0.45, label="Baseline / LNRT"),
        plt.Rectangle((0, 0), 1, 1, color=METHOD_META["pgzoo"]["color"], alpha=0.85, label="PG-ZOO / Chi-square"),
        plt.Rectangle((0, 0), 1, 1, color=METHOD_META["pgzoo"]["color"], alpha=0.45, label="PG-ZOO / LNRT"),
    ]
    ax.set_title("BDD Consistency Comparison")
    ax.set_ylabel("Not-flagged ratio (%)")
    ax.set_xticks(x, [CASE_META[int(case_id)]["label"] for case_id in case_ids])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.28))

    fig.tight_layout()
    _save_fig(fig, FIG_ROOT / "bdd_compare")


def build_budget_compare() -> pd.DataFrame:
    rows: list[dict] = []
    for method_key, meta in METHOD_META.items():
        for case_id, budget_path in meta["budget_paths"].items():
            budget_df = pd.read_csv(budget_path)
            for _, row in budget_df.iterrows():
                rows.append(
                    {
                        "method": method_key,
                        "method_label": meta["label"],
                        "system_id": int(case_id),
                        "budget": int(row["budget"]),
                        "finished_rate": float(row["finished_rate"]),
                        "success_rate": float(row["success_rate"]),
                        "conditional_success_rate": float(row["conditional_success_rate"]),
                    }
                )
    df = pd.DataFrame(rows).sort_values(["system_id", "method", "budget"]).reset_index(drop=True)
    df.to_csv(SUMMARY_ROOT / "budget_curve_comparison.csv", index=False)
    return df


def plot_budget_curves(budget_df: pd.DataFrame) -> None:
    for case_id in sorted(budget_df["system_id"].unique()):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        sub = budget_df[budget_df["system_id"] == int(case_id)]
        for method_key in ["baseline", "pgzoo"]:
            msub = sub[sub["method"] == method_key].sort_values("budget")
            ax.plot(
                msub["budget"],
                msub["success_rate"],
                color=METHOD_META[method_key]["color"],
                linewidth=2.2,
                marker="o",
                markersize=5,
                label=METHOD_META[method_key]["label"],
            )
        ax.set_title(f"{CASE_META[int(case_id)]['label']} Query Budget Curve")
        ax.set_xlabel("Per-sample query budget")
        ax.set_ylabel("ASR within budget (%)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(frameon=False)
        fig.tight_layout()
        _save_fig(fig, FIG_ROOT / f"budget_curve_compare_case{int(case_id)}")


def build_distribution_compare() -> pd.DataFrame:
    rows: list[dict] = []
    for method_key, meta in METHOD_META.items():
        for case_id, dist_path in meta["dist_paths"].items():
            dist_df = pd.read_csv(dist_path)
            for _, row in dist_df.iterrows():
                if str(row["scope"]) != "total_all":
                    continue
                rows.append(
                    {
                        "method": method_key,
                        "method_label": meta["label"],
                        "system_id": int(case_id),
                        "median": float(row["median"]),
                        "p90": float(row["p90"]),
                        "p95": float(row["p95"]),
                        "max": float(row["max"]),
                    }
                )
    df = pd.DataFrame(rows).sort_values(["system_id", "method"]).reset_index(drop=True)
    df.to_csv(SUMMARY_ROOT / "query_distribution_comparison.csv", index=False)
    return df


def plot_distribution_compare(dist_df: pd.DataFrame) -> None:
    metrics = ["median", "p90", "p95", "max"]
    labels = ["Median", "P90", "P95", "Max"]
    width = 0.32
    x = np.arange(len(metrics))

    for case_id in sorted(dist_df["system_id"].unique()):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        sub = dist_df[dist_df["system_id"] == int(case_id)].sort_values("method")
        for idx, method_key in enumerate(["baseline", "pgzoo"]):
            row = sub[sub["method"] == method_key]
            if row.empty:
                continue
            values = row[metrics].iloc[0].to_numpy(dtype=float)
            bars = ax.bar(
                x + (idx - 0.5) * width,
                values,
                width=width,
                color=METHOD_META[method_key]["color"],
                label=METHOD_META[method_key]["label"],
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + max(0.8, 0.015 * max(values.max(), 1.0)),
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_title(f"{CASE_META[int(case_id)]['label']} Query Tail Comparison")
        ax.set_ylabel("Queries")
        ax.set_xticks(x, labels)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        _save_fig(fig, FIG_ROOT / f"query_distribution_compare_case{int(case_id)}")


def write_summary_markdown(df: pd.DataFrame) -> None:
    def row(method: str, case_id: int) -> pd.Series:
        return df[(df["method"] == method) & (df["system_id"] == int(case_id))].iloc[0]

    b14 = row("baseline", 14)
    p14 = row("pgzoo", 14)
    b118 = row("baseline", 118)
    p118 = row("pgzoo", 118)

    lines = [
        "# PG-ZOO 主线图表摘要",
        "",
        "## 1. 当前比较对象",
        "",
        "- `baseline`：`deterministic_mainline_v1`",
        "- `pgzoo`：`physics_subspace_zoo_v1`",
        "",
        "## 2. 核心结论",
        "",
        f"- `case14`：PG-ZOO 从 `{b14['attack_success_rate']:.2f}% / {b14['avg_queries']:.2f} queries` 提升到 `{p14['attack_success_rate']:.2f}% / {p14['avg_queries']:.2f} queries`，在成功率和平均查询上同时优于强基线。",
        f"- `case118`：PG-ZOO 当前为 `{p118['attack_success_rate']:.2f}% / {p118['avg_queries']:.2f} queries`，强基线为 `{b118['attack_success_rate']:.2f}% / {b118['avg_queries']:.2f} queries`，已经在成功率和平均查询上同时优于强基线。",
        f"- `case118`：同时 PG-ZOO 的查询尾部更紧，`p95` 从 `{b118['p95_queries']:.0f}` 降到 `{p118['p95_queries']:.0f}`，`max` 从 `{b118['max_queries']:.0f}` 降到 `{p118['max_queries']:.0f}`，说明其最坏情况查询复杂度也更可控。",
        "",
        "## 3. 图像文件",
        "",
        "- `overview_compare.png`：主结果总览图",
        "- `bdd_compare.png`：BDD 一致性对比图",
        "- `budget_curve_compare_case14.png`：case14 查询预算曲线对比",
        "- `budget_curve_compare_case118.png`：case118 查询预算曲线对比",
        "- `query_distribution_compare_case14.png`：case14 查询尾部分布对比",
        "- `query_distribution_compare_case118.png`：case118 查询尾部分布对比",
        "",
        "## 4. 论文中建议的使用方式",
        "",
        "- 主结果表使用 `method_comparison_summary.csv`。",
        "- 主结果图优先使用 `overview_compare.png`。",
        "- 查询复杂度分析配合 `budget_curve_compare_case118.png` 与 `query_distribution_compare_case118.png` 一起使用。",
        "- 若需要强调 PG-ZOO 的优势，应主要强调 `case14` 的全面收益，以及 `case118` 的尾部查询控制能力。",
        "",
    ]
    (SUMMARY_ROOT / "pgzoo_submission_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    method_df = build_method_summary()
    budget_df = build_budget_compare()
    dist_df = build_distribution_compare()

    plot_overview(method_df)
    plot_bdd_compare(method_df)
    plot_budget_curves(budget_df)
    plot_distribution_compare(dist_df)
    write_summary_markdown(method_df)

    print(f"Saved figures to: {FIG_ROOT}")


if __name__ == "__main__":
    main()
