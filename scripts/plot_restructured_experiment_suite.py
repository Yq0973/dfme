#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "results" / "restructured_experiment_suite_20260329"
FIG = SUITE / "figures"
BUDGET_SWEEP_SUMMARY = ROOT / "results" / "key_results" / "budget_sweep" / "pgzoo_cap20_sweep_summary.csv"


def _setup_style() -> None:
    # Use a broad fallback list for Chinese labels across Windows/macOS/Linux.
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def _short_variant(name: str) -> str:
    mapping = {
        "deterministic_mainline_v1": "确定性主线",
        "physics_subspace_zoo_v1": "PG-ZOO（历史主线）",
        "physics_subspace_zoo_v1_noR": "PG-ZOO（noR，全量）",
        "physics_subspace_zoo_v1_full117": "PG-ZOO（full117）",
        "physics_subspace_zoo_v1_support64": "PG-ZOO（support64）",
        "zoo_baseline": "Vanilla ZOO（无降维）",
        "mainline": "主线 PG-ZOO",
        "sparse_only": "仅稀疏策略",
        "sparse_combined": "稀疏混合策略",
    }
    return mapping.get(name, name)


def _system_labels(systems: list[int]) -> list[str]:
    return [f"IEEE {int(s)}-bus" for s in systems]


def _line_style(i: int) -> tuple[str, str]:
    markers = ["o", "s", "^", "D", "P", "X", "*"]
    colors = ["#4E79A7", "#E15759", "#59A14F", "#F28E2B", "#76B7B2", "#EDC948", "#B07AA1"]
    return markers[i % len(markers)], colors[i % len(colors)]


def plot_main_results() -> None:
    df = pd.read_csv(SUITE / "01_core_performance" / "main_results_with_baselines.csv")
    systems = sorted(int(v) for v in df["system_id"].unique())
    x = np.arange(len(systems))

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))
    metrics = [
        ("attack_success_rate", "攻击成功率 ASR（%）", "主结果：攻击成功率"),
        ("avg_queries", "平均查询次数", "主结果：查询开销"),
    ]

    variants = sorted(df["variant"].unique())
    for ax, (col, ylabel, title) in zip(axes, metrics):
        for i, variant in enumerate(variants):
            sub = df[df["variant"] == variant].set_index("system_id")
            vals = sub.reindex(systems)[col].to_numpy(dtype=float)
            marker, color = _line_style(i)
            ax.plot(
                x,
                vals,
                marker=marker,
                linewidth=2.2,
                markersize=7,
                color=color,
                label=_short_variant(variant),
            )
        ax.set_xticks(x, _system_labels(systems))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25, linestyle="--")
        if col == "attack_success_rate":
            ax.set_ylim(0, 105)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, FIG / "01_main_results_overview")

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))
    tail_metrics = [
        ("median_queries", "查询次数", "查询尾部：中位数（Median）"),
        ("p95_queries", "查询次数", "查询尾部：95分位（P95）"),
    ]
    for ax, (col, ylabel, title) in zip(axes, tail_metrics):
        for i, variant in enumerate(variants):
            sub = df[df["variant"] == variant].set_index("system_id")
            vals = sub.reindex(systems)[col].to_numpy(dtype=float)
            marker, color = _line_style(i)
            ax.plot(
                x,
                vals,
                marker=marker,
                linewidth=2.2,
                markersize=7,
                color=color,
                label=_short_variant(variant),
            )
        ax.set_xticks(x, _system_labels(systems))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, FIG / "02_query_tail_overview")


def plot_budget_curves() -> None:
    df = pd.read_csv(SUITE / "01_core_performance" / "asr_vs_query_budget_noR.csv")
    systems = sorted(df["system_id"].unique())

    fig, axes = plt.subplots(1, len(systems), figsize=(6.4 * len(systems), 4.8), sharey=True)
    if len(systems) == 1:
        axes = [axes]

    for ax, sid in zip(axes, systems):
        sub = df[df["system_id"] == sid].sort_values("budget")
        ax.plot(sub["budget"], sub["success_rate"], color="#E15759", marker="o", linewidth=2.3, label="预算内成功率（ASR）")
        ax.plot(sub["budget"], sub["finished_rate"], color="#9C755F", marker="s", linewidth=2.0, linestyle="--", label="预算内完成率")
        ax.set_title(f"IEEE {int(sid)}-bus")
        ax.set_xlabel("查询预算")
        ax.set_ylabel("比例（%）")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.25, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, FIG / "03_budget_curves_noR")


def plot_asr_vs_perturbation_cap() -> None:
    if not BUDGET_SWEEP_SUMMARY.exists():
        return
    df = pd.read_csv(BUDGET_SWEEP_SUMMARY)
    systems = sorted(int(v) for v in df["system_id"].unique())
    variants = ["minimum_success", "fixed_budget", "boundary_push"]
    colors = {
        "minimum_success": "#4E79A7",
        "fixed_budget": "#59A14F",
        "boundary_push": "#E15759",
    }
    markers = {"minimum_success": "o", "fixed_budget": "^", "boundary_push": "s"}
    labels = {
        "minimum_success": "最小成功准则",
        "fixed_budget": "固定预算准则",
        "boundary_push": "边界推进准则",
    }

    fig, axes = plt.subplots(1, len(systems), figsize=(6.5 * len(systems), 4.9), sharey=True)
    if len(systems) == 1:
        axes = [axes]

    for ax, sid in zip(axes, systems):
        sub = df[df["system_id"] == sid]
        for v in variants:
            cur = sub[sub["variant"] == v].sort_values("epsilon_pct")
            if cur.empty:
                continue
            ax.plot(
                cur["epsilon_pct"],
                cur["attack_success_rate"],
                color=colors[v],
                marker=markers[v],
                linewidth=2.2,
                markersize=7,
                label=labels[v],
            )
        ax.set_title(f"IEEE {int(sid)}-bus")
        ax.set_xlabel("扰动上限（%）")
        ax.set_ylabel("攻击成功率 ASR（%）")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.25, linestyle="--")

    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("不同扰动等级下的 ASR 对比", y=1.10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, FIG / "03b_asr_vs_perturbation_cap")


def _mainline_asr_by_system() -> dict[int, float]:
    main_path = SUITE / "01_core_performance" / "main_results_with_baselines.csv"
    if not main_path.exists():
        return {}
    mdf = pd.read_csv(main_path)
    preferred = ["physics_subspace_zoo_v1_noR", "physics_subspace_zoo_v1"]
    for v in preferred:
        cur = mdf[mdf["variant"] == v]
        if not cur.empty:
            return {
                int(r["system_id"]): float(r["attack_success_rate"])
                for _, r in cur.iterrows()
            }
    return {}


def plot_asr_vs_perturbation_cap_aligned() -> None:
    if not BUDGET_SWEEP_SUMMARY.exists():
        return
    df = pd.read_csv(BUDGET_SWEEP_SUMMARY)
    systems = sorted(int(v) for v in df["system_id"].unique())
    mainline_asr = _mainline_asr_by_system()

    variants = ["minimum_success", "fixed_budget", "boundary_push"]
    colors = {
        "minimum_success": "#4E79A7",
        "fixed_budget": "#59A14F",
        "boundary_push": "#E15759",
    }
    markers = {"minimum_success": "o", "fixed_budget": "^", "boundary_push": "s"}
    labels = {
        "minimum_success": "最小成功准则",
        "fixed_budget": "固定预算准则",
        "boundary_push": "边界推进准则",
    }

    fig, axes = plt.subplots(1, len(systems), figsize=(6.8 * len(systems), 5.1), sharey=True)
    if len(systems) == 1:
        axes = [axes]

    first_sid = systems[0]
    for ax, sid in zip(axes, systems):
        sub = df[df["system_id"] == sid]
        for v in variants:
            cur = sub[sub["variant"] == v].sort_values("epsilon_pct")
            if cur.empty:
                continue
            ax.plot(
                cur["epsilon_pct"],
                cur["attack_success_rate"],
                color=colors[v],
                marker=markers[v],
                linewidth=2.2,
                markersize=7,
                label=labels[v],
            )

        anchor = mainline_asr.get(int(sid))
        if anchor is not None:
            anchor_label = "主结果ASR基线（noR）" if sid == first_sid else "_nolegend_"
            ax.axhline(anchor, color="#2F4B7C", linestyle="--", linewidth=2.0, label=anchor_label)
            ax.text(
                20.4,
                anchor + 0.4,
                f"{anchor:.1f}%",
                color="#2F4B7C",
                fontsize=9,
                va="bottom",
                ha="left",
            )

        ax.set_title(f"IEEE {int(sid)}-bus")
        ax.set_xlabel("扰动上限（%）")
        ax.set_ylabel("攻击成功率 ASR（%）")
        ax.set_xlim(1.0, 22.0)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.25, linestyle="--")

    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("不同扰动等级 ASR 与主结果口径对齐对照", y=1.12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, FIG / "03c_asr_vs_perturbation_cap_aligned")


def plot_asr_marginal_gain() -> None:
    if not BUDGET_SWEEP_SUMMARY.exists():
        return
    df = pd.read_csv(BUDGET_SWEEP_SUMMARY)
    # Use minimum-success as the primary view of attack efficiency under each cap.
    sub = df[df["variant"] == "minimum_success"].copy()
    if sub.empty:
        return

    systems = sorted(int(v) for v in sub["system_id"].unique())
    fig, axes = plt.subplots(1, len(systems), figsize=(6.4 * len(systems), 4.8), sharey=True)
    if len(systems) == 1:
        axes = [axes]

    for ax, sid in zip(axes, systems):
        cur = sub[sub["system_id"] == sid].sort_values("epsilon_pct").copy()
        cur["delta_asr"] = cur["attack_success_rate"].diff()
        gain = cur.dropna(subset=["delta_asr"])
        x = gain["epsilon_pct"].to_numpy(dtype=float)
        y = gain["delta_asr"].to_numpy(dtype=float)

        ax.plot(x, y, color="#2F4B7C", marker="o", linewidth=2.2, markersize=7, label="每+5%扰动上限的ASR增量")
        ax.axhline(0.0, color="#9C755F", linestyle="--", linewidth=1.8, label="零增益参考线")
        for xv, yv in zip(x, y):
            ax.text(xv, yv + (0.8 if yv >= 0 else -1.8), f"{yv:.1f}", fontsize=9, ha="center")

        ax.set_title(f"IEEE {int(sid)}-bus")
        ax.set_xlabel("扰动上限（%）")
        ax.set_ylabel("ASR边际增量（百分点）")
        ax.set_xticks(x)
        ax.grid(alpha=0.25, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("扰动预算边际收益：每+5%上限带来的ASR增量", y=1.10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, FIG / "06_marginal_asr_gain")


def plot_ablation() -> None:
    df = pd.read_csv(SUITE / "02_ablation" / "zoo_vs_mainline_compare_m256.csv")
    systems = sorted(int(v) for v in df["system_id"].unique())
    x = np.arange(len(systems))
    methods = ["zoo_baseline", "mainline", "sparse_only", "sparse_combined"]
    colors = {
        "zoo_baseline": "#4E79A7",
        "mainline": "#E15759",
        "sparse_only": "#59A14F",
        "sparse_combined": "#F28E2B",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))
    for ax, col, title in [
        (axes[0], "attack_success_rate", "消融对比（m=256）：攻击成功率"),
        (axes[1], "avg_queries", "消融对比（m=256）：平均查询次数"),
    ]:
        for i, method in enumerate(methods):
            sub = df[df["method"] == method].sort_values("system_id")
            vals = sub[col].to_numpy(dtype=float)
            marker, _ = _line_style(i)
            ax.plot(
                x,
                vals,
                marker=marker,
                linewidth=2.2,
                markersize=7,
                color=colors[method],
                label=_short_variant(method),
            )
        ax.set_xticks(x, _system_labels(systems))
        ax.set_title(title)
        ax.grid(alpha=0.25, linestyle="--")
        if col == "attack_success_rate":
            ax.set_ylim(0, 105)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, FIG / "04_ablation_m256")


def plot_physical_compliance() -> None:
    df = pd.read_csv(SUITE / "03_physical_compliance" / "physical_compliance_noR.csv")
    systems = sorted(int(v) for v in df["system_id"].unique())
    x = np.arange(len(systems))
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))

    ax = axes[0]
    ax.plot(x, df["clean_chi2_not_flagged_ratio"], marker="o", linewidth=2.2, color="#59A14F", label="清洁样本 Chi-square")
    ax.plot(x, df["adv_chi2_not_flagged_ratio"], marker="s", linewidth=2.2, color="#E15759", label="对抗样本 Chi-square")
    ax.plot(x, df["clean_lnrt_not_flagged_ratio"], marker="^", linewidth=2.0, color="#76B7B2", linestyle="--", label="清洁样本 LNRT")
    ax.plot(x, df["adv_lnrt_not_flagged_ratio"], marker="D", linewidth=2.0, color="#F28E2B", linestyle="--", label="对抗样本 LNRT")
    ax.set_xticks(x, _system_labels(systems))
    ax.set_ylim(0, 105)
    ax.set_title("物理一致性：BDD 未告警比例")
    ax.set_ylabel("未告警比例（%）")
    ax.grid(alpha=0.25, linestyle="--")

    ax = axes[1]
    ax.plot(x, df["delta_over_clean_l2_ratio_mean"] * 100.0, marker="o", linewidth=2.2, color="#4E79A7", label="均值 ||Delta z||/||x||")
    ax.plot(x, df["delta_over_clean_l2_ratio_median"] * 100.0, marker="s", linewidth=2.2, color="#9C755F", label="中位数 ||Delta z||/||x||")
    ax.plot(x, df["state_offsupport_energy_ratio_mean"] * 100.0, marker="^", linewidth=2.0, color="#B07AA1", linestyle="--", label="均值 支撑外能量比")
    ax.set_xticks(x, _system_labels(systems))
    ax.set_title("扰动隐蔽性与骨架保持")
    ax.set_ylabel("比例（%）")
    ax.grid(alpha=0.25, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    fig.legend(handles + h2, labels + l2, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, FIG / "05_physical_compliance")


def write_figure_index() -> None:
    lines = [
        "# 可视化图表索引",
        "",
        "- 01_main_results_overview.png：主结果线图（攻击成功率 + 平均查询）",
        "  - 线条设置：确定性主线、PG-ZOO（历史主线）、PG-ZOO（noR，全量）、PG-ZOO（full117）、PG-ZOO（support64）",
        "- 02_query_tail_overview.png：查询尾部线图（Median + P95）",
        "  - 线条设置同 01 图，展示不同设置下的查询长尾",
        "- 03_budget_curves_noR.png：noR 主线预算曲线线图",
        "  - 红线：预算内成功率（ASR）；棕线虚线：预算内完成率",
        "- 03b_asr_vs_perturbation_cap.png：不同扰动等级（2/5/10/15/20%）下的 ASR 线图",
        "  - 线条设置：最小成功准则、固定预算准则、边界推进准则",
        "- 03c_asr_vs_perturbation_cap_aligned.png：在 03b 基础上叠加主结果 noR ASR 水平虚线",
        "  - 线条设置：最小成功准则、固定预算准则、边界推进准则、主结果ASR基线（noR）",
        "- 04_ablation_m256.png：m=256 消融线图",
        "  - 线条设置：Vanilla ZOO（无降维）、主线 PG-ZOO、仅稀疏策略、稀疏混合策略",
        "- 05_physical_compliance.png：物理一致性与隐蔽性线图",
        "  - 左图：清洁/对抗样本在 Chi-square 与 LNRT 下的未告警比例",
        "  - 右图：扰动均值/中位数与支撑外能量比",
        "- 06_marginal_asr_gain.png：扰动预算边际收益曲线（minimum-success口径）",
        "  - 线条设置：蓝线为每+5%扰动上限带来的ASR增量，棕色虚线为零增益参考",
    ]
    (FIG / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _setup_style()
    FIG.mkdir(parents=True, exist_ok=True)
    plot_main_results()
    plot_budget_curves()
    plot_asr_vs_perturbation_cap()
    plot_asr_vs_perturbation_cap_aligned()
    plot_ablation()
    plot_physical_compliance()
    plot_asr_marginal_gain()
    write_figure_index()
    print(f"Wrote figures to: {FIG}")


if __name__ == "__main__":
    main()
