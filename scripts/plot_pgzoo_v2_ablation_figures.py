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


OUT_ROOT = (
    PACKAGE_ROOT
    / "results"
    / "paper"
    / "physics_subspace_submission_v1"
    / "summary"
    / "ablation_v2"
)
FIG_ROOT = OUT_ROOT / "figures"


ABLATION_META = [
    {
        "system_id": 14,
        "variant": "case14_main",
        "label": "Case14 Mainline",
        "color": "#E15759",
        "summary_path": PACKAGE_ROOT
        / "results"
        / "misc"
        / "misc"
        / "verify_physics_subspace_full_v2"
        / "runs"
        / "ieee_case14_verify_physics_subspace_full_v2"
        / "attack_summary.json",
    },
    {
        "system_id": 14,
        "variant": "case14_no_region_shrink",
        "label": "Case14 No Region Shrink",
        "color": "#4E79A7",
        "summary_path": PACKAGE_ROOT
        / "results"
        / "paper"
        / "misc"
        / "paper_physics_subspace_ablation_case14_noshrink_full"
        / "runs"
        / "ieee_case14_paper_physics_subspace_ablation_case14_noshrink_full"
        / "attack_summary.json",
    },
    {
        "system_id": 118,
        "variant": "case118_hardsupport64",
        "label": "Case118 Hard Support-64",
        "color": "#4E79A7",
        "summary_path": PACKAGE_ROOT
        / "results"
        / "paper"
        / "misc"
        / "paper_physics_subspace_ablation_case118_hardsupport64_full"
        / "runs"
        / "ieee_case118_paper_physics_subspace_ablation_case118_hardsupport64_full"
        / "attack_summary.json",
    },
    {
        "system_id": 118,
        "variant": "case118_main",
        "label": "Case118 Mainline",
        "color": "#E15759",
        "summary_path": PACKAGE_ROOT
        / "results"
        / "misc"
        / "misc"
        / "verify_physics_subspace_full_v2"
        / "runs"
        / "ieee_case118_verify_physics_subspace_full_v2"
        / "attack_summary.json",
    },
    {
        "system_id": 118,
        "variant": "case118_no_semantic",
        "label": "Case118 No Semantic Preserve",
        "color": "#76B7B2",
        "summary_path": PACKAGE_ROOT
        / "results"
        / "paper"
        / "misc"
        / "paper_physics_subspace_ablation_nosemantic_full"
        / "runs"
        / "ieee_case118_paper_physics_subspace_ablation_nosemantic_full"
        / "attack_summary.json",
    },
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_ablation_summary() -> pd.DataFrame:
    rows: list[dict] = []
    for meta in ABLATION_META:
        summary = _load_json(meta["summary_path"])
        qdist = summary["query_distribution"]["total_all"]
        fdia = summary.get("fdia_effect_summary", {})
        rows.append(
            {
                "system_id": int(meta["system_id"]),
                "variant": meta["variant"],
                "label": meta["label"],
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
                "state_projection_ratio_mean": float(fdia.get("state_projection_ratio_mean", np.nan)),
                "measurement_projection_ratio_mean": float(fdia.get("measurement_projection_ratio_mean", np.nan)),
                "state_offsupport_energy_ratio_mean": float(fdia.get("state_offsupport_energy_ratio_mean", np.nan)),
                "result_dir": str(summary.get("result_dir", "")),
            }
        )
    df = pd.DataFrame(rows).sort_values(["system_id", "variant"]).reset_index(drop=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_ROOT / "ablation_summary.csv", index=False)
    return df


def plot_case14_region_shrink(df: pd.DataFrame) -> None:
    sub = df[df["system_id"] == 14].copy()
    order = ["case14_main", "case14_no_region_shrink"]
    sub["variant"] = pd.Categorical(sub["variant"], categories=order, ordered=True)
    sub = sub.sort_values("variant")
    metrics = [
        ("attack_success_rate", "ASR (%)"),
        ("avg_queries", "Avg Queries"),
        ("p95_queries", "P95 Queries"),
    ]
    labels = sub["label"].tolist()
    colors = [next(meta["color"] for meta in ABLATION_META if meta["variant"] == key) for key in sub["variant"]]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))
    for ax, (metric, title) in zip(axes, metrics):
        values = sub[metric].to_numpy(dtype=float)
        bars = ax.bar(np.arange(len(labels)), values, color=colors, width=0.58)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + max(0.6, 0.02 * max(values.max(), 1.0)),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=12, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        if metric == "attack_success_rate":
            ax.set_ylim(0, 105)
    fig.suptitle("Case14 Ablation: Physical Region Shrinkage", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, FIG_ROOT / "case14_region_shrink_ablation")


def plot_case118_support(df: pd.DataFrame) -> None:
    sub = df[df["variant"].isin(["case118_hardsupport64", "case118_main"])].copy()
    order = ["case118_hardsupport64", "case118_main"]
    sub["variant"] = pd.Categorical(sub["variant"], categories=order, ordered=True)
    sub = sub.sort_values("variant")
    metrics = [
        ("attack_success_rate", "ASR (%)"),
        ("avg_queries", "Avg Queries"),
        ("p95_queries", "P95 Queries"),
    ]
    labels = sub["label"].tolist()
    colors = [next(meta["color"] for meta in ABLATION_META if meta["variant"] == key) for key in sub["variant"]]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))
    for ax, (metric, title) in zip(axes, metrics):
        values = sub[metric].to_numpy(dtype=float)
        bars = ax.bar(np.arange(len(labels)), values, color=colors, width=0.58)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + max(0.6, 0.02 * max(values.max(), 1.0)),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=12, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        if metric == "attack_success_rate":
            ax.set_ylim(0, 105)
    fig.suptitle("Case118 Ablation: Hard Support Cut vs Full-Support Physical Subspace", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, FIG_ROOT / "case118_support_ablation")


def plot_case118_semantic(df: pd.DataFrame) -> None:
    sub = df[df["variant"].isin(["case118_main", "case118_no_semantic"])].copy()
    order = ["case118_main", "case118_no_semantic"]
    sub["variant"] = pd.Categorical(sub["variant"], categories=order, ordered=True)
    sub = sub.sort_values("variant")
    labels = sub["label"].tolist()
    colors = [next(meta["color"] for meta in ABLATION_META if meta["variant"] == key) for key in sub["variant"]]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))

    perf_metrics = [
        ("attack_success_rate", "ASR (%)"),
        ("avg_queries", "Avg Queries"),
    ]
    width = 0.34
    x = np.arange(len(perf_metrics))
    for idx, (_, row) in enumerate(sub.iterrows()):
        values = [float(row[m]) for m, _ in perf_metrics]
        bars = axes[0].bar(x + (idx - 0.5) * width, values, width=width, color=colors[idx], label=labels[idx])
        for bar, value in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                value + max(0.25, 0.015 * max(values)),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0].set_xticks(x, [t for _, t in perf_metrics])
    axes[0].set_title("Attack Metrics")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend(frameon=False)

    semantic_metrics = [
        ("state_projection_ratio_mean", "State Proj"),
        ("measurement_projection_ratio_mean", "Meas Proj"),
        ("state_offsupport_energy_ratio_mean", "Off-support"),
    ]
    x2 = np.arange(len(semantic_metrics))
    for idx, (_, row) in enumerate(sub.iterrows()):
        values = [float(row[m]) for m, _ in semantic_metrics]
        bars = axes[1].bar(x2 + (idx - 0.5) * width, values, width=width, color=colors[idx], label=labels[idx])
        for bar, value in zip(bars, values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.004,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[1].set_xticks(x2, [t for _, t in semantic_metrics])
    axes[1].set_title("FDIA Semantic Metrics")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)

    fig.suptitle("Case118 Ablation: FDIA Semantic Preservation", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, FIG_ROOT / "case118_semantic_ablation")


def write_markdown(df: pd.DataFrame) -> None:
    case14_main = df[df["variant"] == "case14_main"].iloc[0]
    case14_noshrink = df[df["variant"] == "case14_no_region_shrink"].iloc[0]
    case118_hard = df[df["variant"] == "case118_hardsupport64"].iloc[0]
    case118_main = df[df["variant"] == "case118_main"].iloc[0]
    case118_nosem = df[df["variant"] == "case118_no_semantic"].iloc[0]

    lines = [
        "# PG-ZOO v2 精炼消融总结",
        "",
        "## 1. 核心结论",
        "",
        f"- `case14`：去掉物理区域收缩后，`ASR` 从 `{case14_main['attack_success_rate']:.2f}%` 降到 `{case14_noshrink['attack_success_rate']:.2f}%`，`avg queries` 从 `{case14_main['avg_queries']:.2f}` 上升到 `{case14_noshrink['avg_queries']:.2f}`，说明小系统上的主收益确实来自局部物理区域选择。",
        f"- `case118`：旧版 `hard support-64` 为 `{case118_hard['attack_success_rate']:.2f}% / {case118_hard['avg_queries']:.2f} queries`，新主线 `full support + physical subspace` 为 `{case118_main['attack_success_rate']:.2f}% / {case118_main['avg_queries']:.2f} queries`，说明过强的 support 硬截断会同时损害成功率与查询效率。",
        f"- `case118`：去掉 FDIA 语义保持后，`ASR` 从 `{case118_main['attack_success_rate']:.2f}%` 轻微下降到 `{case118_nosem['attack_success_rate']:.2f}%`，`avg queries` 从 `{case118_main['avg_queries']:.2f}` 升至 `{case118_nosem['avg_queries']:.2f}`；同时 `state projection` 从 `{case118_main['state_projection_ratio_mean']:.3f}` 降到 `{case118_nosem['state_projection_ratio_mean']:.3f}`，`off-support energy` 从 `{case118_main['state_offsupport_energy_ratio_mean']:.3f}` 升到 `{case118_nosem['state_offsupport_energy_ratio_mean']:.3f}`。",
        "",
        "## 2. 论文写作建议",
        "",
        "- `case14` 重点强调物理区域收缩的必要性。",
        "- `case118` 重点强调不要把物理先验误用为过强的 hard support 截断，而应将其用于构造低维物理搜索坐标系。",
        "- FDIA 语义保持项应表述为“轻量正则项”，作用是稳定语义一致性，而不是主要成功率来源。",
        "",
        "## 3. 图表文件",
        "",
        "- `case14_region_shrink_ablation.png`：case14 去区域收缩消融。",
        "- `case118_support_ablation.png`：case118 旧 hard-support 与新主线对比。",
        "- `case118_semantic_ablation.png`：case118 语义保持项消融。",
        "- `ablation_summary.csv`：所有精炼消融数据表。",
    ]
    (OUT_ROOT / "ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    df = build_ablation_summary()
    plot_case14_region_shrink(df)
    plot_case118_support(df)
    plot_case118_semantic(df)
    write_markdown(df)
    print(f"Saved ablation summary to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
