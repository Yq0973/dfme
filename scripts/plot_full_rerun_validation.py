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
    PACKAGE_ROOT
    / "results"
    / "paper"
    / "results_digest"
    / "paper_results_digest_v1"
)
RUN_ROOT = RESULT_ROOT / "runs"
SUMMARY_ROOT = RESULT_ROOT / "summary"
FIG_ROOT = SUMMARY_ROOT / "figures"
KEY_PGZOO_SUMMARY = PACKAGE_ROOT / "results" / "key_results" / "pgzoo_results_summary.csv"
QUERY_BUDGET_RUN_ROOT = (
    PACKAGE_ROOT
    / "results"
    / "paper"
    / "query_budget"
    / "paper_query_budget_pgzoo_cap20_v1"
    / "runs"
)

CASE_META = {
    14: {"label": "IEEE 14-bus", "color": "#4E79A7"},
    118: {"label": "IEEE 118-bus", "color": "#E15759"},
}

SCOPE_META = {
    "total_all": {"label": "All", "color": "#4E79A7"},
    "total_success": {"label": "Success", "color": "#59A14F"},
    "total_failure": {"label": "Failure", "color": "#E15759"},
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def _run_dir(case_id: int) -> Path:
    return RUN_ROOT / f"ieee_case{int(case_id)}_paper_results_digest_v1_pgzoo_fullvalidate"


def _system_fig_dir(case_id: int) -> Path:
    return _run_dir(case_id) / "figures"


def load_case_summary(case_id: int) -> dict:
    return _load_json(_run_dir(case_id) / "attack_summary.json")


def _budget_sweep_summary_path(case_id: int, epsilon_pct: int) -> Path:
    return (
        QUERY_BUDGET_RUN_ROOT
        / f"ieee_case{int(case_id)}_paper_query_budget_pgzoo_cap20_v1_minimum_success_eps{int(epsilon_pct):02d}"
        / "attack_summary.json"
    )


def _load_budget_sweep_rows(case_id: int) -> list[dict]:
    rows: list[dict] = []
    for epsilon_pct in [2, 5, 10, 15, 20]:
        summary_path = _budget_sweep_summary_path(case_id, epsilon_pct)
        if not summary_path.exists():
            continue
        summary = _load_json(summary_path)
        rows.append(
            {
                "epsilon_pct": float(epsilon_pct),
                "attack_success_rate": float(summary["attack_success_rate"]),
                "realized_pct": float(summary["delta_over_clean_l2_ratio_mean"]) * 100.0,
            }
        )
    return rows


def build_overview_table(case_ids: list[int]) -> pd.DataFrame:
    rows = []
    for case_id in case_ids:
        summary = load_case_summary(case_id)
        rows.append(
            {
                "system_id": int(case_id),
                "label": CASE_META[int(case_id)]["label"],
                "attack_success_rate": float(summary["attack_success_rate"]),
                "avg_queries": float(summary["avg_queries"]),
                "avg_probe_queries": float(summary["avg_probe_queries"]),
                "avg_search_queries": float(summary["avg_search_queries"]),
                "delta_over_clean_l2_ratio_mean": float(
                    summary["delta_over_clean_l2_ratio_mean"]
                ),
                "delta_over_clean_l2_ratio_median": float(
                    summary["delta_over_clean_l2_ratio_median"]
                ),
                "chi2_not_flagged_ratio": float(summary["adv_bdd"]["chi2_not_flagged_ratio"]),
                "lnrt_not_flagged_ratio": float(summary["adv_bdd"]["lnrt_not_flagged_ratio"]),
                "subset_size": int(summary["subset_size"]),
                "result_dir": str(summary["result_dir"]),
            }
        )
    return pd.DataFrame(rows).sort_values("system_id").reset_index(drop=True)


def plot_overview(summary_df: pd.DataFrame) -> None:
    x = np.arange(summary_df.shape[0])
    labels = summary_df["label"].tolist()
    colors = [CASE_META[int(case_id)]["color"] for case_id in summary_df["system_id"]]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
    axes = axes.reshape(-1)

    axes[0].bar(x, summary_df["attack_success_rate"], color=colors, width=0.56)
    axes[0].set_title("Attack Success Rate")
    axes[0].set_ylabel("ASR (%)")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    for xi, yi in zip(x, summary_df["attack_success_rate"]):
        axes[0].text(xi, yi + 1.4, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x, summary_df["avg_probe_queries"], color="#A0CBE8", width=0.56, label="Probe")
    axes[1].bar(
        x,
        summary_df["avg_search_queries"],
        bottom=summary_df["avg_probe_queries"],
        color="#4E79A7",
        width=0.56,
        label="Search",
    )
    axes[1].set_title("Average Query Cost")
    axes[1].set_ylabel("Queries")
    axes[1].set_xticks(x, labels)
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    axes[1].legend(frameon=False)
    totals = summary_df["avg_probe_queries"] + summary_df["avg_search_queries"]
    for xi, yi in zip(x, totals):
        axes[1].text(xi, yi + 0.9, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

    axes[2].bar(x, summary_df["delta_over_clean_l2_ratio_mean"] * 100.0, color=colors, width=0.56)
    axes[2].set_title("Mean Perturbation Ratio")
    axes[2].set_ylabel(r"$\|\Delta z\|_2 / \|x\|_2$ (%)")
    axes[2].set_xticks(x, labels)
    axes[2].grid(axis="y", linestyle="--", alpha=0.25)
    for xi, yi in zip(x, summary_df["delta_over_clean_l2_ratio_mean"] * 100.0):
        axes[2].text(xi, yi + 0.45, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

    width = 0.28
    axes[3].bar(
        x - width / 2,
        summary_df["chi2_not_flagged_ratio"],
        color="#59A14F",
        width=width,
        label="Chi-square",
    )
    axes[3].bar(
        x + width / 2,
        summary_df["lnrt_not_flagged_ratio"],
        color="#9C755F",
        width=width,
        label="LNRT",
    )
    axes[3].set_title("BDD Consistency After Attack")
    axes[3].set_ylabel("Not-flagged ratio (%)")
    axes[3].set_xticks(x, labels)
    axes[3].set_ylim(0, 105)
    axes[3].grid(axis="y", linestyle="--", alpha=0.25)
    axes[3].legend(frameon=False)
    for xi, yi in zip(x - width / 2, summary_df["chi2_not_flagged_ratio"]):
        axes[3].text(xi, yi + 1.0, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, yi in zip(x + width / 2, summary_df["lnrt_not_flagged_ratio"]):
        axes[3].text(xi, yi + 1.0, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Full-Rerun Validation Overview", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, FIG_ROOT / "full_rerun_overview")


def plot_case_summary(case_id: int) -> None:
    meta = CASE_META[int(case_id)]
    summary = load_case_summary(case_id)
    asr = float(summary["attack_success_rate"])
    avg_probe = float(summary["avg_probe_queries"])
    avg_search = float(summary["avg_search_queries"])
    delta_mean = float(summary["delta_over_clean_l2_ratio_mean"]) * 100.0
    delta_median = float(summary["delta_over_clean_l2_ratio_median"]) * 100.0
    chi2_ratio = float(summary["adv_bdd"]["chi2_not_flagged_ratio"])
    lnrt_ratio = float(summary["adv_bdd"]["lnrt_not_flagged_ratio"])

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    axes = axes.reshape(-1)

    axes[0].bar([0], [asr], color=meta["color"], width=0.55)
    axes[0].set_title("Attack Success Rate")
    axes[0].set_ylabel("ASR (%)")
    axes[0].set_xticks([0], [meta["label"]])
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].text(0, asr + 1.4, f"{asr:.2f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar([0], [avg_probe], color="#A0CBE8", width=0.55, label="Probe")
    axes[1].bar([0], [avg_search], bottom=[avg_probe], color="#4E79A7", width=0.55, label="Search")
    axes[1].set_title("Average Query Cost")
    axes[1].set_ylabel("Queries")
    axes[1].set_xticks([0], [meta["label"]])
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    axes[1].legend(frameon=False)
    axes[1].text(0, avg_probe + avg_search + 0.8, f"{avg_probe + avg_search:.2f}", ha="center", va="bottom", fontsize=9)

    axes[2].bar(
        [-0.15, 0.15],
        [delta_mean, delta_median],
        color=[meta["color"], "#9C755F"],
        width=0.26,
    )
    axes[2].set_title("Perturbation Ratio")
    axes[2].set_ylabel(r"$\|\Delta z\|_2 / \|x\|_2$ (%)")
    axes[2].set_xticks([-0.15, 0.15], ["Mean", "Median"])
    axes[2].grid(axis="y", linestyle="--", alpha=0.25)
    axes[2].text(-0.15, delta_mean + 0.35, f"{delta_mean:.2f}", ha="center", va="bottom", fontsize=8)
    axes[2].text(0.15, delta_median + 0.35, f"{delta_median:.2f}", ha="center", va="bottom", fontsize=8)

    axes[3].bar(
        [-0.15, 0.15],
        [chi2_ratio, lnrt_ratio],
        color=["#59A14F", "#9C755F"],
        width=0.26,
    )
    axes[3].set_title("BDD Consistency")
    axes[3].set_ylabel("Not-flagged ratio (%)")
    axes[3].set_xticks([-0.15, 0.15], ["Chi-square", "LNRT"])
    axes[3].set_ylim(0, 105)
    axes[3].grid(axis="y", linestyle="--", alpha=0.25)
    axes[3].text(-0.15, chi2_ratio + 1.0, f"{chi2_ratio:.2f}", ha="center", va="bottom", fontsize=8)
    axes[3].text(0.15, lnrt_ratio + 1.0, f"{lnrt_ratio:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"{meta['label']} Full-Rerun Summary", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, _system_fig_dir(case_id) / "system_summary")


def plot_query_budget_curve_case(case_id: int) -> None:
    meta = CASE_META[int(case_id)]
    budget_df = pd.read_csv(_run_dir(case_id) / "query_budget_curve.csv")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(
        budget_df["budget"],
        budget_df["success_rate"],
        color=meta["color"],
        linewidth=2.2,
        marker="o",
        markersize=6,
        label="ASR within budget",
    )
    ax.plot(
        budget_df["budget"],
        budget_df["finished_rate"],
        color="#9C755F",
        linewidth=1.8,
        marker="s",
        markersize=5,
        linestyle="--",
        label="Finished samples",
    )
    ax.fill_between(
        budget_df["budget"].to_numpy(dtype=float),
        budget_df["success_rate"].to_numpy(dtype=float),
        alpha=0.10,
        color=meta["color"],
    )
    ax.set_title(f"{meta['label']} Query Budget Curve")
    ax.set_xlabel("Query budget")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    for _, row in budget_df.iterrows():
        ax.text(
            float(row["budget"]),
            float(row["success_rate"]) + 1.5,
            f"{float(row['success_rate']):.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    _save_fig(fig, _system_fig_dir(case_id) / "query_budget_curve_plot")


def plot_query_budget_curves(case_ids: list[int]) -> None:
    fig, axes = plt.subplots(1, len(case_ids), figsize=(12.6, 4.5), sharey=True)
    if len(case_ids) == 1:
        axes = [axes]
    for ax, case_id in zip(axes, case_ids):
        meta = CASE_META[int(case_id)]
        budget_df = pd.read_csv(_run_dir(case_id) / "query_budget_curve.csv")
        ax.plot(
            budget_df["budget"],
            budget_df["success_rate"],
            color=meta["color"],
            linewidth=2.2,
            marker="o",
            markersize=5.5,
            label="ASR within budget",
        )
        ax.plot(
            budget_df["budget"],
            budget_df["finished_rate"],
            color="#9C755F",
            linewidth=1.7,
            linestyle="--",
            marker="s",
            markersize=4.8,
            label="Finished samples",
        )
        ax.fill_between(
            budget_df["budget"].to_numpy(dtype=float),
            budget_df["success_rate"].to_numpy(dtype=float),
            alpha=0.10,
            color=meta["color"],
        )
        ax.set_title(meta["label"])
        ax.set_xlabel("Query budget")
        ax.grid(alpha=0.25, linestyle="--")
        for _, row in budget_df.iterrows():
            ax.text(
                float(row["budget"]),
                float(row["success_rate"]) + 1.4,
                f"{float(row['success_rate']):.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_ylim(0, 105)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, FIG_ROOT / "full_rerun_budget_curves")


def plot_query_profile_case(case_id: int) -> None:
    meta = CASE_META[int(case_id)]
    dist_df = pd.read_csv(_run_dir(case_id) / "query_distribution_summary.csv")
    dist_df = dist_df[dist_df["scope"].isin(SCOPE_META.keys())].copy()
    dist_df["scope"] = pd.Categorical(
        dist_df["scope"], categories=list(SCOPE_META.keys()), ordered=True
    )
    dist_df = dist_df.sort_values("scope")

    metrics = ["median", "p90", "p95", "max"]
    metric_labels = ["Median", "P90", "P95", "Max"]
    x = np.arange(len(metrics))
    width = 0.22

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for idx, scope in enumerate(SCOPE_META.keys()):
        row = dist_df[dist_df["scope"] == scope]
        if row.empty:
            continue
        values = row[metrics].iloc[0].to_numpy(dtype=float)
        ax.bar(
            x + (idx - 1) * width,
            values,
            width=width,
            color=SCOPE_META[scope]["color"],
            label=SCOPE_META[scope]["label"],
        )
        for xi, yi in zip(x + (idx - 1) * width, values):
            ax.text(
                xi,
                yi + max(0.6, 0.01 * float(max(values))),
                f"{yi:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_title(f"{meta['label']} Query Profile")
    ax.set_ylabel("Queries")
    ax.set_xticks(x, metric_labels)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)

    fig.tight_layout()
    _save_fig(fig, _system_fig_dir(case_id) / "query_distribution_profile")


def plot_asr_vs_perturbation_case(case_id: int) -> None:
    rows = _load_budget_sweep_rows(case_id)
    if not rows:
        return
    meta = CASE_META[int(case_id)]
    df = pd.DataFrame(rows).sort_values("epsilon_pct").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(
        df["realized_pct"],
        df["attack_success_rate"],
        color=meta["color"],
        linewidth=2.2,
        marker="o",
        markersize=6.5,
    )
    ax.scatter(
        df["realized_pct"],
        df["attack_success_rate"],
        color=meta["color"],
        s=40,
        zorder=3,
    )
    for _, row in df.iterrows():
        ax.text(
            float(row["realized_pct"]) + 0.15,
            float(row["attack_success_rate"]) + 1.0,
            f"cap={int(row['epsilon_pct'])}%",
            fontsize=8,
            ha="left",
            va="bottom",
        )
    ax.set_title(f"{meta['label']} ASR vs Realized Perturbation")
    ax.set_xlabel(r"Mean realized $\|\Delta z\|_2 / \|x\|_2$ (%)")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    _save_fig(fig, _system_fig_dir(case_id) / "asr_vs_perturbation")


def plot_perturbation_distribution_case(case_id: int) -> None:
    summary_rows: list[dict] = []
    series: list[list[float]] = []
    labels: list[str] = []
    cap_values: list[float] = []
    for epsilon_pct in [2, 5, 10, 15, 20]:
        per_sample_path = (
            QUERY_BUDGET_RUN_ROOT
            / f"ieee_case{int(case_id)}_paper_query_budget_pgzoo_cap20_v1_minimum_success_eps{int(epsilon_pct):02d}"
            / "per_sample_metrics.csv"
        )
        if not per_sample_path.exists():
            continue
        df = pd.read_csv(per_sample_path)
        ratio_pct = df["delta_over_clean_l2_ratio"].astype(float) * 100.0
        series.append(ratio_pct.tolist())
        labels.append(f"{int(epsilon_pct)}%")
        cap_values.append(float(epsilon_pct))
        summary_rows.append(
            {
                "budget_cap_pct": float(epsilon_pct),
                "count": int(ratio_pct.shape[0]),
                "mean_realized_pct": float(ratio_pct.mean()),
                "std_realized_pct": float(ratio_pct.std(ddof=0)),
                "min_realized_pct": float(ratio_pct.min()),
                "q10_realized_pct": float(ratio_pct.quantile(0.10)),
                "q25_realized_pct": float(ratio_pct.quantile(0.25)),
                "median_realized_pct": float(ratio_pct.median()),
                "q75_realized_pct": float(ratio_pct.quantile(0.75)),
                "q90_realized_pct": float(ratio_pct.quantile(0.90)),
                "max_realized_pct": float(ratio_pct.max()),
                "success_rate_pct": float(df["success"].astype(float).mean() * 100.0),
            }
        )

    if not series:
        return

    pd.DataFrame(summary_rows).to_csv(
        _run_dir(case_id) / "perturbation_distribution_summary.csv",
        index=False,
    )

    meta = CASE_META[int(case_id)]
    positions = np.arange(1, len(series) + 1)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    box = ax.boxplot(
        series,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1f1f1f", "linewidth": 1.6},
        whiskerprops={"color": "#666666", "linewidth": 1.2},
        capprops={"color": "#666666", "linewidth": 1.2},
        boxprops={"edgecolor": "#666666", "linewidth": 1.2},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(meta["color"])
        patch.set_alpha(0.35)

    means = [float(np.mean(vals)) for vals in series]
    ax.plot(
        positions,
        means,
        color=meta["color"],
        marker="o",
        markersize=6,
        linewidth=1.8,
        label="Mean realized perturbation",
    )
    ax.plot(
        positions,
        cap_values,
        color="#9C755F",
        marker="s",
        markersize=5,
        linewidth=1.6,
        linestyle="--",
        label="Budget cap",
    )

    for pos, mean_val in zip(positions, means):
        ax.text(
            pos,
            mean_val + 0.35,
            f"{mean_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title(f"{meta['label']} Perturbation Distribution")
    ax.set_xlabel("Budget cap")
    ax.set_ylabel(r"Realized $\|\Delta z\|_2 / \|x\|_2$ (%)")
    ax.set_xticks(positions, labels)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, _system_fig_dir(case_id) / "perturbation_distribution")


def plot_query_profiles(case_ids: list[int]) -> None:
    metrics = ["median", "p90", "p95", "max"]
    metric_labels = ["Median", "P90", "P95", "Max"]
    fig, axes = plt.subplots(1, len(case_ids), figsize=(12.8, 4.7), sharey=False)
    if len(case_ids) == 1:
        axes = [axes]
    for ax, case_id in zip(axes, case_ids):
        meta = CASE_META[int(case_id)]
        dist_df = pd.read_csv(_run_dir(case_id) / "query_distribution_summary.csv")
        dist_df = dist_df[dist_df["scope"].isin(SCOPE_META.keys())].copy()
        dist_df["scope"] = pd.Categorical(
            dist_df["scope"], categories=list(SCOPE_META.keys()), ordered=True
        )
        dist_df = dist_df.sort_values("scope")
        x = np.arange(len(metrics))
        width = 0.22
        for idx, scope in enumerate(SCOPE_META.keys()):
            row = dist_df[dist_df["scope"] == scope]
            if row.empty:
                continue
            values = row[metrics].iloc[0].to_numpy(dtype=float)
            ax.bar(
                x + (idx - 1) * width,
                values,
                width=width,
                color=SCOPE_META[scope]["color"],
                label=SCOPE_META[scope]["label"],
            )
            for xi, yi in zip(x + (idx - 1) * width, values):
                ax.text(xi, yi + max(0.8, 0.012 * float(max(values))), f"{yi:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(meta["label"])
        ax.set_xticks(x, metric_labels)
        ax.set_ylabel("Queries")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, FIG_ROOT / "full_rerun_query_profiles")


def plot_baseline_comparison(case_ids: list[int]) -> None:
    baseline_df = pd.read_csv(KEY_PGZOO_SUMMARY)
    rows = []
    for case_id in case_ids:
        rerun = load_case_summary(case_id)
        if int(case_id) == 14:
            baseline_row = baseline_df[baseline_df["system_id"] == 14].iloc[0]
        else:
            baseline_row = baseline_df[baseline_df["variant"] == "physics_subspace_zoo_v1_support64"].iloc[0]
        rows.append(
            {
                "system_id": int(case_id),
                "label": CASE_META[int(case_id)]["label"],
                "baseline_asr": float(baseline_row["attack_success_rate"]),
                "baseline_avg_queries": float(baseline_row["avg_queries"]),
                "rerun_asr": float(rerun["attack_success_rate"]),
                "rerun_avg_queries": float(rerun["avg_queries"]),
            }
        )
    compare_df = pd.DataFrame(rows).sort_values("system_id").reset_index(drop=True)

    x = np.arange(compare_df.shape[0])
    width = 0.28
    labels = compare_df["label"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    axes[0].bar(x - width / 2, compare_df["baseline_asr"], width=width, color="#A0CBE8", label="Historical summary")
    axes[0].bar(x + width / 2, compare_df["rerun_asr"], width=width, color="#4E79A7", label="Full rerun")
    axes[0].set_title("ASR Reproducibility")
    axes[0].set_ylabel("ASR (%)")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)

    axes[1].bar(
        x - width / 2,
        compare_df["baseline_avg_queries"],
        width=width,
        color="#F28E2B",
        label="Historical summary",
    )
    axes[1].bar(
        x + width / 2,
        compare_df["rerun_avg_queries"],
        width=width,
        color="#E15759",
        label="Full rerun",
    )
    axes[1].set_title("Query Cost Reproducibility")
    axes[1].set_ylabel("Average queries")
    axes[1].set_xticks(x, labels)
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)

    for ax in axes:
        handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, FIG_ROOT / "full_rerun_vs_baseline")


def main() -> None:
    case_ids = [14, 118]
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    summary_df = build_overview_table(case_ids)
    plot_overview(summary_df)
    plot_baseline_comparison(case_ids)
    for case_id in case_ids:
        _system_fig_dir(case_id).mkdir(parents=True, exist_ok=True)
        plot_case_summary(case_id)
        plot_query_budget_curve_case(case_id)
        plot_query_profile_case(case_id)
        plot_asr_vs_perturbation_case(case_id)
        plot_perturbation_distribution_case(case_id)


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    main()
