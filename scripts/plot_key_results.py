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


KEY_RESULTS = PACKAGE_ROOT / "results" / "key_results"

CASE_META = {
    14: {"label": "IEEE 14-bus", "color": "#1f77b4"},
    30: {"label": "IEEE 30-bus", "color": "#ff7f0e"},
    118: {"label": "IEEE 118-bus", "color": "#2ca02c"},
}

SCOPE_META = {
    "total_all": {"label": "All", "color": "#4E79A7"},
    "total_success": {"label": "Success", "color": "#59A14F"},
    "total_failure": {"label": "Failure", "color": "#E15759"},
}


def _case_meta(case_id: int) -> dict:
    return CASE_META.get(int(case_id), {"label": f"IEEE {int(case_id)}-bus", "color": "#4E79A7"})


def _save_fig(fig: plt.Figure, stem_path: Path) -> None:
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_mainline_overview(summary_df: pd.DataFrame) -> None:
    case_ids = [int(v) for v in summary_df["system_id"].tolist()]
    labels = [_case_meta(case_id)["label"] for case_id in case_ids]
    colors = [_case_meta(case_id)["color"] for case_id in case_ids]
    x = np.arange(len(case_ids))

    probe_values = []
    search_values = []
    for case_id in case_ids:
        attack_summary = _load_json(KEY_RESULTS / f"case{case_id}" / "main" / "attack_summary.json")
        probe_values.append(float(attack_summary.get("avg_probe_queries", 0.0)))
        search_values.append(float(attack_summary.get("avg_search_queries", 0.0)))

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))

    axes[0].bar(x, summary_df["attack_success_rate"], color=colors, width=0.58)
    axes[0].set_title("Attack Success Rate")
    axes[0].set_ylabel("ASR (%)")
    axes[0].set_xticks(x, labels, rotation=10)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")
    for xi, yi in zip(x, summary_df["attack_success_rate"]):
        axes[0].text(xi, yi + 1.5, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x, summary_df["avg_queries"], color=colors, width=0.58)
    axes[1].set_title("Average Query Cost")
    axes[1].set_ylabel("Queries")
    axes[1].set_xticks(x, labels, rotation=10)
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    for xi, yi in zip(x, summary_df["avg_queries"]):
        axes[1].text(xi, yi + max(0.8, 0.015 * float(max(summary_df["avg_queries"]))), f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

    axes[2].bar(x, probe_values, color="#A0CBE8", width=0.58, label="Probe")
    axes[2].bar(x, search_values, bottom=probe_values, color="#4E79A7", width=0.58, label="Search")
    axes[2].set_title("Query Breakdown")
    axes[2].set_ylabel("Queries")
    axes[2].set_xticks(x, labels, rotation=10)
    axes[2].grid(axis="y", alpha=0.25, linestyle="--")
    axes[2].legend(frameon=False)
    for xi, total in zip(x, np.array(probe_values) + np.array(search_values)):
        axes[2].text(xi, total + max(0.8, 0.015 * float(max(total, 1.0))), f"{total:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    _save_fig(fig, KEY_RESULTS / "figures" / "mainline_overview")


def plot_query_budget(case_id: int) -> None:
    budget_df = pd.read_csv(KEY_RESULTS / f"case{int(case_id)}" / "main" / "query_budget_curve.csv")
    meta = _case_meta(case_id)

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
        budget_df["budget"].to_numpy(),
        budget_df["success_rate"].to_numpy(),
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
        ax.text(float(row["budget"]), float(row["success_rate"]) + 1.5, f"{float(row['success_rate']):.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save_fig(fig, KEY_RESULTS / f"case{int(case_id)}" / "main" / "query_budget_curve_plot")


def plot_query_profile(case_id: int) -> None:
    dist_df = pd.read_csv(KEY_RESULTS / f"case{int(case_id)}" / "main" / "query_distribution_summary.csv")
    dist_df = dist_df[dist_df["scope"].isin(SCOPE_META.keys())].copy()
    dist_df["scope"] = pd.Categorical(dist_df["scope"], categories=list(SCOPE_META.keys()), ordered=True)
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
            ax.text(xi, yi + max(0.6, 0.01 * float(max(values))), f"{yi:.0f}", ha="center", va="bottom", fontsize=8)

    meta = _case_meta(case_id)
    ax.set_title(f"{meta['label']} Query Profile")
    ax.set_ylabel("Queries")
    ax.set_xticks(x, metric_labels)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)

    fig.tight_layout()
    _save_fig(fig, KEY_RESULTS / f"case{int(case_id)}" / "main" / "query_distribution_profile")


def build_query_cap_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, summary_row in summary_df.iterrows():
        case_id = int(summary_row["system_id"])
        budget_df = pd.read_csv(KEY_RESULTS / f"case{case_id}" / "main" / "query_budget_curve.csv")
        for _, row in budget_df.iterrows():
            finished_rate = float(row["finished_rate"])
            rows.append(
                {
                    "system_id": case_id,
                    "query_cap": int(row["budget"]),
                    "cap_label": str(int(row["budget"])),
                    "success_rate": float(row["success_rate"]),
                    "finished_rate": finished_rate,
                    "query_cap_reached_rate": float(max(0.0, 100.0 - finished_rate)),
                    "mode": "capped",
                }
            )
        rows.append(
            {
                "system_id": case_id,
                "query_cap": 225,
                "cap_label": "uncapped",
                "success_rate": float(summary_row["attack_success_rate"]),
                "finished_rate": 100.0,
                "query_cap_reached_rate": 0.0,
                "mode": "uncapped",
            }
        )
    cap_df = pd.DataFrame(rows).sort_values(["system_id", "query_cap"]).reset_index(drop=True)
    cap_df.to_csv(KEY_RESULTS / "query_cap_sweep_summary.csv", index=False)
    return cap_df


def plot_query_cap_effects(cap_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    for case_id in sorted(cap_df["system_id"].unique()):
        sub = cap_df[cap_df["system_id"] == int(case_id)].copy()
        meta = _case_meta(case_id)
        x = sub["query_cap"].to_numpy(dtype=float)
        labels = sub["cap_label"].tolist()

        axes[0].plot(
            x,
            sub["success_rate"].to_numpy(dtype=float),
            color=meta["color"],
            linewidth=2.1,
            marker="o",
            markersize=6,
            label=meta["label"],
        )
        axes[1].plot(
            x,
            sub["query_cap_reached_rate"].to_numpy(dtype=float),
            color=meta["color"],
            linewidth=2.1,
            marker="s",
            markersize=5,
            label=meta["label"],
        )

        for ax in axes:
            ax.set_xticks(x, labels, rotation=0)

    axes[0].set_title("ASR Under Per-Sample Query Cap")
    axes[0].set_ylabel("Attack success rate (%)")
    axes[0].set_xlabel("Per-sample query cap")
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=0.25, linestyle="--")

    axes[1].set_title("Samples Hitting the Query Cap")
    axes[1].set_ylabel("Cap-reached rate (%)")
    axes[1].set_xlabel("Per-sample query cap")
    axes[1].set_ylim(0, 105)
    axes[1].grid(alpha=0.25, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, KEY_RESULTS / "figures" / "query_cap_effects")


def main() -> None:
    summary_path = KEY_RESULTS / "main_results_summary.csv"
    summary_df = pd.read_csv(summary_path).sort_values("system_id").reset_index(drop=True)

    plot_mainline_overview(summary_df)
    cap_df = build_query_cap_summary(summary_df)
    plot_query_cap_effects(cap_df)
    for case_id in summary_df["system_id"].tolist():
        plot_query_budget(int(case_id))
        plot_query_profile(int(case_id))


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
