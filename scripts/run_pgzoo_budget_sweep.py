#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from topo_latent_blackbox.attack import TopologyLatentAttackConfig  # noqa: E402
from topo_latent_blackbox.config import resolve_attack_preset  # noqa: E402
from topo_latent_blackbox.experiment_utils import seed_everything  # noqa: E402
from topo_latent_blackbox.pipeline import run_topology_latent_blackbox_attack  # noqa: E402
from topo_latent_blackbox.results_layout import resolve_run_dir  # noqa: E402


KEY_RESULTS_DIR = PACKAGE_ROOT / "results" / "key_results" / "budget_sweep"

CASE_META = {
    14: {"label": "IEEE 14-bus", "color": "#1f77b4"},
    118: {"label": "IEEE 118-bus", "color": "#d62728"},
}

VARIANT_META = {
    "minimum_success": {
        "label": "Minimum-success",
        "color": "#4E79A7",
        "marker": "o",
        "push_ratio": 0.0,
        "search_steps": 0,
    },
    "fixed_budget": {
        "label": "Fixed-budget",
        "color": "#59A14F",
        "marker": "^",
        "push_ratio": 0.0,
        "search_steps": 0,
    },
    "boundary_push": {
        "label": "Boundary-push",
        "color": "#E15759",
        "marker": "s",
        "push_ratio": 0.98,
        "search_steps": 5,
    },
}


def _save_fig(fig: plt.Figure, stem_path: Path) -> None:
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _build_attack_config(system_id: int, preset_name: str, epsilon: float) -> TopologyLatentAttackConfig:
    cfg = TopologyLatentAttackConfig()
    for key, value in resolve_attack_preset(preset_name, system_id).items():
        setattr(cfg, key, value)
    cfg.measurement_delta_l2_ratio_cap = float(epsilon)
    return cfg


def _exp_tag_for(variant_name: str, epsilon: float) -> str:
    epsilon_tag = f"eps{int(round(float(epsilon) * 100.0)):02d}"
    return f"paper_query_budget_pgzoo_cap20_v1_{variant_name}_{epsilon_tag}"


def _summary_to_row(summary: dict, system_id: int, variant_name: str, epsilon: float) -> dict:
    variant = VARIANT_META[variant_name]
    fdia_summary = summary.get("fdia_effect_summary", {})
    return {
        "system_id": int(system_id),
        "system_label": CASE_META[int(system_id)]["label"],
        "variant": str(variant_name),
        "variant_label": str(variant["label"]),
        "epsilon": float(epsilon),
        "epsilon_pct": float(epsilon) * 100.0,
        "attack_success_rate": float(summary["attack_success_rate"]),
        "avg_queries": float(summary["avg_queries"]),
        "avg_search_queries": float(summary["avg_search_queries"]),
        "avg_probe_queries": float(summary["avg_probe_queries"]),
        "delta_over_clean_l2_ratio_mean": float(summary["delta_over_clean_l2_ratio_mean"]),
        "delta_over_clean_l2_ratio_median": float(summary["delta_over_clean_l2_ratio_median"]),
        "delta_over_clean_l2_ratio_mean_pct": float(summary["delta_over_clean_l2_ratio_mean"]) * 100.0,
        "delta_over_clean_l2_ratio_median_pct": float(summary["delta_over_clean_l2_ratio_median"]) * 100.0,
        "measurement_total_over_base_ratio_mean": float(
            fdia_summary.get("measurement_total_over_base_ratio_mean", 0.0)
        ),
        "measurement_total_over_base_ratio_median": float(
            fdia_summary.get("measurement_total_over_base_ratio_median", 0.0)
        ),
        "measurement_total_over_base_ratio_mean_pct": float(
            fdia_summary.get("measurement_total_over_base_ratio_mean", 0.0)
        )
        * 100.0,
        "budget_boundary_push_applied_ratio": float(
            summary.get("budget_boundary_push_applied_ratio", 0.0)
        ),
        "budget_boundary_push_queries_mean": float(
            summary.get("budget_boundary_push_queries_mean", 0.0)
        ),
        "budget_boundary_push_scale_mean": float(
            summary.get("budget_boundary_push_scale_mean", 1.0)
        ),
        "subset_size": int(summary["subset_size"]),
        "budget_variant": str(summary.get("budget_variant", variant_name)),
        "result_dir": str(summary.get("result_dir", "")),
    }


def _load_existing_row(system_id: int, variant_name: str, epsilon: float) -> dict | None:
    exp_tag = _exp_tag_for(variant_name=variant_name, epsilon=epsilon)
    run_dir = resolve_run_dir(system_id=system_id, exp_tag=exp_tag)
    summary_path = run_dir / "attack_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return _summary_to_row(
        summary=summary,
        system_id=system_id,
        variant_name=variant_name,
        epsilon=epsilon,
    )


def _run_single(
    system_id: int,
    preset_name: str,
    epsilon: float,
    variant_name: str,
    max_samples: int,
    seed: int,
    oracle_arch: str,
    device: torch.device,
    save_level: str,
) -> dict:
    variant = VARIANT_META[variant_name]
    attack_cfg = _build_attack_config(system_id=system_id, preset_name=preset_name, epsilon=epsilon)
    if variant_name == "fixed_budget":
        attack_cfg.budget_objective_mode = "fixed_budget"
        attack_cfg.early_stop_on_success = False
        attack_cfg.max_queries_per_sample = 0
        if int(system_id) == 14:
            attack_cfg.region_size = 2
            attack_cfg.state_basis_dim = 4
            attack_cfg.rounds = 24
            attack_cfg.population = 4
        elif int(system_id) == 118:
            attack_cfg.rounds = 28
            attack_cfg.population = 8
            attack_cfg.state_basis_dim = 16
    exp_tag = _exp_tag_for(variant_name=variant_name, epsilon=epsilon)
    summary = run_topology_latent_blackbox_attack(
        system_id=system_id,
        oracle_arch=oracle_arch,
        attack_class=1,
        max_samples=max_samples,
        exp_tag=exp_tag,
        run_seed=seed,
        device=device,
        attack_config=attack_cfg,
        topology_mode="auto",
        noise_model="known",
        save_level=str(save_level),
        budget_boundary_push_ratio=float(variant["push_ratio"]),
        budget_boundary_search_steps=int(variant["search_steps"]),
        budget_variant_name=str(variant_name),
    )
    return _summary_to_row(
        summary=summary,
        system_id=system_id,
        variant_name=variant_name,
        epsilon=epsilon,
    )


def _plot_metric(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    stem_name: str,
    add_budget_diagonal: bool = False,
) -> None:
    systems = sorted(int(v) for v in df["system_id"].unique())
    fig, axes = plt.subplots(1, len(systems), figsize=(6.0 * len(systems), 4.4), sharey=False)
    if len(systems) == 1:
        axes = [axes]

    for ax, system_id in zip(axes, systems):
        sub = df[df["system_id"] == int(system_id)].copy()
        for variant_name, variant_meta in VARIANT_META.items():
            cur = sub[sub["variant"] == variant_name].sort_values("epsilon_pct")
            if cur.empty:
                continue
            ax.plot(
                cur["epsilon_pct"],
                cur[metric_col],
                color=variant_meta["color"],
                linewidth=2.2,
                marker=variant_meta["marker"],
                markersize=6.5,
                label=variant_meta["label"],
            )
        if add_budget_diagonal:
            x_vals = sorted(float(v) for v in sub["epsilon_pct"].unique())
            ax.plot(
                x_vals,
                x_vals,
                color="#9C755F",
                linewidth=1.6,
                linestyle="--",
                label="y = budget",
            )
        meta = CASE_META[int(system_id)]
        ax.set_title(meta["label"])
        ax.set_xlabel("Budget cap (%)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_xticks(sorted(float(v) for v in sub["epsilon_pct"].unique()))
        if metric_col == "attack_success_rate":
            ax.set_ylim(0, 105)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(2, len(handles)), frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title, y=1.06, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, KEY_RESULTS_DIR / stem_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PG-ZOO budget sweep up to 20%.")
    parser.add_argument("--systems", nargs="+", type=int, default=[14, 118])
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=[0.02, 0.05, 0.10, 0.15, 0.20],
    )
    parser.add_argument("--preset", type=str, default="physics_subspace_zoo_v1")
    parser.add_argument("--oracle_arch", type=str, default="resmlp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument(
        "--save_level",
        type=str,
        default="summary_only",
        choices=["summary_only", "full"],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        type=str,
        default=["minimum_success"],
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    KEY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    variants = [str(v) for v in args.variants]
    for variant_name in variants:
        if variant_name not in VARIANT_META:
            raise ValueError(f"Unsupported variant: {variant_name}")

    rows: list[dict] = []
    for system_id in args.systems:
        if int(system_id) not in CASE_META:
            raise ValueError(f"Unsupported system for this sweep: {system_id}")
        for epsilon in args.epsilons:
            epsilon = float(epsilon)
            if epsilon <= 0.0 or epsilon > 0.20 + 1e-8:
                raise ValueError(f"epsilon must be in (0, 0.20], got {epsilon}")
            for variant_name in variants:
                if args.resume:
                    existing_row = _load_existing_row(
                        system_id=int(system_id),
                        variant_name=variant_name,
                        epsilon=epsilon,
                    )
                    if existing_row is not None:
                        print(
                            f"[budget-sweep] reuse existing result: "
                            f"system={system_id} variant={variant_name} epsilon={epsilon:.3f}"
                        )
                        rows.append(existing_row)
                        continue
                print(
                    f"[budget-sweep] system={system_id} "
                    f"variant={variant_name} epsilon={epsilon:.3f} device={device}"
                )
                row = _run_single(
                    system_id=int(system_id),
                    preset_name=args.preset,
                    epsilon=epsilon,
                    variant_name=variant_name,
                    max_samples=int(args.max_samples),
                    seed=int(args.seed),
                    oracle_arch=str(args.oracle_arch),
                    device=device,
                    save_level=str(args.save_level),
                )
                rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(
        ["system_id", "variant", "epsilon"]
    ).reset_index(drop=True)
    summary_path = KEY_RESULTS_DIR / "pgzoo_cap20_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    _plot_metric(
        df=summary_df,
        metric_col="attack_success_rate",
        ylabel="ASR (%)",
        title="PG-ZOO ASR Under Budget Cap",
        stem_name="asr_vs_budget",
        add_budget_diagonal=False,
    )
    _plot_metric(
        df=summary_df,
        metric_col="delta_over_clean_l2_ratio_mean_pct",
        ylabel="Mean realized ||Δz|| / ||x|| (%)",
        title="Realized Perturbation vs Budget Cap",
        stem_name="realized_vs_budget",
        add_budget_diagonal=True,
    )
    _plot_metric(
        df=summary_df,
        metric_col="avg_queries",
        ylabel="Average queries",
        title="Query Cost vs Budget Cap",
        stem_name="queries_vs_budget",
        add_budget_diagonal=False,
    )

    print(f"Saved summary to: {summary_path}")
    print(f"Saved figures under: {KEY_RESULTS_DIR}")


if __name__ == "__main__":
    main()
