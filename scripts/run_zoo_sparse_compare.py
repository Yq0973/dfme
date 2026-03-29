#!/usr/bin/env python3
import argparse
import json
from dataclasses import replace
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

from topo_latent_blackbox.experiment_utils import (  # noqa: E402
    build_attack_config_from_preset,
    seed_everything,
)
from topo_latent_blackbox.pipeline import run_topology_latent_blackbox_attack  # noqa: E402
from topo_latent_blackbox.results_layout import resolve_summary_dir  # noqa: E402


def _method_configs(base_cfg, zoo_mode: str) -> list[tuple[str, object]]:
    return [
        (
            "zoo_baseline",
            replace(
                base_cfg,
                attack_mode=str(zoo_mode),
                fd_central_gradient=True,
                early_stop_on_success=True,
            ),
        ),
        (
            "mainline",
            replace(
                base_cfg,
                fd_central_gradient=False,
                sparse_support_beta=0.0,
                sparse_seed_weight=0.0,
                sparse_region_mix=0.0,
            ),
        ),
        (
            "sparse_only",
            replace(
                base_cfg,
                sparse_support_beta=0.50,
                sparse_seed_weight=1.0,
                sparse_region_mix=1.0,
                sparse_region_penalty=0.50,
            ),
        ),
        (
            "sparse_combined",
            replace(
                base_cfg,
                sparse_support_beta=0.50,
                sparse_seed_weight=0.10,
                sparse_region_mix=0.10,
                sparse_region_penalty=0.20,
            ),
        ),
    ]


def _collect_row(system_id: int, method_name: str, summary: dict) -> dict:
    effect = summary.get("fdia_effect_summary", {})
    return {
        "system_id": int(system_id),
        "method": str(method_name),
        "subset_size": int(summary["subset_size"]),
        "attack_success_rate": float(summary["attack_success_rate"]),
        "avg_queries": float(summary["avg_queries"]),
        "avg_probe_queries": float(summary.get("avg_probe_queries", 0.0)),
        "avg_search_queries": float(summary.get("avg_search_queries", 0.0)),
        "delta_over_clean_l2_ratio_mean": float(
            summary.get("delta_over_clean_l2_ratio_mean", 0.0)
        ),
        "region_measurement_support_count_mean": float(
            summary.get("region_measurement_support_count_mean", 0.0)
        ),
        "region_measurement_support_ratio_mean": float(
            summary.get("region_measurement_support_ratio_mean", 0.0)
        ),
        "region_sparse_efficiency_mean": float(
            summary.get("region_sparse_efficiency_mean", 0.0)
        ),
        "state_support_retention_ratio_mean": float(
            effect.get("state_support_retention_ratio_mean", 0.0)
        ),
        "state_offsupport_energy_ratio_mean": float(
            effect.get("state_offsupport_energy_ratio_mean", 0.0)
        ),
    }


def _plot_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("attack_success_rate", "ASR (%)", "asr"),
        ("avg_queries", "Avg. queries", "queries"),
        ("region_measurement_support_ratio_mean", "Support ratio", "support_ratio"),
        ("delta_over_clean_l2_ratio_mean", "Delta / clean L2", "delta_ratio"),
    ]

    for metric, ylabel, stem in metrics:
        plt.figure(figsize=(8, 4.8))
        values = df[metric].tolist()
        x = range(len(values))
        plt.bar(list(x), values, color=["#4C72B0", "#55A868", "#C44E52"][: len(values)])
        labels = [f"case{row.system_id}\n{row.method}" for row in df.itertuples()]
        plt.xticks(list(x), labels, rotation=0)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_compare.png", dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ZOO baseline, mainline, and sparse-combined attack."
    )
    parser.add_argument("--systems", nargs="+", type=int, default=[14])
    parser.add_argument("--preset", type=str, default="scale_adaptive_mainline_v1")
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topology_mode", type=str, default="auto")
    parser.add_argument("--noise_model", type=str, default="unknown")
    parser.add_argument("--save_level", type=str, default="summary_only")
    parser.add_argument("--oracle_arch", type=str, default="resmlp")
    parser.add_argument("--exp_tag", type=str, default="main_compare_zoo_sparse_v1")
    parser.add_argument("--region_size_override", type=int, default=0)
    parser.add_argument("--region_candidates_override", type=int, default=0)
    parser.add_argument("--anchor_size_override", type=int, default=0)
    parser.add_argument("--anchor_pool_size_override", type=int, default=0)
    parser.add_argument(
        "--zoo_mode",
        type=str,
        default="full_zoo",
        choices=["full_zoo", "region_zoo", "measurement_full_zoo", "measurement_region_zoo"],
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    all_summaries: dict[str, dict] = {}
    for system_id in args.systems:
        base_cfg = build_attack_config_from_preset(int(system_id), args.preset)
        override_kwargs = {}
        if int(args.region_size_override) > 0:
            override_kwargs["region_size"] = int(args.region_size_override)
        if int(args.region_candidates_override) > 0:
            override_kwargs["region_candidates"] = int(args.region_candidates_override)
        if int(args.anchor_size_override) > 0:
            override_kwargs["anchor_size"] = int(args.anchor_size_override)
        if int(args.anchor_pool_size_override) > 0:
            override_kwargs["anchor_pool_size"] = int(args.anchor_pool_size_override)
        if override_kwargs:
            base_cfg = replace(base_cfg, **override_kwargs)
        for method_name, cfg in _method_configs(base_cfg, args.zoo_mode):
            run_tag = f"{args.exp_tag}_{method_name}"
            summary = run_topology_latent_blackbox_attack(
                system_id=int(system_id),
                oracle_arch=str(args.oracle_arch),
                attack_class=1,
                max_samples=int(args.max_samples),
                run_seed=int(args.seed),
                device=device,
                attack_config=cfg,
                topology_mode=str(args.topology_mode),
                noise_model=str(args.noise_model),
                save_level=str(args.save_level),
                exp_tag=run_tag,
            )
            rows.append(_collect_row(int(system_id), method_name, summary))
            all_summaries[f"case{int(system_id)}_{method_name}"] = summary
            print(
                json.dumps(
                    {
                        "system_id": int(system_id),
                        "method": method_name,
                        "attack_success_rate": float(summary["attack_success_rate"]),
                        "avg_queries": float(summary["avg_queries"]),
                        "support_ratio": float(
                            summary.get("region_measurement_support_ratio_mean", 0.0)
                        ),
                        "delta_ratio": float(
                            summary.get("delta_over_clean_l2_ratio_mean", 0.0)
                        ),
                    },
                    ensure_ascii=False,
                )
            )

    summary_dir = resolve_summary_dir(str(args.exp_tag))
    summary_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(summary_dir / "zoo_sparse_compare_summary.csv", index=False)
    with open(summary_dir / "zoo_sparse_compare_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    _plot_metrics(df, summary_dir)


if __name__ == "__main__":
    main()
