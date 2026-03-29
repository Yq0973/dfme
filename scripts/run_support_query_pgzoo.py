#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from topo_latent_blackbox.attack import TopologyLatentAttackConfig  # noqa: E402
from topo_latent_blackbox.experiment_utils import seed_everything  # noqa: E402
from topo_latent_blackbox.pipeline import run_topology_latent_blackbox_attack  # noqa: E402


SYSTEM_DEFAULTS = {
    14: {
        "support_pool_size": 6,
        "support_final_size": 2,
        "measurement_delta_l2_ratio_cap": 0.20,
        "rounds": 12,
        "state_basis_dim": 2,
        "pgzoo_probe_pairs": 1,
        "pgzoo_line_candidates": 2,
        "probe_scale_ratio": 0.30,
        "support_probe_scale_ratio": 0.30,
    },
    118: {
        "support_pool_size": 32,
        "support_final_size": 8,
        "measurement_delta_l2_ratio_cap": 0.20,
        "rounds": 18,
        "state_basis_dim": 8,
        "pgzoo_probe_pairs": 2,
        "pgzoo_line_candidates": 2,
        "probe_scale_ratio": 0.30,
        "support_probe_scale_ratio": 0.30,
    },
}


def build_config(system_id: int, args: argparse.Namespace) -> TopologyLatentAttackConfig:
    defaults = dict(SYSTEM_DEFAULTS[int(system_id)])
    support_final_size = (
        int(args.support_final_size)
        if int(args.support_final_size) > 0
        else int(defaults["support_final_size"])
    )
    support_pool_size = (
        int(args.support_pool_size)
        if int(args.support_pool_size) > 0
        else int(defaults["support_pool_size"])
    )
    cap = (
        float(args.measurement_delta_l2_ratio_cap)
        if float(args.measurement_delta_l2_ratio_cap) > 0.0
        else float(defaults["measurement_delta_l2_ratio_cap"])
    )
    rounds = int(args.rounds) if int(args.rounds) > 0 else int(defaults["rounds"])
    state_basis_dim = (
        int(args.state_basis_dim)
        if int(args.state_basis_dim) > 0
        else int(defaults["state_basis_dim"])
    )
    pgzoo_probe_pairs = (
        int(args.pgzoo_probe_pairs)
        if int(args.pgzoo_probe_pairs) > 0
        else int(defaults["pgzoo_probe_pairs"])
    )
    pgzoo_line_candidates = (
        int(args.pgzoo_line_candidates)
        if int(args.pgzoo_line_candidates) > 0
        else int(defaults["pgzoo_line_candidates"])
    )
    probe_scale_ratio = (
        float(args.probe_scale_ratio)
        if float(args.probe_scale_ratio) > 0.0
        else float(defaults["probe_scale_ratio"])
    )
    support_probe_scale_ratio = (
        float(args.support_probe_scale_ratio)
        if float(args.support_probe_scale_ratio) > 0.0
        else float(defaults["support_probe_scale_ratio"])
    )

    return TopologyLatentAttackConfig(
        attack_mode="support_identify_pgzoo",
        query_mode="margin",
        early_stop_on_success=True,
        region_size=int(support_final_size),
        support_final_size=int(support_final_size),
        support_pool_size=int(support_pool_size),
        support_diffusion_lambda=float(args.support_diffusion_lambda),
        support_probe_scale_ratio=float(support_probe_scale_ratio),
        support_prior_weight=float(args.support_prior_weight),
        support_diversity_penalty=float(args.support_diversity_penalty),
        support_success_bonus=float(args.support_success_bonus),
        support_keep_base_support=not bool(args.no_support_keep_base_support),
        adaptive_support_selection=bool(args.adaptive_support_selection),
        adaptive_support_mass_threshold=float(args.adaptive_support_mass_threshold),
        adaptive_support_max_size=int(args.adaptive_support_max_size),
        measurement_delta_l2_ratio_cap=float(cap),
        state_subspace_pgzoo=True,
        state_basis_dim=int(state_basis_dim),
        pgzoo_probe_pairs=int(pgzoo_probe_pairs),
        pgzoo_alpha_ratio=float(args.pgzoo_alpha_ratio),
        pgzoo_momentum=float(args.pgzoo_momentum),
        pgzoo_line_candidates=int(pgzoo_line_candidates),
        pgzoo_query_prior_weight=float(args.pgzoo_query_prior_weight),
        pgzoo_structured_covariance=True,
        pgzoo_physical_preconditioner=True,
        pgzoo_covariance_gamma=float(args.pgzoo_covariance_gamma),
        pgzoo_covariance_ridge=float(args.pgzoo_covariance_ridge),
        pgzoo_preconditioner_ridge=float(args.pgzoo_preconditioner_ridge),
        rounds=int(rounds),
        probe_scale_ratio=float(probe_scale_ratio),
        step_ratio=float(args.step_ratio),
        step_decay=float(args.step_decay),
        patience=int(args.patience),
        min_step_ratio=float(args.min_step_ratio),
        max_step_shrinks=int(args.max_step_shrinks),
        radius_floor=float(args.radius_floor),
        fdia_preserve_weight=float(args.fdia_preserve_weight),
        fdia_backbone_lock_ratio=float(args.fdia_backbone_lock_ratio),
        fdia_state_projection_min=float(args.fdia_state_projection_min),
        fdia_measurement_projection_min=float(args.fdia_measurement_projection_min),
        fdia_offsupport_max=float(args.fdia_offsupport_max),
        fdia_state_penalty_weight=float(args.fdia_state_penalty_weight),
        fdia_measurement_penalty_weight=float(args.fdia_measurement_penalty_weight),
        fdia_offsupport_penalty_weight=float(args.fdia_offsupport_penalty_weight),
        max_queries_per_sample=int(args.max_queries_per_sample),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run support-identification PG-ZOO attack."
    )
    parser.add_argument("--systems", nargs="+", type=int, default=[14, 118])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack_class", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--exp_tag", type=str, default="support_query_pgzoo")
    parser.add_argument(
        "--save_level",
        type=str,
        default="summary_only",
        choices=["full", "summary_only"],
    )
    parser.add_argument(
        "--topology_mode",
        type=str,
        default="auto",
        choices=["auto", "explicit", "jacobian"],
    )
    parser.add_argument(
        "--noise_model",
        type=str,
        default="unknown",
        choices=["known", "unknown", "isotropic"],
    )
    parser.add_argument("--support_pool_size", type=int, default=0)
    parser.add_argument("--support_final_size", type=int, default=0)
    parser.add_argument("--support_diffusion_lambda", type=float, default=0.75)
    parser.add_argument("--support_probe_scale_ratio", type=float, default=0.0)
    parser.add_argument("--support_prior_weight", type=float, default=0.25)
    parser.add_argument("--support_diversity_penalty", type=float, default=0.15)
    parser.add_argument("--support_success_bonus", type=float, default=1.0)
    parser.add_argument("--no_support_keep_base_support", action="store_true")
    parser.add_argument("--adaptive_support_selection", action="store_true")
    parser.add_argument("--adaptive_support_mass_threshold", type=float, default=0.85)
    parser.add_argument("--adaptive_support_max_size", type=int, default=0)
    parser.add_argument("--measurement_delta_l2_ratio_cap", type=float, default=0.0)
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--state_basis_dim", type=int, default=0)
    parser.add_argument("--pgzoo_probe_pairs", type=int, default=0)
    parser.add_argument("--pgzoo_line_candidates", type=int, default=0)
    parser.add_argument("--probe_scale_ratio", type=float, default=0.0)
    parser.add_argument("--pgzoo_alpha_ratio", type=float, default=1.0)
    parser.add_argument("--pgzoo_momentum", type=float, default=0.80)
    parser.add_argument("--pgzoo_query_prior_weight", type=float, default=0.35)
    parser.add_argument("--pgzoo_covariance_gamma", type=float, default=0.75)
    parser.add_argument("--pgzoo_covariance_ridge", type=float, default=1e-3)
    parser.add_argument("--pgzoo_preconditioner_ridge", type=float, default=0.10)
    parser.add_argument("--step_ratio", type=float, default=0.40)
    parser.add_argument("--step_decay", type=float, default=0.70)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min_step_ratio", type=float, default=0.10)
    parser.add_argument("--max_step_shrinks", type=int, default=3)
    parser.add_argument("--radius_floor", type=float, default=0.01)
    parser.add_argument("--fdia_preserve_weight", type=float, default=4.0)
    parser.add_argument("--fdia_backbone_lock_ratio", type=float, default=0.15)
    parser.add_argument("--fdia_state_projection_min", type=float, default=0.45)
    parser.add_argument("--fdia_measurement_projection_min", type=float, default=0.45)
    parser.add_argument("--fdia_offsupport_max", type=float, default=0.60)
    parser.add_argument("--fdia_state_penalty_weight", type=float, default=1.0)
    parser.add_argument("--fdia_measurement_penalty_weight", type=float, default=0.50)
    parser.add_argument("--fdia_offsupport_penalty_weight", type=float, default=0.25)
    parser.add_argument("--max_queries_per_sample", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    for system_id in args.systems:
        cfg = build_config(system_id, args)
        summary = run_topology_latent_blackbox_attack(
            system_id=int(system_id),
            attack_class=int(args.attack_class),
            max_samples=int(args.max_samples),
            exp_tag=f"{args.exp_tag}_case{int(system_id)}",
            run_seed=int(args.seed),
            device=device,
            attack_config=cfg,
            topology_mode=str(args.topology_mode),
            noise_model=str(args.noise_model),
            save_level=str(args.save_level),
        )
        print(
            "system={system} subset={subset} asr={asr:.2f}% avg_queries={avg_q:.2f} "
            "avg_probe={avg_probe:.2f} avg_search={avg_search:.2f} result_dir={result_dir}".format(
                system=int(system_id),
                subset=int(summary["subset_size"]),
                asr=float(summary["attack_success_rate"]),
                avg_q=float(summary["avg_queries"]),
                avg_probe=float(summary["avg_probe_queries"]),
                avg_search=float(summary["avg_search_queries"]),
                result_dir=str(summary["result_dir"]),
            )
        )


if __name__ == "__main__":
    main()
