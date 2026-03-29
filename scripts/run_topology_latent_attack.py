#!/usr/bin/env python3
import argparse
from dataclasses import fields as dataclass_fields
from pathlib import Path
import sys

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Topology-constrained latent black-box attack for FDIA NAD"
    )
    parser.add_argument("--systems", nargs="+", type=int, default=[14])
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--input_dim", type=int, default=0)
    parser.add_argument("--oracle_arch", type=str, default="resmlp")
    parser.add_argument("--oracle_ckpt", type=str, default="")
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
    parser.add_argument(
        "--attack_preset",
        type=str,
        default="deterministic_mainline_v1",
        choices=[
            "manual",
            "simple_mainline_v1",
            "scale_adaptive_mainline_v1",
            "layered_active_mainline_v1",
            "one_shot_mainline_v1",
            "deterministic_mainline_v1",
            "physics_subspace_zoo_v1",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack_class", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument(
        "--attack_mode",
        type=str,
        default="region_search",
        choices=[
            "region_search",
            "single_active_region_search",
            "layered_region_search",
            "measurement_region_search",
            "full_fd",
            "region_fd",
            "measurement_full_fd",
            "measurement_region_fd",
            "full_zoo",
            "region_zoo",
            "measurement_full_zoo",
            "measurement_region_zoo",
        ],
    )
    parser.add_argument(
        "--state_seed_mode",
        type=str,
        default="legacy",
        choices=["legacy", "target_radiated"],
    )
    parser.add_argument("--region_size", type=int, default=6)
    parser.add_argument("--anchor_size", type=int, default=2)
    parser.add_argument("--region_candidates", type=int, default=4)
    parser.add_argument("--anchor_pool_size", type=int, default=6)
    parser.add_argument("--hierarchical_probe", action="store_true")
    parser.add_argument("--coarse_probe_directions", type=int, default=1)
    parser.add_argument("--fine_probe_topk", type=int, default=3)
    parser.add_argument("--initial_probe_region_topk", type=int, default=0)
    parser.add_argument("--probe_expand_improvement_ratio", type=float, default=0.0)
    parser.add_argument(
        "--region_proposal_mode",
        type=str,
        default="default",
        choices=["default", "hierarchical_corridor"],
    )
    parser.add_argument("--proposal_diffusion_steps", type=int, default=2)
    parser.add_argument("--proposal_diffusion_alpha", type=float, default=0.35)
    parser.add_argument("--proposal_flow_weight", type=float, default=0.30)
    parser.add_argument("--proposal_corridor_weight", type=float, default=0.25)
    parser.add_argument("--probabilistic_region_prior", action="store_true")
    parser.add_argument("--prior_temperature", type=float, default=1.0)
    parser.add_argument("--prior_uniform_mixing", type=float, default=0.0)
    parser.add_argument("--probe_directions", type=int, default=2)
    parser.add_argument("--probe_scale_ratio", type=float, default=0.25)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--radius_ratio", type=float, default=1.0)
    parser.add_argument("--radius_floor", type=float, default=0.01)
    parser.add_argument("--step_ratio", type=float, default=0.35)
    parser.add_argument("--step_decay", type=float, default=0.7)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_step_ratio", type=float, default=0.08)
    parser.add_argument("--max_step_shrinks", type=int, default=4)
    parser.add_argument("--measurement_delta_l2_ratio_cap", type=float, default=0.0)
    parser.add_argument("--realism_penalty_weight", type=float, default=0.0)
    parser.add_argument("--fdia_preserve_weight", type=float, default=0.0)
    parser.add_argument("--fdia_backbone_lock_ratio", type=float, default=0.0)
    parser.add_argument("--fdia_state_projection_min", type=float, default=0.70)
    parser.add_argument("--fdia_measurement_projection_min", type=float, default=0.70)
    parser.add_argument("--fdia_offsupport_max", type=float, default=0.35)
    parser.add_argument("--fdia_state_penalty_weight", type=float, default=0.45)
    parser.add_argument("--fdia_measurement_penalty_weight", type=float, default=0.35)
    parser.add_argument("--fdia_offsupport_penalty_weight", type=float, default=0.20)
    parser.add_argument("--region_budget_topk", type=int, default=1)
    parser.add_argument("--region_budget_explore_rounds", type=int, default=0)
    parser.add_argument("--feedback_loop", action="store_true")
    parser.add_argument("--feedback_round_chunk", type=int, default=2)
    parser.add_argument("--feedback_branch_topk", type=int, default=2)
    parser.add_argument("--feedback_stagnation_trigger", type=int, default=2)
    parser.add_argument("--feedback_probe_gap_abs_threshold", type=float, default=0.0)
    parser.add_argument("--feedback_probe_gap_ratio_threshold", type=float, default=0.0)
    parser.add_argument(
        "--feedback_keep_probe_best_incumbent",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--feedback_min_state_dim", type=int, default=0)
    parser.add_argument("--detector_feedback_region", action="store_true")
    parser.add_argument("--detector_feedback_prior_mix", type=float, default=0.25)
    parser.add_argument("--detector_feedback_success_bonus", type=float, default=0.75)
    parser.add_argument("--detector_feedback_min_gain", type=float, default=0.0)
    parser.add_argument("--feedback_physics_reward_shaping", action="store_true")
    parser.add_argument("--feedback_physics_weight", type=float, default=0.50)
    parser.add_argument("--region_budget_score_slack", type=float, default=0.0)
    parser.add_argument("--budget_region_max_probe_best_prior", type=float, default=1.0)
    parser.add_argument("--budget_region_prior_tiebreak", action="store_true")
    parser.add_argument("--measurement_suppression_strength", type=float, default=0.0)
    parser.add_argument("--injection_scale", type=float, default=0.82)
    parser.add_argument("--flow_scale", type=float, default=1.0)
    parser.add_argument("--leverage_suppression", type=float, default=0.30)
    parser.add_argument("--min_channel_scale", type=float, default=0.60)
    parser.add_argument(
        "--probe_channel_shaping_only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--structured_search_directions", action="store_true")
    parser.add_argument("--structured_direction_count", type=int, default=4)
    parser.add_argument("--measurement_basis_search", action="store_true")
    parser.add_argument("--measurement_bandit_search", action="store_true")
    parser.add_argument("--state_basis_search", action="store_true")
    parser.add_argument("--state_bandit_search", action="store_true")
    parser.add_argument(
        "--state_subspace_pgzoo",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--state_basis_dim", type=int, default=4)
    parser.add_argument("--bandit_direction_samples", type=int, default=2)
    parser.add_argument("--bandit_momentum", type=float, default=0.75)
    parser.add_argument("--bandit_exploration_ratio", type=float, default=1.0)
    parser.add_argument("--bandit_warmup_rounds", type=int, default=0)
    parser.add_argument("--pgzoo_probe_pairs", type=int, default=2)
    parser.add_argument("--pgzoo_alpha_ratio", type=float, default=1.0)
    parser.add_argument("--pgzoo_momentum", type=float, default=0.75)
    parser.add_argument("--pgzoo_line_candidates", type=int, default=3)
    parser.add_argument("--pgzoo_prior_topology_weight", type=float, default=0.45)
    parser.add_argument("--pgzoo_prior_base_weight", type=float, default=0.35)
    parser.add_argument("--pgzoo_prior_best_weight", type=float, default=0.20)
    parser.add_argument(
        "--pgzoo_structured_covariance",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--pgzoo_physical_preconditioner",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--pgzoo_covariance_gamma", type=float, default=0.75)
    parser.add_argument("--pgzoo_covariance_ridge", type=float, default=1e-3)
    parser.add_argument("--pgzoo_preconditioner_ridge", type=float, default=0.10)
    parser.add_argument("--support_pool_size", type=int, default=0)
    parser.add_argument("--support_final_size", type=int, default=0)
    parser.add_argument("--support_diffusion_lambda", type=float, default=0.75)
    parser.add_argument("--support_probe_scale_ratio", type=float, default=0.30)
    parser.add_argument("--support_prior_weight", type=float, default=0.25)
    parser.add_argument("--support_diversity_penalty", type=float, default=0.15)
    parser.add_argument("--support_success_bonus", type=float, default=1.0)
    parser.add_argument(
        "--support_keep_base_support",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--adaptive_support_selection",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--adaptive_support_mass_threshold", type=float, default=0.85)
    parser.add_argument("--adaptive_support_max_size", type=int, default=0)
    parser.add_argument("--measurement_guided_state_refine", action="store_true")
    parser.add_argument("--guided_state_region_size", type=int, default=0)
    parser.add_argument("--physical_measurement_gate", action="store_true")
    parser.add_argument("--measurement_gate_ratio", type=float, default=1.0)
    parser.add_argument("--measurement_gate_response_weight", type=float, default=0.60)
    parser.add_argument("--measurement_gate_effectiveness_weight", type=float, default=0.40)
    parser.add_argument("--measurement_conditioned_state_gate", action="store_true")
    parser.add_argument("--state_gate_ratio", type=float, default=1.0)
    parser.add_argument("--state_gate_escape_candidates", type=int, default=0)
    parser.add_argument(
        "--adaptive_query_budget",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--easy_probe_improvement_ratio", type=float, default=0.50)
    parser.add_argument("--easy_population", type=int, default=0)
    parser.add_argument("--easy_rounds", type=int, default=0)
    parser.add_argument("--single_active_size", type=int, default=0)
    parser.add_argument("--active_set_sizes", type=str, default="")
    parser.add_argument("--layered_stage_rounds", type=int, default=0)
    parser.add_argument("--layered_final_rounds", type=int, default=0)
    parser.add_argument("--multisource_physical_prior", action="store_true")
    parser.add_argument("--physics_query_allocation", action="store_true")
    parser.add_argument("--physics_query_topk", type=int, default=1)
    parser.add_argument("--physics_query_priority_weight", type=float, default=0.35)
    parser.add_argument("--adaptive_challenger_budget", action="store_true")
    parser.add_argument("--challenger_population_ratio", type=float, default=0.50)
    parser.add_argument("--challenger_rounds", type=int, default=2)
    parser.add_argument("--branch_pruning", action="store_true")
    parser.add_argument("--branch_prune_progress_ratio", type=float, default=0.20)
    parser.add_argument("--branch_prune_score_gap_ratio", type=float, default=0.02)
    parser.add_argument(
        "--score_stagnation_early_stop",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--score_stagnation_rounds", type=int, default=2)
    parser.add_argument("--score_gain_ratio_threshold", type=float, default=0.005)
    parser.add_argument("--search_min_rounds_before_stop", type=int, default=6)
    parser.add_argument("--uncertainty_aware_pruning", action="store_true")
    parser.add_argument("--physics_aware_early_stop", action="store_true")
    parser.add_argument("--termination_uncertainty_ratio_tau", type=float, default=0.25)
    parser.add_argument("--termination_uncertainty_floor", type=float, default=0.30)
    parser.add_argument("--termination_progress_ratio_floor", type=float, default=0.50)
    parser.add_argument("--termination_physics_quality_floor", type=float, default=0.65)
    parser.add_argument("--guarded_boundary_probe", action="store_true")
    parser.add_argument("--guarded_boundary_probe_steps", type=int, default=3)
    parser.add_argument("--guarded_boundary_probe_max_uses", type=int, default=1)
    parser.add_argument("--fd_warmup_rounds", type=int, default=0)
    parser.add_argument("--fd_warmup_topk", type=int, default=0)
    parser.add_argument(
        "--fd_region_selection",
        type=str,
        default="topology_prior",
        choices=["topology_prior", "random_candidate", "probe_best"],
    )
    parser.add_argument("--fd_iterations", type=int, default=3)
    parser.add_argument("--fd_coordinate_eps_ratio", type=float, default=0.10)
    parser.add_argument("--fd_coordinate_eps_floor", type=float, default=0.0025)
    parser.add_argument("--fd_central_gradient", action="store_true")
    parser.add_argument("--fd_line_steps", type=int, default=1)
    parser.add_argument("--fd_line_decay", type=float, default=0.5)
    parser.add_argument("--sparse_support_beta", type=float, default=0.0)
    parser.add_argument("--sparse_seed_weight", type=float, default=0.0)
    parser.add_argument("--sparse_region_mix", type=float, default=0.0)
    parser.add_argument("--sparse_region_penalty", type=float, default=0.35)
    parser.add_argument(
        "--query_mode",
        type=str,
        default="margin",
        choices=["margin", "prob_fdia", "decision"],
    )
    parser.add_argument(
        "--early_stop_on_success",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--max_queries_per_sample", type=int, default=0)
    parser.add_argument("--exp_tag", type=str, default="")
    parser.add_argument(
        "--save_level",
        type=str,
        default="full",
        choices=["full", "summary_only"],
    )
    args = parser.parse_args()
    explicit_arg_names = {
        token[2:].split("=", 1)[0].replace("-", "_")
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    dataclass_field_names = {
        field.name for field in dataclass_fields(TopologyLatentAttackConfig)
    }
    explicit_arg_aliases = {
        "no_feedback_keep_probe_best_incumbent": "feedback_keep_probe_best_incumbent",
    }
    for name in list(explicit_arg_names):
        if name.startswith("no_") and name[3:] in dataclass_field_names:
            explicit_arg_aliases[name] = name[3:]
    canonical_explicit_arg_names = {
        explicit_arg_aliases.get(name, name) for name in explicit_arg_names
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    seed_everything(args.seed)
    default_cfg = TopologyLatentAttackConfig()

    for system_id in args.systems:
        preset_overrides = resolve_attack_preset(args.attack_preset, system_id)
        cfg = TopologyLatentAttackConfig(
            attack_mode=preset_overrides.get("attack_mode", args.attack_mode),
            state_seed_mode=preset_overrides.get(
                "state_seed_mode", args.state_seed_mode
            ),
            region_size=preset_overrides.get("region_size", args.region_size),
            anchor_size=preset_overrides.get("anchor_size", args.anchor_size),
            region_candidates=preset_overrides.get(
                "region_candidates", args.region_candidates
            ),
            anchor_pool_size=preset_overrides.get(
                "anchor_pool_size", args.anchor_pool_size
            ),
            probabilistic_region_prior=preset_overrides.get(
                "probabilistic_region_prior",
                args.probabilistic_region_prior,
            ),
            prior_temperature=preset_overrides.get(
                "prior_temperature",
                args.prior_temperature,
            ),
            prior_uniform_mixing=preset_overrides.get(
                "prior_uniform_mixing",
                args.prior_uniform_mixing,
            ),
            hierarchical_probe=preset_overrides.get(
                "hierarchical_probe", args.hierarchical_probe
            ),
            coarse_probe_directions=preset_overrides.get(
                "coarse_probe_directions", args.coarse_probe_directions
            ),
            fine_probe_topk=preset_overrides.get(
                "fine_probe_topk", args.fine_probe_topk
            ),
            initial_probe_region_topk=preset_overrides.get(
                "initial_probe_region_topk",
                args.initial_probe_region_topk,
            ),
            probe_expand_improvement_ratio=preset_overrides.get(
                "probe_expand_improvement_ratio",
                args.probe_expand_improvement_ratio,
            ),
            region_proposal_mode=preset_overrides.get(
                "region_proposal_mode", args.region_proposal_mode
            ),
            proposal_diffusion_steps=preset_overrides.get(
                "proposal_diffusion_steps", args.proposal_diffusion_steps
            ),
            proposal_diffusion_alpha=preset_overrides.get(
                "proposal_diffusion_alpha", args.proposal_diffusion_alpha
            ),
            proposal_flow_weight=preset_overrides.get(
                "proposal_flow_weight", args.proposal_flow_weight
            ),
            proposal_corridor_weight=preset_overrides.get(
                "proposal_corridor_weight", args.proposal_corridor_weight
            ),
            probe_directions=preset_overrides.get(
                "probe_directions", args.probe_directions
            ),
            probe_scale_ratio=preset_overrides.get(
                "probe_scale_ratio", args.probe_scale_ratio
            ),
            population=preset_overrides.get("population", args.population),
            rounds=preset_overrides.get("rounds", args.rounds),
            radius_ratio=preset_overrides.get("radius_ratio", args.radius_ratio),
            radius_floor=preset_overrides.get("radius_floor", args.radius_floor),
            step_ratio=preset_overrides.get("step_ratio", args.step_ratio),
            step_decay=preset_overrides.get("step_decay", args.step_decay),
            patience=preset_overrides.get("patience", args.patience),
            min_step_ratio=preset_overrides.get(
                "min_step_ratio", args.min_step_ratio
            ),
            max_step_shrinks=preset_overrides.get(
                "max_step_shrinks", args.max_step_shrinks
            ),
            measurement_delta_l2_ratio_cap=preset_overrides.get(
                "measurement_delta_l2_ratio_cap",
                args.measurement_delta_l2_ratio_cap,
            ),
            realism_penalty_weight=preset_overrides.get(
                "realism_penalty_weight",
                args.realism_penalty_weight,
            ),
            fdia_preserve_weight=preset_overrides.get(
                "fdia_preserve_weight",
                args.fdia_preserve_weight,
            ),
            fdia_backbone_lock_ratio=preset_overrides.get(
                "fdia_backbone_lock_ratio",
                args.fdia_backbone_lock_ratio,
            ),
            fdia_state_projection_min=preset_overrides.get(
                "fdia_state_projection_min",
                args.fdia_state_projection_min,
            ),
            fdia_measurement_projection_min=preset_overrides.get(
                "fdia_measurement_projection_min",
                args.fdia_measurement_projection_min,
            ),
            fdia_offsupport_max=preset_overrides.get(
                "fdia_offsupport_max",
                args.fdia_offsupport_max,
            ),
            fdia_state_penalty_weight=preset_overrides.get(
                "fdia_state_penalty_weight",
                args.fdia_state_penalty_weight,
            ),
            fdia_measurement_penalty_weight=preset_overrides.get(
                "fdia_measurement_penalty_weight",
                args.fdia_measurement_penalty_weight,
            ),
            fdia_offsupport_penalty_weight=preset_overrides.get(
                "fdia_offsupport_penalty_weight",
                args.fdia_offsupport_penalty_weight,
            ),
            region_budget_topk=preset_overrides.get(
                "region_budget_topk",
                args.region_budget_topk,
            ),
            region_budget_explore_rounds=preset_overrides.get(
                "region_budget_explore_rounds",
                args.region_budget_explore_rounds,
            ),
            feedback_loop=preset_overrides.get(
                "feedback_loop",
                args.feedback_loop,
            ),
            feedback_round_chunk=preset_overrides.get(
                "feedback_round_chunk",
                args.feedback_round_chunk,
            ),
            feedback_branch_topk=preset_overrides.get(
                "feedback_branch_topk",
                args.feedback_branch_topk,
            ),
            feedback_stagnation_trigger=preset_overrides.get(
                "feedback_stagnation_trigger",
                args.feedback_stagnation_trigger,
            ),
            feedback_probe_gap_abs_threshold=preset_overrides.get(
                "feedback_probe_gap_abs_threshold",
                args.feedback_probe_gap_abs_threshold,
            ),
            feedback_probe_gap_ratio_threshold=preset_overrides.get(
                "feedback_probe_gap_ratio_threshold",
                args.feedback_probe_gap_ratio_threshold,
            ),
            feedback_keep_probe_best_incumbent=preset_overrides.get(
                "feedback_keep_probe_best_incumbent",
                (
                    TopologyLatentAttackConfig().feedback_keep_probe_best_incumbent
                    if args.feedback_keep_probe_best_incumbent is None
                    else args.feedback_keep_probe_best_incumbent
                ),
            ),
            feedback_min_state_dim=preset_overrides.get(
                "feedback_min_state_dim",
                args.feedback_min_state_dim,
            ),
            detector_feedback_region=preset_overrides.get(
                "detector_feedback_region",
                args.detector_feedback_region,
            ),
            detector_feedback_prior_mix=preset_overrides.get(
                "detector_feedback_prior_mix",
                args.detector_feedback_prior_mix,
            ),
            detector_feedback_success_bonus=preset_overrides.get(
                "detector_feedback_success_bonus",
                args.detector_feedback_success_bonus,
            ),
            detector_feedback_min_gain=preset_overrides.get(
                "detector_feedback_min_gain",
                args.detector_feedback_min_gain,
            ),
            feedback_physics_reward_shaping=preset_overrides.get(
                "feedback_physics_reward_shaping",
                args.feedback_physics_reward_shaping,
            ),
            feedback_physics_weight=preset_overrides.get(
                "feedback_physics_weight",
                args.feedback_physics_weight,
            ),
            region_budget_score_slack=preset_overrides.get(
                "region_budget_score_slack",
                args.region_budget_score_slack,
            ),
            budget_region_max_probe_best_prior=preset_overrides.get(
                "budget_region_max_probe_best_prior",
                args.budget_region_max_probe_best_prior,
            ),
            budget_region_prior_tiebreak=preset_overrides.get(
                "budget_region_prior_tiebreak",
                args.budget_region_prior_tiebreak,
            ),
            measurement_suppression_strength=preset_overrides.get(
                "measurement_suppression_strength",
                args.measurement_suppression_strength,
            ),
            injection_scale=preset_overrides.get(
                "injection_scale",
                args.injection_scale,
            ),
            flow_scale=preset_overrides.get(
                "flow_scale",
                args.flow_scale,
            ),
            leverage_suppression=preset_overrides.get(
                "leverage_suppression",
                args.leverage_suppression,
            ),
            min_channel_scale=preset_overrides.get(
                "min_channel_scale",
                args.min_channel_scale,
            ),
            probe_channel_shaping_only=preset_overrides.get(
                "probe_channel_shaping_only",
                (
                    default_cfg.probe_channel_shaping_only
                    if args.probe_channel_shaping_only is None
                    else args.probe_channel_shaping_only
                ),
            ),
            structured_search_directions=preset_overrides.get(
                "structured_search_directions",
                args.structured_search_directions,
            ),
            structured_direction_count=preset_overrides.get(
                "structured_direction_count",
                args.structured_direction_count,
            ),
            measurement_basis_search=preset_overrides.get(
                "measurement_basis_search",
                args.measurement_basis_search,
            ),
            measurement_bandit_search=preset_overrides.get(
                "measurement_bandit_search",
                args.measurement_bandit_search,
            ),
            state_basis_search=preset_overrides.get(
                "state_basis_search",
                args.state_basis_search,
            ),
            state_bandit_search=preset_overrides.get(
                "state_bandit_search",
                args.state_bandit_search,
            ),
            state_subspace_pgzoo=preset_overrides.get(
                "state_subspace_pgzoo",
                (
                    default_cfg.state_subspace_pgzoo
                    if args.state_subspace_pgzoo is None
                    else args.state_subspace_pgzoo
                ),
            ),
            state_basis_dim=preset_overrides.get(
                "state_basis_dim",
                args.state_basis_dim,
            ),
            bandit_direction_samples=preset_overrides.get(
                "bandit_direction_samples",
                args.bandit_direction_samples,
            ),
            bandit_momentum=preset_overrides.get(
                "bandit_momentum",
                args.bandit_momentum,
            ),
            bandit_exploration_ratio=preset_overrides.get(
                "bandit_exploration_ratio",
                args.bandit_exploration_ratio,
            ),
            bandit_warmup_rounds=preset_overrides.get(
                "bandit_warmup_rounds",
                args.bandit_warmup_rounds,
            ),
            pgzoo_probe_pairs=preset_overrides.get(
                "pgzoo_probe_pairs",
                args.pgzoo_probe_pairs,
            ),
            pgzoo_alpha_ratio=preset_overrides.get(
                "pgzoo_alpha_ratio",
                args.pgzoo_alpha_ratio,
            ),
            pgzoo_momentum=preset_overrides.get(
                "pgzoo_momentum",
                args.pgzoo_momentum,
            ),
            pgzoo_line_candidates=preset_overrides.get(
                "pgzoo_line_candidates",
                args.pgzoo_line_candidates,
            ),
            pgzoo_prior_topology_weight=preset_overrides.get(
                "pgzoo_prior_topology_weight",
                args.pgzoo_prior_topology_weight,
            ),
            pgzoo_prior_base_weight=preset_overrides.get(
                "pgzoo_prior_base_weight",
                args.pgzoo_prior_base_weight,
            ),
            pgzoo_prior_best_weight=preset_overrides.get(
                "pgzoo_prior_best_weight",
                args.pgzoo_prior_best_weight,
            ),
            pgzoo_structured_covariance=preset_overrides.get(
                "pgzoo_structured_covariance",
                (
                    default_cfg.pgzoo_structured_covariance
                    if args.pgzoo_structured_covariance is None
                    else args.pgzoo_structured_covariance
                ),
            ),
            pgzoo_physical_preconditioner=preset_overrides.get(
                "pgzoo_physical_preconditioner",
                (
                    default_cfg.pgzoo_physical_preconditioner
                    if args.pgzoo_physical_preconditioner is None
                    else args.pgzoo_physical_preconditioner
                ),
            ),
            pgzoo_covariance_gamma=preset_overrides.get(
                "pgzoo_covariance_gamma",
                args.pgzoo_covariance_gamma,
            ),
            pgzoo_covariance_ridge=preset_overrides.get(
                "pgzoo_covariance_ridge",
                args.pgzoo_covariance_ridge,
            ),
            pgzoo_preconditioner_ridge=preset_overrides.get(
                "pgzoo_preconditioner_ridge",
                args.pgzoo_preconditioner_ridge,
            ),
            support_pool_size=preset_overrides.get(
                "support_pool_size",
                args.support_pool_size,
            ),
            support_final_size=preset_overrides.get(
                "support_final_size",
                args.support_final_size,
            ),
            support_diffusion_lambda=preset_overrides.get(
                "support_diffusion_lambda",
                args.support_diffusion_lambda,
            ),
            support_probe_scale_ratio=preset_overrides.get(
                "support_probe_scale_ratio",
                args.support_probe_scale_ratio,
            ),
            support_prior_weight=preset_overrides.get(
                "support_prior_weight",
                args.support_prior_weight,
            ),
            support_diversity_penalty=preset_overrides.get(
                "support_diversity_penalty",
                args.support_diversity_penalty,
            ),
            support_success_bonus=preset_overrides.get(
                "support_success_bonus",
                args.support_success_bonus,
            ),
            support_keep_base_support=preset_overrides.get(
                "support_keep_base_support",
                (
                    default_cfg.support_keep_base_support
                    if args.support_keep_base_support is None
                    else args.support_keep_base_support
                ),
            ),
            adaptive_support_selection=preset_overrides.get(
                "adaptive_support_selection",
                (
                    default_cfg.adaptive_support_selection
                    if args.adaptive_support_selection is None
                    else args.adaptive_support_selection
                ),
            ),
            adaptive_support_mass_threshold=preset_overrides.get(
                "adaptive_support_mass_threshold",
                args.adaptive_support_mass_threshold,
            ),
            adaptive_support_max_size=preset_overrides.get(
                "adaptive_support_max_size",
                args.adaptive_support_max_size,
            ),
            measurement_guided_state_refine=preset_overrides.get(
                "measurement_guided_state_refine",
                args.measurement_guided_state_refine,
            ),
            guided_state_region_size=preset_overrides.get(
                "guided_state_region_size",
                args.guided_state_region_size,
            ),
            physical_measurement_gate=preset_overrides.get(
                "physical_measurement_gate",
                args.physical_measurement_gate,
            ),
            measurement_gate_ratio=preset_overrides.get(
                "measurement_gate_ratio",
                args.measurement_gate_ratio,
            ),
            measurement_gate_response_weight=preset_overrides.get(
                "measurement_gate_response_weight",
                args.measurement_gate_response_weight,
            ),
            measurement_gate_effectiveness_weight=preset_overrides.get(
                "measurement_gate_effectiveness_weight",
                args.measurement_gate_effectiveness_weight,
            ),
            measurement_conditioned_state_gate=preset_overrides.get(
                "measurement_conditioned_state_gate",
                args.measurement_conditioned_state_gate,
            ),
            state_gate_ratio=preset_overrides.get(
                "state_gate_ratio",
                args.state_gate_ratio,
            ),
            state_gate_escape_candidates=preset_overrides.get(
                "state_gate_escape_candidates",
                args.state_gate_escape_candidates,
            ),
            adaptive_query_budget=preset_overrides.get(
                "adaptive_query_budget",
                (
                    default_cfg.adaptive_query_budget
                    if args.adaptive_query_budget is None
                    else args.adaptive_query_budget
                ),
            ),
            easy_probe_improvement_ratio=preset_overrides.get(
                "easy_probe_improvement_ratio",
                args.easy_probe_improvement_ratio,
            ),
            easy_population=preset_overrides.get(
                "easy_population",
                args.easy_population,
            ),
            easy_rounds=preset_overrides.get(
                "easy_rounds",
                args.easy_rounds,
            ),
            single_active_size=preset_overrides.get(
                "single_active_size",
                args.single_active_size,
            ),
            active_set_sizes=preset_overrides.get(
                "active_set_sizes",
                args.active_set_sizes,
            ),
            layered_stage_rounds=preset_overrides.get(
                "layered_stage_rounds",
                args.layered_stage_rounds,
            ),
            layered_final_rounds=preset_overrides.get(
                "layered_final_rounds",
                args.layered_final_rounds,
            ),
            multisource_physical_prior=preset_overrides.get(
                "multisource_physical_prior",
                args.multisource_physical_prior,
            ),
            physics_query_allocation=preset_overrides.get(
                "physics_query_allocation",
                args.physics_query_allocation,
            ),
            physics_query_topk=preset_overrides.get(
                "physics_query_topk",
                args.physics_query_topk,
            ),
            physics_query_priority_weight=preset_overrides.get(
                "physics_query_priority_weight",
                args.physics_query_priority_weight,
            ),
            adaptive_challenger_budget=preset_overrides.get(
                "adaptive_challenger_budget",
                args.adaptive_challenger_budget,
            ),
            challenger_population_ratio=preset_overrides.get(
                "challenger_population_ratio",
                args.challenger_population_ratio,
            ),
            challenger_rounds=preset_overrides.get(
                "challenger_rounds",
                args.challenger_rounds,
            ),
            branch_pruning=preset_overrides.get(
                "branch_pruning",
                args.branch_pruning,
            ),
            branch_prune_progress_ratio=preset_overrides.get(
                "branch_prune_progress_ratio",
                args.branch_prune_progress_ratio,
            ),
            branch_prune_score_gap_ratio=preset_overrides.get(
                "branch_prune_score_gap_ratio",
                args.branch_prune_score_gap_ratio,
            ),
            score_stagnation_early_stop=preset_overrides.get(
                "score_stagnation_early_stop",
                (
                    default_cfg.score_stagnation_early_stop
                    if args.score_stagnation_early_stop is None
                    else args.score_stagnation_early_stop
                ),
            ),
            score_stagnation_rounds=preset_overrides.get(
                "score_stagnation_rounds",
                args.score_stagnation_rounds,
            ),
            score_gain_ratio_threshold=preset_overrides.get(
                "score_gain_ratio_threshold",
                args.score_gain_ratio_threshold,
            ),
            search_min_rounds_before_stop=preset_overrides.get(
                "search_min_rounds_before_stop",
                args.search_min_rounds_before_stop,
            ),
            uncertainty_aware_pruning=preset_overrides.get(
                "uncertainty_aware_pruning",
                args.uncertainty_aware_pruning,
            ),
            physics_aware_early_stop=preset_overrides.get(
                "physics_aware_early_stop",
                args.physics_aware_early_stop,
            ),
            termination_uncertainty_ratio_tau=preset_overrides.get(
                "termination_uncertainty_ratio_tau",
                args.termination_uncertainty_ratio_tau,
            ),
            termination_uncertainty_floor=preset_overrides.get(
                "termination_uncertainty_floor",
                args.termination_uncertainty_floor,
            ),
            termination_progress_ratio_floor=preset_overrides.get(
                "termination_progress_ratio_floor",
                args.termination_progress_ratio_floor,
            ),
            termination_physics_quality_floor=preset_overrides.get(
                "termination_physics_quality_floor",
                args.termination_physics_quality_floor,
            ),
            guarded_boundary_probe=preset_overrides.get(
                "guarded_boundary_probe",
                args.guarded_boundary_probe,
            ),
            guarded_boundary_probe_steps=preset_overrides.get(
                "guarded_boundary_probe_steps",
                args.guarded_boundary_probe_steps,
            ),
            guarded_boundary_probe_max_uses=preset_overrides.get(
                "guarded_boundary_probe_max_uses",
                args.guarded_boundary_probe_max_uses,
            ),
            fd_warmup_rounds=preset_overrides.get(
                "fd_warmup_rounds",
                args.fd_warmup_rounds,
            ),
            fd_warmup_topk=preset_overrides.get(
                "fd_warmup_topk",
                args.fd_warmup_topk,
            ),
            fd_region_selection=preset_overrides.get(
                "fd_region_selection",
                args.fd_region_selection,
            ),
            fd_iterations=preset_overrides.get("fd_iterations", args.fd_iterations),
            fd_coordinate_eps_ratio=preset_overrides.get(
                "fd_coordinate_eps_ratio",
                args.fd_coordinate_eps_ratio,
            ),
            fd_coordinate_eps_floor=preset_overrides.get(
                "fd_coordinate_eps_floor",
                args.fd_coordinate_eps_floor,
            ),
            fd_central_gradient=preset_overrides.get(
                "fd_central_gradient",
                args.fd_central_gradient,
            ),
            fd_line_steps=preset_overrides.get("fd_line_steps", args.fd_line_steps),
            fd_line_decay=preset_overrides.get("fd_line_decay", args.fd_line_decay),
            sparse_support_beta=preset_overrides.get(
                "sparse_support_beta",
                args.sparse_support_beta,
            ),
            sparse_seed_weight=preset_overrides.get(
                "sparse_seed_weight",
                args.sparse_seed_weight,
            ),
            sparse_region_mix=preset_overrides.get(
                "sparse_region_mix",
                args.sparse_region_mix,
            ),
            sparse_region_penalty=preset_overrides.get(
                "sparse_region_penalty",
                args.sparse_region_penalty,
            ),
            query_mode=preset_overrides.get("query_mode", args.query_mode),
            early_stop_on_success=preset_overrides.get(
                "early_stop_on_success",
                (
                    default_cfg.early_stop_on_success
                    if args.early_stop_on_success is None
                    else args.early_stop_on_success
                ),
            ),
            max_queries_per_sample=preset_overrides.get(
                "max_queries_per_sample", args.max_queries_per_sample
            ),
        )
        for field in dataclass_fields(TopologyLatentAttackConfig):
            field_name = field.name
            if field_name not in canonical_explicit_arg_names:
                continue
            if hasattr(args, field_name):
                setattr(cfg, field_name, getattr(args, field_name))
        summary = run_topology_latent_blackbox_attack(
            system_id=system_id,
            oracle_arch=args.oracle_arch,
            data_dir=args.data_dir,
            input_dim=args.input_dim,
            attack_class=args.attack_class,
            max_samples=args.max_samples,
            checkpoint_path=args.oracle_ckpt,
            exp_tag=args.exp_tag,
            run_seed=args.seed,
            device=device,
            attack_config=cfg,
            topology_mode=args.topology_mode,
            noise_model=args.noise_model,
            save_level=args.save_level,
        )
        print(
            f"IEEE-{system_id}: subset={summary['subset_size']} | "
            f"ASR={summary['attack_success_rate']:.2f}% | "
            f"avg_queries={summary['avg_queries']:.2f}"
        )


if __name__ == "__main__":
    main()
