from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch

from .attack import TopologyLatentAttackConfig
from .config import resolve_attack_preset
from .results_layout import resolve_run_dir


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def build_attack_config_from_preset(
    system_id: int,
    preset_name: str,
) -> TopologyLatentAttackConfig:
    preset = resolve_attack_preset(preset_name, system_id)
    return TopologyLatentAttackConfig(
        attack_mode=str(preset.get("attack_mode", "region_search")),
        state_seed_mode=str(preset.get("state_seed_mode", "legacy")),
        region_size=int(preset.get("region_size", 6)),
        anchor_size=int(preset.get("anchor_size", 2)),
        region_candidates=int(preset.get("region_candidates", 4)),
        anchor_pool_size=int(preset.get("anchor_pool_size", 6)),
        probabilistic_region_prior=bool(
            preset.get("probabilistic_region_prior", False)
        ),
        prior_temperature=float(preset.get("prior_temperature", 1.0)),
        prior_uniform_mixing=float(preset.get("prior_uniform_mixing", 0.0)),
        hierarchical_probe=bool(preset.get("hierarchical_probe", False)),
        coarse_probe_directions=int(preset.get("coarse_probe_directions", 1)),
        fine_probe_topk=int(preset.get("fine_probe_topk", 3)),
        initial_probe_region_topk=int(preset.get("initial_probe_region_topk", 0)),
        probe_expand_improvement_ratio=float(
            preset.get("probe_expand_improvement_ratio", 0.0)
        ),
        region_proposal_mode=str(preset.get("region_proposal_mode", "default")),
        proposal_diffusion_steps=int(preset.get("proposal_diffusion_steps", 2)),
        proposal_diffusion_alpha=float(preset.get("proposal_diffusion_alpha", 0.35)),
        proposal_flow_weight=float(preset.get("proposal_flow_weight", 0.30)),
        proposal_corridor_weight=float(preset.get("proposal_corridor_weight", 0.25)),
        probe_directions=int(preset.get("probe_directions", 2)),
        probe_scale_ratio=float(preset.get("probe_scale_ratio", 0.25)),
        population=int(preset.get("population", 12)),
        rounds=int(preset.get("rounds", 24)),
        radius_ratio=float(preset.get("radius_ratio", 1.0)),
        radius_floor=float(preset.get("radius_floor", 0.01)),
        step_ratio=float(preset.get("step_ratio", 0.35)),
        step_decay=float(preset.get("step_decay", 0.7)),
        patience=int(preset.get("patience", 3)),
        min_step_ratio=float(preset.get("min_step_ratio", 0.08)),
        max_step_shrinks=int(preset.get("max_step_shrinks", 4)),
        measurement_delta_l2_ratio_cap=float(
            preset.get("measurement_delta_l2_ratio_cap", 0.0)
        ),
        realism_penalty_weight=float(preset.get("realism_penalty_weight", 0.0)),
        fdia_preserve_weight=float(preset.get("fdia_preserve_weight", 0.0)),
        fdia_backbone_lock_ratio=float(preset.get("fdia_backbone_lock_ratio", 0.0)),
        fdia_state_projection_min=float(preset.get("fdia_state_projection_min", 0.70)),
        fdia_measurement_projection_min=float(
            preset.get("fdia_measurement_projection_min", 0.70)
        ),
        fdia_offsupport_max=float(preset.get("fdia_offsupport_max", 0.35)),
        fdia_state_penalty_weight=float(preset.get("fdia_state_penalty_weight", 0.45)),
        fdia_measurement_penalty_weight=float(
            preset.get("fdia_measurement_penalty_weight", 0.35)
        ),
        fdia_offsupport_penalty_weight=float(
            preset.get("fdia_offsupport_penalty_weight", 0.20)
        ),
        region_budget_topk=int(preset.get("region_budget_topk", 1)),
        region_budget_explore_rounds=int(preset.get("region_budget_explore_rounds", 0)),
        feedback_loop=bool(preset.get("feedback_loop", False)),
        feedback_round_chunk=int(preset.get("feedback_round_chunk", 2)),
        feedback_branch_topk=int(preset.get("feedback_branch_topk", 2)),
        feedback_stagnation_trigger=int(preset.get("feedback_stagnation_trigger", 2)),
        feedback_probe_gap_abs_threshold=float(
            preset.get("feedback_probe_gap_abs_threshold", 0.0)
        ),
        feedback_probe_gap_ratio_threshold=float(
            preset.get("feedback_probe_gap_ratio_threshold", 0.0)
        ),
        feedback_keep_probe_best_incumbent=bool(
            preset.get("feedback_keep_probe_best_incumbent", True)
        ),
        feedback_min_state_dim=int(preset.get("feedback_min_state_dim", 0)),
        detector_feedback_region=bool(preset.get("detector_feedback_region", False)),
        detector_feedback_prior_mix=float(
            preset.get("detector_feedback_prior_mix", 0.25)
        ),
        detector_feedback_success_bonus=float(
            preset.get("detector_feedback_success_bonus", 0.75)
        ),
        detector_feedback_min_gain=float(
            preset.get("detector_feedback_min_gain", 0.0)
        ),
        feedback_physics_reward_shaping=bool(
            preset.get("feedback_physics_reward_shaping", False)
        ),
        feedback_physics_weight=float(preset.get("feedback_physics_weight", 0.50)),
        region_budget_score_slack=float(preset.get("region_budget_score_slack", 0.0)),
        budget_region_max_probe_best_prior=float(
            preset.get("budget_region_max_probe_best_prior", 1.0)
        ),
        budget_region_prior_tiebreak=bool(
            preset.get("budget_region_prior_tiebreak", False)
        ),
        measurement_suppression_strength=float(
            preset.get("measurement_suppression_strength", 0.0)
        ),
        injection_scale=float(preset.get("injection_scale", 0.82)),
        flow_scale=float(preset.get("flow_scale", 1.0)),
        leverage_suppression=float(preset.get("leverage_suppression", 0.30)),
        min_channel_scale=float(preset.get("min_channel_scale", 0.60)),
        probe_channel_shaping_only=bool(preset.get("probe_channel_shaping_only", False)),
        structured_search_directions=bool(
            preset.get("structured_search_directions", False)
        ),
        structured_direction_count=int(preset.get("structured_direction_count", 4)),
        measurement_basis_search=bool(preset.get("measurement_basis_search", False)),
        measurement_bandit_search=bool(preset.get("measurement_bandit_search", False)),
        state_basis_search=bool(preset.get("state_basis_search", False)),
        state_bandit_search=bool(preset.get("state_bandit_search", False)),
        state_subspace_pgzoo=bool(preset.get("state_subspace_pgzoo", False)),
        state_basis_dim=int(preset.get("state_basis_dim", 4)),
        bandit_direction_samples=int(preset.get("bandit_direction_samples", 2)),
        bandit_momentum=float(preset.get("bandit_momentum", 0.75)),
        bandit_exploration_ratio=float(preset.get("bandit_exploration_ratio", 1.0)),
        bandit_warmup_rounds=int(preset.get("bandit_warmup_rounds", 0)),
        pgzoo_probe_pairs=int(preset.get("pgzoo_probe_pairs", 2)),
        pgzoo_alpha_ratio=float(preset.get("pgzoo_alpha_ratio", 1.0)),
        pgzoo_momentum=float(preset.get("pgzoo_momentum", 0.75)),
        pgzoo_line_candidates=int(preset.get("pgzoo_line_candidates", 3)),
        pgzoo_prior_topology_weight=float(
            preset.get("pgzoo_prior_topology_weight", 0.45)
        ),
        pgzoo_prior_base_weight=float(
            preset.get("pgzoo_prior_base_weight", 0.35)
        ),
        pgzoo_prior_best_weight=float(
            preset.get("pgzoo_prior_best_weight", 0.20)
        ),
        pgzoo_structured_covariance=bool(
            preset.get("pgzoo_structured_covariance", False)
        ),
        pgzoo_physical_preconditioner=bool(
            preset.get("pgzoo_physical_preconditioner", False)
        ),
        pgzoo_covariance_gamma=float(preset.get("pgzoo_covariance_gamma", 0.75)),
        pgzoo_covariance_ridge=float(preset.get("pgzoo_covariance_ridge", 1e-3)),
        pgzoo_preconditioner_ridge=float(
            preset.get("pgzoo_preconditioner_ridge", 0.10)
        ),
        measurement_guided_state_refine=bool(
            preset.get("measurement_guided_state_refine", False)
        ),
        guided_state_region_size=int(preset.get("guided_state_region_size", 0)),
        physical_measurement_gate=bool(
            preset.get("physical_measurement_gate", False)
        ),
        measurement_gate_ratio=float(preset.get("measurement_gate_ratio", 1.0)),
        measurement_gate_response_weight=float(
            preset.get("measurement_gate_response_weight", 0.60)
        ),
        measurement_gate_effectiveness_weight=float(
            preset.get("measurement_gate_effectiveness_weight", 0.40)
        ),
        measurement_conditioned_state_gate=bool(
            preset.get("measurement_conditioned_state_gate", False)
        ),
        state_gate_ratio=float(preset.get("state_gate_ratio", 1.0)),
        state_gate_escape_candidates=int(
            preset.get("state_gate_escape_candidates", 0)
        ),
        adaptive_query_budget=bool(preset.get("adaptive_query_budget", False)),
        easy_probe_improvement_ratio=float(
            preset.get("easy_probe_improvement_ratio", 0.50)
        ),
        easy_population=int(preset.get("easy_population", 0)),
        easy_rounds=int(preset.get("easy_rounds", 0)),
        single_active_size=int(preset.get("single_active_size", 0)),
        active_set_sizes=preset.get("active_set_sizes", ()),
        layered_stage_rounds=int(preset.get("layered_stage_rounds", 0)),
        layered_final_rounds=int(preset.get("layered_final_rounds", 0)),
        multisource_physical_prior=bool(
            preset.get("multisource_physical_prior", False)
        ),
        physics_query_allocation=bool(preset.get("physics_query_allocation", False)),
        physics_query_topk=int(preset.get("physics_query_topk", 1)),
        physics_query_priority_weight=float(
            preset.get("physics_query_priority_weight", 0.35)
        ),
        adaptive_challenger_budget=bool(
            preset.get("adaptive_challenger_budget", False)
        ),
        challenger_population_ratio=float(
            preset.get("challenger_population_ratio", 0.50)
        ),
        challenger_rounds=int(preset.get("challenger_rounds", 2)),
        branch_pruning=bool(preset.get("branch_pruning", False)),
        branch_prune_progress_ratio=float(
            preset.get("branch_prune_progress_ratio", 0.20)
        ),
        branch_prune_score_gap_ratio=float(
            preset.get("branch_prune_score_gap_ratio", 0.02)
        ),
        score_stagnation_early_stop=bool(
            preset.get("score_stagnation_early_stop", False)
        ),
        score_stagnation_rounds=int(preset.get("score_stagnation_rounds", 2)),
        score_gain_ratio_threshold=float(
            preset.get("score_gain_ratio_threshold", 0.005)
        ),
        search_min_rounds_before_stop=int(
            preset.get("search_min_rounds_before_stop", 6)
        ),
        uncertainty_aware_pruning=bool(
            preset.get("uncertainty_aware_pruning", False)
        ),
        physics_aware_early_stop=bool(
            preset.get("physics_aware_early_stop", False)
        ),
        termination_uncertainty_ratio_tau=float(
            preset.get("termination_uncertainty_ratio_tau", 0.25)
        ),
        termination_uncertainty_floor=float(
            preset.get("termination_uncertainty_floor", 0.30)
        ),
        termination_progress_ratio_floor=float(
            preset.get("termination_progress_ratio_floor", 0.50)
        ),
        termination_physics_quality_floor=float(
            preset.get("termination_physics_quality_floor", 0.65)
        ),
        guarded_boundary_probe=bool(preset.get("guarded_boundary_probe", False)),
        guarded_boundary_probe_steps=int(
            preset.get("guarded_boundary_probe_steps", 3)
        ),
        guarded_boundary_probe_max_uses=int(
            preset.get("guarded_boundary_probe_max_uses", 1)
        ),
        fd_warmup_rounds=int(preset.get("fd_warmup_rounds", 0)),
        fd_warmup_topk=int(preset.get("fd_warmup_topk", 0)),
        fd_region_selection=str(preset.get("fd_region_selection", "topology_prior")),
        fd_iterations=int(preset.get("fd_iterations", 3)),
        fd_coordinate_eps_ratio=float(preset.get("fd_coordinate_eps_ratio", 0.10)),
        fd_coordinate_eps_floor=float(preset.get("fd_coordinate_eps_floor", 0.0025)),
        fd_central_gradient=bool(preset.get("fd_central_gradient", False)),
        fd_line_steps=int(preset.get("fd_line_steps", 1)),
        fd_line_decay=float(preset.get("fd_line_decay", 0.5)),
        sparse_support_beta=float(preset.get("sparse_support_beta", 0.0)),
        sparse_seed_weight=float(preset.get("sparse_seed_weight", 0.0)),
        sparse_region_mix=float(preset.get("sparse_region_mix", 0.0)),
        sparse_region_penalty=float(preset.get("sparse_region_penalty", 0.35)),
        budget_objective_mode=str(preset.get("budget_objective_mode", "minimum_success")),
        query_mode=str(preset.get("query_mode", "margin")),
        early_stop_on_success=bool(preset.get("early_stop_on_success", True)),
        max_queries_per_sample=int(preset.get("max_queries_per_sample", 0)),
    )


def build_result_dir(system_id: int, exp_tag: str) -> Path:
    return resolve_run_dir(system_id=system_id, exp_tag=exp_tag)
