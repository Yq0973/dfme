from __future__ import annotations

from copy import deepcopy


# Legacy reconstruction presets are intentionally isolated from
# `config.SYSTEM_ATTACK_PRESETS` so the current mainline remains unchanged.
LEGACY_ATTACK_PRESETS = {
    "legacy_region_funnel_v3": {
        14: {
            "attack_mode": "region_search",
            "region_size": 8,
            "anchor_size": 2,
            "region_candidates": 6,
            "anchor_pool_size": 8,
            "hierarchical_probe": True,
            "coarse_probe_directions": 1,
            "fine_probe_topk": 2,
            "initial_probe_region_topk": 4,
            "probe_directions": 3,
            "probe_scale_ratio": 0.35,
            "population": 8,
            "rounds": 24,
            "region_proposal_mode": "hierarchical_corridor",
            "proposal_diffusion_steps": 2,
            "proposal_diffusion_alpha": 0.35,
            "proposal_flow_weight": 0.30,
            "proposal_corridor_weight": 0.25,
            "region_budget_topk": 1,
            "region_budget_explore_rounds": 0,
            "feedback_loop": False,
            "detector_feedback_region": False,
            "adaptive_query_budget": True,
            "easy_population": 4,
            "easy_rounds": 4,
            "easy_probe_improvement_ratio": 0.10,
            "structured_search_directions": True,
            "state_basis_search": False,
            "state_bandit_search": False,
            "query_mode": "margin",
            "realism_penalty_weight": 0.0,
            "fdia_preserve_weight": 0.0,
            "measurement_delta_l2_ratio_cap": 0.0,
            "early_stop_on_success": True,
        },
        30: {
            "attack_mode": "region_search",
            "region_size": 8,
            "anchor_size": 2,
            "region_candidates": 6,
            "anchor_pool_size": 8,
            "hierarchical_probe": True,
            "coarse_probe_directions": 1,
            "fine_probe_topk": 2,
            "initial_probe_region_topk": 4,
            "probe_directions": 3,
            "probe_scale_ratio": 0.35,
            "population": 8,
            "rounds": 24,
            "region_proposal_mode": "hierarchical_corridor",
            "proposal_diffusion_steps": 2,
            "proposal_diffusion_alpha": 0.35,
            "proposal_flow_weight": 0.30,
            "proposal_corridor_weight": 0.25,
            "region_budget_topk": 1,
            "region_budget_explore_rounds": 0,
            "feedback_loop": False,
            "detector_feedback_region": False,
            "adaptive_query_budget": True,
            "easy_population": 4,
            "easy_rounds": 4,
            "easy_probe_improvement_ratio": 0.10,
            "structured_search_directions": True,
            "state_basis_search": False,
            "state_bandit_search": False,
            "query_mode": "margin",
            "realism_penalty_weight": 0.0,
            "fdia_preserve_weight": 0.0,
            "measurement_delta_l2_ratio_cap": 0.0,
            "early_stop_on_success": True,
        },
        118: {
            "attack_mode": "region_search",
            "region_size": 16,
            "anchor_size": 3,
            "region_candidates": 8,
            "anchor_pool_size": 12,
            "hierarchical_probe": True,
            "coarse_probe_directions": 1,
            "fine_probe_topk": 2,
            "initial_probe_region_topk": 4,
            "probe_directions": 3,
            "probe_scale_ratio": 0.25,
            "population": 8,
            "rounds": 24,
            "region_proposal_mode": "hierarchical_corridor",
            "proposal_diffusion_steps": 2,
            "proposal_diffusion_alpha": 0.35,
            "proposal_flow_weight": 0.30,
            "proposal_corridor_weight": 0.25,
            "region_budget_topk": 1,
            "region_budget_explore_rounds": 0,
            "feedback_loop": False,
            "detector_feedback_region": False,
            "adaptive_query_budget": True,
            "easy_population": 4,
            "easy_rounds": 6,
            "easy_probe_improvement_ratio": 0.08,
            "structured_search_directions": True,
            "state_basis_search": True,
            "state_bandit_search": True,
            "state_basis_dim": 16,
            "query_mode": "margin",
            "realism_penalty_weight": 0.0,
            "fdia_preserve_weight": 0.0,
            "measurement_delta_l2_ratio_cap": 0.0,
            "early_stop_on_success": True,
        },
    }
}


def list_legacy_presets() -> list[str]:
    return sorted(LEGACY_ATTACK_PRESETS.keys())


def resolve_legacy_attack_preset(preset_name: str, system_id: int) -> dict:
    if preset_name not in LEGACY_ATTACK_PRESETS:
        supported = ", ".join(list_legacy_presets())
        raise ValueError(
            f"Unsupported legacy preset: {preset_name}. Supported presets: {supported}"
        )
    preset = LEGACY_ATTACK_PRESETS[preset_name]
    return deepcopy(preset.get(int(system_id), {}))
