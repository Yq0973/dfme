from dataclasses import dataclass, replace

import torch

from .oracle import OracleQueryAdapter, QueryBudgetExceeded
from .topology import StateCouplingTopology


@dataclass
class TopologyLatentAttackConfig:
    attack_mode: str = "region_search"
    state_seed_mode: str = "legacy"
    region_size: int = 6
    anchor_size: int = 2
    region_candidates: int = 4
    anchor_pool_size: int = 6
    probabilistic_region_prior: bool = False
    prior_temperature: float = 1.0
    prior_uniform_mixing: float = 0.0
    hierarchical_probe: bool = False
    coarse_probe_directions: int = 1
    fine_probe_topk: int = 3
    initial_probe_region_topk: int = 0
    probe_expand_improvement_ratio: float = 0.0
    region_proposal_mode: str = "default"
    proposal_diffusion_steps: int = 2
    proposal_diffusion_alpha: float = 0.35
    proposal_flow_weight: float = 0.30
    proposal_corridor_weight: float = 0.25
    probe_directions: int = 2
    probe_scale_ratio: float = 0.25
    population: int = 12
    rounds: int = 24
    radius_ratio: float = 1.0
    radius_floor: float = 0.01
    step_ratio: float = 0.35
    step_decay: float = 0.7
    patience: int = 3
    min_step_ratio: float = 0.08
    max_step_shrinks: int = 4
    measurement_delta_l2_ratio_cap: float = 0.0
    realism_penalty_weight: float = 0.0
    fdia_preserve_weight: float = 0.0
    fdia_backbone_lock_ratio: float = 0.0
    fdia_state_projection_min: float = 0.70
    fdia_measurement_projection_min: float = 0.70
    fdia_offsupport_max: float = 0.35
    fdia_state_penalty_weight: float = 0.45
    fdia_measurement_penalty_weight: float = 0.35
    fdia_offsupport_penalty_weight: float = 0.20
    region_budget_topk: int = 1
    region_budget_explore_rounds: int = 0
    feedback_loop: bool = False
    feedback_round_chunk: int = 2
    feedback_branch_topk: int = 2
    feedback_stagnation_trigger: int = 2
    feedback_probe_gap_abs_threshold: float = 0.0
    feedback_probe_gap_ratio_threshold: float = 0.0
    feedback_keep_probe_best_incumbent: bool = True
    feedback_min_state_dim: int = 0
    detector_feedback_region: bool = False
    detector_feedback_prior_mix: float = 0.25
    detector_feedback_success_bonus: float = 0.75
    detector_feedback_min_gain: float = 0.0
    feedback_physics_reward_shaping: bool = False
    feedback_physics_weight: float = 0.50
    region_budget_score_slack: float = 0.0
    budget_region_max_probe_best_prior: float = 1.0
    budget_region_prior_tiebreak: bool = False
    measurement_suppression_strength: float = 0.0
    injection_scale: float = 0.82
    flow_scale: float = 1.0
    leverage_suppression: float = 0.30
    min_channel_scale: float = 0.60
    probe_channel_shaping_only: bool = False
    structured_search_directions: bool = False
    structured_direction_count: int = 4
    measurement_basis_search: bool = False
    measurement_bandit_search: bool = False
    state_basis_search: bool = False
    state_bandit_search: bool = False
    state_subspace_pgzoo: bool = False
    state_basis_dim: int = 4
    bandit_direction_samples: int = 2
    bandit_momentum: float = 0.75
    bandit_exploration_ratio: float = 1.0
    bandit_warmup_rounds: int = 0
    pgzoo_probe_pairs: int = 2
    pgzoo_alpha_ratio: float = 1.0
    pgzoo_momentum: float = 0.75
    pgzoo_line_candidates: int = 3
    pgzoo_query_prior_weight: float = 0.35
    pgzoo_prior_topology_weight: float = 0.45
    pgzoo_prior_base_weight: float = 0.35
    pgzoo_prior_best_weight: float = 0.20
    pgzoo_structured_covariance: bool = False
    pgzoo_physical_preconditioner: bool = False
    pgzoo_covariance_gamma: float = 0.75
    pgzoo_covariance_ridge: float = 1e-3
    pgzoo_preconditioner_ridge: float = 0.10
    support_pool_size: int = 0
    support_final_size: int = 0
    support_diffusion_lambda: float = 0.75
    support_probe_scale_ratio: float = 0.30
    support_prior_weight: float = 0.25
    support_diversity_penalty: float = 0.15
    support_success_bonus: float = 1.0
    support_keep_base_support: bool = True
    adaptive_support_selection: bool = False
    adaptive_support_mass_threshold: float = 0.85
    adaptive_support_max_size: int = 0
    measurement_guided_state_refine: bool = False
    guided_state_region_size: int = 0
    physical_measurement_gate: bool = False
    measurement_gate_ratio: float = 1.0
    measurement_gate_response_weight: float = 0.60
    measurement_gate_effectiveness_weight: float = 0.40
    measurement_conditioned_state_gate: bool = False
    state_gate_ratio: float = 1.0
    state_gate_escape_candidates: int = 0
    adaptive_query_budget: bool = False
    easy_probe_improvement_ratio: float = 0.50
    easy_population: int = 0
    easy_rounds: int = 0
    single_active_size: int = 0
    active_set_sizes: tuple[int, ...] | str = ()
    layered_stage_rounds: int = 0
    layered_final_rounds: int = 0
    multisource_physical_prior: bool = False
    physics_query_allocation: bool = False
    physics_query_topk: int = 1
    physics_query_priority_weight: float = 0.35
    adaptive_challenger_budget: bool = False
    challenger_population_ratio: float = 0.50
    challenger_rounds: int = 2
    branch_pruning: bool = False
    branch_prune_progress_ratio: float = 0.20
    branch_prune_score_gap_ratio: float = 0.02
    score_stagnation_early_stop: bool = False
    score_stagnation_rounds: int = 2
    score_gain_ratio_threshold: float = 0.005
    search_min_rounds_before_stop: int = 6
    uncertainty_aware_pruning: bool = False
    physics_aware_early_stop: bool = False
    termination_uncertainty_ratio_tau: float = 0.25
    termination_uncertainty_floor: float = 0.30
    termination_progress_ratio_floor: float = 0.50
    termination_physics_quality_floor: float = 0.65
    guarded_boundary_probe: bool = False
    guarded_boundary_probe_steps: int = 3
    guarded_boundary_probe_max_uses: int = 1
    fd_warmup_rounds: int = 0
    fd_warmup_topk: int = 0
    fd_region_selection: str = "topology_prior"
    fd_iterations: int = 3
    fd_coordinate_eps_ratio: float = 0.10
    fd_coordinate_eps_floor: float = 0.0025
    fd_central_gradient: bool = False
    fd_line_steps: int = 1
    fd_line_decay: float = 0.5
    sparse_support_beta: float = 0.0
    sparse_seed_weight: float = 0.0
    sparse_region_mix: float = 0.0
    sparse_region_penalty: float = 0.35
    budget_objective_mode: str = "minimum_success"
    query_mode: str = "margin"
    early_stop_on_success: bool = False
    max_queries_per_sample: int = 0


@dataclass
class _RegionSearchState:
    region: torch.Tensor
    rank: int
    region_prior: float
    initial_score: float
    initial_objective: float
    initial_radius: float
    radius: float
    step: float
    min_step: float
    best_objective: float
    best_score: float
    best_pred: int
    best_delta_c: torch.Tensor
    best_delta_z: torch.Tensor
    best_adv: torch.Tensor
    query_coordinate_prior: torch.Tensor | None = None
    success_adv: torch.Tensor | None = None
    success_delta_c: torch.Tensor | None = None
    success_delta_z: torch.Tensor | None = None
    success_objective: float | None = None
    success_score: float | None = None
    success_pred: int | None = None
    query_budget_population: int = 0
    query_budget_rounds: int = 0
    probe_improvement_ratio: float = 0.0
    physics_quality: float = 0.0
    allocation_priority: float = 0.0
    challenger_branch: bool = False
    active: bool = True
    pruned: bool = False
    early_stopped: bool = False
    query_cap_reached: bool = False
    stagnant_rounds: int = 0
    no_improve: int = 0
    step_shrinks: int = 0
    rounds_used: int = 0
    guard_probe_attempts: int = 0
    clean_adv: torch.Tensor | None = None
    clean_delta_c: torch.Tensor | None = None
    clean_delta_z: torch.Tensor | None = None
    clean_pred: int | None = None


def _project_l2(delta_c: torch.Tensor, radius: float) -> torch.Tensor:
    norm = torch.linalg.norm(delta_c, dim=1, keepdim=True).clamp(min=1e-12)
    scale = torch.minimum(torch.ones_like(norm), torch.full_like(norm, radius) / norm)
    return delta_c * scale


class TopologyLatentQueryAttack:
    def __init__(
        self,
        topology: StateCouplingTopology,
        oracle: OracleQueryAdapter,
        config: TopologyLatentAttackConfig,
    ) -> None:
        self.topology = topology
        self.oracle = oracle
        self.config = config
        self.channel_weights = None
        if float(self.config.measurement_suppression_strength) > 0.0:
            self.channel_weights = self.topology.build_channel_weights(
                injection_scale=self.config.injection_scale,
                flow_scale=self.config.flow_scale,
                leverage_suppression=self.config.leverage_suppression,
                min_channel_scale=self.config.min_channel_scale,
            )
        self.search_channel_weights = self.channel_weights
        self.search_shaping_strength = float(self.config.measurement_suppression_strength)
        if self.config.probe_channel_shaping_only:
            self.search_channel_weights = None
            self.search_shaping_strength = 0.0
        self.measurement_state_helper = None
        if self._uses_measurement_regions() and bool(self.config.measurement_guided_state_refine):
            state_cfg = replace(
                self.config,
                attack_mode="region_search",
                measurement_basis_search=False,
                measurement_bandit_search=False,
                measurement_guided_state_refine=False,
            )
            self.measurement_state_helper = TopologyLatentQueryAttack(
                topology=topology,
                oracle=oracle,
                config=state_cfg,
            )

    def _max_queries_per_sample(self) -> int:
        return max(0, int(getattr(self.config, "max_queries_per_sample", 0)))

    def _begin_sample_query_budget(self) -> None:
        max_queries = self._max_queries_per_sample()
        if max_queries > 0:
            self.oracle.begin_attack_budget(max_queries=max_queries)
        else:
            self.oracle.clear_attack_budget()

    def _end_sample_query_budget(self) -> None:
        self.oracle.clear_attack_budget()

    def _remaining_sample_query_budget(self) -> int | None:
        return self.oracle.remaining_attack_budget()

    def _has_query_budget_for(self, requested_queries: int) -> bool:
        remaining = self._remaining_sample_query_budget()
        if remaining is None:
            return True
        return int(remaining) >= int(max(0, requested_queries))

    def _default_budget_region(self) -> torch.Tensor:
        region_dim = max(1, min(int(self.config.region_size), int(self._region_total_dim())))
        return torch.arange(region_dim, dtype=torch.long)

    def _budget_clean_fallback_result(self, x: torch.Tensor, label: int) -> dict:
        x0 = x.reshape(1, -1).to(dtype=torch.float32)
        clean_query = None
        if self._has_query_budget_for(1):
            try:
                clean_query = self.oracle.query(x0)
            except QueryBudgetExceeded:
                clean_query = None
        clean_score = 0.0
        clean_pred = int(label)
        if clean_query is not None:
            clean_score = float(self._score(clean_query)[0].item())
            clean_pred = int(clean_query.pred[0].item())
        probe_summary = self._empty_probe_summary(clean_score=clean_score)
        region = self._default_budget_region()
        zero_delta_c = torch.zeros(1, int(region.numel()), dtype=torch.float32)
        zero_delta_z = torch.zeros_like(x0)
        return {
            "adv_x": x0.squeeze(0),
            "delta_c": zero_delta_c.squeeze(0),
            "delta_z": zero_delta_z.squeeze(0),
            "success": False,
            "final_pred": int(clean_pred),
            "final_score": float(clean_score),
            "final_objective": float(clean_score),
            "region": region,
            "radius": 0.0,
            "queries_used": int(self.oracle.attack_queries_used()),
            "proposal_queries": 0,
            "selected_region_rank": 0,
            "region_candidate_count": 1,
            "budget_region_ranks": [0],
            "budget_region_count": 1,
            "budget_triggered": False,
            "clean_score": float(probe_summary["clean_score"]),
            "probe_best_score": float(probe_summary["probe_best_score"]),
            "probe_second_score": float(probe_summary["probe_second_score"]),
            "probe_score_gap": float(probe_summary["probe_score_gap"]),
            "probe_improvement": float(probe_summary["probe_improvement"]),
            "probe_best_rank": int(probe_summary["probe_best_rank"]),
            "probe_second_rank": int(probe_summary["probe_second_rank"]),
            "probe_best_prior": float(probe_summary["probe_best_prior"]),
            "probe_second_prior": float(probe_summary["probe_second_prior"]),
            "probe_success_count": int(probe_summary["probe_success_count"]),
            "detector_feedback_used": False,
            "detector_feedback_candidate_count": 0,
            "detector_feedback_total_reward": 0.0,
            "detector_feedback_best_reward": 0.0,
            "probe_improvement_ratio": 0.0,
            "query_budget_population": 0,
            "query_budget_rounds": 0,
            "adaptive_query_budget_used": False,
            "selected_region_physics_quality": 0.0,
            "selected_region_allocation_priority": 0.0,
            "selected_region_progress_ratio": 0.0,
            "selected_region_boundary_ratio": 0.0,
            "selected_region_boundary_uncertainty": 0.0,
            "selected_region_guard_probe_attempts": 0,
            "selected_region_stagnant_rounds": 0,
            "selected_region_challenger_branch": False,
            "selected_region_early_stopped": True,
            "region_space": self._region_space(),
            "candidate_rows": [],
            "measurement_gate_enabled": False,
            "measurement_gate_keep_dim": int(self.topology.n_measurements),
            "measurement_gate_keep_ratio": 1.0,
            "state_gate_enabled": False,
            "state_gate_keep_dim": int(self.topology.n_states),
            "state_gate_keep_ratio": 1.0,
            "active_set_size": int(self.topology.n_states),
            "active_set_stage_index": -1,
            "active_set_keep_ratio": 1.0,
            "layered_success_stage": -1,
            "layered_total_stages": 1,
            "query_cap_reached": True,
        }

    def _measurement_reference(self, x0: torch.Tensor) -> torch.Tensor:
        x0 = x0.reshape(1, -1).to(dtype=torch.float32)
        return self.topology.project_measurement_reference(x0).reshape(-1)

    def _uses_measurement_regions(self) -> bool:
        return str(self.config.attack_mode).lower().startswith("measurement_")

    def _region_space(self) -> str:
        return "measurement" if self._uses_measurement_regions() else "state"

    def _region_base_vector(
        self,
        x0: torch.Tensor,
        c_base: torch.Tensor,
        region: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._uses_measurement_regions():
            base = self._measurement_reference(x0)
        else:
            base = c_base.reshape(-1).to(dtype=torch.float32)
        if region is None:
            return base
        return base[region.reshape(-1).long()]

    def _region_reference_norm(
        self,
        x0: torch.Tensor,
        c_base: torch.Tensor,
        region: torch.Tensor,
    ) -> float:
        base = self._region_base_vector(x0=x0, c_base=c_base, region=region)
        return max(float(torch.linalg.norm(base).item()), 1e-6)

    def _region_adjacency(self, region: torch.Tensor) -> torch.Tensor:
        region = region.reshape(-1).long()
        if self._uses_measurement_regions():
            return self.topology.measurement_adjacency[region][:, region].to(dtype=torch.float32)
        return self.topology.adjacency[region][:, region].to(dtype=torch.float32)

    def _region_support_score(self, region: torch.Tensor) -> torch.Tensor:
        region = region.reshape(-1).long()
        if self._uses_measurement_regions():
            eff = self.topology.measurement_effectiveness.to(dtype=torch.float32)
            topo = self.topology.measurement_topology_score.to(dtype=torch.float32)
            eff = eff / eff.max().clamp(min=1e-8)
            topo = topo / topo.max().clamp(min=1e-8)
            return (0.60 * eff + 0.40 * topo)[region].reshape(1, -1)
        return self.topology.state_flow_support[region].reshape(1, -1).to(dtype=torch.float32)

    def _project_region(
        self,
        region: torch.Tensor,
        delta_region: torch.Tensor,
        use_search_shaping: bool = False,
    ) -> torch.Tensor:
        if use_search_shaping:
            channel_weights = self.search_channel_weights
            shaping_strength = self.search_shaping_strength
        else:
            channel_weights = self.channel_weights
            shaping_strength = self.config.measurement_suppression_strength
        if self._uses_measurement_regions():
            return self.topology.measurement_region_projection(
                region,
                delta_region,
                channel_weights=channel_weights,
                shaping_strength=shaping_strength,
            )
        return self.topology.region_projection(
            region,
            delta_region,
            channel_weights=channel_weights,
            shaping_strength=shaping_strength,
        )

    def _enumerate_candidate_regions(
        self,
        x0: torch.Tensor,
        c_base: torch.Tensor,
        generator: torch.Generator | None = None,
        forced_seed_score: torch.Tensor | None = None,
        forced_state_allowed_mask: torch.Tensor | None = None,
        region_size_override: int | None = None,
        anchor_size_override: int | None = None,
        region_candidates_override: int | None = None,
        anchor_pool_size_override: int | None = None,
    ) -> list[torch.Tensor]:
        region_size = (
            int(self.config.region_size)
            if region_size_override is None
            else max(1, int(region_size_override))
        )
        anchor_size = (
            int(self.config.anchor_size)
            if anchor_size_override is None
            else max(1, int(anchor_size_override))
        )
        region_candidates = (
            int(self.config.region_candidates)
            if region_candidates_override is None
            else max(1, int(region_candidates_override))
        )
        anchor_pool_size = (
            int(self.config.anchor_pool_size)
            if anchor_pool_size_override is None
            else max(1, int(anchor_pool_size_override))
        )
        if self._uses_measurement_regions():
            gate_score = None
            gate_ratio = 1.0
            if bool(self.config.physical_measurement_gate):
                gate_score = self.topology.measurement_physical_gate(
                    x_ref=self._measurement_reference(x0),
                    response_weight=float(self.config.measurement_gate_response_weight),
                    effectiveness_weight=float(
                        self.config.measurement_gate_effectiveness_weight
                    ),
                )
                gate_ratio = float(self.config.measurement_gate_ratio)
            return self.topology.enumerate_measurement_candidate_regions(
                x_ref=self._measurement_reference(x0),
                region_size=region_size,
                anchor_size=anchor_size,
                num_candidates=region_candidates,
                anchor_pool_size=anchor_pool_size,
                diffusion_steps=self.config.proposal_diffusion_steps,
                diffusion_alpha=self.config.proposal_diffusion_alpha,
                gate_score=gate_score,
                gate_ratio=gate_ratio,
            )
        seed_score = (
            self._region_seed_score(x0=x0, c_base=c_base)
            if forced_seed_score is None
            else forced_seed_score.reshape(-1).to(dtype=torch.float32)
        )
        state_gate_score, state_allowed_mask = self._state_gate_mask(x0)
        if forced_state_allowed_mask is not None:
            forced_state_allowed_mask = (
                forced_state_allowed_mask.reshape(-1).to(dtype=torch.bool)
            )
            if state_allowed_mask is None:
                state_allowed_mask = forced_state_allowed_mask
            else:
                state_allowed_mask = state_allowed_mask & forced_state_allowed_mask
        if state_allowed_mask is None:
            return self.topology.enumerate_candidate_regions(
                c_base=c_base,
                region_size=region_size,
                anchor_size=anchor_size,
                num_candidates=region_candidates,
                anchor_pool_size=anchor_pool_size,
                probabilistic_prior=self.config.probabilistic_region_prior,
                prior_temperature=self.config.prior_temperature,
                prior_uniform_mixing=self.config.prior_uniform_mixing,
                generator=generator,
                proposal_mode=self.config.region_proposal_mode,
                diffusion_steps=self.config.proposal_diffusion_steps,
                diffusion_alpha=self.config.proposal_diffusion_alpha,
                flow_weight=self.config.proposal_flow_weight,
                corridor_weight=self.config.proposal_corridor_weight,
                seed_score=seed_score,
                allowed_mask=None,
            )

        total_candidates = max(1, int(region_candidates))
        escape_candidates = min(
            max(0, int(self.config.state_gate_escape_candidates)),
            max(0, total_candidates - 1),
        )
        gated_candidates = max(1, total_candidates - escape_candidates)

        merged_regions: list[torch.Tensor] = []
        seen: set[tuple[int, ...]] = set()

        def extend_unique(regions: list[torch.Tensor]) -> None:
            for region in regions:
                key = tuple(int(i) for i in region.reshape(-1).tolist())
                if key in seen:
                    continue
                seen.add(key)
                merged_regions.append(region)
                if len(merged_regions) >= total_candidates:
                    break

        extend_unique(
            self.topology.enumerate_candidate_regions(
                c_base=c_base,
                region_size=region_size,
                anchor_size=anchor_size,
                num_candidates=gated_candidates,
                anchor_pool_size=anchor_pool_size,
                probabilistic_prior=self.config.probabilistic_region_prior,
                prior_temperature=self.config.prior_temperature,
                prior_uniform_mixing=self.config.prior_uniform_mixing,
                generator=generator,
                proposal_mode=self.config.region_proposal_mode,
                diffusion_steps=self.config.proposal_diffusion_steps,
                diffusion_alpha=self.config.proposal_diffusion_alpha,
                flow_weight=self.config.proposal_flow_weight,
                corridor_weight=self.config.proposal_corridor_weight,
                seed_score=seed_score,
                allowed_mask=state_allowed_mask,
            )
        )

        remaining = total_candidates - len(merged_regions)
        excluded_mask = ~state_allowed_mask
        if remaining > 0 and escape_candidates > 0 and bool(excluded_mask.any().item()):
            escape_seed_score = seed_score
            if state_gate_score is not None:
                boundary_score = self.topology.adjacency[
                    state_allowed_mask.reshape(-1).to(dtype=torch.bool)
                ].sum(dim=0).to(dtype=torch.float32)
                boundary_score = self.topology._normalize_vector(boundary_score)
                residual_score = (seed_score.reshape(-1) - state_gate_score.reshape(-1)).clamp(
                    min=0.0
                )
                escape_seed_score = (
                    0.50 * seed_score.reshape(-1).to(dtype=torch.float32)
                    + 0.30 * boundary_score
                    + 0.20 * residual_score.to(dtype=torch.float32)
                )
            extend_unique(
                self.topology.enumerate_candidate_regions(
                    c_base=c_base,
                    region_size=region_size,
                    anchor_size=anchor_size,
                    num_candidates=min(remaining, escape_candidates),
                    anchor_pool_size=anchor_pool_size,
                    probabilistic_prior=self.config.probabilistic_region_prior,
                    prior_temperature=self.config.prior_temperature,
                    prior_uniform_mixing=self.config.prior_uniform_mixing,
                    generator=generator,
                    proposal_mode=self.config.region_proposal_mode,
                    diffusion_steps=self.config.proposal_diffusion_steps,
                    diffusion_alpha=self.config.proposal_diffusion_alpha,
                    flow_weight=self.config.proposal_flow_weight,
                    corridor_weight=self.config.proposal_corridor_weight,
                    seed_score=escape_seed_score,
                    allowed_mask=excluded_mask,
                )
            )

        remaining = total_candidates - len(merged_regions)
        if remaining > 0:
            extend_unique(
                self.topology.enumerate_candidate_regions(
                    c_base=c_base,
                    region_size=self.config.region_size,
                    anchor_size=self.config.anchor_size,
                    num_candidates=remaining,
                    anchor_pool_size=self.config.anchor_pool_size,
                    probabilistic_prior=self.config.probabilistic_region_prior,
                    prior_temperature=self.config.prior_temperature,
                    prior_uniform_mixing=self.config.prior_uniform_mixing,
                    generator=generator,
                    proposal_mode=self.config.region_proposal_mode,
                    diffusion_steps=self.config.proposal_diffusion_steps,
                    diffusion_alpha=self.config.proposal_diffusion_alpha,
                    flow_weight=self.config.proposal_flow_weight,
                    corridor_weight=self.config.proposal_corridor_weight,
                    seed_score=seed_score,
                    allowed_mask=None,
                )
            )

        return merged_regions[:total_candidates]

    def _build_region_from_priority(
        self,
        priority_score: torch.Tensor,
        x0: torch.Tensor,
        c_base: torch.Tensor,
        respect_state_gate: bool = True,
    ) -> torch.Tensor:
        if self._uses_measurement_regions():
            allowed_mask = None
            if bool(self.config.physical_measurement_gate):
                gate_score = self.topology.measurement_physical_gate(
                    x_ref=self._measurement_reference(x0),
                    response_weight=float(self.config.measurement_gate_response_weight),
                    effectiveness_weight=float(
                        self.config.measurement_gate_effectiveness_weight
                    ),
                )
                gate_ratio = min(max(float(self.config.measurement_gate_ratio), 0.0), 1.0)
                if gate_ratio < 1.0:
                    keep_dim = max(
                        int(self.config.region_size),
                        int(self.config.anchor_size),
                        int(round(gate_ratio * float(self.topology.n_measurements))),
                    )
                    keep_dim = max(1, min(keep_dim, int(self.topology.n_measurements)))
                    gate_idx = torch.topk(gate_score, k=keep_dim).indices
                    allowed_mask = torch.zeros(
                        int(self.topology.n_measurements), dtype=torch.bool
                    )
                    allowed_mask[gate_idx] = True
            return self.topology.build_measurement_region_from_priority(
                priority_score=priority_score,
                region_size=self.config.region_size,
                anchor_size=self.config.anchor_size,
                allowed_mask=allowed_mask,
            )
        state_allowed_mask = self._state_gate_mask(x0)[1] if respect_state_gate else None
        return self.topology.build_state_region_from_priority(
            priority_score=priority_score,
            region_size=self.config.region_size,
            anchor_size=self.config.anchor_size,
            allowed_mask=state_allowed_mask,
        )

    def _region_prior(
        self,
        region: torch.Tensor,
        x0: torch.Tensor,
        c_base: torch.Tensor,
    ) -> float:
        if self._uses_measurement_regions():
            base_prior = self.topology.measurement_region_budget_prior(
                region=region,
                x_ref=self._measurement_reference(x0),
            )
            if not bool(self.config.physical_measurement_gate):
                return base_prior
            gate_score = self.topology.measurement_physical_gate(
                x_ref=self._measurement_reference(x0),
                response_weight=float(self.config.measurement_gate_response_weight),
                effectiveness_weight=float(
                    self.config.measurement_gate_effectiveness_weight
                ),
            )
            gate_mean = float(
                gate_score[region.reshape(-1).long()].mean().item()
            )
            return float(max(0.0, min(1.0, 0.65 * base_prior + 0.35 * gate_mean)))
        base_prior = self.topology.region_budget_prior(region, c_base=c_base)
        sparse_mix = min(max(float(self.config.sparse_region_mix), 0.0), 1.0)
        sparse_beta = max(float(self.config.sparse_support_beta), 0.0)
        if sparse_mix > 0.0 and sparse_beta > 0.0:
            sparse_efficiency = self.topology.state_sparse_efficiency(beta=sparse_beta)
            sparse_eff_mean = float(sparse_efficiency[region.reshape(-1).long()].mean().item())
            sparse_penalty = max(float(self.config.sparse_region_penalty), 0.0)
            support_ratio = self.topology.state_region_measurement_support_ratio(region)
            sparse_prior = float(
                max(0.0, min(1.0, sparse_eff_mean - sparse_penalty * support_ratio))
            )
            base_prior = float(
                max(0.0, min(1.0, (1.0 - sparse_mix) * base_prior + sparse_mix * sparse_prior))
            )
        if not bool(self.config.multisource_physical_prior):
            return base_prior
        multisource_prior = float(
            self.topology.state_multisource_prior(c_base)[region.reshape(-1).long()].mean().item()
        )
        return float(max(0.0, min(1.0, 0.55 * base_prior + 0.45 * multisource_prior)))

    def _region_total_dim(self) -> int:
        return self.topology.n_measurements if self._uses_measurement_regions() else self.topology.n_states

    @staticmethod
    def _parse_positive_int_schedule(raw) -> list[int]:
        if raw is None:
            return []
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            values = []
            for token in text.replace(";", ",").split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    values.append(int(float(token)))
                except ValueError:
                    continue
            return values
        if isinstance(raw, (list, tuple)):
            values = []
            for item in raw:
                try:
                    values.append(int(item))
                except (TypeError, ValueError):
                    continue
            return values
        try:
            return [int(raw)]
        except (TypeError, ValueError):
            return []

    def _layered_active_schedule(self) -> list[int]:
        if self._uses_measurement_regions():
            raise ValueError("layered active-set search is only implemented for state regions.")
        total_dim = int(self.topology.n_states)
        parsed = self._parse_positive_int_schedule(self.config.active_set_sizes)
        normalized = sorted(
            {
                max(1, min(total_dim, int(value)))
                for value in parsed
                if int(value) > 0
            }
        )
        if not normalized:
            normalized = [total_dim]
        if normalized[-1] != total_dim:
            normalized.append(total_dim)
        return normalized

    def _layered_round_budget(self, stage_idx: int, stage_count: int) -> int:
        final_rounds = max(0, int(self.config.layered_final_rounds))
        stage_rounds = max(0, int(self.config.layered_stage_rounds))
        if stage_idx == max(0, stage_count - 1) and final_rounds > 0:
            return final_rounds
        if stage_rounds > 0:
            return stage_rounds
        return max(0, int(self.config.rounds))

    @staticmethod
    def _topk_allowed_mask(score: torch.Tensor, active_size: int) -> torch.Tensor:
        score = score.reshape(-1).to(dtype=torch.float32)
        active_size = max(1, min(int(active_size), int(score.numel())))
        keep_idx = torch.topk(score, k=active_size).indices
        allowed_mask = torch.zeros(int(score.numel()), dtype=torch.bool)
        allowed_mask[keep_idx] = True
        return allowed_mask

    def _region_seed_score(
        self,
        x0: torch.Tensor,
        c_base: torch.Tensor,
    ) -> torch.Tensor:
        if self._uses_measurement_regions():
            return self.topology.measurement_saliency(self._measurement_reference(x0))
        state_seed_mode = str(self.config.state_seed_mode).lower()
        if state_seed_mode == "target_radiated":
            legacy_seed = self.topology.state_target_radiated_prior(c_base)
        else:
            legacy_seed = self.topology._seed_score(c_base)
        sparse_weight = min(max(float(self.config.sparse_seed_weight), 0.0), 1.0)
        sparse_beta = max(float(self.config.sparse_support_beta), 0.0)
        if sparse_weight > 0.0 and sparse_beta > 0.0:
            contextual_seed = self.topology._diffuse_seed_score(
                seed_score=legacy_seed,
                diffusion_steps=max(1, int(self.config.proposal_diffusion_steps)),
                diffusion_alpha=float(self.config.proposal_diffusion_alpha),
            )
            sparse_seed = self._normalize_rows(
                (
                    contextual_seed
                    * self.topology.state_sparse_efficiency(beta=sparse_beta)
                ).reshape(1, -1)
            ).reshape(-1)
            legacy_seed = self._normalize_rows(
                ((1.0 - sparse_weight) * legacy_seed + sparse_weight * sparse_seed).reshape(1, -1)
            ).reshape(-1)
        if bool(self.config.multisource_physical_prior):
            multisource_seed = self.topology.state_multisource_prior(c_base)
            return self._normalize_rows(
                (0.50 * legacy_seed + 0.50 * multisource_seed).reshape(1, -1)
            ).reshape(-1)
        return legacy_seed

    def _measurement_gate_info(self, x0: torch.Tensor) -> dict:
        if not self._uses_measurement_regions() or not bool(self.config.physical_measurement_gate):
            return {
                "measurement_gate_enabled": False,
                "measurement_gate_keep_dim": int(self.topology.n_measurements),
                "measurement_gate_keep_ratio": 1.0,
            }
        gate_ratio = min(max(float(self.config.measurement_gate_ratio), 0.0), 1.0)
        if gate_ratio >= 1.0:
            keep_dim = int(self.topology.n_measurements)
        else:
            keep_dim = max(
                int(self.config.region_size),
                int(self.config.anchor_size),
                int(round(gate_ratio * float(self.topology.n_measurements))),
            )
            keep_dim = max(1, min(keep_dim, int(self.topology.n_measurements)))
        return {
            "measurement_gate_enabled": True,
            "measurement_gate_keep_dim": int(keep_dim),
            "measurement_gate_keep_ratio": float(
                keep_dim / max(1, int(self.topology.n_measurements))
            ),
        }

    def _state_gate_mask(self, x0: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._uses_measurement_regions() or not bool(self.config.measurement_conditioned_state_gate):
            return None, None
        gate_ratio = min(max(float(self.config.state_gate_ratio), 0.0), 1.0)
        gate_score, _ = self.topology.state_guidance_from_measurement(
            self._measurement_reference(x0)
        )
        if gate_ratio >= 1.0:
            return gate_score, None
        keep_dim = max(
            int(self.config.region_size),
            int(self.config.anchor_size),
            int(round(gate_ratio * float(self.topology.n_states))),
        )
        keep_dim = max(1, min(keep_dim, int(self.topology.n_states)))
        gate_idx = torch.topk(gate_score, k=keep_dim).indices
        allowed_mask = torch.zeros(int(self.topology.n_states), dtype=torch.bool)
        allowed_mask[gate_idx] = True
        return gate_score, allowed_mask

    def _state_gate_info(self, x0: torch.Tensor) -> dict:
        if self._uses_measurement_regions() or not bool(self.config.measurement_conditioned_state_gate):
            return {
                "state_gate_enabled": False,
                "state_gate_keep_dim": int(self.topology.n_states),
                "state_gate_keep_ratio": 1.0,
            }
        gate_ratio = min(max(float(self.config.state_gate_ratio), 0.0), 1.0)
        if gate_ratio >= 1.0:
            keep_dim = int(self.topology.n_states)
        else:
            keep_dim = max(
                int(self.config.region_size),
                int(self.config.anchor_size),
                int(round(gate_ratio * float(self.topology.n_states))),
            )
            keep_dim = max(1, min(keep_dim, int(self.topology.n_states)))
        return {
            "state_gate_enabled": True,
            "state_gate_keep_dim": int(keep_dim),
            "state_gate_keep_ratio": float(keep_dim / max(1, int(self.topology.n_states))),
        }

    def _score(self, result) -> torch.Tensor:
        if self.config.query_mode == "decision":
            return torch.zeros_like(result.fdia_margin, dtype=torch.float32)
        if self.config.query_mode == "prob_fdia":
            # Probability outputs are heavily saturated near 0/1.
            # Mapping them to log-odds preserves ranking while recovering
            # a margin-like geometry that is more useful for local search.
            prob = result.fdia_prob.to(dtype=torch.float32).clamp(min=1e-6, max=1.0 - 1e-6)
            return torch.log(prob) - torch.log1p(-prob)
        return result.fdia_margin

    def _selection_objective(
        self,
        score: torch.Tensor,
        x_ref: torch.Tensor,
        delta_z: torch.Tensor,
        c_base: torch.Tensor | None = None,
        delta_c: torch.Tensor | None = None,
        region: torch.Tensor | None = None,
    ) -> torch.Tensor:
        preserve_penalty = self._fdia_backbone_penalty(
            c_base=c_base,
            delta_c=delta_c,
            delta_z=delta_z,
            region=region,
        )
        if self.config.query_mode == "decision":
            ref_norm = torch.linalg.norm(x_ref, dim=1).clamp(min=1e-12)
            delta_norm = torch.linalg.norm(delta_z, dim=1)
            return delta_norm / ref_norm + preserve_penalty
        objective = score.to(dtype=torch.float32)
        penalty_weight = float(self.config.realism_penalty_weight)
        if penalty_weight > 0.0:
            ref_norm = torch.linalg.norm(x_ref, dim=1).clamp(min=1e-12)
            delta_norm = torch.linalg.norm(delta_z, dim=1)
            realism_ratio = delta_norm / ref_norm
            objective = objective + penalty_weight * realism_ratio
        return objective + preserve_penalty

    def _expand_delta_c_full(
        self,
        delta_c: torch.Tensor | None,
        delta_z: torch.Tensor,
        region: torch.Tensor | None,
    ) -> torch.Tensor:
        delta_z = delta_z.to(dtype=torch.float32)
        if delta_z.ndim == 1:
            delta_z = delta_z.unsqueeze(0)
        if delta_c is not None:
            delta_c = delta_c.to(device=delta_z.device, dtype=delta_z.dtype)
            if delta_c.ndim == 1:
                delta_c = delta_c.unsqueeze(0)
            if not self._uses_measurement_regions():
                if delta_c.shape[1] == self.topology.n_states:
                    return delta_c
                if region is None:
                    raise ValueError("State-region delta expansion requires region indices.")
                full = torch.zeros(
                    delta_z.shape[0],
                    self.topology.n_states,
                    device=delta_z.device,
                    dtype=delta_z.dtype,
                )
                full[:, region.reshape(-1).long()] = delta_c
                return full
        return self.topology.estimate_state_from_measurement(delta_z).reshape(
            delta_z.shape[0], -1
        ).to(device=delta_z.device, dtype=delta_z.dtype)

    def _fdia_backbone_penalty(
        self,
        c_base: torch.Tensor | None,
        delta_c: torch.Tensor | None,
        delta_z: torch.Tensor,
        region: torch.Tensor | None,
    ) -> torch.Tensor:
        delta_z = delta_z.to(dtype=torch.float32)
        if delta_z.ndim == 1:
            delta_z = delta_z.unsqueeze(0)
        total_weight = float(self.config.fdia_preserve_weight)
        if total_weight <= 0.0 or c_base is None:
            return torch.zeros(delta_z.shape[0], device=delta_z.device, dtype=delta_z.dtype)

        c_base = c_base.reshape(-1).to(device=delta_z.device, dtype=delta_z.dtype)
        support = c_base.abs() > 1e-10
        if not bool(support.any().item()):
            return torch.zeros(delta_z.shape[0], device=delta_z.device, dtype=delta_z.dtype)

        delta_c_full = self._expand_delta_c_full(delta_c=delta_c, delta_z=delta_z, region=region)
        c_total = c_base.unsqueeze(0) + delta_c_full

        support_base = c_base[support]
        support_total = c_total[:, support]
        support_norm_sq = torch.dot(support_base, support_base).clamp(min=1e-12)
        support_norm = torch.sqrt(support_norm_sq)
        state_projection = (support_total * support_base.unsqueeze(0)).sum(dim=1) / support_norm_sq

        h_mat = self.topology.H.to(device=delta_z.device, dtype=delta_z.dtype)
        a_base = (c_base.unsqueeze(0) @ h_mat.T).reshape(1, -1)
        a_total = a_base + delta_z
        a_base_norm_sq = torch.sum(a_base * a_base, dim=1).clamp(min=1e-12)
        measurement_projection = torch.sum(a_total * a_base, dim=1) / a_base_norm_sq

        if bool((~support).any().item()):
            offsupport_ratio = torch.linalg.norm(c_total[:, ~support], dim=1) / support_norm
        else:
            offsupport_ratio = torch.zeros(delta_z.shape[0], device=delta_z.device, dtype=delta_z.dtype)

        state_penalty = torch.relu(
            torch.full_like(state_projection, float(self.config.fdia_state_projection_min))
            - state_projection
        ).pow(2)
        measurement_penalty = torch.relu(
            torch.full_like(
                measurement_projection,
                float(self.config.fdia_measurement_projection_min),
            )
            - measurement_projection
        ).pow(2)
        offsupport_penalty = torch.relu(
            offsupport_ratio
            - torch.full_like(offsupport_ratio, float(self.config.fdia_offsupport_max))
        ).pow(2)

        combined = (
            float(self.config.fdia_state_penalty_weight) * state_penalty
            + float(self.config.fdia_measurement_penalty_weight) * measurement_penalty
            + float(self.config.fdia_offsupport_penalty_weight) * offsupport_penalty
        )
        return total_weight * combined

    def _fdia_backbone_quality(
        self,
        c_base: torch.Tensor | None,
        delta_c: torch.Tensor | None,
        delta_z: torch.Tensor,
        region: torch.Tensor | None,
    ) -> torch.Tensor:
        delta_z = delta_z.to(dtype=torch.float32)
        if delta_z.ndim == 1:
            delta_z = delta_z.unsqueeze(0)
        if c_base is None:
            return torch.ones(delta_z.shape[0], device=delta_z.device, dtype=delta_z.dtype)

        c_base = c_base.reshape(-1).to(device=delta_z.device, dtype=delta_z.dtype)
        support = c_base.abs() > 1e-10
        if not bool(support.any().item()):
            return torch.ones(delta_z.shape[0], device=delta_z.device, dtype=delta_z.dtype)

        delta_c_full = self._expand_delta_c_full(delta_c=delta_c, delta_z=delta_z, region=region)
        c_total = c_base.unsqueeze(0) + delta_c_full
        support_base = c_base[support]
        support_total = c_total[:, support]
        support_norm_sq = torch.dot(support_base, support_base).clamp(min=1e-12)
        support_norm = torch.sqrt(support_norm_sq)
        state_projection = (support_total * support_base.unsqueeze(0)).sum(dim=1) / support_norm_sq

        h_mat = self.topology.H.to(device=delta_z.device, dtype=delta_z.dtype)
        a_base = (c_base.unsqueeze(0) @ h_mat.T).reshape(1, -1)
        a_total = a_base + delta_z
        a_base_norm_sq = torch.sum(a_base * a_base, dim=1).clamp(min=1e-12)
        measurement_projection = torch.sum(a_total * a_base, dim=1) / a_base_norm_sq

        if bool((~support).any().item()):
            offsupport_ratio = torch.linalg.norm(c_total[:, ~support], dim=1) / support_norm
        else:
            offsupport_ratio = torch.zeros(delta_z.shape[0], device=delta_z.device, dtype=delta_z.dtype)

        state_floor = max(float(self.config.fdia_state_projection_min), 1e-6)
        measurement_floor = max(float(self.config.fdia_measurement_projection_min), 1e-6)
        offsupport_cap = max(float(self.config.fdia_offsupport_max), 1e-6)
        state_quality = (state_projection / state_floor).clamp(min=0.0, max=1.0)
        measurement_quality = (measurement_projection / measurement_floor).clamp(
            min=0.0,
            max=1.0,
        )
        offsupport_quality = (1.0 - offsupport_ratio / offsupport_cap).clamp(min=0.0, max=1.0)
        combined = (
            float(self.config.fdia_state_penalty_weight) * state_quality
            + float(self.config.fdia_measurement_penalty_weight) * measurement_quality
            + float(self.config.fdia_offsupport_penalty_weight) * offsupport_quality
        )
        total = (
            float(self.config.fdia_state_penalty_weight)
            + float(self.config.fdia_measurement_penalty_weight)
            + float(self.config.fdia_offsupport_penalty_weight)
        )
        if total <= 1e-8:
            return torch.ones(delta_z.shape[0], device=delta_z.device, dtype=delta_z.dtype)
        return (combined / total).clamp(min=0.0, max=1.0)

    def _success_priority_value(
        self,
        x_ref: torch.Tensor,
        delta_z: torch.Tensor,
        c_base: torch.Tensor | None = None,
        delta_c: torch.Tensor | None = None,
        region: torch.Tensor | None = None,
    ) -> torch.Tensor:
        delta_z = delta_z.to(dtype=torch.float32)
        if delta_z.ndim == 1:
            delta_z = delta_z.unsqueeze(0)
        ref_norm = torch.linalg.norm(x_ref, dim=1).clamp(min=1e-12)
        delta_norm = torch.linalg.norm(delta_z, dim=1)
        preserve_penalty = self._fdia_backbone_penalty(
            c_base=c_base,
            delta_c=delta_c,
            delta_z=delta_z,
            region=region,
        )
        return delta_norm / ref_norm + preserve_penalty

    def _uses_fixed_budget_objective(self) -> bool:
        return (
            str(getattr(self.config, "budget_objective_mode", "minimum_success")).lower()
            == "fixed_budget"
            and self.config.query_mode != "decision"
        )

    def _should_early_stop_on_success(self) -> bool:
        return bool(self.config.early_stop_on_success) and not self._uses_fixed_budget_objective()

    def _success_candidate_value(
        self,
        x_ref: torch.Tensor,
        objective: torch.Tensor,
        delta_z: torch.Tensor,
        c_base: torch.Tensor | None = None,
        delta_c: torch.Tensor | None = None,
        region: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._uses_fixed_budget_objective():
            value = objective.to(dtype=torch.float32)
            if value.ndim == 0:
                value = value.reshape(1)
            return value
        return self._success_priority_value(
            x_ref=x_ref,
            delta_z=delta_z,
            c_base=c_base,
            delta_c=delta_c,
            region=region,
        )

    def _consume_search_batch(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        region_state: _RegionSearchState,
        batch_c: torch.Tensor,
        batch_delta_z: torch.Tensor,
        batch_adv: torch.Tensor,
        batch_score: torch.Tensor,
        batch_objective: torch.Tensor,
        batch_preds: torch.Tensor,
    ) -> tuple[bool, bool]:
        improved = False
        success_mask = batch_preds != int(label)

        if bool(success_mask.any().item()):
            success_priority = self._success_candidate_value(
                x_ref=x0,
                objective=batch_objective,
                delta_z=batch_delta_z,
                c_base=c_base,
                delta_c=batch_c,
                region=region_state.region,
            )
            success_idx = int(
                torch.argmin(success_priority.masked_fill(~success_mask, float("inf"))).item()
            )
            success_candidate_priority = float(success_priority[success_idx].item())
            current_success_priority = float("inf")
            if region_state.success_delta_z is not None:
                current_success_priority = float(
                    self._success_candidate_value(
                        x_ref=x0,
                        objective=torch.tensor(
                            [float(region_state.success_objective)],
                            dtype=torch.float32,
                        ),
                        delta_z=region_state.success_delta_z,
                        c_base=c_base,
                        delta_c=region_state.success_delta_c,
                        region=region_state.region,
                    )[0].item()
                )

            if (
                region_state.success_adv is None
                or success_candidate_priority + 1e-8 < current_success_priority
            ):
                region_state.success_adv = batch_adv[success_idx : success_idx + 1].clone()
                region_state.success_delta_c = batch_c[success_idx : success_idx + 1].clone()
                region_state.success_delta_z = batch_delta_z[success_idx : success_idx + 1].clone()
                region_state.success_objective = float(batch_objective[success_idx].item())
                region_state.success_score = float(batch_score[success_idx].item())
                region_state.success_pred = int(batch_preds[success_idx].item())
                region_state.best_adv = batch_adv[success_idx : success_idx + 1].clone()
                region_state.best_delta_c = batch_c[success_idx : success_idx + 1].clone()
                region_state.best_delta_z = batch_delta_z[success_idx : success_idx + 1].clone()
                region_state.best_objective = float(batch_objective[success_idx].item())
                region_state.best_score = float(batch_score[success_idx].item())
                region_state.best_pred = int(batch_preds[success_idx].item())
                improved = True
            if self._should_early_stop_on_success():
                return True, improved

        best_idx = int(torch.argmin(batch_objective).item())
        if float(batch_objective[best_idx].item()) + 1e-8 < region_state.best_objective:
            region_state.best_objective = float(batch_objective[best_idx].item())
            region_state.best_score = float(batch_score[best_idx].item())
            region_state.best_pred = int(batch_preds[best_idx].item())
            region_state.best_delta_c = batch_c[best_idx : best_idx + 1].clone()
            region_state.best_delta_z = batch_delta_z[best_idx : best_idx + 1].clone()
            region_state.best_adv = batch_adv[best_idx : best_idx + 1].clone()
            improved = True

        return False, improved

    @staticmethod
    def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
        return x / torch.linalg.norm(x, dim=1, keepdim=True).clamp(min=1e-12)

    def _uses_state_budget_geometry(self) -> bool:
        return (
            not self._uses_measurement_regions()
            and bool(self.config.state_subspace_pgzoo)
            and float(self.config.measurement_delta_l2_ratio_cap) > 0.0
        )

    def _measurement_budget_abs(self, x_ref: torch.Tensor) -> float:
        cap_ratio = float(self.config.measurement_delta_l2_ratio_cap)
        if cap_ratio <= 0.0:
            return 0.0
        x_ref = x_ref.reshape(1, -1).to(dtype=torch.float32)
        return float(cap_ratio * torch.linalg.norm(x_ref, dim=1).item())

    def _normalize_state_budget_rows(
        self,
        region: torch.Tensor,
        delta_c: torch.Tensor,
        use_search_shaping: bool = False,
    ) -> torch.Tensor:
        delta_c = delta_c.to(dtype=torch.float32)
        if delta_c.ndim == 1:
            delta_c = delta_c.unsqueeze(0)
        if delta_c.shape[0] <= 0:
            return delta_c
        if not self._uses_state_budget_geometry():
            return self._normalize_rows(delta_c)
        delta_z = self._project_region(
            region=region,
            delta_region=delta_c,
            use_search_shaping=use_search_shaping,
        )
        budget_norm = torch.linalg.norm(delta_z, dim=1, keepdim=True)
        fallback_norm = torch.linalg.norm(delta_c, dim=1, keepdim=True)
        denom = torch.where(
            budget_norm > 1e-12,
            budget_norm,
            fallback_norm.clamp(min=1e-12),
        )
        return delta_c / denom.clamp(min=1e-12)

    def _project_state_budget_ball(
        self,
        x_ref: torch.Tensor,
        region: torch.Tensor,
        delta_c: torch.Tensor,
        use_search_shaping: bool = False,
        cap_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_c = delta_c.to(dtype=torch.float32)
        if delta_c.ndim == 1:
            delta_c = delta_c.unsqueeze(0)
        delta_z = self._project_region(
            region=region,
            delta_region=delta_c,
            use_search_shaping=use_search_shaping,
        )
        cap_abs = (
            float(cap_override)
            if cap_override is not None
            else self._measurement_budget_abs(x_ref)
        )
        if cap_abs <= 0.0:
            return delta_c, delta_z
        delta_norm = torch.linalg.norm(delta_z, dim=1, keepdim=True).clamp(min=1e-12)
        scale = torch.minimum(
            torch.ones_like(delta_norm),
            torch.full_like(delta_norm, float(cap_abs)) / delta_norm,
        )
        return delta_c * scale, delta_z * scale

    def _candidate_priority(self, candidate: dict) -> tuple[float, float, int]:
        if self._uses_fixed_budget_objective():
            return (
                float(candidate.get("selection_objective", candidate["initial_score"])),
                0.0 if candidate["initial_success"] else 1.0,
                int(candidate["rank"]),
            )
        return (
            0.0 if candidate["initial_success"] else 1.0,
            float(candidate.get("selection_objective", candidate["initial_score"])),
            int(candidate["rank"]),
        )

    @staticmethod
    def _budget_extra_priority(candidate: dict) -> tuple[float, float, int]:
        return (
            -(float(candidate.get("region_prior", 0.0))),
            float(candidate.get("selection_objective", candidate["initial_score"])),
            int(candidate["rank"]),
        )

    def _state_priority(self, state: _RegionSearchState) -> tuple[float, float, float, int]:
        if state.success_adv is not None and state.success_delta_z is not None:
            if self._uses_fixed_budget_objective():
                success_objective = (
                    float(state.success_objective)
                    if state.success_objective is not None
                    else float(state.best_objective)
                )
                return (
                    0.0,
                    success_objective,
                    -float(state.allocation_priority),
                    int(state.rank),
                )
            success_norm = float(torch.linalg.norm(state.success_delta_z).item())
            if (
                float(self.config.fdia_preserve_weight) > 0.0
                and state.success_objective is not None
            ):
                return (
                    0.0,
                    float(state.success_objective),
                    -float(state.allocation_priority),
                    int(state.rank),
                )
            return (0.0, success_norm, -float(state.allocation_priority), int(state.rank))
        if self.config.query_mode == "decision":
            if bool(self.config.physics_query_allocation):
                return (
                    1.0,
                    -float(state.allocation_priority),
                    -float(state.region_prior),
                    int(state.rank),
                )
            frontier_norm = float(torch.linalg.norm(state.best_delta_z).item())
            return (
                1.0,
                -float(state.region_prior),
                -float(state.allocation_priority),
                int(state.rank),
            )
        if bool(self.config.physics_query_allocation):
            return (
                1.0,
                -float(state.allocation_priority),
                float(state.best_score),
                int(state.rank),
            )
        return (
            1.0,
            float(state.best_score),
            -float(state.allocation_priority),
            int(state.rank),
        )

    @staticmethod
    def _decision_probe_scales(probe_scale: float, region_radius: float) -> list[float]:
        scales = [
            max(float(probe_scale), 1e-6),
            max(float(probe_scale), 0.60 * float(region_radius)),
            max(float(probe_scale), float(region_radius)),
        ]
        ordered: list[float] = []
        for scale in sorted(scales):
            if ordered and abs(scale - ordered[-1]) <= 1e-8:
                continue
            ordered.append(float(scale))
        return ordered

    def _decision_binary_refine(
        self,
        x0: torch.Tensor,
        label: int,
        region: torch.Tensor,
        fail_delta_c: torch.Tensor,
        success_delta_c: torch.Tensor,
        success_delta_z: torch.Tensor,
        success_adv: torch.Tensor,
        success_pred: int,
    ) -> dict:
        low = fail_delta_c.clone()
        high = success_delta_c.clone()
        best = {
            "delta_c": success_delta_c.clone(),
            "delta_z": success_delta_z.clone(),
            "adv": success_adv.clone(),
            "score": 0.0,
            "objective": float(self._selection_objective(
                torch.zeros(1, dtype=torch.float32),
                x0,
                success_delta_z,
            )[0].item()),
            "pred": int(success_pred),
            "success": True,
            "queries": 0,
        }

        refine_steps = max(4, min(8, int(self.config.patience) + 2))
        for _ in range(refine_steps):
            mid_delta_c = 0.5 * (low + high)
            mid_delta_z = self._project_region(
                region=region,
                delta_region=mid_delta_c,
                use_search_shaping=True,
            )
            mid_delta_c, mid_delta_z = self._apply_measurement_l2_cap(
                x_ref=x0,
                delta_c=mid_delta_c,
                delta_z=mid_delta_z,
            )
            mid_adv = x0 + mid_delta_z
            mid_query = self.oracle.query(mid_adv)
            best["queries"] += 1
            mid_pred = int(mid_query.pred[0].item())
            if mid_pred != int(label):
                high = mid_delta_c.clone()
                best.update(
                    {
                        "delta_c": mid_delta_c.clone(),
                        "delta_z": mid_delta_z.clone(),
                        "adv": mid_adv.clone(),
                        "objective": float(
                            self._selection_objective(
                                torch.zeros(1, dtype=torch.float32),
                                x0,
                                mid_delta_z,
                            )[0].item()
                        ),
                        "pred": mid_pred,
                    }
                )
            else:
                low = mid_delta_c.clone()
        return best

    def _probe_region_decision(
        self,
        x0: torch.Tensor,
        label: int,
        region: torch.Tensor,
        probe_scale: float,
        region_radius: float,
        probe_dirs: torch.Tensor,
    ) -> dict:
        signed_dirs = torch.cat([probe_dirs, -probe_dirs], dim=0)
        last_failure = torch.zeros_like(signed_dirs)
        best_failure = {
            "delta_c": torch.zeros(1, int(region.numel()), dtype=torch.float32),
            "delta_z": torch.zeros_like(x0),
            "adv": x0.clone(),
            "score": 0.0,
            "objective": 0.0,
            "pred": int(label),
            "success": False,
            "queries": 0,
        }
        best_failure_norm = 0.0

        for scale in self._decision_probe_scales(
            probe_scale=probe_scale,
            region_radius=region_radius,
        ):
            probe_delta_c = scale * signed_dirs
            probe_delta_z = self._project_region(
                region=region,
                delta_region=probe_delta_c,
                use_search_shaping=False,
            )
            probe_delta_c, probe_delta_z = self._apply_measurement_l2_cap(
                x_ref=x0,
                delta_c=probe_delta_c,
                delta_z=probe_delta_z,
            )
            probe_adv = x0 + probe_delta_z
            probe_query = self.oracle.query(probe_adv)
            probe_pred = probe_query.pred
            probe_norm = torch.linalg.norm(probe_delta_z, dim=1)
            success_mask = probe_pred != int(label)
            best_failure["queries"] += int(probe_adv.shape[0])

            if bool(success_mask.any().item()):
                success_idx = int(
                    torch.argmin(probe_norm.masked_fill(~success_mask, float("inf"))).item()
                )
                refined = self._decision_binary_refine(
                    x0=x0,
                    label=label,
                    region=region,
                    fail_delta_c=last_failure[success_idx : success_idx + 1].clone(),
                    success_delta_c=probe_delta_c[success_idx : success_idx + 1].clone(),
                    success_delta_z=probe_delta_z[success_idx : success_idx + 1].clone(),
                    success_adv=probe_adv[success_idx : success_idx + 1].clone(),
                    success_pred=int(probe_pred[success_idx].item()),
                )
                refined["queries"] += int(best_failure["queries"])
                return refined

            failure_idx = int(torch.argmax(probe_norm).item())
            failure_norm = float(probe_norm[failure_idx].item())
            if failure_norm >= best_failure_norm:
                best_failure_norm = failure_norm
                best_failure.update(
                    {
                        "delta_c": probe_delta_c[failure_idx : failure_idx + 1].clone(),
                        "delta_z": probe_delta_z[failure_idx : failure_idx + 1].clone(),
                        "adv": probe_adv[failure_idx : failure_idx + 1].clone(),
                        "objective": float(
                            self._selection_objective(
                                torch.zeros(1, dtype=torch.float32),
                                x0,
                                probe_delta_z[failure_idx : failure_idx + 1],
                            )[0].item()
                        ),
                        "pred": int(probe_pred[failure_idx].item()),
                    }
                )
            last_failure = probe_delta_c.clone()

        return best_failure

    def _build_probe_directions(
        self,
        x0: torch.Tensor,
        c_base: torch.Tensor,
        region: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        region_dim = int(region.numel())
        base = self._region_base_vector(x0=x0.reshape(-1), c_base=c_base, region=region).reshape(1, -1)
        directions = []
        physical_templates = []
        if self._uses_measurement_regions():
            measurement_ref = self._measurement_reference(x0)
            physical_templates = self.topology.measurement_region_direction_templates(
                region=region,
                x_ref=measurement_ref,
            )
            physical_templates.extend(
                self.topology.measurement_region_basis_vectors(
                    region=region,
                    x_ref=measurement_ref,
                    max_vectors=max(1, int(self.config.probe_directions)),
                )
            )
        for cand in physical_templates:
            if len(directions) >= max(1, self.config.probe_directions):
                break
            directions.append(cand.to(dtype=torch.float32))
        if float(torch.linalg.norm(base).item()) > 0:
            directions.append(base)
            directions.append(base.sign())
        while len(directions) < max(1, self.config.probe_directions):
            noise = torch.randn(1, region_dim, dtype=torch.float32, generator=generator)
            if physical_templates:
                noise = noise + 0.35 * physical_templates[0].to(dtype=torch.float32)
            elif float(torch.linalg.norm(base).item()) > 0:
                noise = noise + 0.35 * base
            directions.append(noise)
        dirs = torch.cat(directions[: max(1, self.config.probe_directions)], dim=0)
        return self._normalize_state_budget_rows(
            region=region,
            delta_c=dirs,
            use_search_shaping=False,
        )

    def _measurement_basis_directions(
        self,
        region_state: _RegionSearchState,
        x_ref: torch.Tensor,
        active_population: int,
    ) -> torch.Tensor:
        basis_vectors = self.topology.measurement_region_basis_vectors(
            region=region_state.region,
            x_ref=self._measurement_reference(x_ref),
            max_vectors=max(1, int(active_population)),
        )
        candidate_dirs = []
        best = region_state.best_delta_c.reshape(1, -1).to(dtype=torch.float32)
        if float(torch.linalg.norm(best).item()) > 1e-12:
            candidate_dirs.append(best)
            candidate_dirs.append(best.sign())
        candidate_dirs.extend(basis_vectors)
        if not candidate_dirs:
            candidate_dirs.append(torch.ones(1, int(region_state.region.numel()), dtype=torch.float32))
        dirs = torch.cat(candidate_dirs[: max(1, int(active_population))], dim=0)
        return self._normalize_rows(dirs)

    def _measurement_basis_matrix(
        self,
        region_state: _RegionSearchState,
        x_ref: torch.Tensor,
        max_vectors: int,
    ) -> torch.Tensor:
        basis_vectors = self.topology.measurement_region_basis_vectors(
            region=region_state.region,
            x_ref=self._measurement_reference(x_ref),
            max_vectors=max(1, int(max_vectors)),
        )
        candidate_dirs = []
        best = region_state.best_delta_c.reshape(1, -1).to(dtype=torch.float32)
        if float(torch.linalg.norm(best).item()) > 1e-12:
            candidate_dirs.append(best)
        candidate_dirs.extend(basis_vectors)
        if not candidate_dirs:
            candidate_dirs.append(torch.ones(1, int(region_state.region.numel()), dtype=torch.float32))
        basis = torch.cat(candidate_dirs, dim=0)
        basis = self._normalize_rows(basis)
        unique_rows = []
        for row in basis:
            row = row.reshape(1, -1)
            keep = True
            for exist in unique_rows:
                cosine = float(torch.abs((row * exist).sum()).item())
                if cosine >= 0.985:
                    keep = False
                    break
            if keep:
                unique_rows.append(row)
            if len(unique_rows) >= max(1, int(max_vectors)):
                break
        return torch.cat(unique_rows, dim=0)

    def _state_basis_max_vectors(self, active_population: int | None = None) -> int:
        max_vectors = max(2, int(self.config.state_basis_dim))
        if active_population is not None:
            max_vectors = min(max_vectors, max(1, int(active_population)))
        return max(1, max_vectors)

    def _state_basis_matrix(
        self,
        region_state: _RegionSearchState,
        c_base: torch.Tensor,
        max_vectors: int,
    ) -> torch.Tensor:
        basis_vectors = self.topology.state_region_basis_vectors(
            region=region_state.region,
            c_base=c_base.reshape(-1),
            max_vectors=max(1, int(max_vectors)),
        )
        template_vectors = self.topology.state_region_direction_templates(
            region=region_state.region,
            c_base=c_base.reshape(-1),
        )
        candidate_dirs = []
        best = region_state.best_delta_c.reshape(1, -1).to(dtype=torch.float32)
        if float(torch.linalg.norm(best).item()) > 1e-12:
            candidate_dirs.append(best)
            candidate_dirs.append(best.sign())
        candidate_dirs.extend(template_vectors)
        candidate_dirs.extend(basis_vectors)
        if not candidate_dirs:
            candidate_dirs.append(
                torch.ones(1, int(region_state.region.numel()), dtype=torch.float32)
            )
        basis = torch.cat(candidate_dirs, dim=0)
        basis = self._normalize_state_budget_rows(
            region=region_state.region,
            delta_c=basis,
            use_search_shaping=True,
        )
        unique_rows = []
        for row in basis:
            row = row.reshape(1, -1)
            keep = True
            for exist in unique_rows:
                cosine = float(torch.abs((row * exist).sum()).item())
                if cosine >= 0.985:
                    keep = False
                    break
            if keep:
                unique_rows.append(row)
            if len(unique_rows) >= max(1, int(max_vectors)):
                break
        return torch.cat(unique_rows, dim=0)

    def _state_basis_directions(
        self,
        region_state: _RegionSearchState,
        c_base: torch.Tensor,
        active_population: int,
    ) -> torch.Tensor:
        return self._state_basis_matrix(
            region_state=region_state,
            c_base=c_base,
            max_vectors=self._state_basis_max_vectors(active_population),
        )

    @staticmethod
    def _normalize_vector(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1).to(dtype=torch.float32)
        total = float(x.sum().item())
        if total <= 1e-12:
            return torch.full_like(x, 1.0 / max(1, int(x.numel())))
        return x / total

    @staticmethod
    def _orthonormalize_rows(x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[0] <= 0 or x.shape[1] <= 0:
            return x
        q, _ = torch.linalg.qr(x.T, mode="reduced")
        q = q.T
        keep = torch.linalg.norm(q, dim=1) > 1e-12
        q = q[keep]
        if q.shape[0] <= 0:
            return TopologyLatentQueryAttack._normalize_rows(x[:1])
        return TopologyLatentQueryAttack._normalize_rows(q)

    @staticmethod
    def _lowdim_structured_templates(region_dim: int) -> list[torch.Tensor]:
        region_dim = int(region_dim)
        if region_dim <= 0 or region_dim > 4:
            return []
        templates: list[torch.Tensor] = []
        eye = torch.eye(region_dim, dtype=torch.float32)
        for idx in range(region_dim):
            templates.append(eye[idx : idx + 1].clone())
        if region_dim >= 2:
            ones = torch.ones(1, region_dim, dtype=torch.float32)
            templates.append(ones.clone())
            for flip_idx in range(1, region_dim):
                signed = ones.clone()
                signed[0, flip_idx] = -1.0
                templates.append(signed)
        return templates

    @staticmethod
    def _project_to_row_space(x: torch.Tensor, row_basis: torch.Tensor) -> torch.Tensor:
        if row_basis.numel() <= 0:
            return x
        coeff = x @ row_basis.T
        return coeff @ row_basis

    @staticmethod
    def _pgzoo_structural_mix(region_dim: int) -> float:
        return float(min(max((float(region_dim) - 6.0) / 10.0, 0.0), 1.0))

    def _state_region_coordinate_prior(
        self,
        c_base: torch.Tensor,
        region_state: _RegionSearchState,
    ) -> torch.Tensor:
        region = region_state.region.reshape(-1).long()
        topo_prior = self._normalize_vector(
            self.topology.state_physical_score()[region].to(dtype=torch.float32).clamp(min=1e-8)
        )
        base_prior = self._normalize_vector(
            self.topology.state_physical_saliency(c_base.reshape(-1))[region]
            .to(dtype=torch.float32)
            .clamp(min=1e-8)
        )
        best_prior = self._normalize_vector(region_state.best_delta_c.reshape(-1).abs() + 1e-8)
        query_prior = None
        if region_state.query_coordinate_prior is not None:
            query_prior = self._normalize_vector(
                region_state.query_coordinate_prior.reshape(-1).abs().to(dtype=torch.float32)
                + 1e-8
            )

        query_weight = max(0.0, float(self.config.pgzoo_query_prior_weight))
        topo_weight = max(0.0, float(self.config.pgzoo_prior_topology_weight))
        base_weight = max(0.0, float(self.config.pgzoo_prior_base_weight))
        best_weight = max(0.0, float(self.config.pgzoo_prior_best_weight))
        total_weight = topo_weight + base_weight + best_weight + query_weight
        if total_weight <= 1e-12:
            return torch.full(
                (int(region.numel()),),
                1.0 / max(1, int(region.numel())),
                dtype=torch.float32,
            )
        prior = topo_weight * topo_prior + base_weight * base_prior + best_weight * best_prior
        if query_prior is not None:
            prior = prior + query_weight * query_prior
        prior = prior / total_weight
        return self._normalize_vector(prior.clamp(min=1e-8))

    def _state_region_gmrf_covariance(
        self,
        c_base: torch.Tensor,
        region_state: _RegionSearchState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        region_dim = int(region_state.region.numel())
        eye = torch.eye(region_dim, dtype=torch.float32)
        if region_dim <= 0:
            return eye[:0, :0], eye.new_zeros((0,))
        prior = self._state_region_coordinate_prior(
            c_base=c_base,
            region_state=region_state,
        )
        local_adj = self._region_adjacency(region_state.region)
        local_adj = 0.5 * (local_adj + local_adj.T)
        local_adj = local_adj.to(dtype=torch.float32)
        local_adj = local_adj.clone()
        local_adj.fill_diagonal_(0.0)
        local_deg = local_adj.sum(dim=1)
        local_laplacian = torch.diag(local_deg) - local_adj
        gamma = max(float(self.config.pgzoo_covariance_gamma), 0.0)
        ridge = max(float(self.config.pgzoo_covariance_ridge), 1e-6)
        diffusion = torch.linalg.pinv(eye + gamma * local_laplacian + ridge * eye)
        diffusion = 0.5 * (diffusion + diffusion.T)
        structured_cov = torch.diag(prior.sqrt()) @ diffusion @ torch.diag(prior.sqrt())
        structured_cov = 0.5 * (structured_cov + structured_cov.T)
        diagonal_cov = torch.diag(prior.clamp(min=1e-8))
        mix = self._pgzoo_structural_mix(region_dim)
        cov = mix * structured_cov + (1.0 - mix) * diagonal_cov
        cov = 0.5 * (cov + cov.T) + ridge * eye
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.clamp(min=1e-6)
        cov = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        cov = 0.5 * (cov + cov.T)
        return cov, prior

    def _state_subspace_covariance_factor(
        self,
        region_covariance: torch.Tensor,
        row_basis: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        sub_dim = int(row_basis.shape[0])
        eye = torch.eye(sub_dim, dtype=torch.float32)
        if sub_dim <= 0:
            return eye[:0, :0], 1.0
        ridge = max(float(self.config.pgzoo_covariance_ridge), 1e-6)
        sub_cov = row_basis @ region_covariance @ row_basis.T
        sub_cov = 0.5 * (sub_cov + sub_cov.T) + ridge * eye
        eigvals, eigvecs = torch.linalg.eigh(sub_cov)
        eigvals = eigvals.clamp(min=1e-6)
        sub_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals))
        expected_norm_sq = float(eigvals.sum().item())
        sampling_scale = 1.0 / max(expected_norm_sq ** 0.5, 1e-6)
        return sub_sqrt, sampling_scale

    def _state_region_preconditioner(
        self,
        region_state: _RegionSearchState,
    ) -> torch.Tensor:
        region = region_state.region.reshape(-1).long()
        region_dim = int(region.numel())
        eye = torch.eye(region_dim, dtype=torch.float32)
        if region_dim <= 0:
            return eye[:0, :0]
        region_h = self.topology.H[:, region].to(dtype=torch.float32)
        rinv = self.topology.rinv.reshape(-1, 1).to(dtype=torch.float32)
        region_fisher = region_h.T @ (region_h * rinv)
        region_fisher = 0.5 * (region_fisher + region_fisher.T)
        fisher_scale = max(
            float(region_fisher.diag().mean().item()),
            1e-6,
        )
        ridge = max(float(self.config.pgzoo_preconditioner_ridge), 0.0) * fisher_scale
        preconditioner = torch.linalg.pinv(region_fisher + ridge * eye)
        preconditioner = 0.5 * (preconditioner + preconditioner.T)
        diag_mean = preconditioner.diag().mean().clamp(min=1e-6)
        preconditioner = preconditioner / diag_mean
        mix = self._pgzoo_structural_mix(region_dim)
        return mix * preconditioner + (1.0 - mix) * eye

    def _state_subspace_prior_weights(
        self,
        x0: torch.Tensor,
        c_base: torch.Tensor,
        region_state: _RegionSearchState,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        basis_abs = basis.abs().to(dtype=torch.float32)
        region_support = self._normalize_vector(
            self._region_support_score(region_state.region).reshape(-1).abs()
        )
        base_region = self._normalize_vector(
            self._region_base_vector(
                x0=x0.reshape(-1),
                c_base=c_base,
                region=region_state.region,
            ).abs()
        )
        best_region = self._normalize_vector(region_state.best_delta_c.reshape(-1).abs())

        topo_weight = max(0.0, float(self.config.pgzoo_prior_topology_weight))
        base_weight = max(0.0, float(self.config.pgzoo_prior_base_weight))
        best_weight = max(0.0, float(self.config.pgzoo_prior_best_weight))
        total_weight = topo_weight + base_weight + best_weight
        if total_weight <= 1e-12:
            return torch.full(
                (int(basis.shape[0]),),
                1.0 / max(1, int(basis.shape[0])),
                dtype=torch.float32,
            )

        prior_score = (
            topo_weight * (basis_abs @ region_support.reshape(-1, 1)).reshape(-1)
            + base_weight * (basis_abs @ base_region.reshape(-1, 1)).reshape(-1)
            + best_weight * (basis_abs @ best_region.reshape(-1, 1)).reshape(-1)
        ) / total_weight
        prior_score = prior_score.clamp(min=1e-6)
        return prior_score / prior_score.sum().clamp(min=1e-12)

    def _build_search_directions(
        self,
        c_base: torch.Tensor,
        region_state: _RegionSearchState,
        population: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        region = region_state.region
        region_dim = int(region.numel())
        active_population = max(
            1,
            int(self.config.population if population is None else population),
        )
        if not self.config.structured_search_directions:
            directions = torch.randn(
                active_population,
                region_dim,
                dtype=torch.float32,
                generator=generator,
            )
            return self._normalize_rows(directions)

        directions = []
        best = region_state.best_delta_c.reshape(1, -1).to(dtype=torch.float32)
        base = self._region_base_vector(
            x0=region_state.clean_adv.reshape(-1),
            c_base=c_base,
            region=region,
        ).reshape(1, -1)
        sub_adj = self._region_adjacency(region)
        flow_support = self._region_support_score(region)
        physical_templates = []
        if self._uses_measurement_regions():
            measurement_ref = self._measurement_reference(region_state.clean_adv)
            physical_templates = self.topology.measurement_region_direction_templates(
                region=region,
                x_ref=measurement_ref,
            )

        if self._uses_measurement_regions():
            candidate_dirs = [best]
            if float(torch.linalg.norm(best).item()) > 0:
                candidate_dirs.append(best.sign())
            candidate_dirs.extend(self._lowdim_structured_templates(region_dim))
            candidate_dirs.extend(physical_templates)
            if float(torch.linalg.norm(base).item()) > 0:
                candidate_dirs.append(base)
                candidate_dirs.append(base.sign())
            if physical_templates:
                primary = physical_templates[0].to(dtype=torch.float32)
                candidate_dirs.append((sub_adj @ primary.reshape(-1)).reshape(1, -1))
                candidate_dirs.append(flow_support * primary.sign())
                if float(torch.linalg.norm(best).item()) > 0:
                    candidate_dirs.append(0.60 * best + 0.40 * primary)
                    candidate_dirs.append((sub_adj @ best.reshape(-1)).reshape(1, -1))
                    candidate_dirs.append(flow_support * best.sign())
            elif float(torch.linalg.norm(base).item()) > 0:
                candidate_dirs.extend(
                    [
                        (sub_adj @ base.reshape(-1)).reshape(1, -1),
                        flow_support * base.sign(),
                        flow_support
                        * (
                            best.sign()
                            if float(torch.linalg.norm(best).item()) > 0
                            else base.sign()
                        ),
                    ]
                )
        else:
            candidate_dirs = [
                *self._lowdim_structured_templates(region_dim),
                best,
                best.sign(),
                base,
                base.sign(),
                (sub_adj @ base.reshape(-1)).reshape(1, -1),
                (sub_adj @ best.reshape(-1)).reshape(1, -1),
                flow_support * base.sign(),
                flow_support * (best.sign() if float(torch.linalg.norm(best).item()) > 0 else base.sign()),
            ]
        for cand in candidate_dirs:
            if float(torch.linalg.norm(cand).item()) <= 1e-12:
                continue
            directions.append(cand)
            if len(directions) >= min(
                active_population,
                max(1, int(self.config.structured_direction_count)),
            ):
                break

        while len(directions) < active_population:
            noise = torch.randn(1, region_dim, dtype=torch.float32, generator=generator)
            if physical_templates:
                noise = noise + 0.25 * physical_templates[0].to(dtype=torch.float32)
            elif float(torch.linalg.norm(base).item()) > 0:
                noise = noise + 0.25 * base
            directions.append(noise)

        dirs = torch.cat(directions[:active_population], dim=0)
        return self._normalize_rows(dirs)

    def _evaluate_probe_batch(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        region: torch.Tensor,
        probe_scale: float,
        probe_dirs: torch.Tensor,
        direction_start_idx: int,
        direction_end_idx: int,
    ) -> dict:
        active_dirs = probe_dirs[direction_start_idx:direction_end_idx]
        if active_dirs.numel() == 0:
            return {
                "delta_c": torch.zeros(1, int(region.numel()), dtype=torch.float32),
                "delta_z": torch.zeros_like(x0),
                "adv": x0.clone(),
                "score": float("inf"),
                "objective": float("inf"),
                "pred": int(label),
                "success": False,
                "queries": 0,
            }

        probe_delta_c = torch.cat([probe_scale * active_dirs, -probe_scale * active_dirs], dim=0)
        probe_delta_c = self._apply_state_backbone_lock(
            c_base=c_base,
            region=region,
            delta_c=probe_delta_c,
        )
        if self._uses_state_budget_geometry():
            probe_delta_c, probe_delta_z = self._project_state_budget_ball(
                x_ref=x0,
                region=region,
                delta_c=probe_delta_c,
                use_search_shaping=False,
            )
        else:
            probe_delta_z = self._project_region(
                region=region,
                delta_region=probe_delta_c,
                use_search_shaping=False,
            )
            probe_delta_c, probe_delta_z = self._apply_measurement_l2_cap(
                x_ref=x0,
                delta_c=probe_delta_c,
                delta_z=probe_delta_z,
            )
        probe_adv = x0 + probe_delta_z
        probe_query = self.oracle.query(probe_adv)

        probe_score = self._score(probe_query)
        probe_objective = self._selection_objective(
            probe_score,
            x0,
            probe_delta_z,
            c_base=c_base,
            delta_c=probe_delta_c,
            region=region,
        )
        probe_pred = probe_query.pred
        probe_norm = torch.linalg.norm(probe_delta_z, dim=1)
        success_mask = probe_pred != int(label)
        if success_mask.any():
            success_priority = self._success_priority_value(
                x_ref=x0,
                delta_z=probe_delta_z,
                c_base=c_base,
                delta_c=probe_delta_c,
                region=region,
            )
            best_idx = int(
                torch.argmin(success_priority.masked_fill(~success_mask, float("inf"))).item()
            )
        else:
            best_idx = int(torch.argmin(probe_objective).item())

        return {
            "delta_c": probe_delta_c[best_idx : best_idx + 1].clone(),
            "delta_z": probe_delta_z[best_idx : best_idx + 1].clone(),
            "adv": probe_adv[best_idx : best_idx + 1].clone(),
            "score": float(probe_score[best_idx].item()),
            "objective": float(probe_objective[best_idx].item()),
            "pred": int(probe_pred[best_idx].item()),
            "success": bool(success_mask[best_idx].item()),
            "queries": int(probe_adv.shape[0]),
        }

    def _apply_measurement_l2_cap(
        self,
        x_ref: torch.Tensor,
        delta_c: torch.Tensor,
        delta_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cap_ratio = float(self.config.measurement_delta_l2_ratio_cap)
        if cap_ratio <= 0.0:
            return delta_c, delta_z

        ref_norm = torch.linalg.norm(x_ref, dim=1, keepdim=True).clamp(min=1e-12)
        delta_norm = torch.linalg.norm(delta_z, dim=1, keepdim=True).clamp(min=1e-12)
        cap = cap_ratio * ref_norm
        scale = torch.minimum(torch.ones_like(delta_norm), cap / delta_norm)
        return delta_c * scale, delta_z * scale

    def _apply_state_backbone_lock(
        self,
        c_base: torch.Tensor,
        region: torch.Tensor,
        delta_c: torch.Tensor,
    ) -> torch.Tensor:
        lock_ratio = float(self.config.fdia_backbone_lock_ratio)
        if self._uses_measurement_regions() or lock_ratio <= 0.0:
            return delta_c

        delta_c = delta_c.to(dtype=torch.float32)
        if delta_c.ndim == 1:
            delta_c = delta_c.unsqueeze(0)
        region = region.reshape(-1).long()
        c_region = c_base.reshape(-1).to(dtype=torch.float32)[region]
        support = c_region.abs() > 1e-10
        if not bool(support.any().item()):
            return delta_c

        total = c_region.unsqueeze(0) + delta_c
        sign = torch.sign(c_region[support])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        aligned_total = total[:, support] * sign.unsqueeze(0)
        min_mag = float(min(max(lock_ratio, 0.0), 1.0)) * c_region[support].abs().unsqueeze(0)
        aligned_total = torch.maximum(aligned_total, min_mag)
        total[:, support] = aligned_total * sign.unsqueeze(0)
        return total - c_region.unsqueeze(0)

    def _summarize_probe(
        self,
        candidate_entries: list[dict],
        clean_score: float,
    ) -> dict:
        if not candidate_entries:
            return {
                "clean_score": float(clean_score),
                "probe_best_score": float(clean_score),
                "probe_second_score": float(clean_score),
                "probe_score_gap": 0.0,
                "probe_improvement": 0.0,
                "probe_best_rank": 0,
                "probe_second_rank": 0,
                "probe_best_prior": 0.0,
                "probe_second_prior": 0.0,
                "probe_success_count": 0,
                "detector_feedback_used": False,
                "detector_feedback_candidate_count": 0,
                "detector_feedback_total_reward": 0.0,
                "detector_feedback_best_reward": 0.0,
            }

        ordered = sorted(candidate_entries, key=self._candidate_priority)
        best_entry = ordered[0]
        second_entry = ordered[1] if len(ordered) > 1 else ordered[0]
        return {
            "clean_score": float(clean_score),
            "probe_best_score": float(best_entry["initial_score"]),
            "probe_second_score": float(second_entry["initial_score"]),
            "probe_score_gap": float(second_entry["initial_score"] - best_entry["initial_score"]),
            "probe_improvement": float(clean_score - float(best_entry["initial_score"])),
            "probe_best_rank": int(best_entry["rank"]),
            "probe_second_rank": int(second_entry["rank"]),
            "probe_best_prior": float(best_entry.get("region_prior", 0.0)),
            "probe_second_prior": float(second_entry.get("region_prior", 0.0)),
            "probe_success_count": int(
                sum(1 for entry in ordered if bool(entry["initial_success"]))
            ),
            "detector_feedback_used": False,
            "detector_feedback_candidate_count": 0,
            "detector_feedback_total_reward": 0.0,
            "detector_feedback_best_reward": 0.0,
        }

    @staticmethod
    def _empty_probe_summary(clean_score: float) -> dict:
        return {
            "clean_score": float(clean_score),
            "probe_best_score": float(clean_score),
            "probe_second_score": float(clean_score),
            "probe_score_gap": 0.0,
            "probe_improvement": 0.0,
            "probe_best_rank": 0,
            "probe_second_rank": 0,
            "probe_best_prior": 0.0,
            "probe_second_prior": 0.0,
            "probe_success_count": 0,
            "detector_feedback_used": False,
            "detector_feedback_candidate_count": 0,
            "detector_feedback_total_reward": 0.0,
            "detector_feedback_best_reward": 0.0,
        }

    @staticmethod
    def _build_candidate_entry(
        region: torch.Tensor,
        rank: int,
        x0: torch.Tensor,
        clean_score: float,
        clean_pred: int,
        initial_delta_c: torch.Tensor | None = None,
        initial_delta_z: torch.Tensor | None = None,
        initial_adv: torch.Tensor | None = None,
        initial_score: float | None = None,
        initial_pred: int | None = None,
        initial_success: bool = False,
        selection_objective: float | None = None,
        region_prior: float = 0.0,
        proposal_source: str = "prior",
        feedback_reward: float = 0.0,
    ) -> dict:
        if initial_delta_c is None:
            initial_delta_c = torch.zeros(1, int(region.numel()), dtype=torch.float32)
        if initial_delta_z is None:
            initial_delta_z = torch.zeros_like(x0)
        if initial_adv is None:
            initial_adv = x0.clone()
        if initial_score is None:
            initial_score = float(clean_score)
        if initial_pred is None:
            initial_pred = int(clean_pred)
        if selection_objective is None:
            selection_objective = float(initial_score)
        return {
            "rank": int(rank),
            "region": region,
            "initial_delta_c": initial_delta_c.clone(),
            "initial_delta_z": initial_delta_z.clone(),
            "initial_adv": initial_adv.clone(),
            "initial_score": float(initial_score),
            "selection_objective": float(selection_objective),
            "initial_pred": int(initial_pred),
            "initial_success": bool(initial_success),
            "region_prior": float(region_prior),
            "proposal_source": str(proposal_source),
            "feedback_reward": float(feedback_reward),
        }

    @staticmethod
    def _region_overlap_ratio(region_a: torch.Tensor, region_b: torch.Tensor) -> float:
        region_a_set = set(int(v) for v in region_a.reshape(-1).tolist())
        region_b_set = set(int(v) for v in region_b.reshape(-1).tolist())
        if not region_a_set or not region_b_set:
            return 0.0
        return float(
            len(region_a_set & region_b_set) / max(1, min(len(region_a_set), len(region_b_set)))
        )

    def _evaluate_staged_probe_entry(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        staged_entry: dict,
    ) -> tuple[dict, dict, int]:
        if self.config.query_mode == "decision":
            region_radius = max(
                float(self.config.radius_floor),
                float(self.config.radius_ratio)
                * self._region_reference_norm(
                    x0=x0,
                    c_base=c_base,
                    region=staged_entry["region"],
                ),
            )
            probe_eval = self._probe_region_decision(
                x0=x0,
                label=label,
                region=staged_entry["region"],
                probe_scale=float(staged_entry["probe_scale"]),
                region_radius=region_radius,
                probe_dirs=staged_entry["probe_dirs"],
            )
        else:
            probe_eval = self._evaluate_probe_batch(
                x0=x0,
                label=label,
                c_base=c_base,
                region=staged_entry["region"],
                probe_scale=float(staged_entry["probe_scale"]),
                probe_dirs=staged_entry["probe_dirs"],
                direction_start_idx=0,
                direction_end_idx=int(staged_entry["probe_dirs"].shape[0]),
            )
        candidate_row = {
            "rank": int(staged_entry["rank"]),
            "region": staged_entry["region"],
            "region_space": self._region_space(),
            "best_score": float(probe_eval["score"]),
            "selection_objective": float(probe_eval["objective"]),
            "best_success": bool(probe_eval["success"]),
            "probe_scale": float(staged_entry["probe_scale"]),
            "region_prior": float(staged_entry["region_prior"]),
            "proposal_source": str(staged_entry.get("proposal_source", "prior")),
            "feedback_reward": float(staged_entry.get("feedback_reward", 0.0)),
        }
        candidate_entry = {
            "rank": int(staged_entry["rank"]),
            "region": staged_entry["region"],
            "initial_delta_c": probe_eval["delta_c"].clone(),
            "initial_delta_z": probe_eval["delta_z"].clone(),
            "initial_adv": probe_eval["adv"].clone(),
            "initial_score": float(probe_eval["score"]),
            "selection_objective": float(probe_eval["objective"]),
            "initial_pred": int(probe_eval["pred"]),
            "initial_success": bool(probe_eval["success"]),
            "probe_scale": float(staged_entry["probe_scale"]),
            "region_prior": float(staged_entry["region_prior"]),
            "proposal_source": str(staged_entry.get("proposal_source", "prior")),
            "feedback_reward": float(staged_entry.get("feedback_reward", 0.0)),
        }
        return candidate_row, candidate_entry, int(probe_eval["queries"])

    def _candidate_physics_quality(
        self,
        candidate_entry: dict,
        c_base: torch.Tensor,
    ) -> float:
        backbone_quality = float(
            self._fdia_backbone_quality(
                c_base=c_base,
                delta_c=candidate_entry["initial_delta_c"],
                delta_z=candidate_entry["initial_delta_z"],
                region=candidate_entry["region"],
            )[0].item()
        )
        region_prior = float(candidate_entry.get("region_prior", 0.0))
        return float(max(0.0, min(1.0, 0.65 * backbone_quality + 0.35 * region_prior)))

    def _candidate_detector_signal(
        self,
        candidate_entry: dict,
        clean_score: float,
    ) -> float:
        clean_scale = max(abs(float(clean_score)), 1e-6)
        success_bonus = max(0.0, float(self.config.detector_feedback_success_bonus))
        if self.config.query_mode == "decision":
            return success_bonus if bool(candidate_entry["initial_success"]) else 0.0
        detector_gain = max(float(clean_score) - float(candidate_entry["initial_score"]), 0.0) / clean_scale
        if bool(candidate_entry["initial_success"]):
            detector_gain += success_bonus
        return float(max(0.0, detector_gain))

    def _candidate_feedback_reward(
        self,
        candidate_entry: dict,
        clean_score: float,
        c_base: torch.Tensor,
    ) -> float:
        detector_signal = self._candidate_detector_signal(
            candidate_entry=candidate_entry,
            clean_score=clean_score,
        )
        if detector_signal <= 0.0:
            return 0.0
        if not bool(self.config.feedback_physics_reward_shaping):
            return float(detector_signal)
        physics_quality = self._candidate_physics_quality(
            candidate_entry=candidate_entry,
            c_base=c_base,
        )
        weight = min(max(float(self.config.feedback_physics_weight), 0.0), 1.0)
        gate = (1.0 - weight) + weight * physics_quality
        return float(detector_signal * gate)

    def _candidate_allocation_priority(
        self,
        candidate_entry: dict,
        clean_score: float,
        c_base: torch.Tensor,
    ) -> float:
        detector_signal = self._candidate_detector_signal(
            candidate_entry=candidate_entry,
            clean_score=clean_score,
        )
        detector_signal = min(detector_signal, 1.5) / 1.5
        physics_quality = self._candidate_physics_quality(
            candidate_entry=candidate_entry,
            c_base=c_base,
        )
        weight = min(max(float(self.config.physics_query_priority_weight), 0.0), 1.0)
        return float(
            max(
                0.0,
                min(
                    1.0,
                    (1.0 - weight) * detector_signal + weight * physics_quality,
                ),
            )
        )

    def _build_detector_feedback_candidate(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        clean_score: float,
        candidate_entries: list[dict],
        generator: torch.Generator | None = None,
    ) -> dict | None:
        if not bool(self.config.detector_feedback_region) or not candidate_entries:
            return None

        total_dim = self._region_total_dim()
        evidence = torch.zeros(total_dim, dtype=torch.float32)
        best_reward = 0.0
        total_reward = 0.0
        min_gain = max(0.0, float(self.config.detector_feedback_min_gain))

        for entry in candidate_entries:
            reward = self._candidate_feedback_reward(
                candidate_entry=entry,
                clean_score=clean_score,
                c_base=c_base,
            )
            if reward <= min_gain:
                continue

            region = entry["region"].reshape(-1).long()
            local_pattern = entry["initial_delta_c"].reshape(-1).abs().to(dtype=torch.float32)
            if local_pattern.numel() != region.numel() or float(local_pattern.sum().item()) <= 1e-12:
                local_pattern = torch.ones(int(region.numel()), dtype=torch.float32)
            local_pattern = local_pattern / local_pattern.sum().clamp(min=1e-12)
            evidence[region] += reward * local_pattern
            best_reward = max(best_reward, float(reward))
            total_reward += float(reward)

        if float(evidence.max().item()) <= 0.0:
            return None

        evidence = evidence / evidence.max().clamp(min=1e-12)
        prior_seed = self._region_seed_score(x0=x0, c_base=c_base)
        mix = min(max(float(self.config.detector_feedback_prior_mix), 0.0), 1.0)
        combined = self._normalize_rows(
            ((1.0 - mix) * evidence.reshape(1, -1) + mix * prior_seed.reshape(1, -1))
        ).reshape(-1)

        candidate_scores = [
            ("detector_feedback", combined),
            ("detector_feedback_raw", evidence),
        ]
        synthetic_region = None
        synthetic_source = "detector_feedback"
        for source_name, priority_score in candidate_scores:
            region = self._build_region_from_priority(
                priority_score=priority_score,
                x0=x0,
                c_base=c_base,
                respect_state_gate=False,
            )
            if any(
                self._region_overlap_ratio(region, entry["region"]) >= 0.85
                for entry in candidate_entries
            ):
                continue
            synthetic_region = region
            synthetic_source = source_name
            break
        if synthetic_region is None:
            return None

        region_prior = self._region_prior(region=synthetic_region, x0=x0, c_base=c_base)
        region_radius = max(
            float(self.config.radius_floor),
            float(self.config.radius_ratio)
            * self._region_reference_norm(x0=x0, c_base=c_base, region=synthetic_region),
        )
        probe_scale = max(
            float(self.config.radius_floor) * 0.5,
            float(self.config.probe_scale_ratio) * region_radius,
        )
        probe_dirs = self._build_probe_directions(
            x0=x0.reshape(-1),
            c_base=c_base,
            region=synthetic_region,
            generator=generator,
        )
        staged_entry = {
            "rank": max(int(entry["rank"]) for entry in candidate_entries) + 1,
            "region": synthetic_region,
            "probe_scale": float(probe_scale),
            "region_prior": float(region_prior),
            "probe_dirs": probe_dirs,
            "proposal_source": str(synthetic_source),
            "feedback_reward": float(best_reward),
        }
        candidate_row, candidate_entry, queries = self._evaluate_staged_probe_entry(
            x0=x0,
            label=label,
            c_base=c_base,
            staged_entry=staged_entry,
        )
        return {
            "candidate_row": candidate_row,
            "candidate_entry": candidate_entry,
            "queries": int(queries),
            "total_reward": float(total_reward),
            "best_reward": float(best_reward),
        }

    def _probe_regions(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        clean_query,
        generator: torch.Generator | None = None,
        candidates_override: list[torch.Tensor] | None = None,
    ) -> dict:
        candidates = (
            list(candidates_override)
            if candidates_override is not None
            else self._enumerate_candidate_regions(
                x0=x0,
                c_base=c_base,
                generator=generator,
            )
        )
        if not candidates:
            fallback_dim = max(1, min(int(self.config.region_size), int(self.topology.n_states)))
            candidates = [torch.arange(fallback_dim, dtype=torch.long)]
        clean_score = float(self._score(clean_query)[0].item())
        clean_pred = int(clean_query.pred[0].item())
        best = {
            "region": candidates[0],
            "initial_delta_c": torch.zeros(1, int(candidates[0].numel()), dtype=torch.float32),
            "initial_delta_z": torch.zeros_like(x0),
            "initial_adv": x0.clone(),
            "initial_score": clean_score,
            "initial_pred": clean_pred,
            "initial_success": False,
            "selected_region_rank": 0,
            "proposal_queries": 0,
            "candidate_count": len(candidates),
            "detector_feedback_used": False,
            "detector_feedback_candidate_count": 0,
            "detector_feedback_total_reward": 0.0,
            "detector_feedback_best_reward": 0.0,
            "query_cap_reached": False,
        }

        candidate_rows = []
        candidate_entries = []
        staged_entries = []
        coarse_dirs = min(
            max(1, int(self.config.coarse_probe_directions)),
            max(1, int(self.config.probe_directions)),
        )
        for rank, region in enumerate(candidates):
            region_prior = self._region_prior(region=region, x0=x0, c_base=c_base)
            if self._uses_state_budget_geometry():
                region_radius = max(
                    float(self.config.radius_floor),
                    self._measurement_budget_abs(x0),
                )
            else:
                region_radius = max(
                    float(self.config.radius_floor),
                    float(self.config.radius_ratio)
                    * self._region_reference_norm(x0=x0, c_base=c_base, region=region),
                )
            probe_scale = max(
                float(self.config.radius_floor) * 0.5,
                float(self.config.probe_scale_ratio) * region_radius,
            )
            probe_dirs = self._build_probe_directions(
                x0=x0.reshape(-1),
                c_base=c_base, region=region, generator=generator
            )
            staged_entries.append(
                {
                    "rank": int(rank),
                    "region": region,
                    "probe_scale": float(probe_scale),
                    "region_prior": float(region_prior),
                    "probe_dirs": probe_dirs,
                    "probe_eval": None,
                }
            )

        if self.config.query_mode == "decision":
            configured_probe_topk = int(self.config.initial_probe_region_topk)
            if 0 < configured_probe_topk < len(staged_entries):
                staged_probe_order = sorted(
                    staged_entries,
                    key=lambda entry: (-float(entry["region_prior"]), int(entry["rank"])),
                )
                initial_rank_set = {
                    int(entry["rank"])
                    for entry in staged_probe_order[: max(1, configured_probe_topk)]
                }
            elif len(staged_entries) > 2:
                staged_probe_order = sorted(
                    staged_entries,
                    key=lambda entry: (-float(entry["region_prior"]), int(entry["rank"])),
                )
                initial_rank_set = {
                    int(entry["rank"])
                    for entry in staged_probe_order[: max(1, len(staged_entries) // 2)]
                }
            else:
                initial_rank_set = None

            def run_decision_probe(entry: dict) -> dict:
                return self._probe_region_decision(
                    x0=x0,
                    label=label,
                    region=entry["region"],
                    probe_scale=float(entry["probe_scale"]),
                    region_radius=max(
                        float(self.config.radius_floor),
                        float(self.config.radius_ratio)
                        * self._region_reference_norm(
                            x0=x0,
                            c_base=c_base,
                            region=entry["region"],
                        ),
                    ),
                    probe_dirs=entry["probe_dirs"],
                )

            for entry in staged_entries:
                if initial_rank_set is not None and int(entry["rank"]) not in initial_rank_set:
                    continue
                probe_eval = run_decision_probe(entry)
                best["proposal_queries"] += int(probe_eval["queries"])
                entry["probe_eval"] = probe_eval

            initial_success_found = any(
                entry["probe_eval"] is not None and bool(entry["probe_eval"]["success"])
                for entry in staged_entries
            )
            if not initial_success_found:
                for entry in staged_entries:
                    if entry["probe_eval"] is not None:
                        continue
                    probe_eval = run_decision_probe(entry)
                    best["proposal_queries"] += int(probe_eval["queries"])
                    entry["probe_eval"] = probe_eval

            for entry in staged_entries:
                probe_eval = entry["probe_eval"]
                if probe_eval is None:
                    continue
                candidate_rows.append(
                    {
                        "rank": int(entry["rank"]),
                        "region": entry["region"],
                        "region_space": self._region_space(),
                        "best_score": float(probe_eval["score"]),
                        "selection_objective": float(probe_eval["objective"]),
                        "best_success": bool(probe_eval["success"]),
                        "probe_scale": float(entry["probe_scale"]),
                        "region_prior": float(entry["region_prior"]),
                        "proposal_source": "prior",
                        "feedback_reward": 0.0,
                    }
                )
                candidate_entries.append(
                    {
                        "rank": int(entry["rank"]),
                        "region": entry["region"],
                        "initial_delta_c": probe_eval["delta_c"].clone(),
                        "initial_delta_z": probe_eval["delta_z"].clone(),
                        "initial_adv": probe_eval["adv"].clone(),
                        "initial_score": float(probe_eval["score"]),
                        "selection_objective": float(probe_eval["objective"]),
                        "initial_pred": int(probe_eval["pred"]),
                        "initial_success": bool(probe_eval["success"]),
                        "probe_scale": float(entry["probe_scale"]),
                        "region_prior": float(entry["region_prior"]),
                        "proposal_source": "prior",
                        "feedback_reward": 0.0,
                    }
                )

            feedback_candidate = self._build_detector_feedback_candidate(
                x0=x0,
                label=label,
                c_base=c_base,
                clean_score=clean_score,
                candidate_entries=candidate_entries,
                generator=generator,
            )
            if feedback_candidate is not None:
                candidate_rows.append(feedback_candidate["candidate_row"])
                candidate_entries.append(feedback_candidate["candidate_entry"])
                best["proposal_queries"] += int(feedback_candidate["queries"])
                best["detector_feedback_used"] = True
                best["detector_feedback_candidate_count"] = 1
                best["detector_feedback_total_reward"] = float(
                    feedback_candidate["total_reward"]
                )
                best["detector_feedback_best_reward"] = float(
                    feedback_candidate["best_reward"]
                )

            if candidate_entries:
                success_entries = [
                    entry for entry in candidate_entries if bool(entry["initial_success"])
                ]
                if success_entries:
                    best_entry = min(
                        success_entries,
                        key=lambda entry: (
                            float(torch.linalg.norm(entry["initial_delta_z"]).item()),
                            int(entry["rank"]),
                        ),
                    )
                else:
                    best_entry = max(
                        candidate_entries,
                        key=lambda entry: (
                            float(entry.get("region_prior", 0.0)),
                            float(torch.linalg.norm(entry["initial_delta_z"]).item()),
                            -int(entry["rank"]),
                        ),
                    )
                best.update(
                    {
                        "region": best_entry["region"],
                        "initial_delta_c": best_entry["initial_delta_c"].clone(),
                        "initial_delta_z": best_entry["initial_delta_z"].clone(),
                        "initial_adv": best_entry["initial_adv"].clone(),
                        "initial_score": float(best_entry["initial_score"]),
                        "selection_objective": float(best_entry["selection_objective"]),
                        "initial_pred": int(best_entry["initial_pred"]),
                        "initial_success": bool(best_entry["initial_success"]),
                        "selected_region_rank": int(best_entry["rank"]),
                    }
                )

            best["candidate_rows"] = candidate_rows
            best["candidate_entries"] = candidate_entries
            best["candidate_count"] = len(candidate_entries)
            best["clean_score"] = float(clean_score)
            return best

        def evaluate_entry(entry: dict) -> dict:
            if self.config.hierarchical_probe and int(entry["probe_dirs"].shape[0]) > coarse_dirs:
                return self._evaluate_probe_batch(
                    x0=x0,
                    label=label,
                    c_base=c_base,
                    region=entry["region"],
                    probe_scale=float(entry["probe_scale"]),
                    probe_dirs=entry["probe_dirs"],
                    direction_start_idx=0,
                    direction_end_idx=coarse_dirs,
                )
            return self._evaluate_probe_batch(
                x0=x0,
                label=label,
                c_base=c_base,
                region=entry["region"],
                probe_scale=float(entry["probe_scale"]),
                probe_dirs=entry["probe_dirs"],
                direction_start_idx=0,
                direction_end_idx=int(entry["probe_dirs"].shape[0]),
            )

        initial_rank_set = None
        configured_probe_topk = int(self.config.initial_probe_region_topk)
        if 0 < configured_probe_topk < len(staged_entries):
            ranked_by_prior = sorted(
                staged_entries,
                key=lambda entry: (-float(entry["region_prior"]), int(entry["rank"])),
            )
            initial_rank_set = {
                int(entry["rank"])
                for entry in ranked_by_prior[: max(1, configured_probe_topk)]
            }

        for entry in staged_entries:
            if initial_rank_set is not None and int(entry["rank"]) not in initial_rank_set:
                continue
            required_queries = 2 * int(
                min(
                    int(entry["probe_dirs"].shape[0]),
                    coarse_dirs if self.config.hierarchical_probe else int(entry["probe_dirs"].shape[0]),
                )
            )
            if not self._has_query_budget_for(required_queries):
                best["query_cap_reached"] = True
                break
            probe_eval = evaluate_entry(entry)
            best["proposal_queries"] += int(probe_eval["queries"])
            entry["probe_eval"] = probe_eval

        if initial_rank_set is not None and any(
            entry["probe_eval"] is None for entry in staged_entries
        ):
            initial_evaluated = [
                entry for entry in staged_entries if entry["probe_eval"] is not None
            ]
            initial_success_count = sum(
                1 for entry in initial_evaluated if bool(entry["probe_eval"]["success"])
            )
            if initial_evaluated:
                initial_best = min(
                    initial_evaluated,
                    key=lambda entry: self._candidate_priority(
                        {
                            "initial_success": bool(entry["probe_eval"]["success"]),
                            "selection_objective": float(entry["probe_eval"]["objective"]),
                            "initial_score": float(entry["probe_eval"]["score"]),
                            "rank": int(entry["rank"]),
                        }
                    ),
                )
                initial_improvement_ratio = (
                    clean_score - float(initial_best["probe_eval"]["score"])
                ) / max(abs(clean_score), 1e-6)
            else:
                initial_improvement_ratio = 0.0

            if (
                initial_success_count <= 0
                and initial_improvement_ratio
                < float(self.config.probe_expand_improvement_ratio)
            ):
                for entry in staged_entries:
                    if entry["probe_eval"] is not None:
                        continue
                    required_queries = 2 * int(
                        min(
                            int(entry["probe_dirs"].shape[0]),
                            coarse_dirs if self.config.hierarchical_probe else int(entry["probe_dirs"].shape[0]),
                        )
                    )
                    if not self._has_query_budget_for(required_queries):
                        best["query_cap_reached"] = True
                        break
                    probe_eval = evaluate_entry(entry)
                    best["proposal_queries"] += int(probe_eval["queries"])
                    entry["probe_eval"] = probe_eval

        staged_entries = [
            entry for entry in staged_entries if entry["probe_eval"] is not None
        ]

        if self.config.hierarchical_probe and any(
            int(entry["probe_dirs"].shape[0]) > coarse_dirs for entry in staged_entries
        ):
            shortlist = sorted(
                staged_entries,
                key=lambda entry: self._candidate_priority(
                    {
                        "initial_success": bool(entry["probe_eval"]["success"]),
                        "selection_objective": float(entry["probe_eval"]["objective"]),
                        "initial_score": float(entry["probe_eval"]["score"]),
                        "rank": int(entry["rank"]),
                    }
                ),
            )[: max(1, int(self.config.fine_probe_topk))]
            shortlist_ranks = {int(entry["rank"]) for entry in shortlist}
            for entry in staged_entries:
                final_eval = entry["probe_eval"]
                if int(entry["rank"]) in shortlist_ranks and int(entry["probe_dirs"].shape[0]) > coarse_dirs:
                    fine_queries = 2 * max(0, int(entry["probe_dirs"].shape[0]) - coarse_dirs)
                    if not self._has_query_budget_for(fine_queries):
                        best["query_cap_reached"] = True
                        continue
                    fine_eval = self._evaluate_probe_batch(
                        x0=x0,
                        label=label,
                        c_base=c_base,
                        region=entry["region"],
                        probe_scale=float(entry["probe_scale"]),
                        probe_dirs=entry["probe_dirs"],
                        direction_start_idx=coarse_dirs,
                        direction_end_idx=int(entry["probe_dirs"].shape[0]),
                    )
                    best["proposal_queries"] += int(fine_eval["queries"])
                    if self._candidate_priority(
                        {
                            "initial_success": bool(fine_eval["success"]),
                            "selection_objective": float(fine_eval["objective"]),
                            "initial_score": float(fine_eval["score"]),
                            "rank": int(entry["rank"]),
                        }
                    ) < self._candidate_priority(
                        {
                            "initial_success": bool(final_eval["success"]),
                            "selection_objective": float(final_eval["objective"]),
                            "initial_score": float(final_eval["score"]),
                            "rank": int(entry["rank"]),
                        }
                    ):
                        final_eval = fine_eval
                entry["probe_eval"] = final_eval

        for entry in staged_entries:
            probe_eval = entry["probe_eval"]
            candidate_rows.append(
                {
                    "rank": int(entry["rank"]),
                    "region": entry["region"],
                    "region_space": self._region_space(),
                    "best_score": float(probe_eval["score"]),
                    "selection_objective": float(probe_eval["objective"]),
                    "best_success": bool(probe_eval["success"]),
                    "probe_scale": float(entry["probe_scale"]),
                    "region_prior": float(entry["region_prior"]),
                    "proposal_source": "prior",
                    "feedback_reward": 0.0,
                }
            )
            candidate_entry = {
                "rank": int(entry["rank"]),
                "region": entry["region"],
                "initial_delta_c": probe_eval["delta_c"].clone(),
                "initial_delta_z": probe_eval["delta_z"].clone(),
                "initial_adv": probe_eval["adv"].clone(),
                "initial_score": float(probe_eval["score"]),
                "selection_objective": float(probe_eval["objective"]),
                "initial_pred": int(probe_eval["pred"]),
                "initial_success": bool(probe_eval["success"]),
                "probe_scale": float(entry["probe_scale"]),
                "region_prior": float(entry["region_prior"]),
                "proposal_source": "prior",
                "feedback_reward": 0.0,
            }
            candidate_entries.append(candidate_entry)

            should_replace = False
            if bool(candidate_entry["initial_success"]) and not best["initial_success"]:
                should_replace = True
            elif bool(candidate_entry["initial_success"]) == bool(best["initial_success"]) and float(
                candidate_entry["selection_objective"]
            ) < float(best.get("selection_objective", best["initial_score"])):
                should_replace = True

            if should_replace:
                best.update(
                    {
                        "region": candidate_entry["region"],
                        "initial_delta_c": candidate_entry["initial_delta_c"].clone(),
                        "initial_delta_z": candidate_entry["initial_delta_z"].clone(),
                        "initial_adv": candidate_entry["initial_adv"].clone(),
                        "initial_score": float(candidate_entry["initial_score"]),
                        "selection_objective": float(candidate_entry["selection_objective"]),
                        "initial_pred": int(candidate_entry["initial_pred"]),
                        "initial_success": bool(candidate_entry["initial_success"]),
                        "selected_region_rank": int(candidate_entry["rank"]),
                    }
                )

        feedback_candidate = self._build_detector_feedback_candidate(
            x0=x0,
            label=label,
            c_base=c_base,
            clean_score=clean_score,
            candidate_entries=candidate_entries,
            generator=generator,
        )
        if feedback_candidate is not None:
            candidate_rows.append(feedback_candidate["candidate_row"])
            candidate_entry = feedback_candidate["candidate_entry"]
            candidate_entries.append(candidate_entry)
            best["proposal_queries"] += int(feedback_candidate["queries"])
            best["detector_feedback_used"] = True
            best["detector_feedback_candidate_count"] = 1
            best["detector_feedback_total_reward"] = float(
                feedback_candidate["total_reward"]
            )
            best["detector_feedback_best_reward"] = float(
                feedback_candidate["best_reward"]
            )
            should_replace = False
            if bool(candidate_entry["initial_success"]) and not best["initial_success"]:
                should_replace = True
            elif bool(candidate_entry["initial_success"]) == bool(best["initial_success"]) and float(
                candidate_entry["selection_objective"]
            ) < float(best.get("selection_objective", best["initial_score"])):
                should_replace = True
            if should_replace:
                best.update(
                    {
                        "region": candidate_entry["region"],
                        "initial_delta_c": candidate_entry["initial_delta_c"].clone(),
                        "initial_delta_z": candidate_entry["initial_delta_z"].clone(),
                        "initial_adv": candidate_entry["initial_adv"].clone(),
                        "initial_score": float(candidate_entry["initial_score"]),
                        "selection_objective": float(candidate_entry["selection_objective"]),
                        "initial_pred": int(candidate_entry["initial_pred"]),
                        "initial_success": bool(candidate_entry["initial_success"]),
                        "selected_region_rank": int(candidate_entry["rank"]),
                    }
                )

        best["candidate_rows"] = candidate_rows
        best["candidate_entries"] = candidate_entries
        best["candidate_count"] = len(candidate_entries)
        best["clean_score"] = float(clean_score)
        return best

    def _build_region_state(
        self,
        candidate_entry: dict,
        x0: torch.Tensor,
        clean_pred: int,
        c_base: torch.Tensor,
        query_budget_population: int = 0,
        query_budget_rounds: int = 0,
        probe_improvement_ratio: float = 0.0,
        physics_quality: float = 0.0,
        allocation_priority: float = 0.0,
    ) -> _RegionSearchState:
        region = candidate_entry["region"]
        if self._uses_state_budget_geometry():
            radius = max(
                float(self.config.radius_floor),
                self._measurement_budget_abs(x0),
            )
        else:
            radius = max(
                float(self.config.radius_floor),
                float(self.config.radius_ratio)
                * self._region_reference_norm(x0=x0, c_base=c_base, region=region),
            )
        step = max(radius * float(self.config.step_ratio), self.config.radius_floor * 0.5)
        min_step = max(
            radius * float(self.config.min_step_ratio),
            float(self.config.radius_floor) * 0.25,
        )

        state = _RegionSearchState(
            region=region,
            rank=int(candidate_entry["rank"]),
            region_prior=float(candidate_entry.get("region_prior", 0.0)),
            initial_score=float(candidate_entry["initial_score"]),
            initial_objective=float(
                candidate_entry.get("selection_objective", candidate_entry["initial_score"])
            ),
            initial_radius=float(radius),
            radius=float(radius),
            step=float(step),
            min_step=float(min_step),
            best_score=float(candidate_entry["initial_score"]),
            best_objective=float(
                candidate_entry.get("selection_objective", candidate_entry["initial_score"])
            ),
            best_pred=int(candidate_entry["initial_pred"]),
            best_delta_c=candidate_entry["initial_delta_c"].clone(),
            best_delta_z=candidate_entry["initial_delta_z"].clone(),
            best_adv=candidate_entry["initial_adv"].clone(),
            query_coordinate_prior=(
                candidate_entry["query_coordinate_prior"].clone()
                if candidate_entry.get("query_coordinate_prior") is not None
                else None
            ),
            clean_adv=x0.clone(),
            clean_delta_c=torch.zeros(1, int(region.numel()), dtype=torch.float32),
            clean_delta_z=torch.zeros_like(x0),
            clean_pred=int(clean_pred),
            physics_quality=float(physics_quality),
            allocation_priority=float(allocation_priority),
            challenger_branch=bool(candidate_entry.get("challenger_branch", False)),
        )
        if bool(candidate_entry["initial_success"]):
            state.success_adv = candidate_entry["initial_adv"].clone()
            state.success_delta_c = candidate_entry["initial_delta_c"].clone()
            state.success_delta_z = candidate_entry["initial_delta_z"].clone()
            state.success_objective = float(
                candidate_entry.get("selection_objective", candidate_entry["initial_score"])
            )
            state.success_score = float(candidate_entry["initial_score"])
            state.success_pred = int(candidate_entry["initial_pred"])
        state.query_budget_population = max(0, int(query_budget_population))
        state.query_budget_rounds = max(0, int(query_budget_rounds))
        state.probe_improvement_ratio = float(probe_improvement_ratio)
        return state

    def _annotate_candidate_entries(
        self,
        candidate_entries: list[dict],
        clean_score: float,
        c_base: torch.Tensor,
        default_query_budget_population: int,
        default_query_budget_rounds: int,
    ) -> list[dict]:
        annotated = []
        for entry in candidate_entries:
            item = dict(entry)
            item["physics_quality"] = self._candidate_physics_quality(
                candidate_entry=item,
                c_base=c_base,
            )
            item["allocation_priority"] = self._candidate_allocation_priority(
                candidate_entry=item,
                clean_score=clean_score,
                c_base=c_base,
            )
            item["candidate_query_budget_population"] = max(0, int(default_query_budget_population))
            item["candidate_query_budget_rounds"] = max(0, int(default_query_budget_rounds))
            annotated.append(item)

        if (
            not bool(self.config.physics_query_allocation)
            or default_query_budget_population <= 0
            or default_query_budget_rounds <= 0
            or not annotated
        ):
            return annotated

        topk = max(1, min(int(self.config.physics_query_topk), len(annotated)))
        selected_indices = sorted(
            range(len(annotated)),
            key=lambda idx: (
                -float(annotated[idx].get("allocation_priority", 0.0)),
                -float(annotated[idx].get("physics_quality", 0.0)),
                int(annotated[idx]["rank"]),
            ),
        )[:topk]
        selected_set = set(int(v) for v in selected_indices)
        for idx, item in enumerate(annotated):
            if idx not in selected_set:
                item["candidate_query_budget_population"] = 0
                item["candidate_query_budget_rounds"] = 0
        return annotated

    def _allocate_stage_rounds(
        self,
        region_states: list[_RegionSearchState],
        total_stage_rounds: int,
    ) -> list[int]:
        total_stage_rounds = max(0, int(total_stage_rounds))
        if total_stage_rounds <= 0 or not region_states:
            return [0 for _ in region_states]
        if not bool(self.config.physics_query_allocation):
            base = total_stage_rounds // len(region_states)
            rem = total_stage_rounds % len(region_states)
            allocation = [base for _ in region_states]
            for idx in range(rem):
                allocation[idx] += 1
            return allocation

        scores = [
            max(0.0, float(state.allocation_priority)) + 1e-6 for state in region_states
        ]
        score_sum = sum(scores)
        raw = [score * total_stage_rounds / max(score_sum, 1e-8) for score in scores]
        allocation = [int(value) for value in raw]
        used = sum(allocation)
        remainder = total_stage_rounds - used
        order = sorted(
            range(len(region_states)),
            key=lambda idx: (
                -(raw[idx] - allocation[idx]),
                -scores[idx],
                int(region_states[idx].rank),
            ),
        )
        for idx in order[: max(0, remainder)]:
            allocation[idx] += 1
        return allocation

    def _assign_challenger_query_budgets(
        self,
        candidate_entries: list[dict],
    ) -> list[dict]:
        if not candidate_entries:
            return []
        if (
            not bool(self.config.adaptive_challenger_budget)
            or len(candidate_entries) <= 1
        ):
            for item in candidate_entries:
                item["challenger_branch"] = False
            return candidate_entries

        ordered_indices = sorted(
            range(len(candidate_entries)),
            key=lambda idx: (
                -float(candidate_entries[idx].get("allocation_priority", 0.0)),
                self._candidate_priority(candidate_entries[idx]),
            ),
        )
        primary_idx = int(ordered_indices[0])
        challenger_population = max(
            1,
            int(round(float(self.config.population) * float(self.config.challenger_population_ratio))),
        )
        challenger_rounds = max(1, int(self.config.challenger_rounds))

        for idx, item in enumerate(candidate_entries):
            item["challenger_branch"] = bool(idx != primary_idx)
            if idx == primary_idx:
                continue
            item["candidate_query_budget_population"] = max(
                1,
                min(
                    int(item.get("candidate_query_budget_population", challenger_population))
                    if int(item.get("candidate_query_budget_population", 0)) > 0
                    else challenger_population,
                    challenger_population,
                ),
            )
            item["candidate_query_budget_rounds"] = max(
                int(item.get("candidate_query_budget_rounds", 0)),
                challenger_rounds,
            )
        return candidate_entries

    @staticmethod
    def _score_progress_ratio(region_state: _RegionSearchState) -> float:
        denom = max(abs(float(region_state.initial_score)), 1e-6)
        return max(0.0, float(region_state.initial_score) - float(region_state.best_score)) / denom

    @staticmethod
    def _region_boundary_ratio(region_state: _RegionSearchState) -> float:
        denom = max(abs(float(region_state.initial_score)), 1e-6)
        return abs(float(region_state.best_score)) / denom

    def _region_boundary_uncertainty(self, region_state: _RegionSearchState) -> float:
        if self.config.query_mode == "decision":
            return 0.0
        tau = max(float(self.config.termination_uncertainty_ratio_tau), 1e-6)
        ratio = self._region_boundary_ratio(region_state)
        uncertainty = 1.0 / (1.0 + ratio / tau)
        return float(max(0.0, min(1.0, uncertainty)))

    def _current_region_physics_quality(
        self,
        region_state: _RegionSearchState,
        c_base: torch.Tensor,
    ) -> float:
        if c_base is None:
            return 1.0
        return float(
            self._fdia_backbone_quality(
                c_base=c_base,
                delta_c=region_state.best_delta_c,
                delta_z=region_state.best_delta_z,
                region=region_state.region,
            )[0].item()
        )

    def _should_guard_region_from_termination(
        self,
        region_state: _RegionSearchState,
        c_base: torch.Tensor,
    ) -> bool:
        if not (
            bool(self.config.uncertainty_aware_pruning)
            or bool(self.config.physics_aware_early_stop)
        ):
            return False
        if (
            bool(self.config.guarded_boundary_probe)
            and int(region_state.guard_probe_attempts)
            >= max(1, int(self.config.guarded_boundary_probe_max_uses))
        ):
            return False
        progress_ratio = self._score_progress_ratio(region_state)
        if progress_ratio < max(0.0, float(self.config.termination_progress_ratio_floor)):
            return False
        if bool(self.config.uncertainty_aware_pruning):
            uncertainty = self._region_boundary_uncertainty(region_state)
            if uncertainty < max(0.0, float(self.config.termination_uncertainty_floor)):
                return False
        if bool(self.config.physics_aware_early_stop):
            physics_quality = self._current_region_physics_quality(
                region_state=region_state,
                c_base=c_base,
            )
            if physics_quality < max(
                0.0,
                float(self.config.termination_physics_quality_floor),
            ):
                return False
        return True

    def _should_run_guarded_boundary_probe(
        self,
        region_state: _RegionSearchState,
        c_base: torch.Tensor,
    ) -> bool:
        if not bool(self.config.guarded_boundary_probe):
            return False
        if self.config.query_mode == "decision":
            return False
        if region_state.success_adv is not None:
            return False
        if int(region_state.guard_probe_attempts) >= max(
            1,
            int(self.config.guarded_boundary_probe_max_uses),
        ):
            return False
        if int(region_state.rounds_used) < max(
            1,
            int(self.config.search_min_rounds_before_stop),
        ):
            return False
        if int(region_state.stagnant_rounds) < max(
            1,
            int(self.config.score_stagnation_rounds),
        ):
            return False
        if int(region_state.step_shrinks) <= 0:
            return False
        return self._should_guard_region_from_termination(
            region_state=region_state,
            c_base=c_base,
        )

    def _run_guarded_boundary_probe(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        region_state: _RegionSearchState,
    ) -> bool:
        region_state.guard_probe_attempts += 1
        current_norm = float(torch.linalg.norm(region_state.best_delta_c).item())
        if current_norm <= 1e-12:
            return False
        radius = max(float(region_state.radius), current_norm)
        direction = self._normalize_rows(region_state.best_delta_c.clone())
        step_count = max(1, int(self.config.guarded_boundary_probe_steps))
        step = max(float(region_state.step), float(region_state.min_step))
        norm_candidates: list[float] = []
        for idx in range(1, step_count + 1):
            frac = float(idx) / float(step_count)
            target_norm = min(
                radius,
                max(
                    current_norm + step * float(idx),
                    current_norm + (radius - current_norm) * frac,
                ),
            )
            if target_norm > current_norm + 1e-8:
                norm_candidates.append(target_norm)
        if not norm_candidates:
            return False

        candidate_c = torch.cat(
            [direction * float(target_norm) for target_norm in norm_candidates],
            dim=0,
        )
        candidate_c = _project_l2(candidate_c, region_state.radius)
        candidate_delta_z = self._project_region(
            region=region_state.region,
            delta_region=candidate_c,
            use_search_shaping=True,
        )
        candidate_c, candidate_delta_z = self._apply_measurement_l2_cap(
            x_ref=x0,
            delta_c=candidate_c,
            delta_z=candidate_delta_z,
        )
        candidate_adv = x0 + candidate_delta_z
        candidate_query = self.oracle.query(candidate_adv)
        candidate_score = self._score(candidate_query)
        candidate_objective = self._selection_objective(
            candidate_score,
            x0,
            candidate_delta_z,
            c_base=c_base,
            delta_c=candidate_c,
            region=region_state.region,
        )
        candidate_preds = candidate_query.pred
        success_mask = candidate_preds != int(label)

        if bool(success_mask.any().item()):
            success_priority = self._success_priority_value(
                x_ref=x0,
                delta_z=candidate_delta_z,
                c_base=c_base,
                delta_c=candidate_c,
                region=region_state.region,
            )
            success_idx = int(
                torch.argmin(success_priority.masked_fill(~success_mask, float("inf"))).item()
            )
            region_state.success_adv = candidate_adv[success_idx : success_idx + 1].clone()
            region_state.success_delta_c = candidate_c[success_idx : success_idx + 1].clone()
            region_state.success_delta_z = candidate_delta_z[success_idx : success_idx + 1].clone()
            region_state.success_objective = float(candidate_objective[success_idx].item())
            region_state.success_score = float(candidate_score[success_idx].item())
            region_state.success_pred = int(candidate_preds[success_idx].item())
            region_state.best_adv = candidate_adv[success_idx : success_idx + 1].clone()
            region_state.best_delta_c = candidate_c[success_idx : success_idx + 1].clone()
            region_state.best_delta_z = candidate_delta_z[success_idx : success_idx + 1].clone()
            region_state.best_objective = float(candidate_objective[success_idx].item())
            region_state.best_score = float(candidate_score[success_idx].item())
            region_state.best_pred = int(candidate_preds[success_idx].item())
            region_state.no_improve = 0
            region_state.stagnant_rounds = 0
            region_state.step_shrinks = 0
            return bool(self._should_early_stop_on_success())

        best_idx = int(torch.argmin(candidate_objective).item())
        if float(candidate_objective[best_idx].item()) + 1e-8 < float(region_state.best_objective):
            region_state.best_objective = float(candidate_objective[best_idx].item())
            region_state.best_score = float(candidate_score[best_idx].item())
            region_state.best_pred = int(candidate_preds[best_idx].item())
            region_state.best_delta_c = candidate_c[best_idx : best_idx + 1].clone()
            region_state.best_delta_z = candidate_delta_z[best_idx : best_idx + 1].clone()
            region_state.best_adv = candidate_adv[best_idx : best_idx + 1].clone()
            region_state.no_improve = 0
            region_state.stagnant_rounds = 0
            region_state.step_shrinks = 0
            return False
        region_state.no_improve = 0
        region_state.stagnant_rounds = 0
        return False

    def _update_score_stagnation(
        self,
        region_state: _RegionSearchState,
        previous_best_score: float,
    ) -> None:
        if region_state.success_adv is not None:
            region_state.stagnant_rounds = 0
            return
        gain = max(0.0, float(previous_best_score) - float(region_state.best_score))
        threshold = max(
            0.0,
            float(self.config.score_gain_ratio_threshold)
            * max(abs(float(region_state.initial_score)), 1e-6),
        )
        if gain <= threshold:
            region_state.stagnant_rounds += 1
        else:
            region_state.stagnant_rounds = 0

    def _should_search_early_stop(
        self,
        region_state: _RegionSearchState,
        c_base: torch.Tensor,
    ) -> bool:
        if not bool(self.config.score_stagnation_early_stop):
            return False
        if region_state.success_adv is not None:
            return False
        if int(region_state.rounds_used) < max(
            1,
            int(self.config.search_min_rounds_before_stop),
        ):
            return False
        if int(region_state.stagnant_rounds) < max(
            1,
            int(self.config.score_stagnation_rounds),
        ):
            return False
        if int(region_state.step_shrinks) <= 0:
            return False
        if self._should_guard_region_from_termination(
            region_state=region_state,
            c_base=c_base,
        ):
            return False
        region_state.early_stopped = True
        return True

    @staticmethod
    def _active_region_indices(region_states: list[_RegionSearchState]) -> list[int]:
        return [idx for idx, state in enumerate(region_states) if bool(state.active)]

    def _should_prune_region_state(
        self,
        region_state: _RegionSearchState,
        region_states: list[_RegionSearchState],
        c_base: torch.Tensor,
    ) -> bool:
        if not bool(self.config.branch_pruning):
            return False
        if not bool(region_state.active) or bool(region_state.pruned):
            return False
        if not bool(region_state.challenger_branch):
            return False
        if region_state.success_adv is not None:
            return False
        if int(region_state.rounds_used) < max(1, int(self.config.challenger_rounds)):
            return False
        if int(region_state.stagnant_rounds) < max(1, int(self.config.score_stagnation_rounds)):
            return False
        active_states = [
            state for state in region_states if bool(state.active) and state is not region_state
        ]
        if not active_states:
            return False
        best_other_score = min(float(state.best_score) for state in active_states)
        score_gap_ratio = max(
            0.0,
            float(region_state.best_score) - best_other_score,
        ) / max(abs(float(region_state.initial_score)), 1e-6)
        progress_ratio = self._score_progress_ratio(region_state)
        if (
            progress_ratio <= float(self.config.branch_prune_progress_ratio)
            and score_gap_ratio >= float(self.config.branch_prune_score_gap_ratio)
        ):
            if self._should_guard_region_from_termination(
                region_state=region_state,
                c_base=c_base,
            ):
                return False
            region_state.active = False
            region_state.pruned = True
            return True
        return False

    def _resolve_search_population(self, region_state: _RegionSearchState) -> int:
        active_population = max(1, int(self.config.population))
        if (
            region_state.query_budget_population > 0
            and region_state.query_budget_rounds > 0
            and region_state.rounds_used < region_state.query_budget_rounds
        ):
            active_population = min(
                active_population,
                int(region_state.query_budget_population),
            )
        return max(1, active_population)

    def _select_budget_candidates(
        self,
        candidate_entries: list[dict],
        probe_summary: dict | None = None,
    ) -> list[dict]:
        if not candidate_entries:
            return []

        topk = max(1, int(self.config.region_budget_topk))
        feedback_enabled = bool(self.config.feedback_loop and topk > 1)
        multi_region_allowed = bool(topk > 1) and self._feedback_multi_region_enabled(
            probe_summary
        )
        if self.config.query_mode == "decision":
            if any(bool(entry["initial_success"]) for entry in candidate_entries):
                ranked = sorted(
                    candidate_entries,
                    key=lambda entry: (
                        0.0 if bool(entry["initial_success"]) else 1.0,
                        float(entry.get("selection_objective", entry["initial_score"])),
                        int(entry["rank"]),
                    ),
                )
            else:
                ranked = sorted(
                    candidate_entries,
                    key=lambda entry: (
                        -float(entry.get("region_prior", 0.0)),
                        -float(torch.linalg.norm(entry["initial_delta_z"]).item()),
                        int(entry["rank"]),
                    ),
                )
            return ranked[: topk if multi_region_allowed else 1]

        explore_rounds = max(0, int(self.config.region_budget_explore_rounds))
        if (
            topk <= 1
            or not multi_region_allowed
            or (explore_rounds <= 0 and not feedback_enabled)
        ):
            return sorted(candidate_entries, key=self._candidate_priority)[:1]

        sorted_entries = sorted(candidate_entries, key=self._candidate_priority)
        selected = [sorted_entries[0]]
        best_score = float(sorted_entries[0]["initial_score"])
        best_success = bool(sorted_entries[0]["initial_success"])
        score_slack = max(0.0, float(self.config.region_budget_score_slack))
        best_prior = float(sorted_entries[0].get("region_prior", 0.0))
        if (
            best_prior > float(self.config.budget_region_max_probe_best_prior)
            and not feedback_enabled
        ):
            return selected
        eligible_entries = []

        for entry in sorted_entries[1:]:
            if best_success:
                if bool(entry["initial_success"]):
                    eligible_entries.append(entry)
                continue
            if float(entry["initial_score"]) <= best_score + score_slack:
                eligible_entries.append(entry)

        if self.config.budget_region_prior_tiebreak:
            eligible_entries = sorted(eligible_entries, key=self._budget_extra_priority)
        else:
            eligible_entries = sorted(eligible_entries, key=self._candidate_priority)

        for entry in eligible_entries:
            if len(selected) >= topk:
                break
            selected.append(entry)

        return selected

    def _feedback_multi_region_enabled(self, probe_summary: dict | None) -> bool:
        min_state_dim = max(0, int(self.config.feedback_min_state_dim))
        if min_state_dim > 0 and int(self.topology.n_states) < min_state_dim:
            return False
        if probe_summary is None:
            return True

        abs_threshold = max(0.0, float(self.config.feedback_probe_gap_abs_threshold))
        ratio_threshold = max(0.0, float(self.config.feedback_probe_gap_ratio_threshold))
        if abs_threshold <= 0.0 and ratio_threshold <= 0.0:
            return True

        gap = max(0.0, float(probe_summary.get("probe_score_gap", 0.0)))
        improvement = max(0.0, float(probe_summary.get("probe_improvement", 0.0)))
        if int(probe_summary.get("probe_success_count", 0)) > 1:
            return True
        if abs_threshold > 0.0 and gap <= abs_threshold:
            return True
        if ratio_threshold > 0.0 and gap <= max(improvement, 1e-6) * ratio_threshold:
            return True
        return False

    def _resolve_final_region_state(
        self,
        region_states: list[_RegionSearchState],
    ) -> _RegionSearchState:
        if not region_states:
            raise ValueError("region_states must not be empty.")
        if (
            len(region_states) > 1
            and bool(self.config.feedback_loop)
            and bool(self.config.feedback_keep_probe_best_incumbent)
            and not bool(self.config.physics_query_allocation)
        ):
            active_indices = self._active_region_indices(region_states)
            if active_indices:
                return region_states[int(active_indices[0])]
        return min(region_states, key=self._state_priority)

    def _select_feedback_cycle_indices(
        self,
        region_states: list[_RegionSearchState],
        visit_counts: list[int],
        last_visit: list[int],
        primary_idx: int | None = None,
    ) -> list[int]:
        active_indices = self._active_region_indices(region_states)
        if not active_indices:
            return []

        exploit_order = sorted(
            active_indices,
            key=lambda idx: self._state_priority(region_states[idx]),
        )
        branch_topk = max(
            1,
            min(int(self.config.feedback_branch_topk), len(active_indices)),
        )
        if branch_topk <= 1:
            if primary_idx is not None:
                return [int(primary_idx)]
            return exploit_order[:branch_topk]
        if branch_topk >= len(active_indices):
            if primary_idx is None:
                return exploit_order[:branch_topk]
            remaining = [idx for idx in exploit_order if idx != int(primary_idx)]
            return [int(primary_idx)] + remaining[: max(0, branch_topk - 1)]

        if primary_idx is not None:
            exploit_rank = {idx: rank for rank, idx in enumerate(exploit_order)}
            remaining = [idx for idx in exploit_order if idx != int(primary_idx)]
            ranked_remaining = sorted(
                remaining,
                key=lambda idx: (
                    int(visit_counts[idx]),
                    int(last_visit[idx]),
                    int(exploit_rank[idx]),
                    int(region_states[idx].rank),
                ),
            )
            return [int(primary_idx)] + ranked_remaining[: max(0, branch_topk - 1)]

        main_slots = max(1, branch_topk - 1)
        selected = exploit_order[:main_slots]
        exploit_rank = {idx: rank for rank, idx in enumerate(exploit_order)}
        remaining = [idx for idx in exploit_order if idx not in selected]
        if remaining:
            revisit_idx = min(
                remaining,
                key=lambda idx: (
                    int(visit_counts[idx]),
                    int(last_visit[idx]),
                    int(exploit_rank[idx]),
                    int(region_states[idx].rank),
                ),
            )
            selected.append(revisit_idx)
        return selected

    def _search_region_feedback_cycles(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        region_states: list[_RegionSearchState],
        generator: torch.Generator | None,
        max_rounds: int,
    ) -> _RegionSearchState | None:
        if not region_states or max_rounds <= 0:
            return None

        visit_counts = [0 for _ in region_states]
        last_visit = [-1 for _ in region_states]
        trigger_seen = [-1 for _ in region_states]
        remaining_rounds = max(0, int(max_rounds))
        round_chunk = max(1, int(self.config.feedback_round_chunk))
        stagnation_trigger = max(1, int(self.config.feedback_stagnation_trigger))
        cycle_idx = 0

        while remaining_rounds > 0:
            active_indices = self._active_region_indices(region_states)
            if not active_indices:
                break
            if bool(self.config.feedback_keep_probe_best_incumbent) and not bool(
                self.config.physics_query_allocation
            ):
                primary_idx = int(active_indices[0])
            else:
                exploit_order = sorted(
                    active_indices,
                    key=lambda idx: self._state_priority(region_states[idx]),
                )
                if not exploit_order:
                    break
                primary_idx = exploit_order[0]
            primary_state = region_states[primary_idx]
            should_backtrack = (
                len(region_states) > 1
                and int(self.config.feedback_branch_topk) > 1
                and (
                    int(primary_state.no_improve) >= stagnation_trigger
                    or int(primary_state.step_shrinks) > 0
                )
                and int(primary_state.step_shrinks) > trigger_seen[primary_idx]
            )
            if should_backtrack:
                cycle_indices = self._select_feedback_cycle_indices(
                    region_states=region_states,
                    visit_counts=visit_counts,
                    last_visit=last_visit,
                    primary_idx=primary_idx,
                )
                trigger_seen[primary_idx] = int(primary_state.step_shrinks)
            else:
                cycle_indices = [primary_idx]
            if not cycle_indices:
                break

            for idx in cycle_indices:
                if remaining_rounds <= 0:
                    break
                allocated_rounds = min(round_chunk, remaining_rounds)
                should_stop = self._search_region_rounds(
                    x0=x0,
                    label=label,
                    c_base=c_base,
                    region_state=region_states[idx],
                    generator=generator,
                    max_rounds=allocated_rounds,
                )
                self._should_prune_region_state(
                    region_states[idx],
                    region_states,
                    c_base,
                )
                visit_counts[idx] += 1
                last_visit[idx] = cycle_idx
                remaining_rounds -= allocated_rounds
                if should_stop:
                    return region_states[idx]
            cycle_idx += 1

        return None

    def _search_region_rounds_decision(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        region_state: _RegionSearchState,
        generator: torch.Generator | None,
        max_rounds: int,
    ) -> bool:
        rounds_to_run = max(0, int(max_rounds))
        expand_factor = 1.0 / max(min(float(self.config.step_decay), 0.95), 0.55)

        for _ in range(rounds_to_run):
            region_state.rounds_used += 1
            active_population = self._resolve_search_population(region_state)
            directions = self._build_search_directions(
                c_base=c_base,
                region_state=region_state,
                population=active_population,
                generator=generator,
            )

            if region_state.success_delta_c is None:
                center = region_state.best_delta_c.reshape(1, -1).to(dtype=torch.float32)
                center_norm = float(torch.linalg.norm(center).item())
                max_radius = max(
                    float(self.config.radius_floor),
                    float(region_state.initial_radius) * 4.0,
                )
                search_radius = max(
                    float(region_state.radius),
                    center_norm + max(region_state.step, float(self.config.radius_floor)),
                )
                if center_norm > 1e-12:
                    search_radius = max(search_radius, center_norm * expand_factor)
                else:
                    search_radius = max(search_radius, region_state.step * expand_factor)
                search_radius = min(search_radius, max_radius)

                candidate_blocks = [
                    center + region_state.step * directions,
                    center - region_state.step * directions,
                ]
                if center_norm > 1e-12:
                    center_dir = center / torch.linalg.norm(
                        center,
                        dim=1,
                        keepdim=True,
                    ).clamp(min=1e-12)
                    candidate_blocks.append(center * expand_factor)
                    candidate_blocks.append(
                        center + max(region_state.step, 0.5 * search_radius) * center_dir
                    )
                else:
                    seed_dir = directions[:1]
                    candidate_blocks.append(max(region_state.step, 0.5 * search_radius) * seed_dir)
                    candidate_blocks.append(search_radius * seed_dir)

                candidates_c = torch.cat(candidate_blocks, dim=0)
                candidates_c = _project_l2(candidates_c, search_radius)
                delta_z = self._project_region(
                    region=region_state.region,
                    delta_region=candidates_c,
                    use_search_shaping=True,
                )
                candidates_c, delta_z = self._apply_measurement_l2_cap(
                    x_ref=x0,
                    delta_c=candidates_c,
                    delta_z=delta_z,
                )
                adv_batch = x0 + delta_z
                query = self.oracle.query(adv_batch)
                preds = query.pred
                delta_norm = torch.linalg.norm(delta_z, dim=1)
                success_mask = preds != int(label)

                if bool(success_mask.any().item()):
                    success_idx = int(
                        torch.argmin(delta_norm.masked_fill(~success_mask, float("inf"))).item()
                    )
                    refined = self._decision_binary_refine(
                        x0=x0,
                        label=label,
                        region=region_state.region,
                        fail_delta_c=torch.zeros_like(candidates_c[success_idx : success_idx + 1]),
                        success_delta_c=candidates_c[success_idx : success_idx + 1].clone(),
                        success_delta_z=delta_z[success_idx : success_idx + 1].clone(),
                        success_adv=adv_batch[success_idx : success_idx + 1].clone(),
                        success_pred=int(preds[success_idx].item()),
                    )
                    refined_norm = float(torch.linalg.norm(refined["delta_z"]).item())
                    region_state.success_adv = refined["adv"].clone()
                    region_state.success_delta_c = refined["delta_c"].clone()
                    region_state.success_delta_z = refined["delta_z"].clone()
                    region_state.success_objective = float(refined["objective"])
                    region_state.success_score = float(refined["score"])
                    region_state.success_pred = int(refined["pred"])
                    region_state.best_adv = refined["adv"].clone()
                    region_state.best_delta_c = refined["delta_c"].clone()
                    region_state.best_delta_z = refined["delta_z"].clone()
                    region_state.best_objective = float(refined["objective"])
                    region_state.best_score = float(refined["score"])
                    region_state.best_pred = int(refined["pred"])
                    region_state.radius = max(
                        float(self.config.radius_floor),
                        max(refined_norm * 1.25, float(region_state.min_step) * 2.0),
                    )
                    region_state.step = max(
                        min(region_state.step, 0.5 * max(refined_norm, region_state.min_step)),
                        region_state.min_step,
                    )
                    region_state.no_improve = 0
                    region_state.step_shrinks = 0
                    if self._should_early_stop_on_success():
                        return True
                else:
                    frontier_idx = int(torch.argmax(delta_norm).item())
                    region_state.best_delta_c = candidates_c[frontier_idx : frontier_idx + 1].clone()
                    region_state.best_delta_z = delta_z[frontier_idx : frontier_idx + 1].clone()
                    region_state.best_adv = adv_batch[frontier_idx : frontier_idx + 1].clone()
                    region_state.best_pred = int(preds[frontier_idx].item())
                    region_state.best_score = 0.0
                    region_state.best_objective = float(
                        self._selection_objective(
                            torch.zeros(1, dtype=torch.float32),
                            x0,
                            delta_z[frontier_idx : frontier_idx + 1],
                        )[0].item()
                    )
                    region_state.radius = float(search_radius)
                    region_state.step = min(
                        max(
                            region_state.step * expand_factor,
                            float(self.config.radius_floor) * 0.5,
                        ),
                        max_radius,
                    )
                    if search_radius >= max_radius - 1e-12:
                        region_state.no_improve += 1
                    else:
                        region_state.no_improve = 0
                    if region_state.no_improve >= self.config.patience:
                        break
                continue

            success_c = region_state.success_delta_c.reshape(1, -1).to(dtype=torch.float32)
            success_norm = float(torch.linalg.norm(success_c).item())
            if success_norm <= region_state.min_step + 1e-12:
                break

            normal = success_c / torch.linalg.norm(
                success_c,
                dim=1,
                keepdim=True,
            ).clamp(min=1e-12)
            orth = directions - (directions * normal).sum(dim=1, keepdim=True) * normal
            orth_norm = torch.linalg.norm(orth, dim=1, keepdim=True)
            valid_mask = orth_norm.reshape(-1) > 1e-8

            inward_step = min(region_state.step, 0.60 * success_norm)
            target_norm = max(
                success_norm - inward_step,
                0.20 * success_norm,
                float(region_state.min_step),
            )
            candidate_blocks = [
                success_c * (target_norm / max(success_norm, 1e-12)),
            ]
            if bool(valid_mask.any().item()):
                tangent_dirs = self._normalize_rows(orth[valid_mask])
                tangent_step = min(region_state.step, 0.35 * success_norm)
                tangent = success_c.repeat(tangent_dirs.shape[0], 1) + tangent_step * tangent_dirs
                tangent = tangent / torch.linalg.norm(
                    tangent,
                    dim=1,
                    keepdim=True,
                ).clamp(min=1e-12)
                tangent = tangent * target_norm
                candidate_blocks.append(tangent)

            candidates_c = torch.cat(candidate_blocks, dim=0)
            candidates_c = _project_l2(candidates_c, max(region_state.radius, success_norm))
            delta_z = self._project_region(
                region=region_state.region,
                delta_region=candidates_c,
                use_search_shaping=True,
            )
            candidates_c, delta_z = self._apply_measurement_l2_cap(
                x_ref=x0,
                delta_c=candidates_c,
                delta_z=delta_z,
            )
            adv_batch = x0 + delta_z
            query = self.oracle.query(adv_batch)
            preds = query.pred
            delta_norm = torch.linalg.norm(delta_z, dim=1)
            success_mask = preds != int(label)

            improved = False
            if bool(success_mask.any().item()):
                success_idx = int(
                    torch.argmin(delta_norm.masked_fill(~success_mask, float("inf"))).item()
                )
                refined = self._decision_binary_refine(
                    x0=x0,
                    label=label,
                    region=region_state.region,
                    fail_delta_c=torch.zeros_like(candidates_c[success_idx : success_idx + 1]),
                    success_delta_c=candidates_c[success_idx : success_idx + 1].clone(),
                    success_delta_z=delta_z[success_idx : success_idx + 1].clone(),
                    success_adv=adv_batch[success_idx : success_idx + 1].clone(),
                    success_pred=int(preds[success_idx].item()),
                )
                new_norm = float(torch.linalg.norm(refined["delta_z"]).item())
                current_norm = float(torch.linalg.norm(region_state.success_delta_z).item())
                if new_norm + 1e-8 < current_norm:
                    improved = True
                    region_state.success_adv = refined["adv"].clone()
                    region_state.success_delta_c = refined["delta_c"].clone()
                    region_state.success_delta_z = refined["delta_z"].clone()
                    region_state.success_objective = float(refined["objective"])
                    region_state.success_score = float(refined["score"])
                    region_state.success_pred = int(refined["pred"])
                    region_state.best_adv = refined["adv"].clone()
                    region_state.best_delta_c = refined["delta_c"].clone()
                    region_state.best_delta_z = refined["delta_z"].clone()
                    region_state.best_objective = float(refined["objective"])
                    region_state.best_score = float(refined["score"])
                    region_state.best_pred = int(refined["pred"])
                    region_state.radius = max(
                        float(self.config.radius_floor),
                        max(new_norm * 1.20, float(region_state.min_step) * 2.0),
                    )
                    if self._should_early_stop_on_success():
                        return True

            if improved:
                region_state.step = max(
                    region_state.step * self.config.step_decay,
                    region_state.min_step,
                )
                region_state.no_improve = 0
                region_state.step_shrinks = 0
            else:
                region_state.no_improve += 1
                if region_state.no_improve >= self.config.patience:
                    region_state.step = max(
                        region_state.step * self.config.step_decay,
                        region_state.min_step,
                    )
                    region_state.step_shrinks += 1
                    region_state.no_improve = 0
                    if (
                        region_state.step <= region_state.min_step + 1e-12
                        and region_state.step_shrinks >= self.config.max_step_shrinks
                    ):
                        break

        return False

    def _search_region_rounds(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        region_state: _RegionSearchState,
        generator: torch.Generator | None,
        max_rounds: int,
    ) -> bool:
        if self.config.query_mode == "decision":
            return self._search_region_rounds_decision(
                x0=x0,
                label=label,
                c_base=c_base,
                region_state=region_state,
                generator=generator,
                max_rounds=max_rounds,
            )
        if (not self._uses_measurement_regions()) and self.config.state_subspace_pgzoo:
            return self._search_state_subspace_pgzoo_rounds(
                x0=x0,
                label=label,
                c_base=c_base,
                region_state=region_state,
                generator=generator,
                max_rounds=max_rounds,
            )
        if (
            (not self._uses_measurement_regions())
            and self.config.state_bandit_search
            and self.config.state_basis_search
            and int(self.config.bandit_warmup_rounds) > 0
        ):
            warmup_rounds = min(max_rounds, max(0, int(self.config.bandit_warmup_rounds)))
            if warmup_rounds > 0:
                should_stop = self._search_state_bandit_rounds(
                    x0=x0,
                    label=label,
                    c_base=c_base,
                    region_state=region_state,
                    generator=generator,
                    max_rounds=warmup_rounds,
                )
                if should_stop:
                    return True
            remaining_rounds = max(0, int(max_rounds) - int(warmup_rounds))
            if remaining_rounds > 0:
                return self._search_state_basis_rounds(
                    x0=x0,
                    label=label,
                    c_base=c_base,
                    region_state=region_state,
                    max_rounds=remaining_rounds,
                )
            return False
        if (
            self._uses_measurement_regions()
            and self.config.measurement_bandit_search
            and self.config.measurement_basis_search
            and int(self.config.bandit_warmup_rounds) > 0
        ):
            warmup_rounds = min(max_rounds, max(0, int(self.config.bandit_warmup_rounds)))
            if warmup_rounds > 0:
                should_stop = self._search_measurement_bandit_rounds(
                    x0=x0,
                    label=label,
                    c_base=c_base,
                    region_state=region_state,
                    generator=generator,
                    max_rounds=warmup_rounds,
                )
                if should_stop:
                    return True
            remaining_rounds = max(0, int(max_rounds) - int(warmup_rounds))
            if remaining_rounds > 0:
                return self._search_measurement_basis_rounds(
                    x0=x0,
                    label=label,
                    c_base=c_base,
                    region_state=region_state,
                    max_rounds=remaining_rounds,
                )
            return False
        if self._uses_measurement_regions() and self.config.measurement_bandit_search:
            return self._search_measurement_bandit_rounds(
                x0=x0,
                label=label,
                c_base=c_base,
                region_state=region_state,
                generator=generator,
                max_rounds=max_rounds,
            )
        if (not self._uses_measurement_regions()) and self.config.state_bandit_search:
            return self._search_state_bandit_rounds(
                x0=x0,
                label=label,
                c_base=c_base,
                region_state=region_state,
                generator=generator,
                max_rounds=max_rounds,
            )
        if self._uses_measurement_regions() and self.config.measurement_basis_search:
            return self._search_measurement_basis_rounds(
                x0=x0,
                label=label,
                c_base=c_base,
                region_state=region_state,
                max_rounds=max_rounds,
            )
        if (not self._uses_measurement_regions()) and self.config.state_basis_search:
            return self._search_state_basis_rounds(
                x0=x0,
                label=label,
                c_base=c_base,
                region_state=region_state,
                max_rounds=max_rounds,
            )

        region_dim = int(region_state.region.numel())
        rounds_to_run = max(0, int(max_rounds))

        for _ in range(rounds_to_run):
            region_state.rounds_used += 1
            active_population = self._resolve_search_population(region_state)
            directions = self._build_search_directions(
                c_base=c_base,
                region_state=region_state,
                population=active_population,
                generator=generator,
            )
            candidates_c = torch.cat(
                [
                    region_state.best_delta_c + region_state.step * directions,
                    region_state.best_delta_c - region_state.step * directions,
                ],
                dim=0,
            )
            candidates_c = _project_l2(candidates_c, region_state.radius)
            candidates_c = self._apply_state_backbone_lock(
                c_base=c_base,
                region=region_state.region,
                delta_c=candidates_c,
            )
            delta_z = self._project_region(
                region=region_state.region,
                delta_region=candidates_c,
                use_search_shaping=True,
            )
            candidates_c, delta_z = self._apply_measurement_l2_cap(
                x_ref=x0,
                delta_c=candidates_c,
                delta_z=delta_z,
            )
            adv_batch = x0 + delta_z
            if not self._has_query_budget_for(int(adv_batch.shape[0])):
                region_state.query_cap_reached = True
                region_state.early_stopped = True
                return True

            query = self.oracle.query(adv_batch)
            preds = query.pred
            score = self._score(query)
            objective = self._selection_objective(
                score,
                x0,
                delta_z,
                c_base=c_base,
                delta_c=candidates_c,
                region=region_state.region,
            )
            delta_norm = torch.linalg.norm(delta_z, dim=1)
            success_mask = preds != int(label)

            if success_mask.any():
                success_priority = self._success_priority_value(
                    x_ref=x0,
                    delta_z=delta_z,
                    c_base=c_base,
                    delta_c=candidates_c,
                    region=region_state.region,
                )
                success_idx = torch.argmin(
                    success_priority.masked_fill(~success_mask, float("inf"))
                )
                success_candidate_score = float(score[success_idx].item())
                success_candidate_objective = float(objective[success_idx].item())
                success_candidate_priority = float(success_priority[success_idx].item())
                current_success_priority = float("inf")
                if region_state.success_delta_z is not None:
                    current_success_priority = float(
                        self._success_priority_value(
                            x_ref=x0,
                            delta_z=region_state.success_delta_z,
                            c_base=c_base,
                            delta_c=region_state.success_delta_c,
                            region=region_state.region,
                        )[0].item()
                    )
                if (
                    region_state.success_adv is None
                    or success_candidate_priority + 1e-8 < current_success_priority
                ):
                    region_state.success_adv = adv_batch[success_idx : success_idx + 1].clone()
                    region_state.success_delta_c = candidates_c[
                        success_idx : success_idx + 1
                    ].clone()
                    region_state.success_delta_z = delta_z[
                        success_idx : success_idx + 1
                    ].clone()
                    region_state.success_objective = success_candidate_objective
                    region_state.success_score = success_candidate_score
                    region_state.success_pred = int(preds[success_idx].item())
                if self._should_early_stop_on_success():
                    return True

            if self.config.query_mode == "decision":
                if success_mask.any():
                    best_idx = success_idx
                    region_state.best_delta_c = candidates_c[best_idx : best_idx + 1].clone()
                    region_state.best_delta_z = delta_z[best_idx : best_idx + 1].clone()
                    region_state.best_adv = adv_batch[best_idx : best_idx + 1].clone()
                    region_state.best_objective = float(objective[best_idx].item())
                    region_state.best_pred = int(preds[best_idx].item())
                    region_state.best_score = float(score[best_idx].item())
                    region_state.step = max(
                        region_state.step * self.config.step_decay,
                        self.config.radius_floor * 0.25,
                    )
                    region_state.no_improve = 0
                else:
                    region_state.no_improve += 1
            else:
                best_idx = int(torch.argmin(objective).item())
                candidate_objective = float(objective[best_idx].item())
                candidate_score = float(score[best_idx].item())
                if candidate_objective < region_state.best_objective:
                    region_state.best_objective = candidate_objective
                    region_state.best_score = candidate_score
                    region_state.best_pred = int(preds[best_idx].item())
                    region_state.best_delta_c = candidates_c[best_idx : best_idx + 1].clone()
                    region_state.best_delta_z = delta_z[best_idx : best_idx + 1].clone()
                    region_state.best_adv = adv_batch[best_idx : best_idx + 1].clone()
                    region_state.no_improve = 0
                else:
                    region_state.no_improve += 1

            if region_state.no_improve >= self.config.patience:
                new_step = max(
                    region_state.step * self.config.step_decay,
                    region_state.min_step,
                )
                if (
                    abs(new_step - region_state.step) <= 1e-12
                    and region_state.step <= region_state.min_step + 1e-12
                ):
                    region_state.step_shrinks += 1
                else:
                    region_state.step = new_step
                    region_state.step_shrinks += 1
                region_state.no_improve = 0
                if (
                    region_state.step <= region_state.min_step + 1e-12
                    and region_state.step_shrinks >= self.config.max_step_shrinks
                ):
                    break

        return False

    def _search_measurement_bandit_rounds(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        region_state: _RegionSearchState,
        generator: torch.Generator | None,
        max_rounds: int,
    ) -> bool:
        rounds_to_run = max(0, int(max_rounds))
        momentum_coeff = None
        beta = min(max(float(self.config.bandit_momentum), 0.0), 0.99)
        exploration_ratio = max(float(self.config.bandit_exploration_ratio), 0.1)

        for _ in range(rounds_to_run):
            region_state.rounds_used += 1
            active_population = self._resolve_search_population(region_state)
            basis = self._measurement_basis_matrix(
                region_state=region_state,
                x_ref=x0,
                max_vectors=max(
                    2,
                    min(
                        int(self.config.structured_direction_count),
                        max(2, active_population),
                    ),
                ),
            )
            coeff_dim = int(basis.shape[0])
            direction_samples = max(
                1,
                min(int(self.config.bandit_direction_samples), coeff_dim),
            )
            coeff_noise = torch.randn(
                direction_samples,
                coeff_dim,
                dtype=torch.float32,
                generator=generator,
            )
            coeff_noise = self._normalize_rows(coeff_noise)
            search_dirs = self._normalize_rows(coeff_noise @ basis)

            sigma = max(region_state.step * exploration_ratio, region_state.min_step)
            probe_c = torch.cat(
                [
                    region_state.best_delta_c + sigma * search_dirs,
                    region_state.best_delta_c - sigma * search_dirs,
                ],
                dim=0,
            )
            probe_c = _project_l2(probe_c, region_state.radius)
            probe_c = self._apply_state_backbone_lock(
                c_base=c_base,
                region=region_state.region,
                delta_c=probe_c,
            )
            probe_delta_z = self._project_region(
                region=region_state.region,
                delta_region=probe_c,
                use_search_shaping=True,
            )
            probe_c, probe_delta_z = self._apply_measurement_l2_cap(
                x_ref=x0,
                delta_c=probe_c,
                delta_z=probe_delta_z,
            )
            probe_adv = x0 + probe_delta_z
            if not self._has_query_budget_for(int(probe_adv.shape[0])):
                region_state.query_cap_reached = True
                region_state.early_stopped = True
                return True
            probe_query = self.oracle.query(probe_adv)
            probe_score = self._score(probe_query)
            probe_objective = self._selection_objective(
                probe_score,
                x0,
                probe_delta_z,
                c_base=c_base,
                delta_c=probe_c,
                region=region_state.region,
            )

            obj_plus = probe_objective[:direction_samples]
            obj_minus = probe_objective[direction_samples:]
            grad_coeff = (
                ((obj_plus - obj_minus) / max(2.0 * sigma, 1e-8)).reshape(-1, 1) * coeff_noise
            ).mean(dim=0, keepdim=True)
            if momentum_coeff is None or momentum_coeff.shape[1] != grad_coeff.shape[1]:
                momentum_coeff = grad_coeff.clone()
            else:
                momentum_coeff = beta * momentum_coeff + (1.0 - beta) * grad_coeff

            update_dir = momentum_coeff @ basis
            if float(torch.linalg.norm(update_dir).item()) <= 1e-12:
                region_state.no_improve += 1
                continue
            update_dir = self._normalize_rows(update_dir)

            line_scales = [
                max(region_state.step, region_state.min_step),
                max(region_state.step * self.config.step_decay, region_state.min_step),
            ]
            candidate_c = torch.cat(
                [
                    region_state.best_delta_c - line_scales[0] * update_dir,
                    region_state.best_delta_c - line_scales[1] * update_dir,
                    region_state.best_delta_c + 0.5 * line_scales[1] * update_dir,
                ],
                dim=0,
            )
            candidate_c = _project_l2(candidate_c, region_state.radius)
            candidate_c = self._apply_state_backbone_lock(
                c_base=c_base,
                region=region_state.region,
                delta_c=candidate_c,
            )
            candidate_delta_z = self._project_region(
                region=region_state.region,
                delta_region=candidate_c,
                use_search_shaping=True,
            )
            candidate_c, candidate_delta_z = self._apply_measurement_l2_cap(
                x_ref=x0,
                delta_c=candidate_c,
                delta_z=candidate_delta_z,
            )
            candidate_adv = x0 + candidate_delta_z
            if not self._has_query_budget_for(int(candidate_adv.shape[0])):
                region_state.query_cap_reached = True
                region_state.early_stopped = True
                return True
            candidate_query = self.oracle.query(candidate_adv)
            candidate_score = self._score(candidate_query)
            candidate_objective = self._selection_objective(
                candidate_score,
                x0,
                candidate_delta_z,
                c_base=c_base,
                delta_c=candidate_c,
                region=region_state.region,
            )
            candidate_preds = candidate_query.pred
            candidate_norm = torch.linalg.norm(candidate_delta_z, dim=1)
            success_mask = candidate_preds != int(label)

            improved = False
            if bool(success_mask.any().item()):
                success_priority = self._success_priority_value(
                    x_ref=x0,
                    delta_z=candidate_delta_z,
                    c_base=c_base,
                    delta_c=candidate_c,
                    region=region_state.region,
                )
                success_idx = int(
                    torch.argmin(success_priority.masked_fill(~success_mask, float("inf"))).item()
                )
                success_candidate_priority = float(success_priority[success_idx].item())
                current_success_priority = float("inf")
                if region_state.success_delta_z is not None:
                    current_success_priority = float(
                        self._success_priority_value(
                            x_ref=x0,
                            delta_z=region_state.success_delta_z,
                            c_base=c_base,
                            delta_c=region_state.success_delta_c,
                            region=region_state.region,
                        )[0].item()
                    )
                
                if (
                    region_state.success_adv is None
                    or success_candidate_priority + 1e-8 < current_success_priority
                ):
                    region_state.success_adv = candidate_adv[success_idx : success_idx + 1].clone()
                    region_state.success_delta_c = candidate_c[success_idx : success_idx + 1].clone()
                    region_state.success_delta_z = candidate_delta_z[success_idx : success_idx + 1].clone()
                    region_state.success_objective = float(candidate_objective[success_idx].item())
                    region_state.success_score = float(candidate_score[success_idx].item())
                    region_state.success_pred = int(candidate_preds[success_idx].item())
                    region_state.best_adv = candidate_adv[success_idx : success_idx + 1].clone()
                    region_state.best_delta_c = candidate_c[success_idx : success_idx + 1].clone()
                    region_state.best_delta_z = candidate_delta_z[success_idx : success_idx + 1].clone()
                    region_state.best_objective = float(candidate_objective[success_idx].item())
                    region_state.best_score = float(candidate_score[success_idx].item())
                    region_state.best_pred = int(candidate_preds[success_idx].item())
                    improved = True
                if self._should_early_stop_on_success():
                    return True

            best_idx = int(torch.argmin(candidate_objective).item())
            if float(candidate_objective[best_idx].item()) + 1e-8 < region_state.best_objective:
                region_state.best_objective = float(candidate_objective[best_idx].item())
                region_state.best_score = float(candidate_score[best_idx].item())
                region_state.best_pred = int(candidate_preds[best_idx].item())
                region_state.best_delta_c = candidate_c[best_idx : best_idx + 1].clone()
                region_state.best_delta_z = candidate_delta_z[best_idx : best_idx + 1].clone()
                region_state.best_adv = candidate_adv[best_idx : best_idx + 1].clone()
                improved = True

            if improved:
                region_state.no_improve = 0
                region_state.step_shrinks = 0
            else:
                region_state.no_improve += 1

            if region_state.no_improve >= self.config.patience:
                new_step = max(
                    region_state.step * self.config.step_decay,
                    region_state.min_step,
                )
                if (
                    abs(new_step - region_state.step) <= 1e-12
                    and region_state.step <= region_state.min_step + 1e-12
                ):
                    region_state.step_shrinks += 1
                else:
                    region_state.step = new_step
                    region_state.step_shrinks += 1
                region_state.no_improve = 0
                if (
                    region_state.step <= region_state.min_step + 1e-12
                    and region_state.step_shrinks >= self.config.max_step_shrinks
                ):
                    break

        return False

    def _search_state_bandit_rounds(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        region_state: _RegionSearchState,
        generator: torch.Generator | None,
        max_rounds: int,
    ) -> bool:
        rounds_to_run = max(0, int(max_rounds))
        momentum_coeff = None
        beta = min(max(float(self.config.bandit_momentum), 0.0), 0.99)
        exploration_ratio = max(float(self.config.bandit_exploration_ratio), 0.1)

        for _ in range(rounds_to_run):
            region_state.rounds_used += 1
            previous_best_score = float(region_state.best_score)
            active_population = self._resolve_search_population(region_state)
            basis = self._state_basis_matrix(
                region_state=region_state,
                c_base=c_base,
                max_vectors=self._state_basis_max_vectors(max(2, active_population)),
            )
            coeff_dim = int(basis.shape[0])
            direction_samples = max(
                1,
                min(int(self.config.bandit_direction_samples), coeff_dim),
            )
            coeff_noise = torch.randn(
                direction_samples,
                coeff_dim,
                dtype=torch.float32,
                generator=generator,
            )
            coeff_noise = self._normalize_rows(coeff_noise)
            search_dirs = self._normalize_rows(coeff_noise @ basis)

            sigma = max(region_state.step * exploration_ratio, region_state.min_step)
            probe_c = torch.cat(
                [
                    region_state.best_delta_c + sigma * search_dirs,
                    region_state.best_delta_c - sigma * search_dirs,
                ],
                dim=0,
            )
            probe_c = _project_l2(probe_c, region_state.radius)
            probe_delta_z = self._project_region(
                region=region_state.region,
                delta_region=probe_c,
                use_search_shaping=True,
            )
            probe_c, probe_delta_z = self._apply_measurement_l2_cap(
                x_ref=x0,
                delta_c=probe_c,
                delta_z=probe_delta_z,
            )
            probe_adv = x0 + probe_delta_z
            if not self._has_query_budget_for(int(probe_adv.shape[0])):
                region_state.query_cap_reached = True
                region_state.early_stopped = True
                return True
            probe_query = self.oracle.query(probe_adv)
            probe_score = self._score(probe_query)
            probe_objective = self._selection_objective(
                probe_score,
                x0,
                probe_delta_z,
                c_base=c_base,
                delta_c=probe_c,
                region=region_state.region,
            )

            obj_plus = probe_objective[:direction_samples]
            obj_minus = probe_objective[direction_samples:]
            grad_coeff = (
                ((obj_plus - obj_minus) / max(2.0 * sigma, 1e-8)).reshape(-1, 1) * coeff_noise
            ).mean(dim=0, keepdim=True)
            if momentum_coeff is None or momentum_coeff.shape[1] != grad_coeff.shape[1]:
                momentum_coeff = grad_coeff.clone()
            else:
                momentum_coeff = beta * momentum_coeff + (1.0 - beta) * grad_coeff

            update_dir = momentum_coeff @ basis
            if float(torch.linalg.norm(update_dir).item()) <= 1e-12:
                region_state.no_improve += 1
                continue
            update_dir = self._normalize_rows(update_dir)

            line_scales = [
                max(region_state.step, region_state.min_step),
                max(region_state.step * self.config.step_decay, region_state.min_step),
            ]
            candidate_c = torch.cat(
                [
                    region_state.best_delta_c - line_scales[0] * update_dir,
                    region_state.best_delta_c - line_scales[1] * update_dir,
                    region_state.best_delta_c + 0.5 * line_scales[1] * update_dir,
                ],
                dim=0,
            )
            candidate_c = _project_l2(candidate_c, region_state.radius)
            candidate_delta_z = self._project_region(
                region=region_state.region,
                delta_region=candidate_c,
                use_search_shaping=True,
            )
            candidate_c, candidate_delta_z = self._apply_measurement_l2_cap(
                x_ref=x0,
                delta_c=candidate_c,
                delta_z=candidate_delta_z,
            )
            candidate_adv = x0 + candidate_delta_z
            if not self._has_query_budget_for(int(candidate_adv.shape[0])):
                region_state.query_cap_reached = True
                region_state.early_stopped = True
                return True
            candidate_query = self.oracle.query(candidate_adv)
            candidate_score = self._score(candidate_query)
            candidate_objective = self._selection_objective(
                candidate_score,
                x0,
                candidate_delta_z,
                c_base=c_base,
                delta_c=candidate_c,
                region=region_state.region,
            )
            candidate_preds = candidate_query.pred
            success_mask = candidate_preds != int(label)

            improved = False
            if bool(success_mask.any().item()):
                success_priority = self._success_priority_value(
                    x_ref=x0,
                    delta_z=candidate_delta_z,
                    c_base=c_base,
                    delta_c=candidate_c,
                    region=region_state.region,
                )
                success_idx = int(
                    torch.argmin(success_priority.masked_fill(~success_mask, float("inf"))).item()
                )
                success_candidate_priority = float(success_priority[success_idx].item())
                current_success_priority = float("inf")
                if region_state.success_delta_z is not None:
                    current_success_priority = float(
                        self._success_priority_value(
                            x_ref=x0,
                            delta_z=region_state.success_delta_z,
                            c_base=c_base,
                            delta_c=region_state.success_delta_c,
                            region=region_state.region,
                        )[0].item()
                    )

                if (
                    region_state.success_adv is None
                    or success_candidate_priority + 1e-8 < current_success_priority
                ):
                    region_state.success_adv = candidate_adv[success_idx : success_idx + 1].clone()
                    region_state.success_delta_c = candidate_c[success_idx : success_idx + 1].clone()
                    region_state.success_delta_z = candidate_delta_z[
                        success_idx : success_idx + 1
                    ].clone()
                    region_state.success_objective = float(candidate_objective[success_idx].item())
                    region_state.success_score = float(candidate_score[success_idx].item())
                    region_state.success_pred = int(candidate_preds[success_idx].item())
                    region_state.best_adv = candidate_adv[success_idx : success_idx + 1].clone()
                    region_state.best_delta_c = candidate_c[success_idx : success_idx + 1].clone()
                    region_state.best_delta_z = candidate_delta_z[
                        success_idx : success_idx + 1
                    ].clone()
                    region_state.best_objective = float(candidate_objective[success_idx].item())
                    region_state.best_score = float(candidate_score[success_idx].item())
                    region_state.best_pred = int(candidate_preds[success_idx].item())
                    improved = True
                if self._should_early_stop_on_success():
                    return True

            best_idx = int(torch.argmin(candidate_objective).item())
            if float(candidate_objective[best_idx].item()) + 1e-8 < region_state.best_objective:
                region_state.best_objective = float(candidate_objective[best_idx].item())
                region_state.best_score = float(candidate_score[best_idx].item())
                region_state.best_pred = int(candidate_preds[best_idx].item())
                region_state.best_delta_c = candidate_c[best_idx : best_idx + 1].clone()
                region_state.best_delta_z = candidate_delta_z[best_idx : best_idx + 1].clone()
                region_state.best_adv = candidate_adv[best_idx : best_idx + 1].clone()
                improved = True

            if improved:
                region_state.no_improve = 0
                region_state.step_shrinks = 0
            else:
                region_state.no_improve += 1
            self._update_score_stagnation(region_state, previous_best_score)

            if region_state.no_improve >= self.config.patience:
                new_step = max(
                    region_state.step * self.config.step_decay,
                    region_state.min_step,
                )
                if (
                    abs(new_step - region_state.step) <= 1e-12
                    and region_state.step <= region_state.min_step + 1e-12
                ):
                    region_state.step_shrinks += 1
                else:
                    region_state.step = new_step
                    region_state.step_shrinks += 1
                region_state.no_improve = 0
                if (
                    region_state.step <= region_state.min_step + 1e-12
                    and region_state.step_shrinks >= self.config.max_step_shrinks
                ):
                    break
            if self._should_run_guarded_boundary_probe(region_state, c_base):
                if self._run_guarded_boundary_probe(
                    x0=x0,
                    label=label,
                    c_base=c_base,
                    region_state=region_state,
                ):
                    return True
            if self._should_search_early_stop(region_state, c_base):
                break

        return False

    def _search_measurement_basis_rounds(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        region_state: _RegionSearchState,
        max_rounds: int,
    ) -> bool:
        rounds_to_run = max(0, int(max_rounds))
        for _ in range(rounds_to_run):
            region_state.rounds_used += 1
            active_population = self._resolve_search_population(region_state)
            basis_dirs = self._measurement_basis_directions(
                region_state=region_state,
                x_ref=x0,
                active_population=active_population,
            )
            improved = False

            for direction in basis_dirs:
                direction = direction.reshape(1, -1).to(dtype=torch.float32)
                candidates_c = torch.cat(
                    [
                        region_state.best_delta_c + region_state.step * direction,
                        region_state.best_delta_c - region_state.step * direction,
                    ],
                    dim=0,
                )
                candidates_c = _project_l2(candidates_c, region_state.radius)
                candidates_c = self._apply_state_backbone_lock(
                    c_base=c_base,
                    region=region_state.region,
                    delta_c=candidates_c,
                )
                delta_z = self._project_region(
                    region=region_state.region,
                    delta_region=candidates_c,
                    use_search_shaping=True,
                )
                candidates_c, delta_z = self._apply_measurement_l2_cap(
                    x_ref=x0,
                    delta_c=candidates_c,
                    delta_z=delta_z,
                )
                adv_batch = x0 + delta_z
                if not self._has_query_budget_for(int(adv_batch.shape[0])):
                    region_state.query_cap_reached = True
                    region_state.early_stopped = True
                    return True
                query = self.oracle.query(adv_batch)
                preds = query.pred
                score = self._score(query)
                objective = self._selection_objective(
                    score,
                    x0,
                    delta_z,
                    c_base=c_base,
                    delta_c=candidates_c,
                    region=region_state.region,
                )
                delta_norm = torch.linalg.norm(delta_z, dim=1)
                success_mask = preds != int(label)

                if bool(success_mask.any().item()):
                    success_priority = self._success_priority_value(
                        x_ref=x0,
                        delta_z=delta_z,
                        c_base=c_base,
                        delta_c=candidates_c,
                        region=region_state.region,
                    )
                    success_idx = int(
                        torch.argmin(success_priority.masked_fill(~success_mask, float("inf"))).item()
                    )
                    success_candidate_priority = float(success_priority[success_idx].item())
                    current_success_priority = float("inf")
                    if region_state.success_delta_z is not None:
                        current_success_priority = float(
                            self._success_priority_value(
                                x_ref=x0,
                                delta_z=region_state.success_delta_z,
                                c_base=c_base,
                                delta_c=region_state.success_delta_c,
                                region=region_state.region,
                            )[0].item()
                        )
                    if (
                        region_state.success_adv is None
                        or success_candidate_priority + 1e-8 < current_success_priority
                    ):
                        region_state.success_adv = adv_batch[success_idx : success_idx + 1].clone()
                        region_state.success_delta_c = candidates_c[
                            success_idx : success_idx + 1
                        ].clone()
                        region_state.success_delta_z = delta_z[
                            success_idx : success_idx + 1
                        ].clone()
                        region_state.success_objective = float(objective[success_idx].item())
                        region_state.success_score = float(score[success_idx].item())
                        region_state.success_pred = int(preds[success_idx].item())
                        region_state.best_adv = adv_batch[success_idx : success_idx + 1].clone()
                        region_state.best_delta_c = candidates_c[success_idx : success_idx + 1].clone()
                        region_state.best_delta_z = delta_z[success_idx : success_idx + 1].clone()
                        region_state.best_objective = float(objective[success_idx].item())
                        region_state.best_score = float(score[success_idx].item())
                        region_state.best_pred = int(preds[success_idx].item())
                        improved = True
                    if self._should_early_stop_on_success():
                        return True

                best_idx = int(torch.argmin(objective).item())
                candidate_objective = float(objective[best_idx].item())
                if candidate_objective + 1e-8 < region_state.best_objective:
                    region_state.best_objective = candidate_objective
                    region_state.best_score = float(score[best_idx].item())
                    region_state.best_pred = int(preds[best_idx].item())
                    region_state.best_delta_c = candidates_c[best_idx : best_idx + 1].clone()
                    region_state.best_delta_z = delta_z[best_idx : best_idx + 1].clone()
                    region_state.best_adv = adv_batch[best_idx : best_idx + 1].clone()
                    improved = True

            if improved:
                region_state.no_improve = 0
                region_state.step_shrinks = 0
                region_state.step = max(
                    region_state.step * max(float(self.config.step_decay), 0.85),
                    region_state.min_step,
                )
            else:
                region_state.no_improve += 1
                if region_state.no_improve >= self.config.patience:
                    new_step = max(
                        region_state.step * self.config.step_decay,
                        region_state.min_step,
                    )
                    if (
                        abs(new_step - region_state.step) <= 1e-12
                        and region_state.step <= region_state.min_step + 1e-12
                    ):
                        region_state.step_shrinks += 1
                    else:
                        region_state.step = new_step
                        region_state.step_shrinks += 1
                    region_state.no_improve = 0
                    if (
                        region_state.step <= region_state.min_step + 1e-12
                        and region_state.step_shrinks >= self.config.max_step_shrinks
                    ):
                        break

        return False

    def _search_state_subspace_pgzoo_rounds(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        region_state: _RegionSearchState,
        generator: torch.Generator | None,
        max_rounds: int,
    ) -> bool:
        if not (
            bool(self.config.pgzoo_structured_covariance)
            or bool(self.config.pgzoo_physical_preconditioner)
        ):
            return self._search_state_subspace_pgzoo_legacy_rounds(
                x0=x0,
                label=label,
                c_base=c_base,
                region_state=region_state,
                generator=generator,
                max_rounds=max_rounds,
            )
        rounds_to_run = max(0, int(max_rounds))
        if rounds_to_run <= 0:
            return False
        use_budget_geometry = self._uses_state_budget_geometry()
        momentum_coeff = None
        beta = min(max(float(self.config.pgzoo_momentum), 0.0), 0.99)
        alpha_ratio = max(float(self.config.pgzoo_alpha_ratio), 0.1)
        # Align PG-ZOO with the query budget of the original population search.
        query_budget_units = rounds_to_run * max(
            1,
            int(self._resolve_search_population(region_state)),
        )
        budget_used = 0
        region_covariance, _ = self._state_region_gmrf_covariance(
            c_base=c_base,
            region_state=region_state,
        )
        region_preconditioner = self._state_region_preconditioner(region_state)

        for _ in range(rounds_to_run):
            active_population = self._resolve_search_population(region_state)
            basis = self._state_basis_matrix(
                region_state=region_state,
                c_base=c_base,
                max_vectors=self._state_basis_max_vectors(max(2, active_population)),
            )
            q_basis = self._orthonormalize_rows(basis)
            coeff_dim = int(q_basis.shape[0])
            if coeff_dim <= 0:
                region_state.no_improve += 1
                continue
            probe_pairs = max(
                1,
                min(int(self.config.pgzoo_probe_pairs), coeff_dim),
            )
            line_scales_count = max(1, min(int(self.config.pgzoo_line_candidates), 3))
            candidate_count = line_scales_count + (1 if line_scales_count >= 2 else 0)
            round_query_cost = 2 * probe_pairs + candidate_count
            if budget_used > 0 and budget_used + round_query_cost > query_budget_units:
                break
            budget_used += round_query_cost
            region_state.rounds_used += 1
            previous_best_score = float(region_state.best_score)
            subspace_factor, sampling_scale = self._state_subspace_covariance_factor(
                region_covariance=region_covariance,
                row_basis=q_basis,
            )
            white_noise = torch.randn(
                probe_pairs,
                coeff_dim,
                dtype=torch.float32,
                generator=generator,
            )
            coeff_noise = white_noise @ subspace_factor.T
            search_dirs = sampling_scale * (coeff_noise @ q_basis)
            if use_budget_geometry:
                search_dirs = self._normalize_state_budget_rows(
                    region=region_state.region,
                    delta_c=search_dirs,
                    use_search_shaping=True,
                )

            sigma = max(region_state.step * alpha_ratio, region_state.min_step)
            probe_c = torch.cat(
                [
                    region_state.best_delta_c + sigma * search_dirs,
                    region_state.best_delta_c - sigma * search_dirs,
                ],
                dim=0,
            )
            if use_budget_geometry:
                probe_c, probe_delta_z = self._project_state_budget_ball(
                    x_ref=x0,
                    region=region_state.region,
                    delta_c=probe_c,
                    use_search_shaping=True,
                    cap_override=region_state.radius,
                )
            else:
                probe_c = _project_l2(probe_c, region_state.radius)
                probe_delta_z = self._project_region(
                    region=region_state.region,
                    delta_region=probe_c,
                    use_search_shaping=True,
                )
                probe_c, probe_delta_z = self._apply_measurement_l2_cap(
                    x_ref=x0,
                    delta_c=probe_c,
                    delta_z=probe_delta_z,
                )
            probe_adv = x0 + probe_delta_z
            probe_query = self.oracle.query(probe_adv)
            probe_score = self._score(probe_query)
            probe_objective = self._selection_objective(
                probe_score,
                x0,
                probe_delta_z,
                c_base=c_base,
                delta_c=probe_c,
                region=region_state.region,
            )
            should_stop, probe_improved = self._consume_search_batch(
                x0=x0,
                label=label,
                c_base=c_base,
                region_state=region_state,
                batch_c=probe_c,
                batch_delta_z=probe_delta_z,
                batch_adv=probe_adv,
                batch_score=probe_score,
                batch_objective=probe_objective,
                batch_preds=probe_query.pred,
            )
            if should_stop:
                return True

            obj_plus = probe_objective[:probe_pairs]
            obj_minus = probe_objective[probe_pairs:]
            grad_coeff = (
                ((obj_plus - obj_minus) / max(2.0 * sigma, 1e-8)).reshape(-1, 1) * white_noise
            ).mean(dim=0, keepdim=True)
            if momentum_coeff is None or momentum_coeff.shape[1] != grad_coeff.shape[1]:
                momentum_coeff = grad_coeff.clone()
            else:
                momentum_coeff = beta * momentum_coeff + (1.0 - beta) * grad_coeff

            if float(torch.linalg.norm(momentum_coeff).item()) <= 1e-12:
                region_state.no_improve += 1
                continue

            linear_operator = sampling_scale * (subspace_factor.T @ q_basis)
            grad_region = momentum_coeff @ linear_operator
            grad_region = grad_region @ region_preconditioner
            grad_region = self._project_to_row_space(grad_region, q_basis)
            if float(torch.linalg.norm(grad_region).item()) <= 1e-12:
                region_state.no_improve += 1
                continue
            if use_budget_geometry:
                update_dir = self._normalize_state_budget_rows(
                    region=region_state.region,
                    delta_c=grad_region,
                    use_search_shaping=True,
                )
            else:
                update_dir = self._normalize_rows(grad_region)

            line_scales = [max(region_state.step, region_state.min_step)]
            if line_scales_count >= 2:
                line_scales.append(
                    max(region_state.step * self.config.step_decay, region_state.min_step)
                )
            if line_scales_count >= 3:
                line_scales.append(
                    max(0.5 * region_state.step * self.config.step_decay, region_state.min_step)
                )

            candidate_rows = [region_state.best_delta_c - scale * update_dir for scale in line_scales]
            if len(line_scales) >= 2:
                candidate_rows.append(region_state.best_delta_c + 0.5 * line_scales[-1] * update_dir)
            candidate_c = torch.cat(candidate_rows, dim=0)
            if use_budget_geometry:
                candidate_c, candidate_delta_z = self._project_state_budget_ball(
                    x_ref=x0,
                    region=region_state.region,
                    delta_c=candidate_c,
                    use_search_shaping=True,
                    cap_override=region_state.radius,
                )
            else:
                candidate_c = _project_l2(candidate_c, region_state.radius)
                candidate_delta_z = self._project_region(
                    region=region_state.region,
                    delta_region=candidate_c,
                    use_search_shaping=True,
                )
                candidate_c, candidate_delta_z = self._apply_measurement_l2_cap(
                    x_ref=x0,
                    delta_c=candidate_c,
                    delta_z=candidate_delta_z,
                )
            candidate_adv = x0 + candidate_delta_z
            candidate_query = self.oracle.query(candidate_adv)
            candidate_score = self._score(candidate_query)
            candidate_objective = self._selection_objective(
                candidate_score,
                x0,
                candidate_delta_z,
                c_base=c_base,
                delta_c=candidate_c,
                region=region_state.region,
            )
            candidate_preds = candidate_query.pred
            success_mask = candidate_preds != int(label)

            improved = bool(probe_improved)
            if bool(success_mask.any().item()):
                success_priority = self._success_candidate_value(
                    x_ref=x0,
                    objective=candidate_objective,
                    delta_z=candidate_delta_z,
                    c_base=c_base,
                    delta_c=candidate_c,
                    region=region_state.region,
                )
                success_idx = int(
                    torch.argmin(success_priority.masked_fill(~success_mask, float("inf"))).item()
                )
                success_candidate_priority = float(success_priority[success_idx].item())
                current_success_priority = float("inf")
                if region_state.success_delta_z is not None:
                    current_success_priority = float(
                        self._success_candidate_value(
                            x_ref=x0,
                            objective=torch.tensor(
                                [float(region_state.success_objective)],
                                dtype=torch.float32,
                            ),
                            delta_z=region_state.success_delta_z,
                            c_base=c_base,
                            delta_c=region_state.success_delta_c,
                            region=region_state.region,
                        )[0].item()
                    )
                if (
                    region_state.success_adv is None
                    or success_candidate_priority + 1e-8 < current_success_priority
                ):
                    region_state.success_adv = candidate_adv[success_idx : success_idx + 1].clone()
                    region_state.success_delta_c = candidate_c[success_idx : success_idx + 1].clone()
                    region_state.success_delta_z = candidate_delta_z[
                        success_idx : success_idx + 1
                    ].clone()
                    region_state.success_objective = float(candidate_objective[success_idx].item())
                    region_state.success_score = float(candidate_score[success_idx].item())
                    region_state.success_pred = int(candidate_preds[success_idx].item())
                    region_state.best_adv = candidate_adv[success_idx : success_idx + 1].clone()
                    region_state.best_delta_c = candidate_c[success_idx : success_idx + 1].clone()
                    region_state.best_delta_z = candidate_delta_z[
                        success_idx : success_idx + 1
                    ].clone()
                    region_state.best_objective = float(candidate_objective[success_idx].item())
                    region_state.best_score = float(candidate_score[success_idx].item())
                    region_state.best_pred = int(candidate_preds[success_idx].item())
                    improved = True
                if self._should_early_stop_on_success():
                    return True

            best_idx = int(torch.argmin(candidate_objective).item())
            if float(candidate_objective[best_idx].item()) + 1e-8 < region_state.best_objective:
                region_state.best_objective = float(candidate_objective[best_idx].item())
                region_state.best_score = float(candidate_score[best_idx].item())
                region_state.best_pred = int(candidate_preds[best_idx].item())
                region_state.best_delta_c = candidate_c[best_idx : best_idx + 1].clone()
                region_state.best_delta_z = candidate_delta_z[best_idx : best_idx + 1].clone()
                region_state.best_adv = candidate_adv[best_idx : best_idx + 1].clone()
                improved = True

            if improved:
                region_state.no_improve = 0
                region_state.step_shrinks = 0
            else:
                region_state.no_improve += 1
            self._update_score_stagnation(region_state, previous_best_score)

            if region_state.no_improve >= self.config.patience:
                new_step = max(
                    region_state.step * self.config.step_decay,
                    region_state.min_step,
                )
                if (
                    abs(new_step - region_state.step) <= 1e-12
                    and region_state.step <= region_state.min_step + 1e-12
                ):
                    region_state.step_shrinks += 1
                else:
                    region_state.step = new_step
                    region_state.step_shrinks += 1
                region_state.no_improve = 0
                if (
                    region_state.step <= region_state.min_step + 1e-12
                    and region_state.step_shrinks >= self.config.max_step_shrinks
                ):
                    break
            if self._should_search_early_stop(region_state, c_base):
                break

        return False

    def _search_state_subspace_pgzoo_legacy_rounds(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        region_state: _RegionSearchState,
        generator: torch.Generator | None,
        max_rounds: int,
    ) -> bool:
        rounds_to_run = max(0, int(max_rounds))
        if rounds_to_run <= 0:
            return False
        momentum_coeff = None
        beta = min(max(float(self.config.pgzoo_momentum), 0.0), 0.99)
        alpha_ratio = max(float(self.config.pgzoo_alpha_ratio), 0.1)
        query_budget_units = rounds_to_run * max(
            1,
            int(self._resolve_search_population(region_state)),
        )
        budget_used = 0

        for _ in range(rounds_to_run):
            active_population = self._resolve_search_population(region_state)
            basis = self._state_basis_matrix(
                region_state=region_state,
                c_base=c_base,
                max_vectors=self._state_basis_max_vectors(max(2, active_population)),
            )
            coeff_dim = int(basis.shape[0])
            if coeff_dim <= 0:
                region_state.no_improve += 1
                continue
            probe_pairs = max(
                1,
                min(int(self.config.pgzoo_probe_pairs), coeff_dim),
            )
            line_scales_count = max(1, min(int(self.config.pgzoo_line_candidates), 3))
            candidate_count = line_scales_count + (1 if line_scales_count >= 2 else 0)
            round_query_cost = 2 * probe_pairs + candidate_count
            if budget_used > 0 and budget_used + round_query_cost > query_budget_units:
                break
            budget_used += round_query_cost
            region_state.rounds_used += 1
            previous_best_score = float(region_state.best_score)
            coeff_noise = torch.randn(
                probe_pairs,
                coeff_dim,
                dtype=torch.float32,
                generator=generator,
            )
            prior_weights = self._state_subspace_prior_weights(
                x0=x0,
                c_base=c_base,
                region_state=region_state,
                basis=basis,
            )
            coeff_noise = coeff_noise * prior_weights.sqrt().reshape(1, -1)
            coeff_noise = self._normalize_rows(coeff_noise)
            search_dirs = self._normalize_rows(coeff_noise @ basis)

            sigma = max(region_state.step * alpha_ratio, region_state.min_step)
            probe_c = torch.cat(
                [
                    region_state.best_delta_c + sigma * search_dirs,
                    region_state.best_delta_c - sigma * search_dirs,
                ],
                dim=0,
            )
            probe_c = _project_l2(probe_c, region_state.radius)
            probe_delta_z = self._project_region(
                region=region_state.region,
                delta_region=probe_c,
                use_search_shaping=True,
            )
            probe_c, probe_delta_z = self._apply_measurement_l2_cap(
                x_ref=x0,
                delta_c=probe_c,
                delta_z=probe_delta_z,
            )
            probe_adv = x0 + probe_delta_z
            probe_query = self.oracle.query(probe_adv)
            probe_score = self._score(probe_query)
            probe_objective = self._selection_objective(
                probe_score,
                x0,
                probe_delta_z,
                c_base=c_base,
                delta_c=probe_c,
                region=region_state.region,
            )
            should_stop, probe_improved = self._consume_search_batch(
                x0=x0,
                label=label,
                c_base=c_base,
                region_state=region_state,
                batch_c=probe_c,
                batch_delta_z=probe_delta_z,
                batch_adv=probe_adv,
                batch_score=probe_score,
                batch_objective=probe_objective,
                batch_preds=probe_query.pred,
            )
            if should_stop:
                return True

            obj_plus = probe_objective[:probe_pairs]
            obj_minus = probe_objective[probe_pairs:]
            grad_coeff = (
                ((obj_plus - obj_minus) / max(2.0 * sigma, 1e-8)).reshape(-1, 1) * coeff_noise
            ).mean(dim=0, keepdim=True)
            if momentum_coeff is None or momentum_coeff.shape[1] != grad_coeff.shape[1]:
                momentum_coeff = grad_coeff.clone()
            else:
                momentum_coeff = beta * momentum_coeff + (1.0 - beta) * grad_coeff

            if float(torch.linalg.norm(momentum_coeff).item()) <= 1e-12:
                region_state.no_improve += 1
                continue

            update_coeff = self._normalize_rows(momentum_coeff)
            update_dir = self._normalize_rows(update_coeff @ basis)

            line_scales = [max(region_state.step, region_state.min_step)]
            if line_scales_count >= 2:
                line_scales.append(
                    max(region_state.step * self.config.step_decay, region_state.min_step)
                )
            if line_scales_count >= 3:
                line_scales.append(
                    max(0.5 * region_state.step * self.config.step_decay, region_state.min_step)
                )

            candidate_rows = [region_state.best_delta_c - scale * update_dir for scale in line_scales]
            if len(line_scales) >= 2:
                candidate_rows.append(region_state.best_delta_c + 0.5 * line_scales[-1] * update_dir)
            candidate_c = torch.cat(candidate_rows, dim=0)
            candidate_c = _project_l2(candidate_c, region_state.radius)
            candidate_delta_z = self._project_region(
                region=region_state.region,
                delta_region=candidate_c,
                use_search_shaping=True,
            )
            candidate_c, candidate_delta_z = self._apply_measurement_l2_cap(
                x_ref=x0,
                delta_c=candidate_c,
                delta_z=candidate_delta_z,
            )
            candidate_adv = x0 + candidate_delta_z
            candidate_query = self.oracle.query(candidate_adv)
            candidate_score = self._score(candidate_query)
            candidate_objective = self._selection_objective(
                candidate_score,
                x0,
                candidate_delta_z,
                c_base=c_base,
                delta_c=candidate_c,
                region=region_state.region,
            )
            candidate_preds = candidate_query.pred
            success_mask = candidate_preds != int(label)

            improved = bool(probe_improved)
            if bool(success_mask.any().item()):
                success_priority = self._success_priority_value(
                    x_ref=x0,
                    delta_z=candidate_delta_z,
                    c_base=c_base,
                    delta_c=candidate_c,
                    region=region_state.region,
                )
                success_idx = int(
                    torch.argmin(success_priority.masked_fill(~success_mask, float("inf"))).item()
                )
                success_candidate_priority = float(success_priority[success_idx].item())
                current_success_priority = float("inf")
                if region_state.success_delta_z is not None:
                    current_success_priority = float(
                        self._success_priority_value(
                            x_ref=x0,
                            delta_z=region_state.success_delta_z,
                            c_base=c_base,
                            delta_c=region_state.success_delta_c,
                            region=region_state.region,
                        )[0].item()
                    )
                if (
                    region_state.success_adv is None
                    or success_candidate_priority + 1e-8 < current_success_priority
                ):
                    region_state.success_adv = candidate_adv[success_idx : success_idx + 1].clone()
                    region_state.success_delta_c = candidate_c[success_idx : success_idx + 1].clone()
                    region_state.success_delta_z = candidate_delta_z[
                        success_idx : success_idx + 1
                    ].clone()
                    region_state.success_objective = float(candidate_objective[success_idx].item())
                    region_state.success_score = float(candidate_score[success_idx].item())
                    region_state.success_pred = int(candidate_preds[success_idx].item())
                    region_state.best_adv = candidate_adv[success_idx : success_idx + 1].clone()
                    region_state.best_delta_c = candidate_c[success_idx : success_idx + 1].clone()
                    region_state.best_delta_z = candidate_delta_z[
                        success_idx : success_idx + 1
                    ].clone()
                    region_state.best_objective = float(candidate_objective[success_idx].item())
                    region_state.best_score = float(candidate_score[success_idx].item())
                    region_state.best_pred = int(candidate_preds[success_idx].item())
                    improved = True
                if self._should_early_stop_on_success():
                    return True

            best_idx = int(torch.argmin(candidate_objective).item())
            if float(candidate_objective[best_idx].item()) + 1e-8 < region_state.best_objective:
                region_state.best_objective = float(candidate_objective[best_idx].item())
                region_state.best_score = float(candidate_score[best_idx].item())
                region_state.best_pred = int(candidate_preds[best_idx].item())
                region_state.best_delta_c = candidate_c[best_idx : best_idx + 1].clone()
                region_state.best_delta_z = candidate_delta_z[best_idx : best_idx + 1].clone()
                region_state.best_adv = candidate_adv[best_idx : best_idx + 1].clone()
                improved = True

            if improved:
                region_state.no_improve = 0
                region_state.step_shrinks = 0
            else:
                region_state.no_improve += 1
            self._update_score_stagnation(region_state, previous_best_score)

            if region_state.no_improve >= self.config.patience:
                new_step = max(
                    region_state.step * self.config.step_decay,
                    region_state.min_step,
                )
                if (
                    abs(new_step - region_state.step) <= 1e-12
                    and region_state.step <= region_state.min_step + 1e-12
                ):
                    region_state.step_shrinks += 1
                else:
                    region_state.step = new_step
                    region_state.step_shrinks += 1
                region_state.no_improve = 0
                if (
                    region_state.step <= region_state.min_step + 1e-12
                    and region_state.step_shrinks >= self.config.max_step_shrinks
                ):
                    break
            if self._should_search_early_stop(region_state, c_base):
                break

        return False

    def _search_state_basis_rounds(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        region_state: _RegionSearchState,
        max_rounds: int,
    ) -> bool:
        rounds_to_run = max(0, int(max_rounds))
        for _ in range(rounds_to_run):
            region_state.rounds_used += 1
            previous_best_score = float(region_state.best_score)
            active_population = self._resolve_search_population(region_state)
            basis_dirs = self._state_basis_directions(
                region_state=region_state,
                c_base=c_base,
                active_population=active_population,
            )
            improved = False

            for direction in basis_dirs:
                direction = direction.reshape(1, -1).to(dtype=torch.float32)
                candidates_c = torch.cat(
                    [
                        region_state.best_delta_c + region_state.step * direction,
                        region_state.best_delta_c - region_state.step * direction,
                    ],
                    dim=0,
                )
                candidates_c = _project_l2(candidates_c, region_state.radius)
                delta_z = self._project_region(
                    region=region_state.region,
                    delta_region=candidates_c,
                    use_search_shaping=True,
                )
                candidates_c, delta_z = self._apply_measurement_l2_cap(
                    x_ref=x0,
                    delta_c=candidates_c,
                    delta_z=delta_z,
                )
                adv_batch = x0 + delta_z
                query = self.oracle.query(adv_batch)
                preds = query.pred
                score = self._score(query)
                objective = self._selection_objective(
                    score,
                    x0,
                    delta_z,
                    c_base=c_base,
                    delta_c=candidates_c,
                    region=region_state.region,
                )
                success_mask = preds != int(label)

                if bool(success_mask.any().item()):
                    success_priority = self._success_priority_value(
                        x_ref=x0,
                        delta_z=delta_z,
                        c_base=c_base,
                        delta_c=candidates_c,
                        region=region_state.region,
                    )
                    success_idx = int(
                        torch.argmin(success_priority.masked_fill(~success_mask, float("inf"))).item()
                    )
                    success_candidate_priority = float(success_priority[success_idx].item())
                    current_success_priority = float("inf")
                    if region_state.success_delta_z is not None:
                        current_success_priority = float(
                            self._success_priority_value(
                                x_ref=x0,
                                delta_z=region_state.success_delta_z,
                                c_base=c_base,
                                delta_c=region_state.success_delta_c,
                                region=region_state.region,
                            )[0].item()
                        )
                    if (
                        region_state.success_adv is None
                        or success_candidate_priority + 1e-8 < current_success_priority
                    ):
                        region_state.success_adv = adv_batch[success_idx : success_idx + 1].clone()
                        region_state.success_delta_c = candidates_c[
                            success_idx : success_idx + 1
                        ].clone()
                        region_state.success_delta_z = delta_z[
                            success_idx : success_idx + 1
                        ].clone()
                        region_state.success_objective = float(objective[success_idx].item())
                        region_state.success_score = float(score[success_idx].item())
                        region_state.success_pred = int(preds[success_idx].item())
                        region_state.best_adv = adv_batch[success_idx : success_idx + 1].clone()
                        region_state.best_delta_c = candidates_c[success_idx : success_idx + 1].clone()
                        region_state.best_delta_z = delta_z[success_idx : success_idx + 1].clone()
                        region_state.best_objective = float(objective[success_idx].item())
                        region_state.best_score = float(score[success_idx].item())
                        region_state.best_pred = int(preds[success_idx].item())
                        improved = True
                    if self._should_early_stop_on_success():
                        return True

                best_idx = int(torch.argmin(objective).item())
                if float(objective[best_idx].item()) + 1e-8 < region_state.best_objective:
                    region_state.best_objective = float(objective[best_idx].item())
                    region_state.best_score = float(score[best_idx].item())
                    region_state.best_pred = int(preds[best_idx].item())
                    region_state.best_delta_c = candidates_c[best_idx : best_idx + 1].clone()
                    region_state.best_delta_z = delta_z[best_idx : best_idx + 1].clone()
                    region_state.best_adv = adv_batch[best_idx : best_idx + 1].clone()
                    improved = True

            if improved:
                region_state.no_improve = 0
                region_state.step_shrinks = 0
            else:
                region_state.no_improve += 1
            self._update_score_stagnation(region_state, previous_best_score)

            if region_state.no_improve >= self.config.patience:
                new_step = max(
                    region_state.step * self.config.step_decay,
                    region_state.min_step,
                )
                if (
                    abs(new_step - region_state.step) <= 1e-12
                    and region_state.step <= region_state.min_step + 1e-12
                ):
                    region_state.step_shrinks += 1
                else:
                    region_state.step = new_step
                    region_state.step_shrinks += 1
                region_state.no_improve = 0
                if (
                    region_state.step <= region_state.min_step + 1e-12
                    and region_state.step_shrinks >= self.config.max_step_shrinks
                ):
                    break
            if self._should_run_guarded_boundary_probe(region_state, c_base):
                if self._run_guarded_boundary_probe(
                    x0=x0,
                    label=label,
                    c_base=c_base,
                    region_state=region_state,
                ):
                    return True
            if self._should_search_early_stop(region_state, c_base):
                break

        return False

    def _build_fd_coordinate_eps(
        self,
        x0: torch.Tensor,
        c_base: torch.Tensor,
        region_state: _RegionSearchState,
    ) -> torch.Tensor:
        region_dim = max(1, int(region_state.region.numel()))
        base_region = self._region_base_vector(
            x0=x0.reshape(-1),
            c_base=c_base,
            region=region_state.region,
        ).abs().to(dtype=torch.float32)
        ref_scale = max(region_state.radius / (region_dim ** 0.5), 1e-6)
        ref_tensor = torch.full_like(base_region, ref_scale)
        eps = float(self.config.fd_coordinate_eps_ratio) * torch.maximum(base_region, ref_tensor)
        eps = eps.clamp(min=float(self.config.fd_coordinate_eps_floor))
        eps = eps.clamp(max=max(region_state.radius, float(self.config.fd_coordinate_eps_floor)))
        return eps

    def _search_region_fd_rounds(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        region_state: _RegionSearchState,
        max_rounds: int,
    ) -> bool:
        if self.config.query_mode == "decision":
            raise ValueError("Finite-difference baselines require score-based queries.")

        region_dim = int(region_state.region.numel())
        if region_dim <= 0:
            return False

        fd_rounds = max(0, int(max_rounds))
        line_steps = max(1, int(self.config.fd_line_steps))
        line_decay = min(max(float(self.config.fd_line_decay), 1e-3), 1.0)
        coord_eps = self._build_fd_coordinate_eps(
            x0=x0,
            c_base=c_base,
            region_state=region_state,
        )

        for _ in range(fd_rounds):
            region_state.rounds_used += 1
            base_delta_c = region_state.best_delta_c.reshape(1, -1).to(dtype=torch.float32)
            central_gradient = bool(self.config.fd_central_gradient)

            grad_candidates_plus = base_delta_c.repeat(region_dim, 1) + torch.diag(coord_eps)
            grad_candidates_plus = _project_l2(grad_candidates_plus, region_state.radius)
            grad_delta_z_plus = self._project_region(
                region=region_state.region,
                delta_region=grad_candidates_plus,
                use_search_shaping=True,
            )
            grad_candidates_plus, grad_delta_z_plus = self._apply_measurement_l2_cap(
                x_ref=x0,
                delta_c=grad_candidates_plus,
                delta_z=grad_delta_z_plus,
            )
            grad_adv_plus = x0 + grad_delta_z_plus
            grad_query_plus = self.oracle.query(grad_adv_plus)
            grad_score_plus = self._score(grad_query_plus)
            grad_objective_plus = self._selection_objective(
                grad_score_plus,
                x0,
                grad_delta_z_plus,
                c_base=c_base,
                delta_c=grad_candidates_plus,
                region=region_state.region,
            )

            if central_gradient:
                grad_candidates_minus = base_delta_c.repeat(region_dim, 1) - torch.diag(coord_eps)
                grad_candidates_minus = _project_l2(grad_candidates_minus, region_state.radius)
                grad_delta_z_minus = self._project_region(
                    region=region_state.region,
                    delta_region=grad_candidates_minus,
                    use_search_shaping=True,
                )
                grad_candidates_minus, grad_delta_z_minus = self._apply_measurement_l2_cap(
                    x_ref=x0,
                    delta_c=grad_candidates_minus,
                    delta_z=grad_delta_z_minus,
                )
                grad_adv_minus = x0 + grad_delta_z_minus
                grad_query_minus = self.oracle.query(grad_adv_minus)
                grad_score_minus = self._score(grad_query_minus)
                grad_objective_minus = self._selection_objective(
                    grad_score_minus,
                    x0,
                    grad_delta_z_minus,
                    c_base=c_base,
                    delta_c=grad_candidates_minus,
                    region=region_state.region,
                )
                step_plus = torch.diag(grad_candidates_plus - base_delta_c).abs()
                step_minus = torch.diag(base_delta_c - grad_candidates_minus).abs()
                denom = (step_plus + step_minus).clamp(min=1e-12)
                grad_est = (grad_objective_plus - grad_objective_minus) / denom
            else:
                actual_steps = grad_candidates_plus - base_delta_c
                denom = torch.diag(actual_steps).abs().clamp(min=1e-12)
                grad_est = (grad_objective_plus - float(region_state.best_objective)) / denom
            direction = -grad_est.reshape(1, -1)
            direction_norm = float(torch.linalg.norm(direction).item())
            if direction_norm <= 1e-12:
                break
            direction = self._normalize_rows(direction)

            step_scales = [
                max(region_state.step * (line_decay**idx), region_state.min_step)
                for idx in range(line_steps)
            ]
            line_candidates_list = []
            expanded_step_scales = []
            for step_scale in step_scales:
                for sign in (1.0, -1.0):
                    line_candidates_list.append(
                        _project_l2(
                            base_delta_c + float(sign * step_scale) * direction,
                            region_state.radius,
                        )
                    )
                    expanded_step_scales.append(float(step_scale))
            line_candidates = torch.cat(line_candidates_list, dim=0)
            line_delta_z = self._project_region(
                region=region_state.region,
                delta_region=line_candidates,
                use_search_shaping=True,
            )
            line_candidates, line_delta_z = self._apply_measurement_l2_cap(
                x_ref=x0,
                delta_c=line_candidates,
                delta_z=line_delta_z,
            )
            line_adv = x0 + line_delta_z
            line_query = self.oracle.query(line_adv)
            line_score = self._score(line_query)
            line_objective = self._selection_objective(
                line_score,
                x0,
                line_delta_z,
                c_base=c_base,
                delta_c=line_candidates,
                region=region_state.region,
            )
            line_preds = line_query.pred
            line_norm = torch.linalg.norm(line_delta_z, dim=1)
            success_mask = line_preds != int(label)

            if success_mask.any():
                success_priority = self._success_priority_value(
                    x_ref=x0,
                    delta_z=line_delta_z,
                    c_base=c_base,
                    delta_c=line_candidates,
                    region=region_state.region,
                )
                success_idx = int(
                    torch.argmin(success_priority.masked_fill(~success_mask, float("inf"))).item()
                )
                success_candidate_priority = float(success_priority[success_idx].item())
                current_success_priority = float("inf")
                if region_state.success_delta_z is not None:
                    current_success_priority = float(
                        self._success_priority_value(
                            x_ref=x0,
                            delta_z=region_state.success_delta_z,
                            c_base=c_base,
                            delta_c=region_state.success_delta_c,
                            region=region_state.region,
                        )[0].item()
                    )
                if (
                    region_state.success_adv is None
                    or success_candidate_priority < current_success_priority
                ):
                    region_state.success_adv = line_adv[success_idx : success_idx + 1].clone()
                    region_state.success_delta_c = line_candidates[
                        success_idx : success_idx + 1
                    ].clone()
                    region_state.success_delta_z = line_delta_z[
                        success_idx : success_idx + 1
                    ].clone()
                    region_state.success_objective = float(line_objective[success_idx].item())
                    region_state.success_score = float(line_score[success_idx].item())
                    region_state.success_pred = int(line_preds[success_idx].item())
                if self._should_early_stop_on_success():
                    return True

            best_idx = int(torch.argmin(line_objective).item())
            candidate_objective = float(line_objective[best_idx].item())
            if candidate_objective < float(region_state.best_objective):
                region_state.best_objective = candidate_objective
                region_state.best_score = float(line_score[best_idx].item())
                region_state.best_pred = int(line_preds[best_idx].item())
                region_state.best_delta_c = line_candidates[best_idx : best_idx + 1].clone()
                region_state.best_delta_z = line_delta_z[best_idx : best_idx + 1].clone()
                region_state.best_adv = line_adv[best_idx : best_idx + 1].clone()
                region_state.step = max(float(expanded_step_scales[best_idx]), region_state.min_step)
                region_state.no_improve = 0
            else:
                region_state.no_improve += 1
                region_state.step = max(region_state.step * line_decay, region_state.min_step)

            if region_state.no_improve >= self.config.patience:
                region_state.step_shrinks += 1
                region_state.no_improve = 0
                region_state.step = max(region_state.step * self.config.step_decay, region_state.min_step)
                if (
                    region_state.step <= region_state.min_step + 1e-12
                    and region_state.step_shrinks >= self.config.max_step_shrinks
                ):
                    break

        return False

    def _finalize_region_state(
        self,
        region_state: _RegionSearchState,
        start_queries: int,
        proposal_queries: int,
        candidate_count: int,
        budget_region_ranks: list[int],
        probe_summary: dict,
    ) -> dict:
        final_adv = region_state.success_adv if region_state.success_adv is not None else region_state.best_adv
        final_delta_c = (
            region_state.success_delta_c
            if region_state.success_delta_c is not None
            else region_state.best_delta_c
        )
        final_delta_z = (
            region_state.success_delta_z
            if region_state.success_delta_z is not None
            else region_state.best_delta_z
        )
        final_pred = (
            region_state.success_pred
            if region_state.success_pred is not None
            else region_state.best_pred
        )
        final_score = (
            region_state.success_score
            if region_state.success_score is not None
            else region_state.best_score
        )
        final_objective = (
            region_state.success_objective
            if region_state.success_objective is not None
            else region_state.best_objective
        )

        if self.config.query_mode == "decision" and region_state.success_adv is None:
            final_adv = region_state.clean_adv
            final_delta_c = region_state.clean_delta_c
            final_delta_z = region_state.clean_delta_z
            final_pred = region_state.clean_pred
            final_score = 0.0
            final_objective = 0.0

        return {
            "adv_x": final_adv.squeeze(0),
            "delta_c": final_delta_c.squeeze(0),
            "delta_z": final_delta_z.squeeze(0),
            "success": bool(region_state.success_adv is not None),
            "final_pred": int(final_pred),
            "final_score": float(final_score),
            "final_objective": float(final_objective),
            "region": region_state.region,
            "radius": float(region_state.radius),
            "queries_used": int(self.oracle.query_count - start_queries),
            "proposal_queries": int(proposal_queries),
            "selected_region_rank": int(region_state.rank),
            "region_candidate_count": int(candidate_count),
            "budget_region_ranks": [int(rank) for rank in budget_region_ranks],
            "budget_region_count": int(len(budget_region_ranks)),
            "budget_triggered": bool(len(budget_region_ranks) > 1),
            "clean_score": float(probe_summary["clean_score"]),
            "probe_best_score": float(probe_summary["probe_best_score"]),
            "probe_second_score": float(probe_summary["probe_second_score"]),
            "probe_score_gap": float(probe_summary["probe_score_gap"]),
            "probe_improvement": float(probe_summary["probe_improvement"]),
            "probe_best_rank": int(probe_summary["probe_best_rank"]),
            "probe_second_rank": int(probe_summary["probe_second_rank"]),
            "probe_best_prior": float(probe_summary["probe_best_prior"]),
            "probe_second_prior": float(probe_summary["probe_second_prior"]),
            "probe_success_count": int(probe_summary["probe_success_count"]),
            "detector_feedback_used": bool(probe_summary["detector_feedback_used"]),
            "detector_feedback_candidate_count": int(
                probe_summary["detector_feedback_candidate_count"]
            ),
            "detector_feedback_total_reward": float(
                probe_summary["detector_feedback_total_reward"]
            ),
            "detector_feedback_best_reward": float(
                probe_summary["detector_feedback_best_reward"]
            ),
            "probe_improvement_ratio": float(region_state.probe_improvement_ratio),
            "query_budget_population": int(region_state.query_budget_population),
            "query_budget_rounds": int(region_state.query_budget_rounds),
            "adaptive_query_budget_used": bool(region_state.query_budget_population > 0),
            "selected_region_physics_quality": float(region_state.physics_quality),
            "selected_region_allocation_priority": float(region_state.allocation_priority),
            "selected_region_progress_ratio": float(self._score_progress_ratio(region_state)),
            "selected_region_boundary_ratio": float(self._region_boundary_ratio(region_state)),
            "selected_region_boundary_uncertainty": float(
                self._region_boundary_uncertainty(region_state)
            ),
            "selected_region_guard_probe_attempts": int(region_state.guard_probe_attempts),
            "selected_region_stagnant_rounds": int(region_state.stagnant_rounds),
            "selected_region_challenger_branch": bool(region_state.challenger_branch),
            "selected_region_early_stopped": bool(region_state.early_stopped),
            "region_space": self._region_space(),
            "candidate_rows": list(probe_summary.get("candidate_rows", [])),
            "measurement_gate_enabled": bool(
                probe_summary.get("measurement_gate_enabled", False)
            ),
            "measurement_gate_keep_dim": int(
                probe_summary.get("measurement_gate_keep_dim", self.topology.n_measurements)
            ),
            "measurement_gate_keep_ratio": float(
                probe_summary.get("measurement_gate_keep_ratio", 1.0)
            ),
            "state_gate_enabled": bool(probe_summary.get("state_gate_enabled", False)),
            "state_gate_keep_dim": int(
                probe_summary.get("state_gate_keep_dim", self.topology.n_states)
            ),
            "state_gate_keep_ratio": float(
                probe_summary.get("state_gate_keep_ratio", 1.0)
            ),
            "query_cap_reached": bool(region_state.query_cap_reached),
        }

    def _lift_measurement_candidates_to_state(
        self,
        candidate_entries: list[dict],
        x0: torch.Tensor,
        clean_pred: int,
        c_base: torch.Tensor,
    ) -> tuple[torch.Tensor, list[dict]]:
        state_base = c_base.reshape(-1).to(dtype=torch.float32)
        state_region_size = int(self.config.guided_state_region_size)
        if state_region_size <= 0:
            state_region_size = min(int(self.config.region_size), self.topology.n_states)

        lifted_entries = []
        for entry in candidate_entries:
            measurement_region = entry["region"]
            delta_state_full = self.topology.estimate_state_from_measurement(
                entry["initial_delta_z"]
            ).reshape(-1)
            state_region = self.topology.measurement_region_to_state_region(
                region=measurement_region,
                x_ref=x0.reshape(-1),
                region_size=state_region_size,
                delta_state_hint=delta_state_full,
            )
            initial_delta_c, restricted_delta_z = self.topology.project_measurement_to_state_region(
                delta_z=entry["initial_delta_z"],
                state_region=state_region,
            )
            if self.measurement_state_helper is not None:
                initial_delta_c, consistent_delta_z = self.measurement_state_helper._apply_measurement_l2_cap(
                    x_ref=x0,
                    delta_c=initial_delta_c,
                    delta_z=restricted_delta_z,
                )
                consistent_adv = x0 + consistent_delta_z
                consistent_query = self.oracle.query(consistent_adv)
                consistent_score = float(self._score(consistent_query)[0].item())
                consistent_pred = int(consistent_query.pred[0].item())
            else:
                consistent_delta_z = entry["initial_delta_z"].clone()
                consistent_adv = entry["initial_adv"].clone()
                consistent_score = float(entry["initial_score"])
                consistent_pred = int(entry["initial_pred"])

            lifted_entries.append(
                {
                    "rank": int(entry["rank"]),
                    "region": state_region,
                    "measurement_region": measurement_region,
                    "initial_delta_c": initial_delta_c,
                    "initial_delta_z": consistent_delta_z.clone(),
                    "initial_adv": consistent_adv.clone(),
                    "initial_score": consistent_score,
                    "selection_objective": float(
                        self._selection_objective(
                            torch.tensor([consistent_score], dtype=torch.float32),
                            x0,
                            consistent_delta_z,
                            c_base=state_base,
                            delta_c=initial_delta_c,
                            region=state_region,
                        )[0].item()
                    ),
                    "initial_pred": consistent_pred,
                    "initial_success": bool(consistent_pred != int(clean_pred)),
                    "region_prior": float(entry.get("region_prior", 0.0)),
                }
            )
        return state_base, lifted_entries

    def _attack_sample_measurement_guided_state_search(
        self,
        x: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict:
        if self.measurement_state_helper is None:
            return self._attack_sample_region_search(
                x=x,
                label=label,
                c_base=c_base,
                generator=generator,
            )

        start_queries = self.oracle.query_count
        x0 = x.reshape(1, -1).to(dtype=torch.float32)
        clean_query = self.oracle.query(x0)
        region_probe = self._probe_regions(
            x0=x0,
            label=label,
            c_base=c_base,
            clean_query=clean_query,
            generator=generator,
        )
        probe_summary = self._summarize_probe(
            candidate_entries=region_probe["candidate_entries"],
            clean_score=float(region_probe["clean_score"]),
        )
        probe_summary.update(self._measurement_gate_info(x0))
        probe_summary.update(self._state_gate_info(x0))
        probe_summary["candidate_rows"] = list(region_probe.get("candidate_rows", []))
        probe_summary.update(
            {
                "detector_feedback_used": bool(region_probe.get("detector_feedback_used", False)),
                "detector_feedback_candidate_count": int(
                    region_probe.get("detector_feedback_candidate_count", 0)
                ),
                "detector_feedback_total_reward": float(
                    region_probe.get("detector_feedback_total_reward", 0.0)
                ),
                "detector_feedback_best_reward": float(
                    region_probe.get("detector_feedback_best_reward", 0.0)
                ),
            }
        )
        clean_score = float(probe_summary["clean_score"])
        probe_improvement_ratio = float(probe_summary["probe_improvement"]) / max(
            abs(clean_score),
            1e-6,
        )
        adaptive_query_budget_used = bool(
            self.config.adaptive_query_budget
            and int(self.config.easy_population) > 0
            and int(self.config.easy_rounds) > 0
            and (
                int(probe_summary["probe_success_count"]) > 0
                or probe_improvement_ratio
                >= float(self.config.easy_probe_improvement_ratio)
            )
        )
        query_budget_population = (
            int(self.config.easy_population) if adaptive_query_budget_used else 0
        )
        query_budget_rounds = (
            int(self.config.easy_rounds) if adaptive_query_budget_used else 0
        )
        fallback_region_prior = self._region_prior(
            region=region_probe["region"],
            x0=x0,
            c_base=c_base,
        )
        early_candidate_entry = self._annotate_candidate_entries(
            candidate_entries=[
                {
                    "rank": int(region_probe["selected_region_rank"]),
                    "region": region_probe["region"],
                    "initial_delta_c": region_probe["initial_delta_c"],
                    "initial_delta_z": region_probe["initial_delta_z"],
                    "initial_adv": region_probe["initial_adv"],
                    "initial_score": float(region_probe["initial_score"]),
                    "selection_objective": float(
                        region_probe.get("selection_objective", region_probe["initial_score"])
                    ),
                    "initial_pred": int(region_probe["initial_pred"]),
                    "initial_success": bool(region_probe["initial_success"]),
                    "region_prior": float(fallback_region_prior),
                }
            ],
            clean_score=clean_score,
            c_base=c_base,
            default_query_budget_population=query_budget_population,
            default_query_budget_rounds=query_budget_rounds,
        )[0]

        if region_probe["initial_success"] and self._should_early_stop_on_success():
            region_state = self._build_region_state(
                early_candidate_entry,
                x0=x0,
                clean_pred=int(clean_query.pred[0].item()),
                c_base=c_base,
                query_budget_population=int(
                    early_candidate_entry.get(
                        "candidate_query_budget_population",
                        query_budget_population,
                    )
                ),
                query_budget_rounds=int(
                    early_candidate_entry.get(
                        "candidate_query_budget_rounds",
                        query_budget_rounds,
                    )
                ),
                probe_improvement_ratio=probe_improvement_ratio,
                physics_quality=float(early_candidate_entry.get("physics_quality", 0.0)),
                allocation_priority=float(
                    early_candidate_entry.get("allocation_priority", 0.0)
                ),
            )
            return self._finalize_region_state(
                region_state=region_state,
                start_queries=start_queries,
                proposal_queries=int(region_probe["proposal_queries"]),
                candidate_count=int(region_probe["candidate_count"]),
                budget_region_ranks=[int(region_probe["selected_region_rank"])],
                probe_summary=probe_summary,
            )

        candidate_entries = self._select_budget_candidates(
            region_probe["candidate_entries"],
            probe_summary=probe_summary,
        )
        if not candidate_entries:
            candidate_entries = [dict(early_candidate_entry)]
        candidate_entries = self._annotate_candidate_entries(
            candidate_entries=candidate_entries,
            clean_score=clean_score,
            c_base=c_base,
            default_query_budget_population=query_budget_population,
            default_query_budget_rounds=query_budget_rounds,
        )
        candidate_entries = self._assign_challenger_query_budgets(candidate_entries)

        state_base, lifted_entries = self._lift_measurement_candidates_to_state(
            candidate_entries=candidate_entries,
            x0=x0,
            clean_pred=int(clean_query.pred[0].item()),
            c_base=c_base,
        )
        lifted_entries = self.measurement_state_helper._annotate_candidate_entries(
            candidate_entries=lifted_entries,
            clean_score=clean_score,
            c_base=state_base,
            default_query_budget_population=query_budget_population,
            default_query_budget_rounds=query_budget_rounds,
        )
        lifted_entries = self.measurement_state_helper._assign_challenger_query_budgets(
            lifted_entries
        )
        region_states = [
            self.measurement_state_helper._build_region_state(
                candidate_entry,
                x0=x0,
                clean_pred=int(clean_query.pred[0].item()),
                c_base=state_base,
                query_budget_population=int(
                    candidate_entry.get(
                        "candidate_query_budget_population",
                        query_budget_population,
                    )
                ),
                query_budget_rounds=int(
                    candidate_entry.get(
                        "candidate_query_budget_rounds",
                        query_budget_rounds,
                    )
                ),
                probe_improvement_ratio=probe_improvement_ratio,
                physics_quality=float(candidate_entry.get("physics_quality", 0.0)),
                allocation_priority=float(
                    candidate_entry.get("allocation_priority", 0.0)
                ),
            )
            for candidate_entry in lifted_entries
        ]
        budget_region_ranks = [int(entry["rank"]) for entry in lifted_entries]

        if int(self.config.rounds) > 0:
            if bool(self.config.feedback_loop) and len(region_states) > 1:
                feedback_success = self.measurement_state_helper._search_region_feedback_cycles(
                    x0=x0,
                    label=label,
                    c_base=state_base,
                    region_states=region_states,
                    generator=generator,
                    max_rounds=max(0, int(self.config.rounds)),
                )
                if feedback_success is not None:
                    return self.measurement_state_helper._finalize_region_state(
                        region_state=feedback_success,
                        start_queries=start_queries,
                        proposal_queries=int(region_probe["proposal_queries"]),
                        candidate_count=int(region_probe["candidate_count"]),
                        budget_region_ranks=budget_region_ranks,
                        probe_summary=probe_summary,
                    )
            else:
                selected_state = min(
                    region_states,
                    key=self.measurement_state_helper._state_priority,
                )
                should_stop = self.measurement_state_helper._search_region_rounds(
                    x0=x0,
                    label=label,
                    c_base=state_base,
                    region_state=selected_state,
                    generator=generator,
                    max_rounds=int(self.config.rounds),
                )
                if should_stop:
                    return self.measurement_state_helper._finalize_region_state(
                        region_state=selected_state,
                        start_queries=start_queries,
                        proposal_queries=int(region_probe["proposal_queries"]),
                        candidate_count=int(region_probe["candidate_count"]),
                        budget_region_ranks=budget_region_ranks,
                        probe_summary=probe_summary,
                    )

        final_state = self.measurement_state_helper._resolve_final_region_state(
            region_states
        )
        return self.measurement_state_helper._finalize_region_state(
            region_state=final_state,
            start_queries=start_queries,
            proposal_queries=int(region_probe["proposal_queries"]),
            candidate_count=int(region_probe["candidate_count"]),
            budget_region_ranks=budget_region_ranks,
            probe_summary=probe_summary,
        )

    def _execute_region_search_stage(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        clean_query,
        region_probe: dict,
        start_queries: int,
        generator: torch.Generator | None = None,
        max_rounds: int | None = None,
    ) -> dict:
        probe_summary = self._summarize_probe(
            candidate_entries=region_probe["candidate_entries"],
            clean_score=float(region_probe["clean_score"]),
        )
        probe_summary.update(self._measurement_gate_info(x0))
        probe_summary.update(self._state_gate_info(x0))
        probe_summary["candidate_rows"] = list(region_probe.get("candidate_rows", []))
        probe_summary.update(
            {
                "detector_feedback_used": bool(region_probe.get("detector_feedback_used", False)),
                "detector_feedback_candidate_count": int(
                    region_probe.get("detector_feedback_candidate_count", 0)
                ),
                "detector_feedback_total_reward": float(
                    region_probe.get("detector_feedback_total_reward", 0.0)
                ),
                "detector_feedback_best_reward": float(
                    region_probe.get("detector_feedback_best_reward", 0.0)
                ),
            }
        )
        clean_score = float(probe_summary["clean_score"])
        probe_improvement_ratio = float(probe_summary["probe_improvement"]) / max(
            abs(clean_score),
            1e-6,
        )
        adaptive_query_budget_used = bool(
            self.config.adaptive_query_budget
            and int(self.config.easy_population) > 0
            and int(self.config.easy_rounds) > 0
            and (
                int(probe_summary["probe_success_count"]) > 0
                or probe_improvement_ratio
                >= float(self.config.easy_probe_improvement_ratio)
            )
        )
        query_budget_population = (
            int(self.config.easy_population) if adaptive_query_budget_used else 0
        )
        query_budget_rounds = (
            int(self.config.easy_rounds) if adaptive_query_budget_used else 0
        )
        fallback_region_prior = self._region_prior(
            region=region_probe["region"],
            x0=x0,
            c_base=c_base,
        )
        early_candidate_entry = self._annotate_candidate_entries(
            candidate_entries=[
                {
                    "rank": int(region_probe["selected_region_rank"]),
                    "region": region_probe["region"],
                    "initial_delta_c": region_probe["initial_delta_c"],
                    "initial_delta_z": region_probe["initial_delta_z"],
                    "initial_adv": region_probe["initial_adv"],
                    "initial_score": float(region_probe["initial_score"]),
                    "selection_objective": float(
                        region_probe.get("selection_objective", region_probe["initial_score"])
                    ),
                    "initial_pred": int(region_probe["initial_pred"]),
                    "initial_success": bool(region_probe["initial_success"]),
                    "region_prior": float(fallback_region_prior),
                }
            ],
            clean_score=clean_score,
            c_base=c_base,
            default_query_budget_population=query_budget_population,
            default_query_budget_rounds=query_budget_rounds,
        )[0]
        if region_probe["initial_success"] and self._should_early_stop_on_success():
            region_state = self._build_region_state(
                early_candidate_entry,
                x0=x0,
                clean_pred=int(clean_query.pred[0].item()),
                c_base=c_base,
                query_budget_population=int(
                    early_candidate_entry.get(
                        "candidate_query_budget_population",
                        query_budget_population,
                    )
                ),
                query_budget_rounds=int(
                    early_candidate_entry.get(
                        "candidate_query_budget_rounds",
                        query_budget_rounds,
                    )
                ),
                probe_improvement_ratio=probe_improvement_ratio,
                physics_quality=float(early_candidate_entry.get("physics_quality", 0.0)),
                allocation_priority=float(
                    early_candidate_entry.get("allocation_priority", 0.0)
                ),
            )
            return self._finalize_region_state(
                region_state=region_state,
                start_queries=start_queries,
                proposal_queries=int(region_probe["proposal_queries"]),
                candidate_count=int(region_probe["candidate_count"]),
                budget_region_ranks=[int(region_probe["selected_region_rank"])],
                probe_summary=probe_summary,
            )

        candidate_entries = self._select_budget_candidates(
            region_probe["candidate_entries"],
            probe_summary=probe_summary,
        )
        if not candidate_entries:
            candidate_entries = [dict(early_candidate_entry)]
        candidate_entries = self._annotate_candidate_entries(
            candidate_entries=candidate_entries,
            clean_score=clean_score,
            c_base=c_base,
            default_query_budget_population=query_budget_population,
            default_query_budget_rounds=query_budget_rounds,
        )
        candidate_entries = self._assign_challenger_query_budgets(candidate_entries)

        region_states = [
            self._build_region_state(
                candidate_entry,
                x0=x0,
                clean_pred=int(clean_query.pred[0].item()),
                c_base=c_base,
                query_budget_population=int(
                    candidate_entry.get(
                        "candidate_query_budget_population",
                        query_budget_population,
                    )
                ),
                query_budget_rounds=int(
                    candidate_entry.get(
                        "candidate_query_budget_rounds",
                        query_budget_rounds,
                    )
                ),
                probe_improvement_ratio=probe_improvement_ratio,
                physics_quality=float(candidate_entry.get("physics_quality", 0.0)),
                allocation_priority=float(
                    candidate_entry.get("allocation_priority", 0.0)
                ),
            )
            for candidate_entry in candidate_entries
        ]
        budget_region_ranks = [int(state.rank) for state in region_states]
        if region_probe.get("query_cap_reached", False) and not self._has_query_budget_for(1):
            final_state = self._resolve_final_region_state(region_states)
            final_state.query_cap_reached = True
            return self._finalize_region_state(
                region_state=final_state,
                start_queries=start_queries,
                proposal_queries=int(region_probe["proposal_queries"]),
                candidate_count=int(region_probe["candidate_count"]),
                budget_region_ranks=budget_region_ranks,
                probe_summary=probe_summary,
            )

        round_budget = int(self.config.rounds) if max_rounds is None else max(0, int(max_rounds))
        fd_warmup_rounds = min(max(0, int(self.config.fd_warmup_rounds)), round_budget)
        fd_warmup_topk = max(0, int(self.config.fd_warmup_topk))
        if (
            fd_warmup_rounds > 0
            and fd_warmup_topk > 0
            and self.config.query_mode != "decision"
            and region_states
        ):
            warmup_states = sorted(region_states, key=self._state_priority)[:fd_warmup_topk]
            for region_state in warmup_states:
                should_stop = self._search_region_fd_rounds(
                    x0=x0,
                    label=label,
                    c_base=c_base,
                    region_state=region_state,
                    max_rounds=fd_warmup_rounds,
                )
                if should_stop:
                    return self._finalize_region_state(
                        region_state=region_state,
                        start_queries=start_queries,
                        proposal_queries=int(region_probe["proposal_queries"]),
                        candidate_count=int(region_probe["candidate_count"]),
                        budget_region_ranks=budget_region_ranks,
                        probe_summary=probe_summary,
                    )

        explore_rounds = max(0, int(self.config.region_budget_explore_rounds))
        if len(region_states) > 1 and explore_rounds > 0:
            stage_rounds = min(
                explore_rounds,
                max(1, round_budget // len(region_states)),
            )
        else:
            stage_rounds = 0

        remaining_rounds = round_budget
        if len(region_states) > 1 and stage_rounds > 0:
            total_stage_rounds = stage_rounds * len(region_states)
            stage_allocations = self._allocate_stage_rounds(
                region_states=region_states,
                total_stage_rounds=total_stage_rounds,
            )
            remaining_rounds = max(0, round_budget - sum(stage_allocations))
            for region_state, allocated_rounds in zip(region_states, stage_allocations):
                if allocated_rounds <= 0:
                    continue
                should_stop = self._search_region_rounds(
                    x0=x0,
                    label=label,
                    c_base=c_base,
                    region_state=region_state,
                    generator=generator,
                    max_rounds=int(allocated_rounds),
                )
                self._should_prune_region_state(
                    region_state,
                    region_states,
                    c_base,
                )
                if should_stop:
                    return self._finalize_region_state(
                        region_state=region_state,
                        start_queries=start_queries,
                        proposal_queries=int(region_probe["proposal_queries"]),
                        candidate_count=int(region_probe["candidate_count"]),
                        budget_region_ranks=budget_region_ranks,
                        probe_summary=probe_summary,
                    )

        if remaining_rounds > 0:
            if bool(self.config.feedback_loop) and len(region_states) > 1:
                feedback_success = self._search_region_feedback_cycles(
                    x0=x0,
                    label=label,
                    c_base=c_base,
                    region_states=region_states,
                    generator=generator,
                    max_rounds=remaining_rounds,
                )
                if feedback_success is not None:
                    return self._finalize_region_state(
                        region_state=feedback_success,
                        start_queries=start_queries,
                        proposal_queries=int(region_probe["proposal_queries"]),
                        candidate_count=int(region_probe["candidate_count"]),
                        budget_region_ranks=budget_region_ranks,
                        probe_summary=probe_summary,
                    )
            else:
                selected_state = min(region_states, key=self._state_priority)
                should_stop = self._search_region_rounds(
                    x0=x0,
                    label=label,
                    c_base=c_base,
                    region_state=selected_state,
                    generator=generator,
                    max_rounds=remaining_rounds,
                )
                if should_stop:
                    return self._finalize_region_state(
                        region_state=selected_state,
                        start_queries=start_queries,
                        proposal_queries=int(region_probe["proposal_queries"]),
                        candidate_count=int(region_probe["candidate_count"]),
                        budget_region_ranks=budget_region_ranks,
                        probe_summary=probe_summary,
                    )

        final_state = self._resolve_final_region_state(region_states)
        return self._finalize_region_state(
            region_state=final_state,
            start_queries=start_queries,
            proposal_queries=int(region_probe["proposal_queries"]),
            candidate_count=int(region_probe["candidate_count"]),
            budget_region_ranks=budget_region_ranks,
            probe_summary=probe_summary,
        )

    def _layered_state_stage_candidates(
        self,
        x0: torch.Tensor,
        c_base: torch.Tensor,
        active_size: int,
        generator: torch.Generator | None = None,
        seed_score: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if self._uses_measurement_regions():
            raise ValueError("layered active-set search is only implemented for state regions.")
        if seed_score is None:
            seed_score = self._region_seed_score(x0=x0, c_base=c_base)
        else:
            seed_score = seed_score.reshape(-1).to(dtype=torch.float32)
        allowed_mask = self._topk_allowed_mask(seed_score, active_size)
        local_region_size = max(1, min(int(self.config.region_size), int(active_size)))
        local_anchor_size = max(1, min(int(self.config.anchor_size), local_region_size))
        local_anchor_pool = max(
            local_anchor_size,
            min(int(self.config.anchor_pool_size), int(active_size)),
        )
        candidates = self._enumerate_candidate_regions(
            x0=x0,
            c_base=c_base,
            generator=generator,
            forced_seed_score=seed_score,
            forced_state_allowed_mask=allowed_mask,
            region_size_override=local_region_size,
            anchor_size_override=local_anchor_size,
            region_candidates_override=int(self.config.region_candidates),
            anchor_pool_size_override=local_anchor_pool,
        )
        if not candidates:
            priority_score = torch.where(
                allowed_mask,
                seed_score.reshape(-1).to(dtype=torch.float32),
                torch.zeros_like(seed_score.reshape(-1).to(dtype=torch.float32)),
            )
            fallback_region = self.topology.build_state_region_from_priority(
                priority_score=priority_score,
                region_size=local_region_size,
                anchor_size=local_anchor_size,
                allowed_mask=allowed_mask,
            )
            candidates = [fallback_region]
        return candidates, allowed_mask

    def _support_final_size(self) -> int:
        if self._uses_measurement_regions():
            raise ValueError("support identification is only implemented for state regions.")
        total_dim = int(self.topology.n_states)
        requested = int(getattr(self.config, "support_final_size", 0))
        if requested <= 0:
            requested = int(self.config.region_size)
        return max(1, min(total_dim, requested))

    def _support_pool_size(self) -> int:
        if self._uses_measurement_regions():
            raise ValueError("support identification is only implemented for state regions.")
        total_dim = int(self.topology.n_states)
        requested = int(getattr(self.config, "support_pool_size", 0))
        if requested <= 0:
            requested = int(self.config.single_active_size)
        if requested <= 0:
            requested = max(self._support_final_size(), int(self.config.region_size))
        return max(self._support_final_size(), min(total_dim, requested))

    def _support_diffused_pool_score(self, c_base: torch.Tensor) -> torch.Tensor:
        if self._uses_measurement_regions():
            raise ValueError("support identification is only implemented for state regions.")
        c_base = c_base.reshape(-1).to(dtype=torch.float32)
        support_mass = c_base.abs()
        if float(support_mass.max().item()) <= 0.0:
            return self.topology._normalize_vector(self.topology.state_importance)

        sym_adj = 0.5 * (self.topology.adjacency + self.topology.adjacency.T)
        diffusion_lambda = max(float(self.config.support_diffusion_lambda), 0.0)
        pool_score = support_mass + diffusion_lambda * (sym_adj @ support_mass)
        if bool(self.config.support_keep_base_support):
            base_bonus = support_mass.max().clamp(min=1e-8)
            pool_score = torch.where(
                support_mass > 1e-10,
                pool_score + base_bonus,
                pool_score,
            )
        return self.topology._normalize_vector(pool_score)

    def _support_candidate_pool(
        self,
        c_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pool_score = self._support_diffused_pool_score(c_base)
        pool_size = self._support_pool_size()
        ordered = torch.argsort(pool_score, descending=True).tolist()
        support_mass = c_base.reshape(-1).abs().to(dtype=torch.float32)
        support_idx = torch.nonzero(support_mass > 1e-10, as_tuple=False).reshape(-1).tolist()
        support_idx = sorted(
            support_idx,
            key=lambda idx: (-float(pool_score[int(idx)].item()), int(idx)),
        )

        pool: list[int] = []
        if bool(self.config.support_keep_base_support):
            for idx in support_idx:
                if idx not in pool:
                    pool.append(int(idx))
                if len(pool) >= pool_size:
                    break
        for idx in ordered:
            if int(idx) in pool:
                continue
            pool.append(int(idx))
            if len(pool) >= pool_size:
                break
        if not pool:
            pool = list(range(pool_size))
        return torch.tensor(pool[:pool_size], dtype=torch.long), pool_score

    def _support_probe_scale(
        self,
        x0: torch.Tensor,
        c_base: torch.Tensor,
        region: torch.Tensor,
    ) -> float:
        scale_ratio = max(
            float(
                getattr(
                    self.config,
                    "support_probe_scale_ratio",
                    self.config.probe_scale_ratio,
                )
            ),
            0.0,
        )
        measurement_budget = self._measurement_budget_abs(x0)
        if measurement_budget > 0.0:
            return max(float(self.config.radius_floor) * 0.5, scale_ratio * measurement_budget)
        return max(
            float(self.config.radius_floor) * 0.5,
            scale_ratio * self._region_reference_norm(x0=x0, c_base=c_base, region=region),
        )

    def _probe_support_pool(
        self,
        x0: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        clean_query,
    ) -> dict:
        pool_idx, pool_score = self._support_candidate_pool(c_base=c_base)
        clean_score = float(self._score(clean_query)[0].item())
        clean_pred = int(clean_query.pred[0].item())
        candidate_rows: list[dict] = []
        candidate_entries: list[dict] = []
        proposal_queries = 0

        for rank, state_idx in enumerate(pool_idx.tolist(), start=1):
            region = torch.tensor([int(state_idx)], dtype=torch.long)
            probe_dir = self._normalize_state_budget_rows(
                region=region,
                delta_c=torch.ones(1, 1, dtype=torch.float32),
                use_search_shaping=False,
            )
            probe_scale = self._support_probe_scale(
                x0=x0,
                c_base=c_base,
                region=region,
            )
            probe_eval = self._evaluate_probe_batch(
                x0=x0,
                label=label,
                c_base=c_base,
                region=region,
                probe_scale=float(probe_scale),
                probe_dirs=probe_dir,
                direction_start_idx=0,
                direction_end_idx=int(probe_dir.shape[0]),
            )
            proposal_queries += int(probe_eval["queries"])
            region_prior = float(pool_score[int(state_idx)].item())
            candidate_rows.append(
                {
                    "rank": int(rank),
                    "region": region,
                    "region_space": "state",
                    "best_score": float(probe_eval["score"]),
                    "selection_objective": float(probe_eval["objective"]),
                    "best_success": bool(probe_eval["success"]),
                    "probe_scale": float(probe_scale),
                    "region_prior": float(region_prior),
                    "proposal_source": "support_probe",
                    "feedback_reward": 0.0,
                }
            )
            candidate_entries.append(
                {
                    "rank": int(rank),
                    "region": region,
                    "initial_delta_c": probe_eval["delta_c"].clone(),
                    "initial_delta_z": probe_eval["delta_z"].clone(),
                    "initial_adv": probe_eval["adv"].clone(),
                    "initial_score": float(probe_eval["score"]),
                    "selection_objective": float(probe_eval["objective"]),
                    "initial_pred": int(probe_eval["pred"]),
                    "initial_success": bool(probe_eval["success"]),
                    "region_prior": float(region_prior),
                    "proposal_source": "support_probe",
                    "feedback_reward": 0.0,
                }
            )

        return {
            "pool_idx": pool_idx,
            "pool_score": pool_score,
            "candidate_rows": candidate_rows,
            "candidate_entries": candidate_entries,
            "proposal_queries": int(proposal_queries),
            "clean_score": float(clean_score),
            "clean_pred": int(clean_pred),
        }

    def _select_support_from_probes(
        self,
        candidate_entries: list[dict],
        clean_score: float,
    ) -> tuple[list[dict], torch.Tensor]:
        if not candidate_entries:
            raise RuntimeError("support probe stage produced no candidate entries.")

        min_support_size = min(self._support_final_size(), len(candidate_entries))
        improvement = torch.tensor(
            [
                max(
                    float(clean_score)
                    - float(entry.get("selection_objective", entry["initial_score"])),
                    0.0,
                )
                for entry in candidate_entries
            ],
            dtype=torch.float32,
        )
        if float(improvement.max().item()) > 0.0:
            improvement = improvement / improvement.max().clamp(min=1e-8)

        prior = torch.tensor(
            [float(entry.get("region_prior", 0.0)) for entry in candidate_entries],
            dtype=torch.float32,
        )
        if float(prior.max().item()) > 0.0:
            prior = prior / prior.max().clamp(min=1e-8)
        success = torch.tensor(
            [1.0 if bool(entry["initial_success"]) else 0.0 for entry in candidate_entries],
            dtype=torch.float32,
        )

        utility = (
            improvement
            + float(self.config.support_prior_weight) * prior
            + float(self.config.support_success_bonus) * success
        )
        support_size = int(min_support_size)
        if bool(self.config.adaptive_support_selection):
            positive_utility = utility.clamp(min=0.0)
            threshold = min(
                max(float(self.config.adaptive_support_mass_threshold), 0.0),
                1.0,
            )
            max_support_size = int(self.config.adaptive_support_max_size)
            if max_support_size <= 0:
                max_support_size = len(candidate_entries)
            max_support_size = max(min_support_size, min(max_support_size, len(candidate_entries)))
            if float(positive_utility.sum().item()) > 1e-8:
                sorted_utility, _ = torch.sort(positive_utility, descending=True)
                cumulative = torch.cumsum(
                    sorted_utility / sorted_utility.sum().clamp(min=1e-8),
                    dim=0,
                )
                target_index = int(torch.nonzero(cumulative >= threshold, as_tuple=False)[0].item()) + 1
                support_size = max(min_support_size, min(max_support_size, target_index))
            else:
                support_size = max_support_size
        sym_adj = 0.5 * (self.topology.adjacency + self.topology.adjacency.T)
        diversity_penalty = max(float(self.config.support_diversity_penalty), 0.0)

        selected_positions: list[int] = []
        remaining = set(range(len(candidate_entries)))
        while len(selected_positions) < support_size and remaining:
            best_pos = None
            best_key = None
            for pos in sorted(remaining):
                state_idx = int(candidate_entries[pos]["region"].reshape(-1)[0].item())
                redundancy = 0.0
                if selected_positions:
                    selected_state_ids = [
                        int(candidate_entries[idx]["region"].reshape(-1)[0].item())
                        for idx in selected_positions
                    ]
                    redundancy = float(sym_adj[state_idx, selected_state_ids].max().item())
                marginal = float(utility[pos].item()) - diversity_penalty * redundancy
                key = (
                    float(marginal),
                    float(utility[pos].item()),
                    -int(state_idx),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_pos = int(pos)
            if best_pos is None:
                break
            selected_positions.append(int(best_pos))
            remaining.remove(int(best_pos))

        if not selected_positions:
            selected_positions = [0]
        return [candidate_entries[idx] for idx in selected_positions], utility

    def _initial_candidate_value(
        self,
        x0: torch.Tensor,
        c_base: torch.Tensor,
        candidate_entry: dict,
    ) -> tuple[float, float, int]:
        if bool(candidate_entry["initial_success"]):
            priority = float(
                self._success_candidate_value(
                    x_ref=x0,
                    objective=torch.tensor(
                        [float(candidate_entry["selection_objective"])],
                        dtype=torch.float32,
                    ),
                    delta_z=candidate_entry["initial_delta_z"],
                    c_base=c_base,
                    delta_c=candidate_entry["initial_delta_c"],
                    region=candidate_entry["region"],
                )[0].item()
            )
            return (0.0, priority, int(candidate_entry["rank"]))
        return (
            1.0,
            float(candidate_entry["selection_objective"]),
            int(candidate_entry["rank"]),
        )

    def _single_active_size(self, seed_score: torch.Tensor | None = None) -> int:
        if self._uses_measurement_regions():
            raise ValueError("single active-set search is only implemented for state regions.")
        total_dim = int(self.topology.n_states)
        requested = int(self.config.single_active_size)
        if requested <= 0:
            requested = int(self.config.region_size)
        requested = max(1, min(total_dim, requested))
        if (not bool(self.config.adaptive_support_selection)) or seed_score is None:
            return requested

        score = seed_score.reshape(-1).to(dtype=torch.float32).clamp(min=0.0)
        if score.numel() != total_dim or float(score.sum().item()) <= 1e-8:
            return requested

        threshold = min(max(float(self.config.adaptive_support_mass_threshold), 0.0), 1.0)
        max_support_size = int(self.config.adaptive_support_max_size)
        if max_support_size <= 0:
            max_support_size = requested
        max_support_size = max(1, min(total_dim, requested, max_support_size))
        min_support_size = max(1, min(total_dim, int(self.config.anchor_size)))

        sorted_score, _ = torch.sort(score, descending=True)
        cumulative = torch.cumsum(sorted_score / score.sum().clamp(min=1e-8), dim=0)
        target_index = int(
            torch.nonzero(cumulative >= threshold, as_tuple=False)[0].item()
        ) + 1
        return max(min_support_size, min(max_support_size, target_index))

    def _attack_sample_support_identify_pgzoo(
        self,
        x: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict:
        if self._uses_measurement_regions():
            raise ValueError("support-identify PG-ZOO currently only supports state regions.")

        start_queries = self.oracle.query_count
        x0 = x.reshape(1, -1).to(dtype=torch.float32)
        clean_query = self.oracle.query(x0)
        clean_score = float(self._score(clean_query)[0].item())
        clean_pred = int(clean_query.pred[0].item())

        support_probe = self._probe_support_pool(
            x0=x0,
            label=label,
            c_base=c_base,
            clean_query=clean_query,
        )
        singleton_entries = support_probe["candidate_entries"]
        singleton_rows = list(support_probe["candidate_rows"])
        selected_entries, support_utility = self._select_support_from_probes(
            candidate_entries=singleton_entries,
            clean_score=clean_score,
        )

        support_region = torch.tensor(
            [
                int(entry["region"].reshape(-1)[0].item())
                for entry in selected_entries
            ],
            dtype=torch.long,
        )
        support_region_prior = self._region_prior(
            region=support_region,
            x0=x0,
            c_base=c_base,
        )
        probe_summary = self._summarize_probe(
            candidate_entries=singleton_entries,
            clean_score=clean_score,
        )
        probe_summary.update(self._measurement_gate_info(x0))
        probe_summary.update(self._state_gate_info(x0))

        init_candidates: list[dict] = [
            {
                "rank": 0,
                "region": support_region,
                "initial_delta_c": torch.zeros(
                    1,
                    int(support_region.numel()),
                    dtype=torch.float32,
                ),
                "initial_delta_z": torch.zeros_like(x0),
                "initial_adv": x0.clone(),
                "initial_score": float(clean_score),
                "selection_objective": float(clean_score),
                "initial_pred": int(clean_pred),
                "initial_success": False,
                "region_prior": float(support_region_prior),
                "proposal_source": "support_selected",
                "feedback_reward": 0.0,
            }
        ]

        best_singleton_pos = min(
            range(len(selected_entries)),
            key=lambda idx: self._candidate_priority(selected_entries[idx]),
        )
        best_singleton = selected_entries[best_singleton_pos]
        best_singleton_delta = torch.zeros(
            1,
            int(support_region.numel()),
            dtype=torch.float32,
        )
        best_singleton_delta[0, best_singleton_pos] = float(
            best_singleton["initial_delta_c"].reshape(-1)[0].item()
        )
        init_candidates.append(
            {
                "rank": 0,
                "region": support_region,
                "initial_delta_c": best_singleton_delta,
                "initial_delta_z": best_singleton["initial_delta_z"].clone(),
                "initial_adv": best_singleton["initial_adv"].clone(),
                "initial_score": float(best_singleton["initial_score"]),
                "selection_objective": float(
                    best_singleton.get(
                        "selection_objective",
                        best_singleton["initial_score"],
                    )
                ),
                "initial_pred": int(best_singleton["initial_pred"]),
                "initial_success": bool(best_singleton["initial_success"]),
                "region_prior": float(support_region_prior),
                "proposal_source": "support_selected_singleton",
                "feedback_reward": 0.0,
            }
        )

        combined_delta_c = torch.zeros(
            1,
            int(support_region.numel()),
            dtype=torch.float32,
        )
        utility_by_rank = {
            int(entry["rank"]): float(support_utility[int(entry["rank"]) - 1].item())
            for entry in singleton_entries
        }
        max_utility = max(utility_by_rank.values()) if utility_by_rank else 0.0
        query_coordinate_prior = torch.zeros(
            1,
            int(support_region.numel()),
            dtype=torch.float32,
        )
        for idx, entry in enumerate(selected_entries):
            sign = float(torch.sign(entry["initial_delta_c"].reshape(-1)[0]).item())
            if abs(sign) <= 1e-12:
                sign = 1.0
            weight = float(utility_by_rank.get(int(entry["rank"]), 0.0))
            if max_utility > 1e-8:
                weight = weight / max_utility
            else:
                weight = 1.0
            combined_delta_c[0, idx] = sign * max(weight, 1e-4)
            query_coordinate_prior[0, idx] = max(weight, 1e-4)

        proposal_queries = int(support_probe["proposal_queries"])
        if float(torch.linalg.norm(combined_delta_c).item()) > 1e-12:
            if self._uses_state_budget_geometry():
                combined_delta_c, combined_delta_z = self._project_state_budget_ball(
                    x_ref=x0,
                    region=support_region,
                    delta_c=combined_delta_c,
                    use_search_shaping=False,
                )
            else:
                combined_delta_z = self._project_region(
                    region=support_region,
                    delta_region=combined_delta_c,
                    use_search_shaping=False,
                )
                combined_delta_c, combined_delta_z = self._apply_measurement_l2_cap(
                    x_ref=x0,
                    delta_c=combined_delta_c,
                    delta_z=combined_delta_z,
                )
            combined_adv = x0 + combined_delta_z
            combined_query = self.oracle.query(combined_adv)
            combined_score = self._score(combined_query)
            combined_objective = self._selection_objective(
                combined_score,
                x0,
                combined_delta_z,
                c_base=c_base,
                delta_c=combined_delta_c,
                region=support_region,
            )
            init_candidates.append(
                {
                    "rank": 0,
                    "region": support_region,
                    "initial_delta_c": combined_delta_c.clone(),
                    "initial_delta_z": combined_delta_z.clone(),
                    "initial_adv": combined_adv.clone(),
                    "initial_score": float(combined_score[0].item()),
                    "selection_objective": float(combined_objective[0].item()),
                    "initial_pred": int(combined_query.pred[0].item()),
                    "initial_success": bool(combined_query.pred[0].item() != int(label)),
                    "region_prior": float(support_region_prior),
                    "proposal_source": "support_selected_combined",
                    "feedback_reward": 0.0,
                }
            )
            proposal_queries += 1

        selected_candidate = min(
            init_candidates,
            key=lambda entry: self._initial_candidate_value(
                x0=x0,
                c_base=c_base,
                candidate_entry=entry,
            ),
        )
        final_candidate_entry = {
            **selected_candidate,
            "rank": 0,
            "region": support_region,
            "region_prior": float(support_region_prior),
            "query_coordinate_prior": query_coordinate_prior.clone(),
            "proposal_source": "support_selected",
            "feedback_reward": 0.0,
        }

        singleton_rows.append(
            {
                "rank": 0,
                "region": support_region,
                "region_space": "state",
                "best_score": float(final_candidate_entry["initial_score"]),
                "selection_objective": float(final_candidate_entry["selection_objective"]),
                "best_success": bool(final_candidate_entry["initial_success"]),
                "probe_scale": 0.0,
                "region_prior": float(support_region_prior),
                "proposal_source": "support_selected",
                "feedback_reward": 0.0,
            }
        )
        probe_summary["candidate_rows"] = singleton_rows
        probe_improvement_ratio = float(probe_summary["probe_improvement"]) / max(
            abs(clean_score),
            1e-6,
        )

        annotated_entry = self._annotate_candidate_entries(
            candidate_entries=[final_candidate_entry],
            clean_score=clean_score,
            c_base=c_base,
            default_query_budget_population=0,
            default_query_budget_rounds=0,
        )[0]
        region_state = self._build_region_state(
            candidate_entry=annotated_entry,
            x0=x0,
            clean_pred=clean_pred,
            c_base=c_base,
            query_budget_population=0,
            query_budget_rounds=0,
            probe_improvement_ratio=probe_improvement_ratio,
            physics_quality=float(annotated_entry.get("physics_quality", 0.0)),
            allocation_priority=float(annotated_entry.get("allocation_priority", 0.0)),
        )

        if bool(final_candidate_entry["initial_success"]) and self._should_early_stop_on_success():
            result = self._finalize_region_state(
                region_state=region_state,
                start_queries=start_queries,
                proposal_queries=int(proposal_queries),
                candidate_count=int(len(singleton_entries) + 1),
                budget_region_ranks=[0],
                probe_summary=probe_summary,
            )
        else:
            should_stop = self._search_region_rounds(
                x0=x0,
                label=label,
                c_base=c_base,
                region_state=region_state,
                generator=generator,
                max_rounds=int(self.config.rounds),
            )
            result = self._finalize_region_state(
                region_state=region_state,
                start_queries=start_queries,
                proposal_queries=int(proposal_queries),
                candidate_count=int(len(singleton_entries) + 1),
                budget_region_ranks=[0],
                probe_summary=probe_summary,
            )
            if should_stop:
                result["selected_region_early_stopped"] = True

        result["active_set_size"] = int(support_probe["pool_idx"].numel())
        result["active_set_stage_index"] = 0
        result["active_set_keep_ratio"] = float(
            support_probe["pool_idx"].numel() / max(1, int(self.topology.n_states))
        )
        result["layered_total_stages"] = 1
        result["layered_success_stage"] = 0 if bool(result["success"]) else -1
        result["support_pool_size"] = int(support_probe["pool_idx"].numel())
        result["support_final_size"] = int(support_region.numel())
        return result

    def _attack_sample_single_active_region_search(
        self,
        x: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict:
        if self._uses_measurement_regions():
            raise ValueError("single active-set search currently only supports state regions.")
        start_queries = self.oracle.query_count
        x0 = x.reshape(1, -1).to(dtype=torch.float32)
        clean_query = self.oracle.query(x0)
        seed_score = self._region_seed_score(x0=x0, c_base=c_base).reshape(-1).to(dtype=torch.float32)
        active_size = self._single_active_size(seed_score=seed_score)
        candidates, _ = self._layered_state_stage_candidates(
            x0=x0,
            c_base=c_base,
            active_size=active_size,
            generator=generator,
            seed_score=seed_score,
        )
        region_probe = self._probe_regions(
            x0=x0,
            label=label,
            c_base=c_base,
            clean_query=clean_query,
            generator=generator,
            candidates_override=candidates,
        )
        stage_result = self._execute_region_search_stage(
            x0=x0,
            label=label,
            c_base=c_base,
            clean_query=clean_query,
            region_probe=region_probe,
            start_queries=start_queries,
            generator=generator,
            max_rounds=int(self.config.rounds),
        )
        stage_rows = []
        for candidate_row in stage_result.get("candidate_rows", []):
            stage_rows.append(
                {
                    **candidate_row,
                    "active_set_size": int(active_size),
                    "active_set_stage_index": 0,
                }
            )
        single_result = dict(stage_result)
        single_result["candidate_rows"] = stage_rows
        single_result["active_set_size"] = int(active_size)
        single_result["active_set_stage_index"] = 0
        single_result["active_set_schedule"] = [int(active_size)]
        single_result["active_set_keep_ratio"] = float(
            int(active_size) / max(1, int(self.topology.n_states))
        )
        single_result["layered_total_stages"] = 1
        single_result["layered_success_stage"] = 0 if bool(stage_result["success"]) else -1
        return single_result

    def _attack_sample_layered_region_search(
        self,
        x: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict:
        if self._uses_measurement_regions():
            raise ValueError("layered active-set search currently only supports state regions.")
        start_queries = self.oracle.query_count
        x0 = x.reshape(1, -1).to(dtype=torch.float32)
        clean_query = self.oracle.query(x0)
        seed_score = self._region_seed_score(x0=x0, c_base=c_base).reshape(-1).to(dtype=torch.float32)
        active_schedule = self._layered_active_schedule()

        accumulated_proposal_queries = 0
        aggregated_candidate_rows = []
        best_result = None
        best_key = None

        for stage_idx, active_size in enumerate(active_schedule):
            stage_round_budget = self._layered_round_budget(stage_idx, len(active_schedule))
            candidates, _ = self._layered_state_stage_candidates(
                x0=x0,
                c_base=c_base,
                active_size=int(active_size),
                generator=generator,
                seed_score=seed_score,
            )
            region_probe = self._probe_regions(
                x0=x0,
                label=label,
                c_base=c_base,
                clean_query=clean_query,
                generator=generator,
                candidates_override=candidates,
            )
            stage_result = self._execute_region_search_stage(
                x0=x0,
                label=label,
                c_base=c_base,
                clean_query=clean_query,
                region_probe=region_probe,
                start_queries=start_queries,
                generator=generator,
                max_rounds=stage_round_budget,
            )

            stage_rows = []
            for candidate_row in stage_result.get("candidate_rows", []):
                stage_rows.append(
                    {
                        **candidate_row,
                        "active_set_size": int(active_size),
                        "active_set_stage_index": int(stage_idx),
                    }
                )
            aggregated_candidate_rows.extend(stage_rows)

            cumulative_result = dict(stage_result)
            cumulative_result["proposal_queries"] = int(
                accumulated_proposal_queries + int(stage_result["proposal_queries"])
            )
            cumulative_result["candidate_rows"] = list(aggregated_candidate_rows)
            cumulative_result["active_set_size"] = int(active_size)
            cumulative_result["active_set_stage_index"] = int(stage_idx)
            cumulative_result["active_set_schedule"] = [int(value) for value in active_schedule]
            cumulative_result["active_set_keep_ratio"] = float(
                int(active_size) / max(1, int(self.topology.n_states))
            )
            cumulative_result["layered_total_stages"] = int(len(active_schedule))
            cumulative_result["layered_success_stage"] = (
                int(stage_idx) if bool(stage_result["success"]) else -1
            )

            if bool(stage_result["success"]):
                return cumulative_result

            stage_key = (
                float(stage_result["final_objective"]),
                float(stage_result["final_score"]),
                int(active_size),
            )
            if best_result is None or stage_key < best_key:
                best_result = dict(cumulative_result)
                best_key = stage_key

            accumulated_proposal_queries += int(stage_result["proposal_queries"])

        if best_result is None:
            raise RuntimeError("layered active-set search failed to produce any stage result.")

        best_result["queries_used"] = int(self.oracle.query_count - start_queries)
        best_result["proposal_queries"] = int(accumulated_proposal_queries)
        best_result["candidate_rows"] = list(aggregated_candidate_rows)
        best_result["layered_total_stages"] = int(len(active_schedule))
        best_result["layered_success_stage"] = -1
        return best_result

    def _attack_sample_region_search(
        self,
        x: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict:
        if self._uses_measurement_regions() and bool(self.config.measurement_guided_state_refine):
            return self._attack_sample_measurement_guided_state_search(
                x=x,
                label=label,
                c_base=c_base,
                generator=generator,
            )
        start_queries = self.oracle.query_count
        x0 = x.reshape(1, -1).to(dtype=torch.float32)
        clean_query = self.oracle.query(x0)
        region_probe = self._probe_regions(
            x0=x0,
            label=label,
            c_base=c_base,
            clean_query=clean_query,
            generator=generator,
        )
        return self._execute_region_search_stage(
            x0=x0,
            label=label,
            c_base=c_base,
            clean_query=clean_query,
            region_probe=region_probe,
            start_queries=start_queries,
            generator=generator,
            max_rounds=int(self.config.rounds),
        )

    def _attack_sample_fd(
        self,
        x: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict:
        start_queries = self.oracle.query_count
        x0 = x.reshape(1, -1).to(dtype=torch.float32)
        clean_query = self.oracle.query(x0)
        clean_score = float(self._score(clean_query)[0].item())
        clean_pred = int(clean_query.pred[0].item())

        attack_mode = str(self.config.attack_mode).lower()
        probe_summary = self._empty_probe_summary(clean_score=clean_score)
        proposal_queries = 0
        candidate_count = 1
        budget_region_ranks = [0]

        if attack_mode in {"full_fd", "measurement_full_fd", "full_zoo", "measurement_full_zoo"}:
            region = torch.arange(self._region_total_dim(), dtype=torch.long)
            candidate_entry = self._build_candidate_entry(
                region=region,
                rank=0,
                x0=x0,
                clean_score=clean_score,
                clean_pred=clean_pred,
            )
        elif attack_mode in {"region_fd", "measurement_region_fd", "region_zoo", "measurement_region_zoo"}:
            region_selection = str(self.config.fd_region_selection).lower()
            if region_selection == "probe_best":
                region_probe = self._probe_regions(
                    x0=x0,
                    label=label,
                    c_base=c_base,
                    clean_query=clean_query,
                    generator=generator,
                )
                proposal_queries = int(region_probe["proposal_queries"])
                candidate_count = int(region_probe["candidate_count"])
                probe_summary = self._summarize_probe(
                    candidate_entries=region_probe["candidate_entries"],
                    clean_score=float(region_probe["clean_score"]),
                )
                probe_summary.update(self._measurement_gate_info(x0))
                probe_summary.update(self._state_gate_info(x0))
                probe_summary["candidate_rows"] = list(region_probe.get("candidate_rows", []))
                probe_summary.update(
                    {
                        "detector_feedback_used": bool(
                            region_probe.get("detector_feedback_used", False)
                        ),
                        "detector_feedback_candidate_count": int(
                            region_probe.get("detector_feedback_candidate_count", 0)
                        ),
                        "detector_feedback_total_reward": float(
                            region_probe.get("detector_feedback_total_reward", 0.0)
                        ),
                        "detector_feedback_best_reward": float(
                            region_probe.get("detector_feedback_best_reward", 0.0)
                        ),
                    }
                )
                candidate_entry = self._build_candidate_entry(
                    region=region_probe["region"],
                    rank=int(region_probe["selected_region_rank"]),
                    x0=x0,
                    clean_score=clean_score,
                    clean_pred=clean_pred,
                    initial_delta_c=region_probe["initial_delta_c"],
                    initial_delta_z=region_probe["initial_delta_z"],
                    initial_adv=region_probe["initial_adv"],
                    initial_score=float(region_probe["initial_score"]),
                    initial_pred=int(region_probe["initial_pred"]),
                    initial_success=bool(region_probe["initial_success"]),
                    selection_objective=float(
                        region_probe.get("selection_objective", region_probe["initial_score"])
                    ),
                )
                budget_region_ranks = [int(region_probe["selected_region_rank"])]
            else:
                candidates = self._enumerate_candidate_regions(
                    x0=x0,
                    c_base=c_base,
                    generator=generator,
                )
                candidate_count = max(1, len(candidates))
                selected_rank = 0
                if region_selection == "random_candidate" and len(candidates) > 1:
                    if generator is not None:
                        selected_rank = int(
                            torch.randint(
                                low=0,
                                high=len(candidates),
                                size=(1,),
                                generator=generator,
                            ).item()
                        )
                    else:
                        selected_rank = int(torch.randint(low=0, high=len(candidates), size=(1,)).item())
                region = candidates[selected_rank]
                region_prior = self._region_prior(region=region, x0=x0, c_base=c_base)
                candidate_entry = self._build_candidate_entry(
                    region=region,
                    rank=selected_rank,
                    x0=x0,
                    clean_score=clean_score,
                    clean_pred=clean_pred,
                    region_prior=region_prior,
                )
                budget_region_ranks = [int(selected_rank)]
                probe_summary = {
                    **probe_summary,
                    "probe_best_rank": int(selected_rank),
                    "probe_second_rank": int(selected_rank),
                    "probe_best_prior": float(region_prior),
                    "probe_second_prior": float(region_prior),
                }
        else:
            raise ValueError(f"Unsupported finite-difference attack mode: {self.config.attack_mode}")

        region_state = self._build_region_state(
            candidate_entry=candidate_entry,
            x0=x0,
            clean_pred=int(clean_pred),
            c_base=c_base,
            query_budget_population=0,
            query_budget_rounds=0,
            probe_improvement_ratio=0.0,
        )
        if bool(candidate_entry["initial_success"]) and self._should_early_stop_on_success():
            return self._finalize_region_state(
                region_state=region_state,
                start_queries=start_queries,
                proposal_queries=proposal_queries,
                candidate_count=candidate_count,
                budget_region_ranks=budget_region_ranks,
                probe_summary=probe_summary,
            )

        should_stop = self._search_region_fd_rounds(
            x0=x0,
            label=label,
            c_base=c_base,
            region_state=region_state,
            max_rounds=int(self.config.fd_iterations),
        )
        if should_stop:
            return self._finalize_region_state(
                region_state=region_state,
                start_queries=start_queries,
                proposal_queries=proposal_queries,
                candidate_count=candidate_count,
                budget_region_ranks=budget_region_ranks,
                probe_summary=probe_summary,
            )

        return self._finalize_region_state(
            region_state=region_state,
            start_queries=start_queries,
            proposal_queries=proposal_queries,
            candidate_count=candidate_count,
            budget_region_ranks=budget_region_ranks,
            probe_summary=probe_summary,
        )

    def attack_sample(
        self,
        x: torch.Tensor,
        label: int,
        c_base: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict:
        attack_mode = str(self.config.attack_mode).lower()
        self._begin_sample_query_budget()
        try:
            if attack_mode in {"region_search", "measurement_region_search"}:
                return self._attack_sample_region_search(
                    x=x,
                    label=label,
                    c_base=c_base,
                    generator=generator,
                )
            if attack_mode in {"support_identify_pgzoo"}:
                return self._attack_sample_support_identify_pgzoo(
                    x=x,
                    label=label,
                    c_base=c_base,
                    generator=generator,
                )
            if attack_mode in {"single_active_region_search"}:
                return self._attack_sample_single_active_region_search(
                    x=x,
                    label=label,
                    c_base=c_base,
                    generator=generator,
                )
            if attack_mode in {"layered_region_search"}:
                return self._attack_sample_layered_region_search(
                    x=x,
                    label=label,
                    c_base=c_base,
                    generator=generator,
                )
            if attack_mode in {
                "full_fd",
                "region_fd",
                "measurement_full_fd",
                "measurement_region_fd",
                "full_zoo",
                "region_zoo",
                "measurement_full_zoo",
                "measurement_region_zoo",
            }:
                return self._attack_sample_fd(
                    x=x,
                    label=label,
                    c_base=c_base,
                    generator=generator,
                )
            raise ValueError(f"Unsupported attack mode: {self.config.attack_mode}")
        except QueryBudgetExceeded:
            return self._budget_clean_fallback_result(x=x, label=label)
        finally:
            self._end_sample_query_budget()
