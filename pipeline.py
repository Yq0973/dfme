import json
from pathlib import Path
from typing import Optional
import sys

import pandas as pd
import torch
from scipy.stats import chi2, norm

from .attack import TopologyLatentAttackConfig, TopologyLatentQueryAttack
from .config import DFME_DIR, resolve_system_config
from .data import load_matlab_fdia_bundle
from .oracle import build_oracle_adapter
from .results_layout import resolve_run_dir
from .topology import StateCouplingTopology


if str(DFME_DIR) not in sys.path:
    sys.path.insert(0, str(DFME_DIR))

from core.dc_physics import try_build_dc_physics_analyzer  # noqa: E402


def _series_distribution(values: list[float | int]) -> dict:
    numeric = [float(v) for v in values]
    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "q25": None,
            "q75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    series = pd.Series(numeric, dtype="float64")
    return {
        "count": int(series.shape[0]),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "q25": float(series.quantile(0.25)),
        "q75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def _query_distribution_rows(per_sample: list[dict]) -> list[dict]:
    scopes = {
        "total_all": [float(row["queries_used"]) for row in per_sample],
        "probe_all": [float(row["proposal_queries"]) for row in per_sample],
        "search_all": [float(row["search_queries"]) for row in per_sample],
        "total_success": [
            float(row["queries_used"]) for row in per_sample if int(row["success"]) == 1
        ],
        "total_failure": [
            float(row["queries_used"]) for row in per_sample if int(row["success"]) == 0
        ],
        "search_success": [
            float(row["search_queries"]) for row in per_sample if int(row["success"]) == 1
        ],
        "search_failure": [
            float(row["search_queries"]) for row in per_sample if int(row["success"]) == 0
        ],
    }
    rows = []
    for scope_name, values in scopes.items():
        rows.append({"scope": scope_name, **_series_distribution(values)})
    return rows


def _query_budget_curve(per_sample: list[dict], budgets: list[int]) -> list[dict]:
    total = max(1, len(per_sample))
    rows = []
    for budget in budgets:
        finished = [row for row in per_sample if float(row["queries_used"]) <= float(budget)]
        success = [row for row in finished if int(row["success"]) == 1]
        rows.append(
            {
                "budget": int(budget),
                "finished_count": int(len(finished)),
                "finished_rate": float(len(finished) * 100.0 / total),
                "success_count": int(len(success)),
                "success_rate": float(len(success) * 100.0 / total),
                "conditional_success_rate": float(len(success) * 100.0 / max(1, len(finished))),
            }
        )
    return rows


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).to(dtype=torch.float32)
    b = b.reshape(-1).to(dtype=torch.float32)
    denom = float(torch.linalg.norm(a).item() * torch.linalg.norm(b).item())
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def _safe_projection_ratio(total: torch.Tensor, base: torch.Tensor) -> float:
    total = total.reshape(-1).to(dtype=torch.float32)
    base = base.reshape(-1).to(dtype=torch.float32)
    denom = float(torch.dot(base, base).item())
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(total, base).item() / denom)


def _fdia_effect_metrics(
    c_base: torch.Tensor,
    delta_c: torch.Tensor,
    a_base: torch.Tensor,
    delta_z: torch.Tensor,
) -> dict:
    c_base = c_base.reshape(-1).to(dtype=torch.float32)
    delta_c = delta_c.reshape(-1).to(dtype=torch.float32)
    a_base = a_base.reshape(-1).to(dtype=torch.float32)
    delta_z = delta_z.reshape(-1).to(dtype=torch.float32)

    c_total = c_base + delta_c
    a_total = a_base + delta_z
    support = c_base.abs() > 1e-10

    state_base_norm = float(torch.linalg.norm(c_base).item())
    state_total_norm = float(torch.linalg.norm(c_total).item())
    meas_base_norm = float(torch.linalg.norm(a_base).item())
    meas_total_norm = float(torch.linalg.norm(a_total).item())

    metrics = {
        "state_base_l2": state_base_norm,
        "state_total_l2": state_total_norm,
        "state_total_over_base_ratio": state_total_norm / max(state_base_norm, 1e-12),
        "state_projection_ratio": _safe_projection_ratio(c_total, c_base),
        "state_cosine": _safe_cosine(c_total, c_base),
        "measurement_base_l2": meas_base_norm,
        "measurement_total_l2": meas_total_norm,
        "measurement_total_over_base_ratio": meas_total_norm / max(meas_base_norm, 1e-12),
        "measurement_projection_ratio": _safe_projection_ratio(a_total, a_base),
        "measurement_cosine": _safe_cosine(a_total, a_base),
        "state_support_size": int(support.sum().item()),
    }

    if bool(support.any().item()):
        c_base_support = c_base[support]
        c_total_support = c_total[support]
        c_total_offsupport = c_total[~support]
        base_support_norm = float(torch.linalg.norm(c_base_support).item())
        total_support_norm = float(torch.linalg.norm(c_total_support).item())
        flip_rate = float(
            ((c_base_support * c_total_support) < 0).float().mean().item()
        )
        metrics.update(
            {
                "state_support_retention_ratio": total_support_norm
                / max(base_support_norm, 1e-12),
                "state_support_projection_ratio": _safe_projection_ratio(
                    c_total_support, c_base_support
                ),
                "state_support_cosine": _safe_cosine(c_total_support, c_base_support),
                "state_support_flip_rate": flip_rate,
                "state_offsupport_energy_ratio": float(
                    torch.linalg.norm(c_total_offsupport).item()
                    / max(base_support_norm, 1e-12)
                ),
            }
        )
    else:
        metrics.update(
            {
                "state_support_retention_ratio": 0.0,
                "state_support_projection_ratio": 0.0,
                "state_support_cosine": 0.0,
                "state_support_flip_rate": 0.0,
                "state_offsupport_energy_ratio": 0.0,
            }
        )

    return metrics


def _expand_delta_c_full(
    topology: StateCouplingTopology,
    delta_c_region: torch.Tensor,
    region: torch.Tensor,
    region_space: str,
    delta_z: torch.Tensor,
) -> torch.Tensor:
    region_space = str(region_space).lower()
    if region_space == "state":
        full = torch.zeros(topology.n_states, dtype=torch.float32)
        full[region.reshape(-1).long()] = delta_c_region.reshape(-1).to(dtype=torch.float32)
        return full
    return topology.estimate_state_from_measurement(delta_z).reshape(-1).to(dtype=torch.float32)


def _select_correct_fdia_samples(
    oracle_adapter,
    bundle,
    attack_class: int,
    max_samples: int,
) -> dict:
    oracle_adapter.reset_queries()
    query = oracle_adapter.query(bundle.test_x)
    oracle_adapter.reset_queries()

    mask = (query.pred == bundle.test_y.cpu()) & (bundle.test_y.cpu() == int(attack_class))
    indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if max_samples > 0:
        indices = indices[:max_samples]

    if indices.numel() == 0:
        raise RuntimeError("No correctly classified FDIA samples available.")

    selected = {
        "indices": indices,
        "x": bundle.test_x[indices].clone(),
        "y": bundle.test_y[indices].clone(),
    }
    if bundle.test_c is not None:
        selected["c"] = bundle.test_c[indices].clone()
    else:
        selected["c"] = torch.zeros(indices.numel(), bundle.input_dim, dtype=torch.float32)
    if bundle.test_fdia_vectors is not None:
        selected["fdia"] = bundle.test_fdia_vectors[indices].clone()
    return selected


def _bdd_summary(physics_analyzer, z: torch.Tensor, alpha: float = 0.05) -> dict:
    z = z.to(dtype=torch.float32)
    dof = physics_analyzer.n_meas - physics_analyzer.n_states
    chi2_threshold = float(chi2.ppf(1.0 - alpha, dof))
    lnrt_threshold = float(norm.ppf(1.0 - alpha / (2.0 * physics_analyzer.n_meas)))
    weighted_residual = physics_analyzer.weighted_residual_energy(z)
    max_norm_res = physics_analyzer.max_normalized_residual(z)
    return {
        "chi2_threshold": chi2_threshold,
        "lnrt_threshold": lnrt_threshold,
        "chi2_not_flagged_ratio": float(
            (weighted_residual <= chi2_threshold).float().mean().item() * 100.0
        ),
        "lnrt_not_flagged_ratio": float(
            (max_norm_res <= lnrt_threshold).float().mean().item() * 100.0
        ),
    }


def _push_success_to_budget_boundary(
    attacker: TopologyLatentQueryAttack,
    oracle_adapter,
    x_ref: torch.Tensor,
    label: int,
    c_base: torch.Tensor,
    attack_out: dict,
    push_ratio: float,
    search_steps: int,
) -> dict:
    updated = dict(attack_out)
    updated.update(
        {
            "budget_boundary_push_used": False,
            "budget_boundary_push_queries": 0,
            "budget_boundary_push_target_ratio": 0.0,
            "budget_boundary_push_steps": max(0, int(search_steps)),
            "budget_boundary_push_scale": 1.0,
        }
    )

    cap_ratio = float(attacker.config.measurement_delta_l2_ratio_cap)
    push_ratio = max(0.0, float(push_ratio))
    search_steps = max(0, int(search_steps))
    if (
        not bool(updated.get("success", False))
        or cap_ratio <= 0.0
        or push_ratio <= 0.0
    ):
        return updated

    x0 = x_ref.reshape(1, -1).to(dtype=torch.float32)
    c_base = c_base.reshape(-1).to(dtype=torch.float32)
    delta_c0 = updated["delta_c"].reshape(1, -1).to(dtype=torch.float32)
    delta_z0 = updated["delta_z"].reshape(1, -1).to(dtype=torch.float32)
    delta_z_norm = float(torch.linalg.norm(delta_z0).item())
    if delta_z_norm <= 1e-12:
        return updated

    ref_norm = float(torch.linalg.norm(x0).item())
    if ref_norm <= 1e-12:
        return updated

    target_ratio = min(1.0, push_ratio) * cap_ratio
    target_norm = target_ratio * ref_norm
    updated["budget_boundary_push_target_ratio"] = float(target_ratio)
    if target_norm <= delta_z_norm * (1.0 + 1e-6):
        return updated

    scale_measure = target_norm / max(delta_z_norm, 1e-12)
    scale_state = float("inf")
    delta_c_norm = float(torch.linalg.norm(delta_c0).item())
    radius = float(updated.get("radius", 0.0))
    if delta_c_norm > 1e-12 and radius > 0.0:
        scale_state = radius / delta_c_norm
    scale_max = min(scale_measure, scale_state)
    if scale_max <= 1.0 + 1e-6:
        return updated

    query_count = 0
    region = updated["region"]

    def _evaluate(scale: float) -> dict:
        nonlocal query_count
        scaled_delta_c = delta_c0 * float(scale)
        scaled_delta_z = delta_z0 * float(scale)
        scaled_delta_c, scaled_delta_z = attacker._apply_measurement_l2_cap(
            x_ref=x0,
            delta_c=scaled_delta_c,
            delta_z=scaled_delta_z,
        )
        adv = x0 + scaled_delta_z
        query = oracle_adapter.query(adv)
        query_count += 1
        score = attacker._score(query)
        objective = attacker._selection_objective(
            score,
            x0,
            scaled_delta_z,
            c_base=c_base,
            delta_c=scaled_delta_c,
            region=region,
        )
        pred = int(query.pred[0].item())
        return {
            "success": bool(pred != int(label)),
            "pred": pred,
            "score": float(score[0].item()),
            "objective": float(objective[0].item()),
            "delta_c": scaled_delta_c.clone(),
            "delta_z": scaled_delta_z.clone(),
            "adv_x": adv.clone(),
            "scale": float(scale),
        }

    best = {
        "delta_c": delta_c0.clone(),
        "delta_z": delta_z0.clone(),
        "adv_x": updated["adv_x"].reshape(1, -1).to(dtype=torch.float32).clone(),
        "pred": int(updated["final_pred"]),
        "score": float(updated["final_score"]),
        "objective": float(updated["final_objective"]),
        "scale": 1.0,
    }

    scale_max_eval = _evaluate(scale_max)
    if bool(scale_max_eval["success"]):
        best = scale_max_eval
    elif search_steps > 0:
        low = 1.0
        high = float(scale_max)
        for _ in range(search_steps):
            mid = 0.5 * (low + high)
            if mid <= low + 1e-6:
                break
            candidate = _evaluate(mid)
            if bool(candidate["success"]):
                low = float(mid)
                best = candidate
            else:
                high = float(mid)

    updated["queries_used"] = int(updated["queries_used"]) + int(query_count)
    updated["budget_boundary_push_queries"] = int(query_count)
    updated["budget_boundary_push_scale"] = float(best["scale"])
    if float(best["scale"]) <= 1.0 + 1e-6:
        return updated

    updated["budget_boundary_push_used"] = True
    updated["adv_x"] = best["adv_x"].squeeze(0).clone()
    updated["delta_c"] = best["delta_c"].squeeze(0).clone()
    updated["delta_z"] = best["delta_z"].squeeze(0).clone()
    updated["final_pred"] = int(best["pred"])
    updated["final_score"] = float(best["score"])
    updated["final_objective"] = float(best["objective"])
    return updated


def run_topology_latent_blackbox_attack(
    system_id: int,
    oracle_arch: str = "resmlp",
    data_dir: str = "",
    input_dim: int = 0,
    attack_class: int = 1,
    max_samples: int = 64,
    checkpoint_path: str = "",
    exp_tag: str = "",
    run_seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    attack_config: Optional[TopologyLatentAttackConfig] = None,
    topology_mode: str = "auto",
    noise_model: str = "known",
    save_level: str = "full",
    budget_boundary_push_ratio: float = 0.0,
    budget_boundary_search_steps: int = 0,
    budget_variant_name: Optional[str] = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if attack_config is None:
        attack_config = TopologyLatentAttackConfig()
    if save_level not in {"full", "summary_only"}:
        raise ValueError("Unsupported save_level. Expected one of: full, summary_only.")

    cfg = resolve_system_config(system_id, data_dir=data_dir, input_dim=input_dim)
    bundle = load_matlab_fdia_bundle(str(cfg["data_dir"]), cfg["input_dim"])
    oracle_adapter = build_oracle_adapter(
        system_id=system_id,
        data_dir=Path(cfg["data_dir"]),
        input_dim=cfg["input_dim"],
        oracle_arch=oracle_arch,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    selected = _select_correct_fdia_samples(
        oracle_adapter=oracle_adapter,
        bundle=bundle,
        attack_class=attack_class,
        max_samples=max_samples,
    )
    topology = StateCouplingTopology.from_data_dir(
        str(cfg["data_dir"]),
        topology_mode=topology_mode,
        noise_model=noise_model,
    )
    if "fdia" not in selected:
        selected["fdia"] = selected["c"].to(dtype=torch.float32) @ topology.H.T.to(
            dtype=torch.float32
        )
    attacker = TopologyLatentQueryAttack(
        topology=topology,
        oracle=oracle_adapter,
        config=attack_config,
    )
    physics_analyzer = try_build_dc_physics_analyzer(
        data_dir=str(cfg["data_dir"]),
        prefer_adjusted_h=True,
        expected_input_dim=cfg["input_dim"],
        device=device,
    )

    result_dir = resolve_run_dir(system_id=system_id, exp_tag=exp_tag)
    result_dir.mkdir(parents=True, exist_ok=True)
    oracle_adapter.reset_queries()

    adv_x = []
    delta_x = []
    per_sample = []
    candidate_probe_rows = []
    for row_id in range(selected["x"].shape[0]):
        sample_generator = None
        if run_seed is not None:
            sample_generator = torch.Generator(device="cpu")
            sample_generator.manual_seed(int(run_seed) + int(selected["indices"][row_id].item()))
        attack_out = attacker.attack_sample(
            x=selected["x"][row_id],
            label=int(selected["y"][row_id].item()),
            c_base=selected["c"][row_id],
            generator=sample_generator,
        )
        attack_out = _push_success_to_budget_boundary(
            attacker=attacker,
            oracle_adapter=oracle_adapter,
            x_ref=selected["x"][row_id],
            label=int(selected["y"][row_id].item()),
            c_base=selected["c"][row_id],
            attack_out=attack_out,
            push_ratio=budget_boundary_push_ratio,
            search_steps=budget_boundary_search_steps,
        )
        delta_c_full = _expand_delta_c_full(
            topology=topology,
            delta_c_region=attack_out["delta_c"],
            region=attack_out["region"],
            region_space=str(attack_out.get("region_space", "state")),
            delta_z=attack_out["delta_z"],
        )
        effect_metrics = _fdia_effect_metrics(
            c_base=selected["c"][row_id],
            delta_c=delta_c_full,
            a_base=selected["fdia"][row_id],
            delta_z=attack_out["delta_z"],
        )
        adv_x.append(attack_out["adv_x"].unsqueeze(0))
        delta_x.append(attack_out["delta_z"].unsqueeze(0))
        region_summary = topology.summarize_region(
            attack_out["region"],
            region_space=str(attack_out.get("region_space", "state")),
            x_ref=selected["x"][row_id],
            c_base=selected["c"][row_id],
        )
        per_sample.append(
            {
                "dataset_index": int(selected["indices"][row_id].item()),
                "attack_mode": str(attack_config.attack_mode),
                "region_space": str(attack_out.get("region_space", "state")),
                "success": int(attack_out["success"]),
                "queries_used": int(attack_out["queries_used"]),
                "proposal_queries": int(attack_out["proposal_queries"]),
                "search_queries": int(
                    attack_out["queries_used"] - attack_out["proposal_queries"]
                ),
                "final_pred": int(attack_out["final_pred"]),
                "final_score": float(attack_out["final_score"]),
                "final_objective": float(attack_out["final_objective"]),
                "radius": float(attack_out["radius"]),
                "selected_region_rank": int(attack_out["selected_region_rank"]),
                "region_candidate_count": int(attack_out["region_candidate_count"]),
                "budget_region_count": int(attack_out["budget_region_count"]),
                "budget_triggered": int(attack_out["budget_triggered"]),
                "budget_region_ranks": attack_out["budget_region_ranks"],
                "clean_score": float(attack_out["clean_score"]),
                "probe_best_score": float(attack_out["probe_best_score"]),
                "probe_second_score": float(attack_out["probe_second_score"]),
                "probe_score_gap": float(attack_out["probe_score_gap"]),
                "probe_improvement": float(attack_out["probe_improvement"]),
                "probe_best_rank": int(attack_out["probe_best_rank"]),
                "probe_second_rank": int(attack_out["probe_second_rank"]),
                "probe_best_prior": float(attack_out["probe_best_prior"]),
                "probe_second_prior": float(attack_out["probe_second_prior"]),
                "probe_success_count": int(attack_out["probe_success_count"]),
                "detector_feedback_used": int(
                    bool(attack_out.get("detector_feedback_used", False))
                ),
                "detector_feedback_candidate_count": int(
                    attack_out.get("detector_feedback_candidate_count", 0)
                ),
                "detector_feedback_total_reward": float(
                    attack_out.get("detector_feedback_total_reward", 0.0)
                ),
                "detector_feedback_best_reward": float(
                    attack_out.get("detector_feedback_best_reward", 0.0)
                ),
                "probe_improvement_ratio": float(attack_out["probe_improvement_ratio"]),
                "query_budget_population": int(attack_out["query_budget_population"]),
                "query_budget_rounds": int(attack_out["query_budget_rounds"]),
                "adaptive_query_budget_used": int(attack_out["adaptive_query_budget_used"]),
                "query_cap_reached": int(bool(attack_out.get("query_cap_reached", False))),
                "selected_region_physics_quality": float(
                    attack_out.get("selected_region_physics_quality", 0.0)
                ),
                "selected_region_allocation_priority": float(
                    attack_out.get("selected_region_allocation_priority", 0.0)
                ),
                "selected_region_progress_ratio": float(
                    attack_out.get("selected_region_progress_ratio", 0.0)
                ),
                "selected_region_boundary_ratio": float(
                    attack_out.get("selected_region_boundary_ratio", 0.0)
                ),
                "selected_region_boundary_uncertainty": float(
                    attack_out.get("selected_region_boundary_uncertainty", 0.0)
                ),
                "selected_region_guard_probe_attempts": int(
                    attack_out.get("selected_region_guard_probe_attempts", 0)
                ),
                "selected_region_stagnant_rounds": int(
                    attack_out.get("selected_region_stagnant_rounds", 0)
                ),
                "selected_region_challenger_branch": int(
                    bool(attack_out.get("selected_region_challenger_branch", False))
                ),
                "selected_region_early_stopped": int(
                    bool(attack_out.get("selected_region_early_stopped", False))
                ),
                "active_set_size": int(
                    attack_out.get("active_set_size", topology.n_states)
                ),
                "active_set_stage_index": int(
                    attack_out.get("active_set_stage_index", -1)
                ),
                "active_set_keep_ratio": float(
                    attack_out.get("active_set_keep_ratio", 1.0)
                ),
                "layered_success_stage": int(
                    attack_out.get("layered_success_stage", -1)
                ),
                "layered_total_stages": int(
                    attack_out.get("layered_total_stages", 1)
                ),
                "measurement_gate_enabled": int(
                    bool(attack_out.get("measurement_gate_enabled", False))
                ),
                "measurement_gate_keep_dim": int(
                    attack_out.get("measurement_gate_keep_dim", topology.n_measurements)
                ),
                "measurement_gate_keep_ratio": float(
                    attack_out.get("measurement_gate_keep_ratio", 1.0)
                ),
                "state_gate_enabled": int(bool(attack_out.get("state_gate_enabled", False))),
                "state_gate_keep_dim": int(
                    attack_out.get("state_gate_keep_dim", topology.n_states)
                ),
                "state_gate_keep_ratio": float(
                    attack_out.get("state_gate_keep_ratio", 1.0)
                ),
                "clean_x_l2": float(torch.linalg.norm(selected["x"][row_id]).item()),
                "delta_c_l2": float(torch.linalg.norm(delta_c_full).item()),
                "delta_z_l2": float(torch.linalg.norm(attack_out["delta_z"]).item()),
                "delta_over_clean_l2_ratio": float(
                    torch.linalg.norm(attack_out["delta_z"]).item()
                    / max(torch.linalg.norm(selected["x"][row_id]).item(), 1e-12)
                ),
                "budget_boundary_push_used": int(
                    bool(attack_out.get("budget_boundary_push_used", False))
                ),
                "budget_boundary_push_queries": int(
                    attack_out.get("budget_boundary_push_queries", 0)
                ),
                "budget_boundary_push_target_ratio": float(
                    attack_out.get("budget_boundary_push_target_ratio", 0.0)
                ),
                "budget_boundary_push_steps": int(
                    attack_out.get("budget_boundary_push_steps", 0)
                ),
                "budget_boundary_push_scale": float(
                    attack_out.get("budget_boundary_push_scale", 1.0)
                ),
                **effect_metrics,
                **region_summary,
            }
        )
        candidate_rows = attack_out.get("candidate_rows", [])
        for candidate_row in candidate_rows:
            candidate_region_space = str(
                candidate_row.get("region_space", attack_out.get("region_space", "state"))
            )
            candidate_region_summary = topology.summarize_region(
                candidate_row["region"],
                region_space=candidate_region_space,
                x_ref=selected["x"][row_id],
                c_base=selected["c"][row_id],
            )
            candidate_probe_rows.append(
                {
                    "dataset_index": int(selected["indices"][row_id].item()),
                    "attack_mode": str(attack_config.attack_mode),
                    "region_space": candidate_region_space,
                    "sample_success": int(attack_out["success"]),
                    "candidate_rank": int(candidate_row["rank"]),
                    "candidate_is_probe_best": int(
                        int(candidate_row["rank"]) == int(attack_out["probe_best_rank"])
                    ),
                    "candidate_is_selected": int(
                        int(candidate_row["rank"]) == int(attack_out["selected_region_rank"])
                    ),
                    "candidate_in_budget": int(
                        int(candidate_row["rank"]) in set(int(v) for v in attack_out["budget_region_ranks"])
                    ),
                    "candidate_source": str(candidate_row.get("proposal_source", "prior")),
                    "candidate_feedback_reward": float(
                        candidate_row.get("feedback_reward", 0.0)
                    ),
                    "candidate_best_success": int(bool(candidate_row["best_success"])),
                    "candidate_best_score": float(candidate_row["best_score"]),
                    "candidate_selection_objective": float(
                        candidate_row["selection_objective"]
                    ),
                    "candidate_probe_improvement": float(
                        attack_out["clean_score"] - float(candidate_row["best_score"])
                    ),
                    "candidate_probe_scale": float(candidate_row["probe_scale"]),
                    "candidate_region_prior": float(candidate_row["region_prior"]),
                    "active_set_size": int(
                        candidate_row.get(
                            "active_set_size",
                            attack_out.get("active_set_size", topology.n_states),
                        )
                    ),
                    "active_set_stage_index": int(
                        candidate_row.get(
                            "active_set_stage_index",
                            attack_out.get("active_set_stage_index", -1),
                        )
                    ),
                    "selected_region_rank": int(attack_out["selected_region_rank"]),
                    "probe_best_rank": int(attack_out["probe_best_rank"]),
                    "probe_second_rank": int(attack_out["probe_second_rank"]),
                    "region_candidate_count": int(attack_out["region_candidate_count"]),
                    "budget_region_ranks": attack_out["budget_region_ranks"],
                    **candidate_region_summary,
                }
            )

    adv_x = torch.cat(adv_x, dim=0)
    delta_x = torch.cat(delta_x, dim=0)
    adv_query = oracle_adapter.query(adv_x)

    success_mask = adv_query.pred != selected["y"].cpu()
    asr = float(success_mask.float().mean().item() * 100.0)
    avg_search_queries = float(
        sum(float(row["search_queries"]) for row in per_sample) / max(1, len(per_sample))
    )
    avg_attack_queries = float(
        sum(float(row["queries_used"]) for row in per_sample) / max(1, len(per_sample))
    )
    avg_probe_queries = float(
        sum(float(row["proposal_queries"]) for row in per_sample)
        / max(1, len(per_sample))
    )
    query_distribution_rows = _query_distribution_rows(per_sample)
    query_distribution_map = {
        str(row["scope"]): {k: v for k, v in row.items() if k != "scope"}
        for row in query_distribution_rows
    }
    query_budget_curve = _query_budget_curve(
        per_sample=per_sample,
        budgets=[25, 40, 50, 60, 75, 100, 150, 200],
    )
    summary = {
        "system_id": int(system_id),
        "data_dir": str(cfg["data_dir"]),
        "result_dir": str(result_dir),
        "exp_tag": str(exp_tag),
        "subset_size": int(selected["x"].shape[0]),
        "run_seed": None if run_seed is None else int(run_seed),
        "oracle_arch": str(oracle_arch),
        "oracle_checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "attack_success_rate": asr,
        "avg_queries": avg_attack_queries,
        "avg_search_queries": avg_search_queries,
        "avg_probe_queries": avg_probe_queries,
        "avg_total_oracle_queries": float(
            oracle_adapter.query_count / max(1, selected["x"].shape[0])
        ),
        "query_cap_reached_ratio": float(
            sum(float(row.get("query_cap_reached", 0.0)) for row in per_sample)
            / max(1, len(per_sample))
        ),
        "topology_mode": topology_mode,
        "topology_source": topology.graph_source,
        "noise_model": str(topology.noise_model),
        "attack_mode": str(attack_config.attack_mode),
        "query_mode": attack_config.query_mode,
        "save_level": save_level,
        "budget_boundary_push_ratio": float(budget_boundary_push_ratio),
        "budget_boundary_search_steps": int(budget_boundary_search_steps),
        "budget_variant": (
            str(budget_variant_name)
            if budget_variant_name is not None
            else (
                "boundary_push"
                if float(budget_boundary_push_ratio) > 0.0
                and int(budget_boundary_search_steps) > 0
                else "minimum_success"
            )
        ),
        "query_distribution": query_distribution_map,
        "query_budget_curve": query_budget_curve,
        "attack_config": {
            "attack_mode": attack_config.attack_mode,
            "state_seed_mode": attack_config.state_seed_mode,
            "max_queries_per_sample": attack_config.max_queries_per_sample,
            "region_size": attack_config.region_size,
            "anchor_size": attack_config.anchor_size,
            "region_candidates": attack_config.region_candidates,
            "anchor_pool_size": attack_config.anchor_pool_size,
            "probabilistic_region_prior": attack_config.probabilistic_region_prior,
            "prior_temperature": attack_config.prior_temperature,
            "prior_uniform_mixing": attack_config.prior_uniform_mixing,
            "hierarchical_probe": attack_config.hierarchical_probe,
            "coarse_probe_directions": attack_config.coarse_probe_directions,
            "fine_probe_topk": attack_config.fine_probe_topk,
            "initial_probe_region_topk": attack_config.initial_probe_region_topk,
            "probe_expand_improvement_ratio": attack_config.probe_expand_improvement_ratio,
            "region_proposal_mode": attack_config.region_proposal_mode,
            "proposal_diffusion_steps": attack_config.proposal_diffusion_steps,
            "proposal_diffusion_alpha": attack_config.proposal_diffusion_alpha,
            "proposal_flow_weight": attack_config.proposal_flow_weight,
            "proposal_corridor_weight": attack_config.proposal_corridor_weight,
            "probe_directions": attack_config.probe_directions,
            "probe_scale_ratio": attack_config.probe_scale_ratio,
            "population": attack_config.population,
            "rounds": attack_config.rounds,
            "radius_ratio": attack_config.radius_ratio,
            "radius_floor": attack_config.radius_floor,
            "step_ratio": attack_config.step_ratio,
            "step_decay": attack_config.step_decay,
            "patience": attack_config.patience,
            "min_step_ratio": attack_config.min_step_ratio,
            "max_step_shrinks": attack_config.max_step_shrinks,
            "measurement_delta_l2_ratio_cap": attack_config.measurement_delta_l2_ratio_cap,
            "realism_penalty_weight": attack_config.realism_penalty_weight,
            "fdia_preserve_weight": attack_config.fdia_preserve_weight,
            "fdia_backbone_lock_ratio": attack_config.fdia_backbone_lock_ratio,
            "fdia_state_projection_min": attack_config.fdia_state_projection_min,
            "fdia_measurement_projection_min": attack_config.fdia_measurement_projection_min,
            "fdia_offsupport_max": attack_config.fdia_offsupport_max,
            "fdia_state_penalty_weight": attack_config.fdia_state_penalty_weight,
            "fdia_measurement_penalty_weight": attack_config.fdia_measurement_penalty_weight,
            "fdia_offsupport_penalty_weight": attack_config.fdia_offsupport_penalty_weight,
            "region_budget_topk": attack_config.region_budget_topk,
            "region_budget_explore_rounds": attack_config.region_budget_explore_rounds,
            "feedback_loop": attack_config.feedback_loop,
            "feedback_round_chunk": attack_config.feedback_round_chunk,
            "feedback_branch_topk": attack_config.feedback_branch_topk,
            "feedback_stagnation_trigger": attack_config.feedback_stagnation_trigger,
            "feedback_probe_gap_abs_threshold": attack_config.feedback_probe_gap_abs_threshold,
            "feedback_probe_gap_ratio_threshold": attack_config.feedback_probe_gap_ratio_threshold,
            "feedback_keep_probe_best_incumbent": attack_config.feedback_keep_probe_best_incumbent,
            "feedback_min_state_dim": attack_config.feedback_min_state_dim,
            "detector_feedback_region": attack_config.detector_feedback_region,
            "detector_feedback_prior_mix": attack_config.detector_feedback_prior_mix,
            "detector_feedback_success_bonus": attack_config.detector_feedback_success_bonus,
            "detector_feedback_min_gain": attack_config.detector_feedback_min_gain,
            "feedback_physics_reward_shaping": attack_config.feedback_physics_reward_shaping,
            "feedback_physics_weight": attack_config.feedback_physics_weight,
            "region_budget_score_slack": attack_config.region_budget_score_slack,
            "budget_region_max_probe_best_prior": attack_config.budget_region_max_probe_best_prior,
            "budget_region_prior_tiebreak": attack_config.budget_region_prior_tiebreak,
            "measurement_suppression_strength": attack_config.measurement_suppression_strength,
            "injection_scale": attack_config.injection_scale,
            "flow_scale": attack_config.flow_scale,
            "leverage_suppression": attack_config.leverage_suppression,
            "min_channel_scale": attack_config.min_channel_scale,
            "probe_channel_shaping_only": attack_config.probe_channel_shaping_only,
            "structured_search_directions": attack_config.structured_search_directions,
            "structured_direction_count": attack_config.structured_direction_count,
            "measurement_basis_search": attack_config.measurement_basis_search,
            "measurement_bandit_search": attack_config.measurement_bandit_search,
            "state_basis_search": attack_config.state_basis_search,
            "state_bandit_search": attack_config.state_bandit_search,
            "state_subspace_pgzoo": attack_config.state_subspace_pgzoo,
            "state_basis_dim": attack_config.state_basis_dim,
            "bandit_direction_samples": attack_config.bandit_direction_samples,
            "bandit_momentum": attack_config.bandit_momentum,
            "bandit_exploration_ratio": attack_config.bandit_exploration_ratio,
            "bandit_warmup_rounds": attack_config.bandit_warmup_rounds,
            "pgzoo_probe_pairs": attack_config.pgzoo_probe_pairs,
            "pgzoo_alpha_ratio": attack_config.pgzoo_alpha_ratio,
            "pgzoo_momentum": attack_config.pgzoo_momentum,
            "pgzoo_line_candidates": attack_config.pgzoo_line_candidates,
            "pgzoo_query_prior_weight": attack_config.pgzoo_query_prior_weight,
            "pgzoo_prior_topology_weight": attack_config.pgzoo_prior_topology_weight,
            "pgzoo_prior_base_weight": attack_config.pgzoo_prior_base_weight,
            "pgzoo_prior_best_weight": attack_config.pgzoo_prior_best_weight,
            "pgzoo_structured_covariance": attack_config.pgzoo_structured_covariance,
            "pgzoo_physical_preconditioner": attack_config.pgzoo_physical_preconditioner,
            "pgzoo_covariance_gamma": attack_config.pgzoo_covariance_gamma,
            "pgzoo_covariance_ridge": attack_config.pgzoo_covariance_ridge,
            "pgzoo_preconditioner_ridge": attack_config.pgzoo_preconditioner_ridge,
            "support_pool_size": attack_config.support_pool_size,
            "support_final_size": attack_config.support_final_size,
            "support_diffusion_lambda": attack_config.support_diffusion_lambda,
            "support_probe_scale_ratio": attack_config.support_probe_scale_ratio,
            "support_prior_weight": attack_config.support_prior_weight,
            "support_diversity_penalty": attack_config.support_diversity_penalty,
            "support_success_bonus": attack_config.support_success_bonus,
            "support_keep_base_support": attack_config.support_keep_base_support,
            "adaptive_support_selection": attack_config.adaptive_support_selection,
            "adaptive_support_mass_threshold": attack_config.adaptive_support_mass_threshold,
            "adaptive_support_max_size": attack_config.adaptive_support_max_size,
            "measurement_guided_state_refine": attack_config.measurement_guided_state_refine,
            "guided_state_region_size": attack_config.guided_state_region_size,
            "physical_measurement_gate": attack_config.physical_measurement_gate,
            "measurement_gate_ratio": attack_config.measurement_gate_ratio,
            "measurement_gate_response_weight": attack_config.measurement_gate_response_weight,
            "measurement_gate_effectiveness_weight": attack_config.measurement_gate_effectiveness_weight,
            "measurement_conditioned_state_gate": attack_config.measurement_conditioned_state_gate,
            "state_gate_ratio": attack_config.state_gate_ratio,
            "state_gate_escape_candidates": attack_config.state_gate_escape_candidates,
            "adaptive_query_budget": attack_config.adaptive_query_budget,
            "easy_probe_improvement_ratio": attack_config.easy_probe_improvement_ratio,
            "easy_population": attack_config.easy_population,
            "easy_rounds": attack_config.easy_rounds,
            "single_active_size": attack_config.single_active_size,
            "active_set_sizes": list(attack_config.active_set_sizes)
            if isinstance(attack_config.active_set_sizes, (list, tuple))
            else attack_config.active_set_sizes,
            "layered_stage_rounds": attack_config.layered_stage_rounds,
            "layered_final_rounds": attack_config.layered_final_rounds,
            "multisource_physical_prior": attack_config.multisource_physical_prior,
            "physics_query_allocation": attack_config.physics_query_allocation,
            "physics_query_topk": attack_config.physics_query_topk,
            "physics_query_priority_weight": attack_config.physics_query_priority_weight,
            "adaptive_challenger_budget": attack_config.adaptive_challenger_budget,
            "challenger_population_ratio": attack_config.challenger_population_ratio,
            "challenger_rounds": attack_config.challenger_rounds,
            "branch_pruning": attack_config.branch_pruning,
            "branch_prune_progress_ratio": attack_config.branch_prune_progress_ratio,
            "branch_prune_score_gap_ratio": attack_config.branch_prune_score_gap_ratio,
            "score_stagnation_early_stop": attack_config.score_stagnation_early_stop,
            "score_stagnation_rounds": attack_config.score_stagnation_rounds,
            "score_gain_ratio_threshold": attack_config.score_gain_ratio_threshold,
            "search_min_rounds_before_stop": attack_config.search_min_rounds_before_stop,
            "uncertainty_aware_pruning": attack_config.uncertainty_aware_pruning,
            "physics_aware_early_stop": attack_config.physics_aware_early_stop,
            "termination_uncertainty_ratio_tau": attack_config.termination_uncertainty_ratio_tau,
            "termination_uncertainty_floor": attack_config.termination_uncertainty_floor,
            "termination_progress_ratio_floor": attack_config.termination_progress_ratio_floor,
            "termination_physics_quality_floor": attack_config.termination_physics_quality_floor,
            "guarded_boundary_probe": attack_config.guarded_boundary_probe,
            "guarded_boundary_probe_steps": attack_config.guarded_boundary_probe_steps,
            "guarded_boundary_probe_max_uses": attack_config.guarded_boundary_probe_max_uses,
            "fd_warmup_rounds": attack_config.fd_warmup_rounds,
            "fd_warmup_topk": attack_config.fd_warmup_topk,
            "fd_region_selection": attack_config.fd_region_selection,
            "fd_iterations": attack_config.fd_iterations,
            "fd_coordinate_eps_ratio": attack_config.fd_coordinate_eps_ratio,
            "fd_coordinate_eps_floor": attack_config.fd_coordinate_eps_floor,
            "fd_central_gradient": attack_config.fd_central_gradient,
            "fd_line_steps": attack_config.fd_line_steps,
            "fd_line_decay": attack_config.fd_line_decay,
            "sparse_support_beta": attack_config.sparse_support_beta,
            "sparse_seed_weight": attack_config.sparse_seed_weight,
            "sparse_region_mix": attack_config.sparse_region_mix,
            "sparse_region_penalty": attack_config.sparse_region_penalty,
            "budget_objective_mode": attack_config.budget_objective_mode,
            "early_stop_on_success": attack_config.early_stop_on_success,
        },
    }
    summary["adaptive_query_budget_ratio"] = float(
        sum(float(row["adaptive_query_budget_used"]) for row in per_sample)
        / max(1, len(per_sample))
    )
    summary["budget_boundary_push_applied_ratio"] = float(
        sum(float(row["budget_boundary_push_used"]) for row in per_sample)
        / max(1, len(per_sample))
    )
    summary["budget_boundary_push_queries_mean"] = float(
        sum(float(row["budget_boundary_push_queries"]) for row in per_sample)
        / max(1, len(per_sample))
    )
    summary["budget_boundary_push_scale_mean"] = float(
        sum(float(row["budget_boundary_push_scale"]) for row in per_sample)
        / max(1, len(per_sample))
    )

    if physics_analyzer is not None:
        summary["clean_bdd"] = _bdd_summary(physics_analyzer, selected["x"])
        summary["adv_bdd"] = _bdd_summary(physics_analyzer, adv_x)
        summary["clean_samples"] = physics_analyzer.summarize_samples(selected["x"])
        summary["adv_samples"] = physics_analyzer.summarize_samples(adv_x)
        summary["delta_summary"] = physics_analyzer.summarize_delta(delta_x)
    summary["delta_over_clean_l2_ratio_mean"] = float(
        sum(float(row["delta_over_clean_l2_ratio"]) for row in per_sample) / max(1, len(per_sample))
    )
    summary["delta_over_clean_l2_ratio_median"] = float(
        pd.Series([float(row["delta_over_clean_l2_ratio"]) for row in per_sample]).median()
    )
    effect_keys = [
        "state_base_l2",
        "state_total_l2",
        "state_total_over_base_ratio",
        "state_projection_ratio",
        "state_cosine",
        "state_support_size",
        "state_support_retention_ratio",
        "state_support_projection_ratio",
        "state_support_cosine",
        "state_support_flip_rate",
        "state_offsupport_energy_ratio",
        "measurement_base_l2",
        "measurement_total_l2",
        "measurement_total_over_base_ratio",
        "measurement_projection_ratio",
        "measurement_cosine",
    ]
    summary["fdia_effect_summary"] = {}
    for key in effect_keys:
        values = [float(row[key]) for row in per_sample]
        summary["fdia_effect_summary"][f"{key}_mean"] = float(pd.Series(values).mean())
        summary["fdia_effect_summary"][f"{key}_median"] = float(pd.Series(values).median())

    region_metric_keys = [
        "region_measurement_support_count",
        "region_measurement_support_ratio",
        "region_sparse_efficiency_mean",
        "region_boundary_penalty",
        "region_connectivity_score",
    ]
    for key in region_metric_keys:
        values = [float(row[key]) for row in per_sample if key in row]
        if values:
            summary[f"{key}_mean"] = float(pd.Series(values).mean())
            summary[f"{key}_median"] = float(pd.Series(values).median())

    with open(result_dir / "attack_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame(query_distribution_rows).to_csv(
        result_dir / "query_distribution_summary.csv",
        index=False,
    )
    pd.DataFrame(query_budget_curve).to_csv(
        result_dir / "query_budget_curve.csv",
        index=False,
    )
    if save_level == "full":
        pd.DataFrame(per_sample).to_csv(result_dir / "per_sample_metrics.csv", index=False)
        if candidate_probe_rows:
            pd.DataFrame(candidate_probe_rows).to_csv(
                result_dir / "candidate_probe_metrics.csv",
                index=False,
            )

    return summary
