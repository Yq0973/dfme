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


from topo_latent_blackbox.config import resolve_system_config  # noqa: E402
from topo_latent_blackbox.data import load_matlab_fdia_bundle  # noqa: E402
from topo_latent_blackbox.experiment_utils import seed_everything  # noqa: E402
from topo_latent_blackbox.oracle import build_oracle_adapter  # noqa: E402
from topo_latent_blackbox.results_layout import resolve_run_dir, resolve_summary_dir  # noqa: E402
from topo_latent_blackbox.topology import StateCouplingTopology  # noqa: E402


METHOD_META = {
    "random": {"label": "Random", "color": "#9C755F", "marker": "o"},
    "fdia_support": {"label": "|c|", "color": "#4E79A7", "marker": "s"},
    "support_diffused": {"label": "|c| + A|c|", "color": "#E15759", "marker": "^"},
    "state_importance": {"label": "State importance", "color": "#76B7B2", "marker": "D"},
    "physical_pool": {"label": "Physical pool", "color": "#59A14F", "marker": "P"},
}

DEFAULT_POOL_SIZES = {
    14: [2, 4, 6, 8, 10, 13],
    118: [8, 16, 24, 32, 48, 64, 117],
}

DEFAULT_K_VALUES = {
    14: [1, 2, 3, 4],
    118: [4, 8, 12, 16],
}

DEFAULT_TARGET_K = {14: 2, 118: 8}
DEFAULT_REPORT_POOL = {14: 6, 118: 32}


def _parse_int_list(text: str, fallback: list[int]) -> list[int]:
    text = str(text).strip()
    if not text:
        return list(fallback)
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def _safe_name(method_name: str) -> str:
    return str(method_name).replace("|", "abs").replace(" ", "_")


def _series(values: torch.Tensor) -> dict[str, float]:
    values = values.reshape(-1).to(dtype=torch.float32)
    if values.numel() <= 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0}
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "median": float(values.median().item()),
        "q25": float(torch.quantile(values, 0.25).item()),
        "q75": float(torch.quantile(values, 0.75).item()),
    }


def _row_normalize(adjacency: torch.Tensor) -> torch.Tensor:
    adjacency = adjacency.to(dtype=torch.float32).clone()
    adjacency.fill_diagonal_(0.0)
    row_sum = adjacency.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return adjacency / row_sum


def _physical_adjacency(topology: StateCouplingTopology) -> torch.Tensor:
    if topology.explicit_adjacency is not None:
        return _row_normalize(topology.explicit_adjacency)
    return topology.adjacency.to(dtype=torch.float32)


def _state_footprint_counts(h_mat: torch.Tensor, tau: float) -> torch.Tensor:
    h_abs = h_mat.abs().to(dtype=torch.float32)
    col_max = h_abs.max(dim=0).values.clamp(min=1e-8)
    support = h_abs >= (float(tau) * col_max.unsqueeze(0))
    return support.sum(dim=0).to(dtype=torch.float32).clamp(min=1.0)


def _select_correct_fdia_indices(
    model: torch.nn.Module,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    attack_class: int,
    batch_size: int,
    device: torch.device,
    max_samples: int,
) -> torch.Tensor:
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(test_x.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(test_x.shape[0]))
            logits = model(test_x[start:end].to(device=device, dtype=torch.float32))
            preds.append(logits.argmax(dim=1).detach().cpu())
    pred = torch.cat(preds, dim=0)
    mask = (pred == test_y.cpu()) & (test_y.cpu() == int(attack_class))
    indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if max_samples > 0:
        indices = indices[: int(max_samples)]
    return indices


def _compute_state_margin_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    h_mat: torch.Tensor,
    batch_size: int,
    attack_class: int,
    device: torch.device,
) -> torch.Tensor:
    all_grads = []
    h_device = h_mat.to(device=device, dtype=torch.float32)
    model.eval()
    for start in range(0, int(x.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(x.shape[0]))
        x_batch = x[start:end].to(device=device, dtype=torch.float32).clone().detach()
        x_batch.requires_grad_(True)
        logits = model(x_batch)
        target = logits[:, int(attack_class)]
        other_idx = 1 - int(attack_class)
        margin = target - logits[:, other_idx]
        grad_x = torch.autograd.grad(margin.sum(), x_batch)[0]
        grad_state = grad_x @ h_device
        all_grads.append(grad_state.detach().abs().cpu())
    return torch.cat(all_grads, dim=0)


def _evaluate_rank_scores(
    method_name: str,
    grad_abs: torch.Tensor,
    score_rank: torch.Tensor,
    pool_sizes: list[int],
    k_values: list[int],
) -> tuple[list[dict], list[dict]]:
    n_samples, n_states = grad_abs.shape
    total_grad = grad_abs.sum(dim=1).clamp(min=1e-12)
    true_rank = torch.argsort(grad_abs, dim=1, descending=True)
    energy_rows: list[dict] = []
    recall_rows: list[dict] = []

    for pool_size in pool_sizes:
        pool_size = max(1, min(int(pool_size), int(n_states)))
        pool_idx = score_rank[:, :pool_size]
        captured = grad_abs.gather(1, pool_idx).sum(dim=1) / total_grad
        energy_stats = _series(captured)
        energy_rows.append(
            {
                "method": str(method_name),
                "pool_size": int(pool_size),
                "metric": "energy",
                **energy_stats,
            }
        )
        pool_mask = torch.zeros(n_samples, n_states, dtype=torch.bool)
        pool_mask.scatter_(1, pool_idx, True)
        for topk in k_values:
            if int(topk) > int(pool_size):
                continue
            topk = max(1, min(int(topk), int(n_states), int(pool_size)))
            true_idx = true_rank[:, :topk]
            recall = pool_mask.gather(1, true_idx).float().sum(dim=1) / float(topk)
            recall_stats = _series(recall)
            recall_rows.append(
                {
                    "method": str(method_name),
                    "pool_size": int(pool_size),
                    "metric": "recall",
                    "topk": int(topk),
                    **recall_stats,
                }
            )
    return energy_rows, recall_rows


def _evaluate_random_scores(
    grad_abs: torch.Tensor,
    pool_sizes: list[int],
    k_values: list[int],
    repeats: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    energy_acc: dict[int, list[torch.Tensor]] = {}
    recall_acc: dict[tuple[int, int], list[torch.Tensor]] = {}
    n_states = int(grad_abs.shape[1])
    true_rank = torch.argsort(grad_abs, dim=1, descending=True)
    total_grad = grad_abs.sum(dim=1).clamp(min=1e-12)

    for _ in range(max(1, int(repeats))):
        random_rank = torch.argsort(
            torch.rand(
                grad_abs.shape,
                generator=generator,
                dtype=grad_abs.dtype,
            ),
            dim=1,
            descending=True,
        )
        for pool_size in pool_sizes:
            pool_size = max(1, min(int(pool_size), int(n_states)))
            pool_idx = random_rank[:, :pool_size]
            captured = grad_abs.gather(1, pool_idx).sum(dim=1) / total_grad
            energy_acc.setdefault(int(pool_size), []).append(captured)
            pool_mask = torch.zeros_like(grad_abs, dtype=torch.bool)
            pool_mask.scatter_(1, pool_idx, True)
            for topk in k_values:
                if int(topk) > int(pool_size):
                    continue
                topk = max(1, min(int(topk), int(n_states), int(pool_size)))
                true_idx = true_rank[:, :topk]
                recall = pool_mask.gather(1, true_idx).float().sum(dim=1) / float(topk)
                recall_acc.setdefault((int(pool_size), int(topk)), []).append(recall)

    energy_rows = []
    for pool_size, values in energy_acc.items():
        merged = torch.stack(values, dim=0).mean(dim=0)
        stats = _series(merged)
        energy_rows.append(
            {
                "method": "random",
                "pool_size": int(pool_size),
                "metric": "energy",
                **stats,
            }
        )

    recall_rows = []
    for (pool_size, topk), values in recall_acc.items():
        merged = torch.stack(values, dim=0).mean(dim=0)
        stats = _series(merged)
        recall_rows.append(
            {
                "method": "random",
                "pool_size": int(pool_size),
                "metric": "recall",
                "topk": int(topk),
                **stats,
            }
        )
    return energy_rows, recall_rows


def _build_method_ranks(
    c_base: torch.Tensor,
    topology: StateCouplingTopology,
    footprint_counts: torch.Tensor,
    diffusion_lambda: float,
) -> dict[str, torch.Tensor]:
    c_abs = c_base.abs().to(dtype=torch.float32)
    physical_adj = _physical_adjacency(topology).to(dtype=torch.float32)
    support_diffused = c_abs + float(diffusion_lambda) * (c_abs @ physical_adj.T)
    state_importance = topology.state_importance.reshape(1, -1).to(dtype=torch.float32)
    footprint = footprint_counts.reshape(1, -1).to(dtype=torch.float32)
    physical_pool = support_diffused / footprint.clamp(min=1.0)
    scores = {
        "fdia_support": c_abs,
        "support_diffused": support_diffused,
        "state_importance": state_importance.expand(c_abs.shape[0], -1),
        "physical_pool": physical_pool,
    }
    return {
        name: torch.argsort(value, dim=1, descending=True)
        for name, value in scores.items()
    }


def _state_frequency(indices: torch.Tensor, n_states: int) -> torch.Tensor:
    counts = torch.zeros(n_states, dtype=torch.float32)
    if indices.numel() <= 0:
        return counts
    flat = indices.reshape(-1).to(dtype=torch.long)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    return counts


def _save_fig(fig: plt.Figure, stem_path: Path) -> None:
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_curve(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    stem_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for method_name, meta in METHOD_META.items():
        cur = df[df["method"] == method_name].sort_values("pool_size")
        if cur.empty:
            continue
        ax.plot(
            cur["pool_size"],
            cur[metric_col] * 100.0,
            color=meta["color"],
            linewidth=2.2,
            marker=meta["marker"],
            markersize=6.0,
            label=meta["label"],
        )
    ax.set_xlabel("Candidate pool size M")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, stem_path)


def _plot_frequency_bar(
    freq_df: pd.DataFrame,
    value_col: str,
    title: str,
    stem_path: Path,
    topn: int = 12,
) -> None:
    cur = freq_df.sort_values(value_col, ascending=False).head(int(topn)).copy()
    if cur.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.bar(cur["state_id"].astype(str), cur[value_col], color="#4E79A7")
    ax.set_xlabel("State id")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    _save_fig(fig, stem_path)


def _run_single_system(
    system_id: int,
    attack_class: int,
    oracle_arch: str,
    topology_mode: str,
    noise_model: str,
    batch_size: int,
    max_samples: int,
    footprint_tau: float,
    diffusion_lambda: float,
    random_repeats: int,
    seed: int,
    exp_tag: str,
    device: torch.device,
    pool_sizes: list[int],
    k_values: list[int],
) -> dict:
    cfg = resolve_system_config(system_id)
    bundle = load_matlab_fdia_bundle(str(cfg["data_dir"]), cfg["input_dim"])
    if bundle.test_c is None:
        raise RuntimeError("Dataset does not contain test_c_vectors.csv.")
    oracle_adapter = build_oracle_adapter(
        system_id=system_id,
        data_dir=Path(cfg["data_dir"]),
        input_dim=cfg["input_dim"],
        oracle_arch=oracle_arch,
        device=device,
    )
    model = oracle_adapter.model
    topology = StateCouplingTopology.from_data_dir(
        str(cfg["data_dir"]),
        topology_mode=topology_mode,
        noise_model=noise_model,
    )

    selected_idx = _select_correct_fdia_indices(
        model=model,
        test_x=bundle.test_x,
        test_y=bundle.test_y,
        attack_class=attack_class,
        batch_size=batch_size,
        device=device,
        max_samples=max_samples,
    )
    if selected_idx.numel() <= 0:
        raise RuntimeError(f"No correctly classified FDIA samples available for case{system_id}.")

    selected_x = bundle.test_x[selected_idx].clone().to(dtype=torch.float32)
    selected_c = bundle.test_c[selected_idx].clone().to(dtype=torch.float32)
    grad_abs = _compute_state_margin_gradients(
        model=model,
        x=selected_x,
        h_mat=topology.H,
        batch_size=batch_size,
        attack_class=attack_class,
        device=device,
    )

    footprint_counts = _state_footprint_counts(topology.H, tau=footprint_tau)
    method_ranks = _build_method_ranks(
        c_base=selected_c,
        topology=topology,
        footprint_counts=footprint_counts,
        diffusion_lambda=diffusion_lambda,
    )

    energy_rows, recall_rows = _evaluate_random_scores(
        grad_abs=grad_abs,
        pool_sizes=pool_sizes,
        k_values=k_values,
        repeats=random_repeats,
        seed=seed + int(system_id),
    )
    for method_name, score_rank in method_ranks.items():
        method_energy, method_recall = _evaluate_rank_scores(
            method_name=method_name,
            grad_abs=grad_abs,
            score_rank=score_rank,
            pool_sizes=pool_sizes,
            k_values=k_values,
        )
        energy_rows.extend(method_energy)
        recall_rows.extend(method_recall)

    run_dir = resolve_run_dir(system_id=system_id, exp_tag=exp_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    energy_df = pd.DataFrame(energy_rows).sort_values(["method", "pool_size"]).reset_index(drop=True)
    recall_df = pd.DataFrame(recall_rows).sort_values(["method", "topk", "pool_size"]).reset_index(drop=True)
    energy_df.to_csv(run_dir / "energy_metrics.csv", index=False)
    recall_df.to_csv(run_dir / "recall_metrics.csv", index=False)

    state_meta = pd.DataFrame(
        {
            "state_id": list(range(1, int(topology.n_states) + 1)),
            "bus_id": topology.state_bus_ids.reshape(-1).cpu().numpy(),
            "footprint_count": footprint_counts.reshape(-1).cpu().numpy(),
            "state_importance": topology.state_importance.reshape(-1).cpu().numpy(),
            "state_flow_support": topology.state_flow_support.reshape(-1).cpu().numpy(),
            "state_corridor_centrality": topology.state_corridor_centrality.reshape(-1).cpu().numpy(),
            "state_degree": topology.state_degree.reshape(-1).cpu().numpy(),
        }
    )
    state_meta.to_csv(run_dir / "state_metadata.csv", index=False)

    target_k = int(DEFAULT_TARGET_K.get(int(system_id), int(k_values[0])))
    target_k = max(1, min(target_k, int(topology.n_states)))
    representative_pool = int(DEFAULT_REPORT_POOL.get(int(system_id), int(pool_sizes[min(len(pool_sizes) - 1, 0)])))
    representative_pool = max(1, min(representative_pool, int(topology.n_states)))

    recall_target_df = recall_df[recall_df["topk"] == int(target_k)].copy()
    _plot_metric_curve(
        df=energy_df,
        metric_col="mean",
        ylabel="Energy capture (%)",
        title=f"Case {int(system_id)}: Energy@M",
        stem_path=figures_dir / "energy_vs_pool_size",
    )
    _plot_metric_curve(
        df=recall_target_df,
        metric_col="mean",
        ylabel=f"Recall@{int(target_k)} (%)",
        title=f"Case {int(system_id)}: Recall@{int(target_k)}",
        stem_path=figures_dir / f"recall_at_{int(target_k)}_vs_pool_size",
    )

    true_rank = torch.argsort(grad_abs, dim=1, descending=True)
    top_true = true_rank[:, :target_k]
    top_true_counts = _state_frequency(top_true, int(topology.n_states))
    physical_rank = method_ranks["physical_pool"][:, :representative_pool]
    physical_counts = _state_frequency(physical_rank, int(topology.n_states))
    freq_df = pd.DataFrame(
        {
            "state_id": list(range(1, int(topology.n_states) + 1)),
            "whitebox_topk_frequency": top_true_counts.cpu().numpy(),
            "physical_pool_frequency": physical_counts.cpu().numpy(),
        }
    )
    freq_df.to_csv(run_dir / "state_frequency.csv", index=False)
    _plot_frequency_bar(
        freq_df=freq_df,
        value_col="whitebox_topk_frequency",
        title=f"Case {int(system_id)}: white-box top-{int(target_k)} frequency",
        stem_path=figures_dir / "whitebox_topk_frequency",
    )
    _plot_frequency_bar(
        freq_df=freq_df,
        value_col="physical_pool_frequency",
        title=f"Case {int(system_id)}: physical-pool frequency",
        stem_path=figures_dir / "physical_pool_frequency",
    )

    summary = {
        "system_id": int(system_id),
        "subset_size": int(selected_idx.numel()),
        "n_states": int(topology.n_states),
        "n_measurements": int(topology.n_measurements),
        "graph_source": str(topology.graph_source),
        "noise_model": str(topology.noise_model),
        "attack_class": int(attack_class),
        "oracle_arch": str(oracle_arch),
        "batch_size": int(batch_size),
        "max_samples": int(max_samples),
        "footprint_tau": float(footprint_tau),
        "diffusion_lambda": float(diffusion_lambda),
        "random_repeats": int(random_repeats),
        "target_k": int(target_k),
        "representative_pool": int(representative_pool),
        "result_dir": str(run_dir),
    }
    (run_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "summary": summary,
        "energy_df": energy_df,
        "recall_df": recall_df,
        "run_dir": run_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="White-box diagnostic for physical candidate pool coverage."
    )
    parser.add_argument("--systems", nargs="+", type=int, default=[14, 118])
    parser.add_argument("--attack_class", type=int, default=1)
    parser.add_argument("--oracle_arch", type=str, default="resmlp")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--footprint_tau", type=float, default=0.10)
    parser.add_argument("--diffusion_lambda", type=float, default=0.50)
    parser.add_argument("--random_repeats", type=int, default=16)
    parser.add_argument(
        "--exp_tag",
        type=str,
        default="paper_query_frontier_whitebox_pool_diag_v1",
    )
    parser.add_argument("--pool_sizes_case14", type=str, default="")
    parser.add_argument("--pool_sizes_case118", type=str, default="")
    parser.add_argument("--k_values_case14", type=str, default="")
    parser.add_argument("--k_values_case118", type=str, default="")
    args = parser.parse_args()

    seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    combined_energy = []
    combined_recall = []
    run_summaries = []
    for system_id in args.systems:
        if int(system_id) == 14:
            pool_sizes = _parse_int_list(args.pool_sizes_case14, DEFAULT_POOL_SIZES[14])
            k_values = _parse_int_list(args.k_values_case14, DEFAULT_K_VALUES[14])
        elif int(system_id) == 118:
            pool_sizes = _parse_int_list(args.pool_sizes_case118, DEFAULT_POOL_SIZES[118])
            k_values = _parse_int_list(args.k_values_case118, DEFAULT_K_VALUES[118])
        else:
            raise ValueError(f"Unsupported system for diagnostic: {system_id}")

        result = _run_single_system(
            system_id=int(system_id),
            attack_class=int(args.attack_class),
            oracle_arch=str(args.oracle_arch),
            topology_mode=str(args.topology_mode),
            noise_model=str(args.noise_model),
            batch_size=int(args.batch_size),
            max_samples=int(args.max_samples),
            footprint_tau=float(args.footprint_tau),
            diffusion_lambda=float(args.diffusion_lambda),
            random_repeats=int(args.random_repeats),
            seed=int(args.seed),
            exp_tag=str(args.exp_tag),
            device=device,
            pool_sizes=pool_sizes,
            k_values=k_values,
        )
        energy_df = result["energy_df"].copy()
        energy_df.insert(0, "system_id", int(system_id))
        recall_df = result["recall_df"].copy()
        recall_df.insert(0, "system_id", int(system_id))
        combined_energy.append(energy_df)
        combined_recall.append(recall_df)
        run_summaries.append(result["summary"])
        print(
            f"case{int(system_id)} | subset={int(result['summary']['subset_size'])} | "
            f"target_k={int(result['summary']['target_k'])} | "
            f"representative_pool={int(result['summary']['representative_pool'])}"
        )

    summary_dir = resolve_summary_dir(str(args.exp_tag))
    summary_dir.mkdir(parents=True, exist_ok=True)
    if combined_energy:
        pd.concat(combined_energy, axis=0).to_csv(summary_dir / "combined_energy_metrics.csv", index=False)
    if combined_recall:
        pd.concat(combined_recall, axis=0).to_csv(summary_dir / "combined_recall_metrics.csv", index=False)
    (summary_dir / "run_summaries.json").write_text(
        json.dumps(run_summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved summary under: {summary_dir}")


if __name__ == "__main__":
    main()
