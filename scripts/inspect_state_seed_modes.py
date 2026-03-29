#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from topo_latent_blackbox.config import resolve_system_config  # noqa: E402
from topo_latent_blackbox.data import load_matlab_fdia_bundle  # noqa: E402
from topo_latent_blackbox.topology import StateCouplingTopology  # noqa: E402


def _first_attack_sample(bundle) -> int:
    mask = bundle.test_y == 1
    indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if indices.numel() <= 0:
        raise RuntimeError("No FDIA-positive samples found in test set.")
    return int(indices[0].item())


def _print_mode_summary(
    score_name: str,
    score: torch.Tensor,
    support: torch.Tensor,
    topk: int,
) -> None:
    top_idx = torch.topk(score.reshape(-1), k=min(int(topk), int(score.numel()))).indices.tolist()
    print(f"{score_name}:")
    for rank, idx in enumerate(top_idx, start=1):
        print(
            f"  {rank:>2d}. state={int(idx):>3d} "
            f"score={float(score[idx]):.4f} support={int(support[idx].item())}"
        )
    non_support_hits = sum(int((~support)[idx].item()) for idx in top_idx)
    print(f"  top{len(top_idx)} non-support count = {non_support_hits}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect legacy and target-radiated state seeds.")
    parser.add_argument("--systems", nargs="+", type=int, default=[14, 118])
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--noise_model",
        type=str,
        default="known",
        choices=["known", "unknown", "isotropic"],
    )
    parser.add_argument(
        "--topology_mode",
        type=str,
        default="auto",
        choices=["auto", "explicit", "jacobian"],
    )
    args = parser.parse_args()

    for system_id in args.systems:
        cfg = resolve_system_config(system_id)
        bundle = load_matlab_fdia_bundle(str(cfg["data_dir"]), cfg["input_dim"])
        topology = StateCouplingTopology.from_data_dir(
            str(cfg["data_dir"]),
            topology_mode=args.topology_mode,
            noise_model=args.noise_model,
        )
        if bundle.test_c is None:
            raise RuntimeError("Dataset does not contain test_c_vectors.csv.")

        sample_idx = _first_attack_sample(bundle)
        c_base = bundle.test_c[sample_idx].reshape(-1).to(dtype=torch.float32)
        support = c_base.abs() > 1e-10
        legacy = topology._legacy_seed_score(c_base)
        target_radiated = topology.state_target_radiated_prior(c_base)

        print(
            f"=== IEEE-{system_id} sample_index={sample_idx} "
            f"support_size={int(support.sum().item())} ==="
        )
        _print_mode_summary("legacy", legacy, support, args.topk)
        _print_mode_summary("target_radiated", target_radiated, support, args.topk)
        print("")


if __name__ == "__main__":
    main()
