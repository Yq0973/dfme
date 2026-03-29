#!/usr/bin/env python3
import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate shrunken region settings.")
    parser.add_argument("--system", type=int, required=True)
    parser.add_argument("--preset", type=str, default="scale_adaptive_mainline_v1")
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--region_size", type=int, required=True)
    parser.add_argument("--region_candidates", type=int, required=True)
    parser.add_argument("--anchor_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topology_mode", type=str, default="auto")
    parser.add_argument("--noise_model", type=str, default="unknown")
    parser.add_argument("--save_level", type=str, default="summary_only")
    parser.add_argument("--exp_tag", type=str, default="")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_cfg = build_attack_config_from_preset(args.system, args.preset)
    anchor_size = int(args.anchor_size) if int(args.anchor_size) > 0 else min(
        int(base_cfg.anchor_size), int(args.region_size)
    )
    region_candidates = max(1, int(args.region_candidates))
    cfg = replace(
        base_cfg,
        region_size=int(args.region_size),
        anchor_size=anchor_size,
        region_candidates=region_candidates,
        region_budget_topk=min(max(1, int(base_cfg.region_budget_topk)), region_candidates),
    )

    summary = run_topology_latent_blackbox_attack(
        system_id=int(args.system),
        oracle_arch="resmlp",
        attack_class=1,
        max_samples=int(args.max_samples),
        run_seed=int(args.seed),
        device=device,
        attack_config=cfg,
        topology_mode=str(args.topology_mode),
        noise_model=str(args.noise_model),
        save_level=str(args.save_level),
        exp_tag=str(args.exp_tag),
    )
    compact = {
        "system_id": int(summary["system_id"]),
        "subset_size": int(summary["subset_size"]),
        "region_size": int(summary["attack_config"]["region_size"]),
        "region_candidates": int(summary["attack_config"]["region_candidates"]),
        "attack_success_rate": float(summary["attack_success_rate"]),
        "avg_queries": float(summary["avg_queries"]),
        "avg_probe_queries": float(summary["avg_probe_queries"]),
        "avg_search_queries": float(summary["avg_search_queries"]),
        "delta_over_clean_l2_ratio_mean": float(summary["delta_over_clean_l2_ratio_mean"]),
    }
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
