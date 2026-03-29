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
from topo_latent_blackbox.legacy_config import (  # noqa: E402
    list_legacy_presets,
    resolve_legacy_attack_preset,
)
from topo_latent_blackbox.pipeline import run_topology_latent_blackbox_attack  # noqa: E402


def _build_legacy_config(
    system_id: int,
    preset_name: str,
    args: argparse.Namespace,
) -> TopologyLatentAttackConfig:
    overrides = resolve_legacy_attack_preset(preset_name, system_id)
    cfg = TopologyLatentAttackConfig()
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    if int(args.region_size) > 0:
        cfg.region_size = int(args.region_size)
    if int(args.region_candidates) > 0:
        cfg.region_candidates = int(args.region_candidates)
    if int(args.anchor_pool_size) > 0:
        cfg.anchor_pool_size = int(args.anchor_pool_size)
    if int(args.state_basis_dim) > 0:
        cfg.state_basis_dim = int(args.state_basis_dim)
    if int(args.probe_directions) > 0:
        cfg.probe_directions = int(args.probe_directions)
    if int(args.population) > 0:
        cfg.population = int(args.population)
    if int(args.rounds) > 0:
        cfg.rounds = int(args.rounds)
    if args.feedback_loop:
        cfg.feedback_loop = True
    if args.disable_hierarchical_probe:
        cfg.hierarchical_probe = False
    if args.full_state_region:
        cfg.region_size = int(max(1, system_id - 1))

    return cfg


def _default_exp_tag(preset_name: str, system_id: int) -> str:
    return f"legacy_{preset_name}_case{int(system_id)}"


def main() -> None:
    preset_choices = list_legacy_presets()
    parser = argparse.ArgumentParser(
        description="Isolated legacy topology-latent black-box attack runner"
    )
    parser.add_argument("--systems", nargs="+", type=int, default=[14, 118])
    parser.add_argument(
        "--attack_preset",
        type=str,
        default=preset_choices[0],
        choices=preset_choices,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack_class", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument(
        "--topology_mode",
        type=str,
        default="auto",
        choices=["auto", "explicit", "jacobian"],
    )
    parser.add_argument(
        "--noise_model",
        type=str,
        default="known",
        choices=["known", "unknown", "isotropic"],
    )
    parser.add_argument("--oracle_arch", type=str, default="resmlp")
    parser.add_argument("--oracle_ckpt", type=str, default="")
    parser.add_argument("--save_level", type=str, default="summary_only", choices=["full", "summary_only"])
    parser.add_argument("--exp_tag", type=str, default="")

    # Optional overrides for quick ablation, without touching legacy preset source.
    parser.add_argument("--region_size", type=int, default=0)
    parser.add_argument("--region_candidates", type=int, default=0)
    parser.add_argument("--anchor_pool_size", type=int, default=0)
    parser.add_argument("--state_basis_dim", type=int, default=0)
    parser.add_argument("--probe_directions", type=int, default=0)
    parser.add_argument("--population", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--feedback_loop", action="store_true")
    parser.add_argument("--disable_hierarchical_probe", action="store_true")
    parser.add_argument("--full_state_region", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    print(f"Device: {device}")

    for system_id in args.systems:
        cfg = _build_legacy_config(
            system_id=system_id,
            preset_name=args.attack_preset,
            args=args,
        )
        exp_tag = (
            str(args.exp_tag).strip()
            if str(args.exp_tag).strip()
            else _default_exp_tag(preset_name=args.attack_preset, system_id=system_id)
        )
        summary = run_topology_latent_blackbox_attack(
            system_id=system_id,
            oracle_arch=args.oracle_arch,
            checkpoint_path=args.oracle_ckpt,
            attack_class=args.attack_class,
            max_samples=args.max_samples,
            exp_tag=exp_tag,
            run_seed=args.seed,
            device=device,
            attack_config=cfg,
            topology_mode=args.topology_mode,
            noise_model=args.noise_model,
            save_level=args.save_level,
        )
        print(
            f"[legacy] case{system_id} | preset={args.attack_preset} | "
            f"ASR={summary['attack_success_rate']:.2f}% | avg_queries={summary['avg_queries']:.2f}"
        )


if __name__ == "__main__":
    main()
