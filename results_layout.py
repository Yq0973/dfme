from __future__ import annotations

import re
from pathlib import Path

from .config import RESULTS_DIR


_SERIES_RE = re.compile(r"^(paper_[a-z0-9_]+?_v\d+)(?:_|$)")

_CATEGORY_RULES = [
    ("main_compare", "main_compare"),
    ("noise_model", "noise_model"),
    ("query_frontier", "query_frontier"),
    ("query_feedback", "query_feedback"),
    ("closed_loop", "closed_loop"),
    ("perturb_tradeoff", "perturb_tradeoff"),
    ("sample_figures", "sample_figures"),
    ("results_digest", "results_digest"),
    ("cross_detector", "cross_detector"),
    ("query_budget", "query_budget"),
    ("oracle_seed", "oracle_seed"),
    ("hardlabel", "hardlabel"),
    ("visual", "visualization"),
]


def normalize_exp_tag(exp_tag: str) -> str:
    exp_tag = str(exp_tag).strip()
    if exp_tag.endswith("_summary"):
        return exp_tag[: -len("_summary")]
    return exp_tag


def infer_series_key(exp_tag: str) -> str:
    exp_tag = normalize_exp_tag(exp_tag)
    if not exp_tag:
        return "adhoc"
    matched = _SERIES_RE.match(exp_tag)
    if matched:
        return matched.group(1)
    return exp_tag


def infer_category(exp_tag: str) -> str:
    series_key = infer_series_key(exp_tag)
    for needle, category in _CATEGORY_RULES:
        if needle in series_key:
            return category
    return "misc"


def resolve_series_root(exp_tag: str) -> Path:
    series_key = infer_series_key(exp_tag)
    category = infer_category(series_key)
    if series_key.startswith("paper_"):
        namespace = "paper"
    elif series_key.startswith("legacy_"):
        namespace = "legacy"
    else:
        namespace = "misc"
    return RESULTS_DIR / namespace / category / series_key


def resolve_summary_dir(exp_tag: str) -> Path:
    return resolve_series_root(exp_tag) / "summary"


def resolve_run_dir(system_id: int, exp_tag: str) -> Path:
    exp_tag = normalize_exp_tag(exp_tag)
    if not exp_tag:
        return RESULTS_DIR / "adhoc" / f"ieee_case{int(system_id)}"
    run_name = f"ieee_case{int(system_id)}_{exp_tag}"
    return resolve_series_root(exp_tag) / "runs" / run_name


def locate_summary_dir(exp_tag: str) -> Path:
    new_dir = resolve_summary_dir(exp_tag)
    old_dir = RESULTS_DIR / f"{normalize_exp_tag(exp_tag)}_summary"
    if new_dir.exists():
        return new_dir
    return old_dir


def locate_run_dir(system_id: int, exp_tag: str) -> Path:
    exp_tag = normalize_exp_tag(exp_tag)
    new_dir = resolve_run_dir(system_id=system_id, exp_tag=exp_tag)
    old_dir = RESULTS_DIR / f"ieee_case{int(system_id)}_{exp_tag}"
    if new_dir.exists():
        return new_dir
    return old_dir
