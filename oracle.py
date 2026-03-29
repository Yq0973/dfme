from dataclasses import dataclass
from pathlib import Path
import sys

import torch

from .config import DFME_DIR


if str(DFME_DIR) not in sys.path:
    sys.path.insert(0, str(DFME_DIR))

from core.oracle_pipeline import load_oracle_model, resolve_system_config  # noqa: E402


@dataclass
class OracleQueryResult:
    pred: torch.Tensor
    fdia_prob: torch.Tensor
    normal_prob: torch.Tensor
    fdia_margin: torch.Tensor


class QueryBudgetExceeded(RuntimeError):
    def __init__(self, requested: int, remaining: int) -> None:
        super().__init__(
            f"Per-sample query budget exceeded: requested {int(requested)} queries, "
            f"but only {int(remaining)} remain."
        )
        self.requested = int(requested)
        self.remaining = int(remaining)


class OracleQueryAdapter:
    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.query_count = 0
        self._attack_query_budget: int | None = None
        self._attack_query_start = 0

    def reset_queries(self) -> None:
        self.query_count = 0
        self._attack_query_start = 0

    def begin_attack_budget(self, max_queries: int | None) -> None:
        max_queries = None if max_queries is None else int(max_queries)
        if max_queries is not None and max_queries <= 0:
            max_queries = 0
        self._attack_query_budget = max_queries
        self._attack_query_start = int(self.query_count)

    def clear_attack_budget(self) -> None:
        self._attack_query_budget = None
        self._attack_query_start = int(self.query_count)

    def attack_queries_used(self) -> int:
        return max(0, int(self.query_count) - int(self._attack_query_start))

    def remaining_attack_budget(self) -> int | None:
        if self._attack_query_budget is None:
            return None
        used = self.attack_queries_used()
        return max(0, int(self._attack_query_budget) - int(used))

    def query(self, x: torch.Tensor) -> OracleQueryResult:
        requested = int(x.shape[0])
        remaining = self.remaining_attack_budget()
        if remaining is not None and requested > remaining:
            raise QueryBudgetExceeded(requested=requested, remaining=remaining)
        x = x.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
        self.query_count += int(x.shape[0])
        return OracleQueryResult(
            pred=logits.argmax(dim=1).detach().cpu(),
            fdia_prob=probs[:, 1].detach().cpu(),
            normal_prob=probs[:, 0].detach().cpu(),
            fdia_margin=(logits[:, 1] - logits[:, 0]).detach().cpu(),
        )


def build_oracle_adapter(
    system_id: int,
    data_dir: Path,
    input_dim: int,
    oracle_arch: str = "resmlp",
    checkpoint_path: str = "",
    device: torch.device | None = None,
) -> OracleQueryAdapter:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = resolve_system_config(system_id, data_dir=str(data_dir), input_dim=input_dim)
    model, _ = load_oracle_model(
        system_id=system_id,
        cfg=cfg,
        oracle_arch=oracle_arch,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return OracleQueryAdapter(model=model, device=device)
