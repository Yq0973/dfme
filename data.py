from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys

import pandas as pd
import torch

from .config import DFME_DIR


if str(DFME_DIR) not in sys.path:
    sys.path.insert(0, str(DFME_DIR))

from core.dataloader import ensure_1d_labels, load_fdia_data  # noqa: E402


@dataclass
class MatlabFDIABundle:
    data_dir: Path
    input_dim: int
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    train_mean: torch.Tensor
    train_std: torch.Tensor
    test_c: Optional[torch.Tensor]
    test_fdia_vectors: Optional[torch.Tensor]


def _read_optional_csv(path: Path, input_dim: int) -> Optional[torch.Tensor]:
    if not path.exists():
        return None
    values = pd.read_csv(path, header=None).values
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if input_dim > 0 and values.shape[1] >= input_dim:
        values = values[:, :input_dim]
    return torch.tensor(values, dtype=torch.float32)


def load_matlab_fdia_bundle(data_dir: str, input_dim: int) -> MatlabFDIABundle:
    data_path = Path(data_dir)
    train_x, train_y, test_x, test_y, mean_np, std_np = load_fdia_data(
        str(data_path), input_dim
    )
    test_y = ensure_1d_labels(test_y, name="matlab_test_labels")
    train_y = ensure_1d_labels(train_y, name="matlab_train_labels")

    return MatlabFDIABundle(
        data_dir=data_path,
        input_dim=input_dim,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        train_mean=torch.tensor(mean_np[:input_dim], dtype=torch.float32),
        train_std=torch.tensor(std_np[:input_dim], dtype=torch.float32),
        test_c=_read_optional_csv(data_path / "test_c_vectors.csv", input_dim=0),
        test_fdia_vectors=_read_optional_csv(
            data_path / "test_fdia_vectors.csv", input_dim=input_dim
        ),
    )

