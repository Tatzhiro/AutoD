import random
import re
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils import data


def make_dirs(config: DictConfig) -> None:
    """
    output_dir, save_dir, log_dirを作成
    """
    save_dir: Path = config.output_dir / config.model_name
    for module_name in config.module_names:
        log_dir = save_dir / f"log_{module_name}"
        if not log_dir.exists():
            log_dir.mkdir(parents=True)


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_dataset(valid_split: float, dataset: data.Dataset, seed: int):
    valid_size = int(len(dataset) * valid_split)
    return data.random_split(dataset=dataset, lengths=[len(dataset) - valid_size, valid_size], generator=torch.Generator().manual_seed(seed))


def convert_config_str_to_path(config: DictConfig):
    for key, value in config.items():
        if re.match(r".*dir$", key):
            config[key] = Path(value)
    return config


def inverse_normalize(tensor: torch.Tensor, scaler: MinMaxScaler, col_indices: np.ndarray) -> torch.Tensor:
    min_values = scaler.min_[col_indices]
    scale_values = scaler.scale_[col_indices]
    return (tensor - min_values) / scale_values


def camel_to_snake(text: str) -> str:
    return re.sub(r"(?<!^)([^A-Z]*)([A-Z])([^A-Z])", r"\1_\2\3", text).lower()


def load_best_module(model: nn.Module, config: DictConfig, module_name: str, device) -> nn.Module:
    save_dir = config.output_dir / config.model_name
    best_module_path = save_dir / f"best_{module_name}.pth"
    model.load_state_dict(torch.load(best_module_path, map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model
