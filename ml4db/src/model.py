from __future__ import annotations

from abc import abstractmethod

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BaseModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError

    def __str__(self) -> str:
        """
        prints the number of trainable parameters
        """
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_params])
        return super().__str__() + f"\nTrainable parameters: {params}"


class WorkloadEmbedder(BaseModule):
    def __init__(self, n_metrics: int, n_knobs: int, workload_dim: int, n_hidden_units: list[int]) -> None:
        super().__init__()
        layers = [
            nn.Linear(n_metrics + n_knobs, n_hidden_units[0]),
            nn.BatchNorm1d(n_hidden_units[0]),
            nn.ReLU(),
        ]
        for i in range(1, len(n_hidden_units)):
            layers += [
                nn.Linear(n_hidden_units[i - 1], n_hidden_units[i]),
                nn.BatchNorm1d(n_hidden_units[i]),
                nn.ReLU(),
            ]
        layers += [nn.Linear(n_hidden_units[-1], workload_dim), nn.ReLU()]
        self.model = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedded_workload = self.model(features)
        return F.normalize(embedded_workload)


class TPSEstimator(BaseModule):
    def __init__(self, workload_dim, n_knobs, n_hidden_units: list[int]) -> None:
        super().__init__()
        layers = [nn.Linear(workload_dim + n_knobs, n_hidden_units[0]), nn.BatchNorm1d(n_hidden_units[0]), nn.ReLU()]
        for i in range(1, len(n_hidden_units)):
            layers += [nn.Linear(n_hidden_units[i - 1], n_hidden_units[i]), nn.BatchNorm1d(n_hidden_units[i]), nn.ReLU()]
        layers += [nn.Linear(n_hidden_units[-1], 1), nn.ReLU()]  # 出力がnon-negativeであることを保証する
        self.model = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def forward(self, workload, knobs) -> torch.Tensor:
        return self.model(torch.cat((workload, knobs), axis=1))
