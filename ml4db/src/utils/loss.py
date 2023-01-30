import torch
from torch import nn


class MeanAbsoluteRelativeError(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, preds, targets) -> torch.Tensor:
        if (targets == 0).any():
            raise ZeroDivisionError("The ground truth has 0.")
        abs_error = torch.abs(preds.view_as(targets) - targets) / torch.abs(targets)
        # abs_error = torch.abs(preds - targets.view_as(preds)) / torch.abs(targets.view_as(preds))
        return abs_error.mean()


class RMSLELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        return torch.sqrt(self.mse(torch.log(preds.view_as(targets) + 1), torch.log(targets + 1)))
