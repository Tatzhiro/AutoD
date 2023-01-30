from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import LightSource
from model import TPSEstimator
from sklearn.preprocessing import MinMaxScaler

from utils.general import inverse_normalize


def plot_3d(
    workload: torch.Tensor,
    tps_estimator: TPSEstimator,
    scaler: MinMaxScaler,
    knob_col_indices: np.ndarray,
    knob_names: list[str],
    knob_pred_history: list[torch.Tensor] | None = None,  # Noneの時，勾配上昇法で辿るルートが表示されない
    save_dir: Path | None = None,
    fig_number: str | None = None,
):
    steps = 100
    grid = torch.linspace(0, 1, steps=steps)
    knob_0, knob_1 = torch.meshgrid(grid, grid, indexing="xy")
    knobs = torch.hstack([knob_0.reshape(-1, 1), knob_1.reshape(-1, 1)])
    # plot
    workload_expanded = workload.expand(knobs.shape[0], -1)
    preds = tps_estimator(workload_expanded, knobs)
    preds = preds.reshape_as(knob_0)

    knob_0 = inverse_normalize(knob_0, scaler, knob_col_indices[0])
    knob_1 = inverse_normalize(knob_1, scaler, knob_col_indices[1])

    ax = plt.axes(projection="3d")
    fontsize = 12
    ax.set_xlabel(knob_names[0].replace("_", " ").replace("innodb", "").lstrip(), labelpad=5, fontsize=fontsize)
    ax.set_ylabel(knob_names[1].replace("_", " ").replace("innodb", "").lstrip(), labelpad=10, fontsize=fontsize)
    ax.set_zlabel("TPS", labelpad=10, fontsize=fontsize)
    ax.xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ls = LightSource(270, 45)
    rgb = ls.shade(preds.numpy(), cmap=cm.gist_earth, vert_exag=0.1, blend_mode="soft")
    ax.plot_surface(knob_0, knob_1, preds, facecolors=rgb, linewidth=0, antialiased=True, shade=False, alpha=0.75)
    if knob_pred_history is not None and len(knob_pred_history) > 0:
        # list to tensor
        knob_pred_history = torch.stack(knob_pred_history).T
        ax.plot(knob_pred_history[0], knob_pred_history[1], knob_pred_history[2], "o-", color="red", linewidth=2, markersize=4)
    ax.view_init(30, 250)

    # when saving fig, not show it.
    if save_dir is not None and fig_number is not None:
        if not save_dir.exists():
            save_dir.mkdir()
        plt.savefig(save_dir / f"TPS_dist_{fig_number}.png", dpi=300, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()
