from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch
from torch.utils import data


class WorkloadEmbedderDataset(data.Dataset):
    def __init__(self, metric_knob_df: pd.DataFrame, labels: np.ndarray) -> None:
        super().__init__()
        self.features = torch.from_numpy(metric_knob_df.values).float()
        self.labels = torch.from_numpy(labels).int()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        return self.features[idx], self.labels[idx]


class TPSEstimatorDataset(data.Dataset):
    def __init__(self, metric_df: pd.DataFrame, knob_df: pd.DataFrame, target_col: pd.Series | None = None) -> None:
        super().__init__()
        self.metrics = torch.from_numpy(metric_df.values).float()
        self.knobs = torch.from_numpy(knob_df.values).float()
        self.targets = target_col.to_numpy(dtype=np.float32) if target_col is not None else None

    def __len__(self) -> int:
        return len(self.metrics)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, float] | tuple[torch.Tensor, torch.Tensor]:
        if self.targets is not None:
            return self.metrics[idx], self.knobs[idx], self.targets[idx]
        else:
            return self.metrics[idx], self.knobs[idx]


def get_drop_metric_names(
    df: pd.DataFrame,
    drop_metrics_prior: list[str] = ["timestamp"],
    not_drop_metric_pattern: str = r"disk_used_gb-used|query-",
) -> list[str]:

    # 値が一定のメトリックを取り除く
    drop_metrics_all: list[str] = [*df.loc[:, df.nunique() == 1].columns]
    # 事前知識として不要であると考えられるメトリックを削除
    drop_metrics_all += drop_metrics_prior
    df = df.drop(drop_metrics_all, axis=1, inplace=False)

    step = 0.04
    ths = np.arange(1.0, 0.95 - step, -step)

    tmp_drop_metrics_all = []

    drop_lim = 30
    drop_num = 0
    df_corr = df.corr()
    for th in ths:
        corr_mat = df_corr.to_numpy()
        cols = df_corr.columns

        # 相関が th 以上 or -th 以下のメトリックを取り出す
        high_corr_dict = {k: set() for k in cols}
        for i, j in zip(*np.where((corr_mat >= th) | (corr_mat <= -th))):
            if i < j:
                # 事前知識として削除したくないメトリックは削除しないようにする
                if not re.match(not_drop_metric_pattern, cols[i]):
                    high_corr_dict[cols[i]].add(cols[j])
                if not re.match(not_drop_metric_pattern, cols[j]):
                    high_corr_dict[cols[j]].add(cols[i])
        drop_metrics = []
        while drop_num < drop_lim:
            # 相関が高いメトリック間の関係数をメトリック別に列挙
            # （メトリックごとの関係数を相関係数の和で代用してもいい）
            drop_metric = max(high_corr_dict.items(), key=lambda item: len(item[1]))[0]
            if len(high_corr_dict[drop_metric]) == 0:
                break
            # keyを削除
            high_corr_dict.pop(drop_metric, None)
            # value(=set)の要素を削除
            for v_set in high_corr_dict.values():
                if drop_metric in v_set:
                    v_set.discard(drop_metric)
            drop_metrics.append(drop_metric)
            drop_num += 1
        df_corr = df_corr.drop(drop_metrics, axis=0).drop(drop_metrics, axis=1)
        drop_metrics_all += drop_metrics
        tmp_drop_metrics_all += drop_metrics
    return drop_metrics_all
