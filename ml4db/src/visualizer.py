from __future__ import annotations

from pathlib import Path

import hydra
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from omegaconf import DictConfig

from model import TPSEstimator, WorkloadEmbedder
from utils.general import camel_to_snake, convert_config_str_to_path, fix_seed, load_best_module


@hydra.main(version_base=None, config_path=Path("./config"), config_name="config")
def visualizer(config: DictConfig):
    config = convert_config_str_to_path(config)
    fix_seed(config.seed)

    # make dataset
    dataset_dir = config.dataset_dir
    df_train_path = dataset_dir / "train.csv"
    df = pd.read_csv(df_train_path)

    # 同じワークロードで同じconfigのメトリックデータは平均を取る
    df = df.groupby(df["innodb_io_capacity"].diff().fillna(0).abs().cumsum(), as_index=False).mean()

    ## delete useless columns
    target_col = df[config.target_name]
    df.drop([config.target_name, "label"], axis=1, inplace=True)

    ## convert disk_used_gb-used to data_disk_used_gb
    df["disk_used_gb-used"] -= config.system_disk_used
    df.rename(columns={"disk_used_gb-used": "data_disk_used_gb"}, inplace=True)

    ## load metrics which will have been dropped
    drop_metric_names_path = config.output_dir / "drop_metric_names.txt"
    with open(drop_metric_names_path, "r") as f:
        drop_metric_names = [line.rstrip() for line in f.readlines()]
    df.drop(drop_metric_names, axis=1, inplace=True)

    ## normalize data with scaler fit to training data
    scaler_path = config.output_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path)
    df[df.columns.to_list()] = scaler.transform(df)
    knob_df = df[config.knob_names]

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workload_embedder = WorkloadEmbedder(
        n_metrics=config.n_metrics,
        n_knobs=config.n_knobs,
        workload_dim=config.workload_dim,
        n_hidden_units=config.workload_embedder.hidden_units,
    ).to(device)
    workload_embedder_module_name = camel_to_snake(workload_embedder.__class__.__name__)
    workload_embedder = load_best_module(model=workload_embedder, config=config, module_name=workload_embedder_module_name, device=device)

    tps_estimator = TPSEstimator(
        workload_dim=config.workload_dim,
        n_knobs=config.n_knobs,
        n_hidden_units=config.tps_estimator.hidden_units,
    ).to(device)
    tps_estimator_module_name = camel_to_snake(tps_estimator.__class__.__name__)
    tps_estimator = load_best_module(model=tps_estimator, config=config, module_name=tps_estimator_module_name, device=device)

    # fix workload
    features = torch.from_numpy(df.values).float().to(device)
    workloads: torch.Tensor = workload_embedder(features)

    tmp_knobs = []
    steps = 100
    for knob_name in config.knob_names:
        knob_col = knob_df[knob_name]
        knob_min, knob_max = knob_col.min(), knob_col.max()
        knob_linspace = torch.linspace(knob_min, knob_max, steps=steps)
        tmp_knobs.append(knob_linspace)

    knob_0, knob_1 = torch.meshgrid(tmp_knobs[0], tmp_knobs[1], indexing="xy")
    knobs = torch.hstack([knob_0.reshape(-1, 1), knob_1.reshape(-1, 1)])

    # plot
    for i in range(0, 5000, 500):
        print(f"TPS Observed Value(GT): {target_col.iloc[i]}")
        workload = workloads[i]
        workload_expanded = workload.expand(knobs.shape[0], -1)
        preds = tps_estimator(workload_expanded, knobs)
        ax = plt.axes(projection="3d")
        ax.plot_surface(knob_0, knob_1, preds.reshape_as(knob_0))
        ax.view_init(30, 45)
        plt.show()


if __name__ == "__main__":
    visualizer()
