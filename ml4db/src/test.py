from __future__ import annotations

from pathlib import Path

import hydra
import joblib
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler
from torch.nn import L1Loss
from torch.utils import data
from tqdm import tqdm

from model import TPSEstimator, WorkloadEmbedder
from utils.dataset import TPSEstimatorDataset
from utils.general import convert_config_str_to_path, fix_seed
from utils.loss import RMSLELoss


@hydra.main(version_base=None, config_path=Path("./config"), config_name="config")
def main(config: DictConfig) -> None:
    config = convert_config_str_to_path(config)
    fix_seed(config.seed)

    # read csv data
    dataset_dir = Path(config.dataset_dir)
    df_train_path = dataset_dir / "test.csv"
    df = pd.read_csv(df_train_path)

    # encode label column
    target_col = df[config.target_name]
    # DataFrame is composed of metrics and knobs
    df.drop(["label", config.target_name], axis=1, inplace=True)

    # convert disk_used_gb-used to data_disk_used_gb
    df["disk_used_gb-used"] -= config.system_disk_used
    df.rename(columns={"disk_used_gb-used": "data_disk_used_gb"}, inplace=True)

    # load metrics which will have been dropped
    drop_metric_names_path = config.output_dir / "drop_metric_names.txt"
    with open(drop_metric_names_path, "r") as f:
        drop_metric_names = [line.rstrip() for line in f.readlines()]
    df.drop(drop_metric_names, axis=1, inplace=True)

    # normalize data
    scaler_path = config.output_dir / "scaler.pkl"
    scaler: MinMaxScaler = joblib.load(scaler_path)
    df[df.columns.to_list()] = scaler.transform(df)

    # make tps_estimator dataset
    knob_df = df[config.knob_names]
    metric_df = df.drop(config.knob_names, axis=1)
    tps_estimator_dataset = TPSEstimatorDataset(metric_df=metric_df, knob_df=knob_df, target_col=target_col)
    dataloader = data.DataLoader(
        tps_estimator_dataset,
        batch_size=config.tps_estimator.batch_size,
        num_workers=config.tps_estimator.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    # prepare model for testing
    save_dir = config.output_dir / config.model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workload_embedder = WorkloadEmbedder(
        n_metrics=config.n_metrics,
        n_knobs=config.n_knobs,
        workload_dim=config.workload_dim,
        n_hidden_units=config.workload_embedder.hidden_units,
    ).to(device)
    best_workload_embedder_path = save_dir / f"best_workload_embedder.pth"
    workload_embedder.load_state_dict(torch.load(best_workload_embedder_path, map_location=device))
    workload_embedder.eval()

    tps_estimator = TPSEstimator(
        workload_dim=config.workload_dim,
        n_knobs=config.n_knobs,
        n_hidden_units=config.tps_estimator.hidden_units,
    ).to(device)
    best_tps_estimator_path = save_dir / f"best_tps_estimator.pth"
    tps_estimator.load_state_dict(torch.load(best_tps_estimator_path, map_location=device))
    tps_estimator.eval()

    loss_fn = RMSLELoss()
    metric_fn = L1Loss()

    log_file = config.output_dir / config.model_name / "log_test" / "loss_metric.txt"
    if not log_file.parent.exists():
        log_file.parent.mkdir()

    # test
    total_loss = 0.0
    total_l1_loss = 0.0
    with torch.no_grad():
        for metrics, knobs, targets in tqdm(dataloader):
            metrics, knobs, targets = metrics.to(device), knobs.to(device), targets.to(device)
            features = torch.cat((metrics, knobs), axis=1)
            workloads = workload_embedder(features)
            preds = tps_estimator(workloads, knobs)
            total_loss += loss_fn(preds.squeeze(), targets).item() * config.tps_estimator.batch_size
            total_l1_loss += metric_fn(preds.squeeze(), targets).item() * config.tps_estimator.batch_size
    avg_loss = total_loss / len(dataloader.dataset)
    avg_l1_loss = total_l1_loss / len(dataloader.dataset)
    with open(log_file, "w") as f:
        f.write(f"RMSLELoss: {avg_loss}\n")
        f.write(f"MAE(metric): {avg_l1_loss}")


if __name__ == "__main__":
    main()
