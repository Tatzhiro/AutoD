from __future__ import annotations

from pathlib import Path

import hydra
import joblib
import pandas as pd
import torch
from adabound import AdaBound
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import TPSEstimator, WorkloadEmbedder
from utils.dataset import TPSEstimatorDataset
from utils.general import camel_to_snake, convert_config_str_to_path, fix_seed, inverse_normalize, load_best_module
from utils.visualization import plot_3d


@hydra.main(version_base=None, config_path=Path("./config"), config_name="config")
def main(config: DictConfig):
    config = convert_config_str_to_path(config)
    fix_seed(config.seed)

    # read csv data
    dataset_dir = config.dataset_dir
    save_dir = config.output_dir / config.model_name
    df_test_path = dataset_dir / "test.csv"
    df = pd.read_csv(df_test_path)

    # encode label column
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"].values)
    # average metric data among the same workload and the same knob settings
    df = df.groupby(df["innodb_io_capacity"].diff().fillna(0).abs().cumsum(), as_index=False).mean()

    # DataFrame is composed of metrics and knobs
    labels_encoded = df["label"]
    df.drop(["label", config.target_name], axis=1, inplace=True)

    # convert disk_used_gb-used to data_disk_used_gb
    df["disk_used_gb-used"] -= config.system_disk_used
    df.rename(columns={"disk_used_gb-used": "data_disk_used_gb"}, inplace=True)

    # load metrics which will have been dropped
    drop_metric_names_path = config.output_dir / "drop_metric_names.txt"
    with open(drop_metric_names_path, "r") as f:
        drop_metric_names = [line.rstrip() for line in f.readlines()]
    df.drop(drop_metric_names, axis=1, inplace=True)

    # normalize data with scaler fit to training data
    scaler_path = config.output_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path)
    df[df.columns.to_list()] = scaler.transform(df)
    # get the indices of knob column for inverse_normalize
    knob_col_indices = df.columns.get_indexer(config.knob_names)

    # make dataset
    knob_df = df[config.knob_names]
    metric_df = df.drop(config.knob_names, axis=1)
    tps_estimator_dataset = TPSEstimatorDataset(metric_df, knob_df)

    # make dataloader
    tps_estimator_dataloader = data.DataLoader(
        tps_estimator_dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

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

    # set writer
    log_dir = save_dir / "log_tune_knobs"
    writer = SummaryWriter(log_dir)

    # 最初にembedderを使ってworkloadを推定し，その後固定したlatent_workloadを使ってpredを推定し続ける
    tuned_knob_matrix = torch.zeros((len(tps_estimator_dataloader), config.n_knobs), dtype=torch.float32)

    workloads = torch.zeros((len(df), config.workload_dim))
    with tqdm(tps_estimator_dataloader, unit="data") as pbar:
        for data_idx, (metric, knob) in enumerate(pbar):
            pbar.set_description(f"[{data_idx}/{len(tps_estimator_dataloader)}]")

            metric, knob = metric.to(device), knob.to(device)
            workload = workload_embedder(torch.cat((metric, knob), axis=1))
            workloads[data_idx] = workload.cpu().detach()
            knob.requires_grad_(True)

            # optimizer
            optimizer = AdaBound(
                # [torch.nn.Parameter(knob)],
                [knob],
                lr=config.tune_knobs.lr,
                betas=config.tune_knobs.betas,
                final_lr=config.tune_knobs.final_lr,
                gamma=config.tune_knobs.gamma,
                eps=config.tune_knobs.eps,
                weight_decay=config.tune_knobs.weight_decay,
            )

            knob_pred_history = []
            not_improved_cnt = 0
            patience = 10
            max_target = 0

            for epoch in range(config.tune_knobs.epochs):
                pred = tps_estimator(workload, knob)
                knob_inverse_normalized = inverse_normalize(knob.detach().cpu().squeeze(), scaler, knob_col_indices)
                if pred >= max_target:
                    not_improved_cnt = 0
                    max_target = pred
                    tuned_knob_matrix[data_idx] = knob_inverse_normalized
                else:
                    not_improved_cnt += 1
                if not_improved_cnt > patience:
                    print(f"Early stopped at {epoch}/{config.tune_knobs.epochs}")
                    break
                optimizer.zero_grad()
                (-pred).backward()  # maximize pred
                optimizer.step()

                if data_idx % 100 == 0:
                    knob_pred_history.append(torch.Tensor([*knob_inverse_normalized, pred.detach().cpu().squeeze().clone()]))
                    writer.add_scalar(f"pred_tps (No.{data_idx})", pred, global_step=epoch)
                    for i in range(0, len(config.knob_names)):
                        writer.add_scalar(f"tuned_knob{i} (No.{data_idx})", knob_inverse_normalized[i], global_step=epoch)
            if config.tune_knobs.make_graph == True and data_idx % 100 == 0:
                plot_3d(
                    workload=workload,
                    knob_pred_history=knob_pred_history,
                    tps_estimator=tps_estimator,
                    scaler=scaler,
                    knob_col_indices=knob_col_indices,
                    knob_names=config.knob_names,
                    save_dir=save_dir / "figs",
                    fig_number=str(data_idx),
                )
    writer.close()
    # 異なるConfigで同じワークロードのものを平均する
    tuned_knob_df = pd.DataFrame(tuned_knob_matrix.numpy(), columns=config.knob_names)
    tuned_knob_df["label"] = labels_encoded
    tuned_knob_df = tuned_knob_df.groupby(tuned_knob_df["label"].diff().fillna(0).abs().cumsum(), as_index=False).mean()
    tuned_knob_df["label"] = label_encoder.inverse_transform(tuned_knob_df["label"].astype(int))

    # save tuned knob matrix
    save_path = save_dir / "tuned_knob.csv"
    tuned_knob_df.to_csv(save_path, index=False)

    # TODO: 学習率の異なる複数の optimizer を使うことによって local optimum に落ちることを回避


if __name__ == "__main__":
    main()
