from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import optuna
import pandas as pd
import torch
from adabound import AdaBound
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.optim import lr_scheduler
from torch.utils import data
from tqdm import tqdm

from model import TPSEstimator, WorkloadEmbedder
from utils.dataset import TPSEstimatorDataset, WorkloadEmbedderDataset, get_drop_metric_names
from utils.general import fix_seed, split_dataset
from utils.loss import RMSLELoss

optuna.logging.disable_default_handler()

"""
ハイパラ固定ずみのWorkloadEmbedderを「一度だけ」学習何度もOptunaを回す
"""


seed = 3407
fix_seed(seed)

epochs = 100
n_metrics = 30
n_knobs = 2
workload_dim = 40
workload_embedder_n_hidden_units = [40, 50, 50]
num_workers = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

acc_metric_name = "mean_average_precision_at_r"
knob_names = ["innodb_buffer_pool", "innodb_io_capacity"]

save_dir = Path("../output/")
if not save_dir.exists():
    save_dir.mkdir()


def make_dataframes():
    df_train_path = Path("../dataset/train.csv")
    df = pd.read_csv(df_train_path)
    # encode label column
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["label"].values)
    # extract target column
    target_name = "tps"
    target_col = df[target_name]
    # DataFrame is composed of metrics and knobs
    df.drop(["label", target_name], axis=1, inplace=True)

    # convert disk_used_gb-used to data_disk_used_gb
    system_disk_used = 6.8
    df["disk_used_gb-used"] -= system_disk_used
    df.rename(columns={"disk_used_gb-used": "data_disk_used_gb"}, inplace=True)

    drop_metric_names = get_drop_metric_names(df.drop(knob_names, axis=1))
    df.drop(drop_metric_names, axis=1, inplace=True)
    scaler = MinMaxScaler()
    df[df.columns.to_list()] = scaler.fit_transform(df)
    return df, labels, target_col


def get_optimizer(trial, model):
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    final_lr = trial.suggest_float("final_lr", 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
    return AdaBound(model.parameters(), lr=lr, final_lr=final_lr, gamma=gamma, weight_decay=weight_decay)


def get_scheduler(trial, optimizer):
    step_size = trial.suggest_int("step_size", 10, 50)
    gamma = trial.suggest_float("scheduler_gamma", 0.1, 0.9)
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def train_one_epoch(tps_estimator, workload_embedder, loss_fn, optimizer, scheduler, train_loader):
    tps_estimator.train()
    for metrics, knobs, targets in train_loader:
        metrics, knobs, targets = metrics.to(device), knobs.to(device), targets.to(device)
        features = torch.cat((metrics, knobs), axis=1)
        workloads = workload_embedder(features)
        preds = tps_estimator(workloads, knobs)
        loss = loss_fn(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()


def valid_one_epoch(tps_estimator, workload_embedder, loss_fn, valid_loader):
    tps_estimator.eval()
    total_loss = 0.0
    with torch.no_grad():
        for metrics, knobs, targets in valid_loader:
            metrics, knobs, targets = metrics.to(device), knobs.to(device), targets.to(device)
            features = torch.cat((metrics, knobs), axis=1)
            workloads = workload_embedder(features)
            preds = tps_estimator(workloads, knobs)
            total_loss += loss_fn(preds, targets)
    avg_loss = total_loss / len(valid_loader)
    return avg_loss


def objective(trial):
    # load workload_embedder
    workload_embedder = WorkloadEmbedder(
        n_metrics=n_metrics,
        n_knobs=n_knobs,
        workload_dim=workload_dim,
        n_hidden_units=workload_embedder_n_hidden_units,
    ).to(device)
    workload_embedder_path = save_dir / "best_workload_embedder.pth"
    workload_embedder.load_state_dict(torch.load(workload_embedder_path, map_location=device))
    workload_embedder.eval()

    # architecture
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 2, 5)
    n_hidden_units = [trial.suggest_int(f"layer_{i+1}", 20, 100) for i in range(n_hidden_layers)]

    # model
    tps_estimator = TPSEstimator(workload_dim, n_knobs, n_hidden_units).to(device)

    # loss
    loss_fn = RMSLELoss()

    # optimizer
    optimizer = get_optimizer(trial, tps_estimator)
    scheduler = get_scheduler(trial, optimizer)

    # dataloader
    df, _, target_col = make_dataframes()
    knob_df = df[knob_names]
    metric_df = df.drop(knob_names, axis=1)
    dataset = TPSEstimatorDataset(metric_df=metric_df, knob_df=knob_df, target_col=target_col)
    train_dataset, valid_dataset = split_dataset(valid_split=0.1, dataset=dataset, seed=seed)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    not_improved_cnt = 0
    patience = 10
    min_valid_loss = 100000000
    for epoch in range(1, epochs + 1):
        train_one_epoch(
            tps_estimator=tps_estimator,
            workload_embedder=workload_embedder,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
        )
        valid_loss = valid_one_epoch(
            tps_estimator=tps_estimator,
            workload_embedder=workload_embedder,
            loss_fn=loss_fn,
            valid_loader=valid_loader,
        )
        if valid_loss <= min_valid_loss:
            min_valid_loss = valid_loss
            not_improved_cnt = 0
        else:
            not_improved_cnt += 1
        if not_improved_cnt > patience:
            print(f"Training TPSEstimator has been early-stopped at {epoch}/{epochs}epochs")
            break
    return min_valid_loss


def train_workload_embedder(df, labels, load_existing_model):
    save_path = save_dir / "best_workload_embedder.pth"
    if load_existing_model and save_path.exists():
        return None
    workload_embedder = WorkloadEmbedder(
        n_metrics=n_metrics,
        n_knobs=n_knobs,
        workload_dim=workload_dim,
        n_hidden_units=workload_embedder_n_hidden_units,
    ).to(device)
    dataset = WorkloadEmbedderDataset(metric_knob_df=df, labels=labels)
    train_dataset, valid_dataset = split_dataset(valid_split=0.1, dataset=dataset, seed=seed)
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=128,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    optimizer = AdaBound(
        params=workload_embedder.parameters(),
        lr=0.001,
        betas=[0.9, 0.999],
        final_lr=0.002,
        gamma=6e-6,
        weight_decay=8e-9,
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=24, gamma=0.7)
    # loss functions
    distance = distances.CosineSimilarity()  # default
    reducer = reducers.AvgNonZeroReducer()  # default
    margin = 0.25
    loss_fn = losses.CircleLoss(m=margin, gamma=256, distance=distance, reducer=reducer)
    miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="hard")
    tester = testers.BaseTester(normalize_embeddings=True, data_device=device)

    # train
    not_improved_cnt = 0
    patience = 10
    max_acc = 0
    accuracy_calculator = AccuracyCalculator(include=[acc_metric_name])
    workload_embedder.train()
    for epoch in tqdm(range(1, epochs + 1)):
        for features, labels in train_dataloader:
            features, labels = features.to(device), labels.to(device)
            embeddings = workload_embedder(features)
            indices_tuple = miner(embeddings, labels)
            loss = loss_fn(embeddings, labels, indices_tuple)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # valid
        train_embeddings, train_labels = tester.get_all_embeddings(train_dataset, workload_embedder)
        valid_embeddings, valid_labels = tester.get_all_embeddings(valid_dataset, workload_embedder)
        train_labels = train_labels.squeeze(1)
        valid_labels = valid_labels.squeeze(1)
        accuracies = accuracy_calculator.get_accuracy(
            query=valid_embeddings,
            reference=train_embeddings,
            query_labels=valid_labels,
            reference_labels=train_labels,
            embeddings_come_from_same_source=False,
            include=[acc_metric_name],
        )
        valid_acc = accuracies[acc_metric_name]
        print(f"Validation accuracy (MAP@R): {valid_acc:>7f}")
        if valid_acc >= max_acc:
            max_acc = valid_acc
            not_improved_cnt = 0
            best_state_dict = deepcopy(workload_embedder.state_dict())
        else:
            not_improved_cnt += 1
        if not_improved_cnt > patience:
            print(f"Training WorkloadEmbedder has been early-stopped at {epoch}/{epochs}epochs")
            break
    print("\nTraining WorkloadEmbedder has done!\n")
    torch.save(best_state_dict, save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_existing_model", type=bool, help="whether using existing model", default=False)
    args = parser.parse_args()
    df, labels, _ = make_dataframes()
    train_workload_embedder(df, labels, args.load_existing_model)

    n_trials = 80
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(study.best_params)
    print("")
    print(study.best_value)
    print("")
    print(study.trials)
    tuning_results_dir = save_dir / "log_tuning_results" / "tps_layers" / "RMSLELoss"
    if not tuning_results_dir.exists():
        tuning_results_dir.mkdir(parents=True)
    best_params_path = tuning_results_dir / "best_params.json"
    best_value_path = tuning_results_dir / "best_value.txt"
    trials_path = tuning_results_dir / "trials.txt"
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    with open(best_value_path, "w") as f:
        f.write(str(study.best_value))
    with open(trials_path, "w") as f:
        for trial in study.trials:
            f.write(str(trial) + "\n")
