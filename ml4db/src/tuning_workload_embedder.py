from __future__ import annotations

import json
from pathlib import Path

import optuna
import pandas as pd
import torch
import torch.nn as nn
from adabound import AdaBound
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils import data

from utils.dataset import WorkloadEmbedderDataset, get_drop_metric_names
from utils.general import fix_seed

optuna.logging.disable_default_handler()

seed = 3407
fix_seed(seed)

epochs = 80


# モデルの定義
class Model(nn.Module):
    def __init__(self, n_metrics, n_knobs, workload_dim, n_hidden_units: list[int]):
        super().__init__()
        layer = [
            nn.Linear(n_metrics + n_knobs, n_hidden_units[0]),
            nn.BatchNorm1d(n_hidden_units[0]),
            nn.ReLU(),
        ]
        # workload encoder
        for i in range(1, len(n_hidden_units)):
            layer += [
                nn.Linear(n_hidden_units[i - 1], n_hidden_units[i]),
                nn.BatchNorm1d(n_hidden_units[i]),
                nn.ReLU(),
            ]
        layer += [nn.Linear(n_hidden_units[-1], workload_dim), nn.ReLU()]
        self.model = nn.Sequential(*layer)

        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, features) -> torch.Tensor:
        embedded_workload = self.model(features)
        return F.normalize(embedded_workload)


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


def make_dataset():
    df_train_path = Path("../dataset/train.csv")
    df = pd.read_csv(df_train_path)
    # encode label column
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["label"].values)
    # DataFrame is composed of metrics and knobs
    df.drop(["label", "tps"], axis=1, inplace=True)

    # convert disk_used_gb-used to data_disk_used_gb
    system_disk_used = 6.8
    df["disk_used_gb-used"] -= system_disk_used
    df.rename(columns={"disk_used_gb-used": "data_disk_used_gb"}, inplace=True)

    knob_names = ["innodb_buffer_pool", "innodb_io_capacity"]
    drop_metric_names = get_drop_metric_names(df.drop(knob_names, axis=1))
    df.drop(drop_metric_names, axis=1, inplace=True)
    scaler = MinMaxScaler()
    df[df.columns.to_list()] = scaler.fit_transform(df)

    dataset = WorkloadEmbedderDataset(metric_knob_df=df, labels=labels)
    valid_split = 0.1
    valid_size = int(len(dataset) * valid_split)
    train_dataset, valid_dataset = data.random_split(
        dataset=dataset, lengths=[len(dataset) - valid_size, valid_size], generator=torch.Generator().manual_seed(seed)
    )
    return train_dataset, valid_dataset


def train_one_epoch(device, model, loss_fn, miner, optimizer, scheduler, train_loader):
    model.train()
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        embeddings = model(features)
        indices_tuple = miner(embeddings, labels)
        loss = loss_fn(embeddings, labels, indices_tuple)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def valid(
    train_dataset: WorkloadEmbedderDataset,
    valid_dataset: WorkloadEmbedderDataset,
    model: nn.Module,
    accuracy_calculator: AccuracyCalculator,
    acc_metric_name="mean_average_precision_at_r",
):
    train_embeddings, train_labels = get_all_embeddings(train_dataset, model)
    valid_embeddings, valid_labels = get_all_embeddings(valid_dataset, model)
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
    return accuracies[acc_metric_name]


def objective(trial):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # workload encoder
    n_hidden_layers = trial.suggest_int("num_encoder_layer", 2, 4)
    n_hidden_units = [trial.suggest_int(f"encoder_{i}", 20, 100) for i in range(n_hidden_layers)]
    workload_dim = trial.suggest_int("workload_dim", 20, 100)

    # model
    n_metrics = 30
    n_knobs = 2
    model = Model(n_metrics, n_knobs, workload_dim, n_hidden_units).to(device)

    # optimizer
    optimizer = get_optimizer(trial, model)
    scheduler = get_scheduler(trial, optimizer)

    # dataloader
    train_dataset, valid_dataset = make_dataset()
    batch_size = trial.suggest_int("batch_size", 16, 256)
    num_workers = 4

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    distance = distances.CosineSimilarity()
    reducer = reducers.AvgNonZeroReducer()
    margin = 0.25
    loss_fn = losses.CircleLoss(m=margin, gamma=256, distance=distance, reducer=reducer)
    miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="hard", distance=distance)

    not_improved_cnt = 0
    patience = 10
    max_acc = 0
    acc_metric_name = "mean_average_precision_at_r"
    accuracy_calculator = AccuracyCalculator(include=[acc_metric_name])
    for epoch in range(1, epochs + 1):
        train_one_epoch(
            device=device,
            model=model,
            loss_fn=loss_fn,
            miner=miner,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
        )
        acc = valid(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            model=model,
            accuracy_calculator=accuracy_calculator,
            acc_metric_name=acc_metric_name,
        )
        if acc >= max_acc:
            max_acc = acc
            not_improved_cnt = 0
        else:
            not_improved_cnt += 1
        if not_improved_cnt > patience:
            print(f"Training WorkloadEmbedder has been early-stopped at {epoch}/{epochs}epochs")
            break
    return max_acc


if __name__ == "__main__":
    n_trials = 100
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(study.best_params)
    print("")
    print(study.best_value)
    print("")
    print(study.trials)
    output_dir = Path("../output")
    tuning_results_dir = output_dir / "log_tuning_results" / "workload_embedder" / "mean_average_precision_at_r"
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
