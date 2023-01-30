from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from adabound import AdaBound
from sklearn.preprocessing import MinMaxScaler
from torch.optim import lr_scheduler
from torch.utils import data

from utils.dataset import Dataset, get_drop_metric_names
from utils.loss import MeanAbsoluteRelativeError

optuna.logging.disable_default_handler()

epochs = 80


# モデルの定義
class Model(nn.Module):
    def __init__(self, n_metrics, n_knobs, num_encoder_units: list[int], num_tps_layers_units: list[int]):
        super().__init__()
        encoder = [
            nn.Linear(n_metrics + n_knobs, num_encoder_units[0]),
            nn.BatchNorm1d(num_encoder_units[0]),
            nn.ReLU(),
        ]
        # workload encoder
        for i in range(1, len(num_encoder_units) - 1):
            encoder += [
                nn.Linear(num_encoder_units[i - 1], num_encoder_units[i]),
                nn.BatchNorm1d(num_encoder_units[i]),
                nn.ReLU(),
            ]
        encoder += [nn.Linear(num_encoder_units[-2], num_encoder_units[-1]), nn.ReLU()]
        self.encoder = nn.Sequential(*encoder)

        # tps layers
        tps_layers = [
            nn.Linear(num_encoder_units[-1] + n_knobs, num_tps_layers_units[0]),
            nn.BatchNorm1d(num_tps_layers_units[0]),
            nn.ReLU(),
        ]
        for i in range(1, len(num_tps_layers_units)):
            tps_layers += [
                nn.Linear(num_tps_layers_units[i - 1], num_tps_layers_units[i]),
                nn.BatchNorm1d(num_tps_layers_units[i]),
                nn.ReLU(),
            ]
        tps_layers += [nn.Linear(num_tps_layers_units[-1], 1), nn.ReLU()]
        self.tps_layers = nn.Sequential(*tps_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, metrics, knobs) -> torch.Tensor:
        latent_workload = self.encoder(torch.cat((metrics, knobs), axis=1))
        pred_target = self.tps_layers(torch.cat((latent_workload, knobs), axis=1))
        return pred_target


def train_one_epoch(device, model, loss_fn, optimizer, scheduler, train_loader):
    model.train()

    for metrics, knobs, targets in train_loader:
        metrics, knobs, targets = metrics.to(device), knobs.to(device), targets.to(device)
        preds = model(metrics, knobs)
        loss = loss_fn(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()


def valid_one_epoch(device, model, loss_fn, valid_loader):
    loss_fn = MeanAbsoluteRelativeError()
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for metrics, knobs, targets in valid_loader:
            metrics, knobs, targets = metrics.to(device), knobs.to(device), targets.to(device)
            preds = model(metrics, knobs)
            total_loss += loss_fn(preds, targets).item()
    avg_loss = total_loss / len(valid_loader)
    return avg_loss


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


def make_dataset(seed):
    df_train_path = Path("../dataset/train.csv")
    df = pd.read_csv(df_train_path)
    target_col = df["tps"]
    df.drop("tps", axis=1, inplace=True)

    knob_names = ["innodb_buffer_pool", "innodb_io_capacity"]
    drop_metric_names = get_drop_metric_names(df.drop(knob_names, axis=1))
    df.drop(drop_metric_names, axis=1, inplace=True)
    output_dir = Path("../output")
    if not output_dir.exists():
        output_dir.mkdir()
    scaler_path = output_dir / "scaler.pkl"
    scaler = MinMaxScaler()
    scaler.fit(df)
    joblib.dump(scaler, scaler_path)
    df[df.columns.to_list()] = scaler.transform(df)

    knob_df = df[knob_names]
    metric_df = df.drop(knob_names, axis=1)
    dataset = Dataset(metric_df, knob_df, target_col)
    valid_split = 0.1
    valid_size = int(len(dataset) * valid_split)
    train_dataset, valid_dataset = data.random_split(
        dataset=dataset, lengths=[len(dataset) - valid_size, valid_size], generator=torch.Generator().manual_seed(seed)
    )
    return train_dataset, valid_dataset


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def objective(trial):
    seed = 3407
    fix_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # workload encoder
    num_encoder_layer = trial.suggest_int("num_encoder_layer", 2, 5)
    num_encoder_units = [trial.suggest_int(f"encoder_{i}", 20, 100) for i in range(num_encoder_layer)]

    # tps layer
    num_tps_layer = trial.suggest_int("num_tps_layer", 3, 5)
    num_tps_layers_units = [trial.suggest_int(f"tps_{i}", 40, 150) for i in range(num_tps_layer)]

    # model
    n_metrics = 30
    n_knobs = 2
    model = Model(n_metrics, n_knobs, num_encoder_units, num_tps_layers_units).to(device)

    # optimizer
    optimizer = get_optimizer(trial, model)
    scheduler = get_scheduler(trial, optimizer)

    # dataloader
    train_dataset, valid_dataset = make_dataset(seed)
    batch_size = trial.suggest_int("batch_size", 16, 512)
    num_workers = 4
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

    min_valid_loss = 1000000
    not_improved_cnt = 0
    early_stop = 15
    loss_fn = MeanAbsoluteRelativeError()
    for _ in range(1, epochs + 1):
        train_one_epoch(
            device=device,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
        )
        avg_valid_loss = valid_one_epoch(
            device=device,
            model=model,
            loss_fn=loss_fn,
            valid_loader=valid_loader,
        )
        if avg_valid_loss <= min_valid_loss:
            min_valid_loss = avg_valid_loss
            not_improved_cnt = 0
        else:
            not_improved_cnt += 1

        if not_improved_cnt > early_stop:
            break
    return min_valid_loss


if __name__ == "__main__":
    n_trials = 200
    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(study.best_params)
    print("")
    print(study.best_value)
    print("")
    print(study.trials)
    output_dir = Path("../output")
    tuning_results_dir = output_dir / "tuning_results"
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
