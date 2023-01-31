from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from adabound import AdaBound
from cycler import cycler
from omegaconf import DictConfig, OmegaConf
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.nn import L1Loss
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import TPSEstimator, WorkloadEmbedder
from utils.dataset import TPSEstimatorDataset, WorkloadEmbedderDataset
from utils.general import camel_to_snake, load_best_module
from utils.loss import RMSLELoss


class BaseTrainer:
    def __init__(self, config: DictConfig, train_dataset: data.Dataset, valid_dataset: data.Dataset) -> None:
        # system
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # config
        self.config = config
        module_name_camel: str = re.findall("(.*)Trainer", self.__class__.__name__)[0]
        self.module_name = camel_to_snake(module_name_camel)
        self.module_config = self.config[self.module_name]

        # directories
        self.save_dir: Path = config.output_dir / config.model_name

        # datasets
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        # train dataloader
        self.train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=self.module_config.batch_size,
            num_workers=self.module_config.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        # workload embedder
        self.workload_embedder = WorkloadEmbedder(
            n_metrics=config.n_metrics,
            n_knobs=config.n_knobs,
            workload_dim=config.workload_dim,
            n_hidden_units=config[camel_to_snake(WorkloadEmbedder.__name__)].hidden_units,
        ).to(self.device)

        # tensorboard writer
        log_dir = self.save_dir / f"log_{self.module_name}"
        self.writer = SummaryWriter(log_dir)

    def save_config(self) -> None:
        save_path = self.save_dir / f"config.yaml"
        save_config = deepcopy(self.config)
        # measures to avoid "PosixPath  is not JSON serializable"
        for k, v in save_config.items():
            if isinstance(v, Path):
                save_config[k] = str(v)
        with open(save_path, "w") as f:
            OmegaConf.save(save_config, f)

    def save_best_state_dict(self, state_dict: dict[str, Any]) -> None:
        save_path = self.save_dir / f"best_{self.module_name}.pth"
        torch.save(state_dict, save_path)

    # TODO: 学習をcheckpointから再開させるメソッドを作成(というよりも，trainメソッドにエポックを渡すとそこから学習を再開できるようにしたい)


class WorkloadEmbedderTrainer(BaseTrainer):
    def __init__(self, config: DictConfig, train_dataset: WorkloadEmbedderDataset, valid_dataset: WorkloadEmbedderDataset) -> None:
        super().__init__(config, train_dataset, valid_dataset)

        # optimizer & scheduler
        self.optimizer = AdaBound(
            self.workload_embedder.parameters(),
            lr=self.module_config.lr,
            betas=self.module_config.betas,
            final_lr=self.module_config.final_lr,
            gamma=self.module_config.gamma,
            eps=self.module_config.eps,
            weight_decay=self.module_config.weight_decay,
        )
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.module_config.scheduler_step_size, gamma=0.5)

        # loss functions
        distance = distances.CosineSimilarity()  # default
        reducer = reducers.AvgNonZeroReducer()  # default
        margin = 0.25
        self.loss_fn = losses.CircleLoss(m=margin, gamma=256, distance=distance, reducer=reducer)
        self.miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="hard")
        # self.sampler = samplers.MPerClassSampler(train_dataset.labels, m=4, length_before_new_iter=len(train_dataset))

        # dataloader
        self.train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=self.module_config.batch_size,
            num_workers=self.module_config.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        self.tester = testers.BaseTester(normalize_embeddings=True, data_device=self.device)

    def _train_one_epoch(self, epoch: int) -> None:
        self.workload_embedder.train()
        total_loss = 0.0

        with tqdm(self.train_dataloader, unit="batch") as pbar:
            pbar.set_description(f"[Epoch {epoch}/{self.module_config.epochs}]")
            for features, labels in pbar:
                features, labels = features.to(self.device), labels.to(self.device)
                # compute error
                embeddings = self.workload_embedder(features)
                indices_tuple = self.miner(embeddings, labels)
                loss = self.loss_fn(embeddings, labels, indices_tuple)
                # backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
        avg_loss = total_loss / len(self.train_dataloader)
        print(f"Avg training circle loss: {avg_loss:>7f}")
        self.lr_scheduler.step()
        return avg_loss

    def valid(self, accuracy_calculator: AccuracyCalculator):
        train_embeddings, train_labels = self.tester.get_all_embeddings(self.train_dataset, self.workload_embedder)
        valid_embeddings, valid_labels = self.tester.get_all_embeddings(self.valid_dataset, self.workload_embedder)
        train_labels = train_labels.squeeze(1)
        valid_labels = valid_labels.squeeze(1)
        accuracies = accuracy_calculator.get_accuracy(
            query=valid_embeddings,
            reference=train_embeddings,
            query_labels=valid_labels,
            reference_labels=train_labels,
            embeddings_come_from_same_source=False,
            include=[self.module_config.acc_metric_name],
        )
        valid_acc = accuracies[self.module_config.acc_metric_name]
        print(f"Validation accuracy (MAP@R): {valid_acc:>7f}")
        embeddings = torch.cat((train_embeddings, valid_embeddings), axis=0)
        labels = torch.cat((train_labels, valid_labels), axis=0)
        return (embeddings, labels), valid_acc

    def train(self) -> None:
        # save config before training
        self.save_config()

        not_improved_cnt = 0
        patience = 10
        max_acc = 0
        accuracy_calculator = AccuracyCalculator(include=[self.module_config.acc_metric_name])
        for epoch in range(1, self.module_config.epochs + 1):
            train_circle_loss = self._train_one_epoch(epoch)
            tmp_embeddings_labels, valid_acc = self.valid(accuracy_calculator=accuracy_calculator)
            if valid_acc >= max_acc:
                max_acc = valid_acc
                not_improved_cnt = 0
                embeddings, labels = tmp_embeddings_labels
                best_state_dict = deepcopy(self.workload_embedder.state_dict())
            else:
                not_improved_cnt += 1
            if not_improved_cnt > patience:
                print(f"Training WorkloadEmbedder has been early-stopped at {epoch}/{self.module_config.epochs}epochs")
                break
            self.writer.add_scalar("train_circle_loss", train_circle_loss, global_step=epoch)
            self.writer.add_scalar("valid_accuracy", valid_acc, global_step=epoch)
            self.save_checkpoint(epoch, train_circle_loss, valid_acc)
        self.save_best_state_dict(best_state_dict)
        self.writer.close()

        # save UMAP-embeddings figure
        self.save_umap_fig(embeddings.cpu(), labels.cpu())
        print("\nTraining WorkloadEmbedder has done!\n")

    def save_umap_fig(self, embeddings: torch.Tensor, labels: torch.Tensor):
        umapper = umap.UMAP(
            n_neighbors=15,  # default
            n_components=2,  # default
            metric="euclidean",  # default
            random_state=self.config.seed,
        )
        umap_embeddings = umapper.fit_transform(embeddings)
        label_set = np.unique(labels)
        num_classes = len(label_set)
        plt.figure(figsize=(10, 10))
        plt.gca().set_prop_cycle(cycler("color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]))
        for i in range(num_classes):
            idx = labels == label_set[i]
            plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
        plt.savefig(self.save_dir / "valid_UMAP.png", dpi=300)

    def save_checkpoint(self, epoch: int, train_loss: float, valid_acc: float) -> None:
        # checkpointごとに保存することにより，性能が低下するよりも前の状態のモデルが取り出せる
        save_path = self.save_dir / f"{self.module_name}_epoch_{epoch}.pth"
        save_dict = {
            "model_state_dict": self.workload_embedder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_acc": valid_acc,
        }
        torch.save(save_dict, save_path)


class TPSEstimatorTrainer(BaseTrainer):
    def __init__(self, config: DictConfig, train_dataset: TPSEstimatorDataset, valid_dataset: TPSEstimatorDataset) -> None:
        super().__init__(config, train_dataset, valid_dataset)

        # modules
        ## load trained WorkloadEmbedder module
        trained_module_name = camel_to_snake(self.workload_embedder.__class__.__name__)
        self.workload_embedder = load_best_module(model=self.workload_embedder, config=config, module_name=trained_module_name, device=self.device)

        self.tps_estimator = TPSEstimator(
            workload_dim=config.workload_dim,
            n_knobs=config.n_knobs,
            n_hidden_units=self.module_config.hidden_units,
        ).to(self.device)

        # optimizer & scheduler
        self.optimizer = AdaBound(
            self.tps_estimator.parameters(),
            lr=self.module_config.lr,
            betas=self.module_config.betas,
            final_lr=self.module_config.final_lr,
            gamma=self.module_config.gamma,
            eps=self.module_config.eps,
            weight_decay=self.module_config.weight_decay,
        )
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.module_config.scheduler_step_size, gamma=0.5)

        # loss
        self.loss_fn = RMSLELoss()

        # metric
        self.metric_fn = L1Loss()

        # dataloader
        self.valid_dataloader = data.DataLoader(
            valid_dataset,
            batch_size=self.module_config.batch_size,
            num_workers=self.module_config.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

    def _train_one_epoch(self, epoch: int) -> None:
        self.tps_estimator.train()
        total_loss = 0.0
        total_l1_loss = 0.0

        with tqdm(self.train_dataloader, unit="batch") as pbar:
            pbar.set_description(f"[Epoch {epoch}/{self.module_config.epochs}]")
            for metrics, knobs, targets in pbar:
                metrics, knobs, targets = metrics.to(self.device), knobs.to(self.device), targets.to(self.device)
                features = torch.cat((metrics, knobs), axis=1)
                # compute error
                workloads = self.workload_embedder(features)
                preds = self.tps_estimator(workloads, knobs)
                loss = self.loss_fn(preds, targets)
                # backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_l1_loss += self.metric_fn(preds, targets).item()
        avg_loss = total_loss / len(self.train_dataloader)
        avg_l1_loss = total_l1_loss / len(self.train_dataloader)
        print(f"Avg training loss: {avg_loss:>7f}")
        print(f"Avg training MAE: {avg_l1_loss:>7f}")
        self.lr_scheduler.step()
        return avg_loss, avg_l1_loss

    def _valid_one_epoch(self) -> float:
        self.tps_estimator.eval()
        total_loss = 0.0
        total_l1_loss = 0.0
        with torch.no_grad():
            for metrics, knobs, targets in self.valid_dataloader:
                metrics, knobs, targets = metrics.to(self.device), knobs.to(self.device), targets.to(self.device)
                features = torch.cat((metrics, knobs), axis=1)
                workloads = self.workload_embedder(features)
                preds = self.tps_estimator(workloads, knobs)
                total_loss += self.loss_fn(preds, targets).item()
                total_l1_loss += self.metric_fn(preds, targets).item()
        avg_loss = total_loss / len(self.valid_dataloader)
        avg_l1_loss = total_l1_loss / len(self.valid_dataloader)
        print(f"Avg validation loss: {avg_loss:>7f}")
        print(f"Avg validation MAE: {avg_l1_loss:>7f}\n")
        return avg_loss, avg_l1_loss

    def train(self) -> None:
        not_improved_cnt = 0
        patience = 10
        min_valid_loss = 1000
        for epoch in range(1, self.module_config.epochs + 1):
            train_loss, train_l1_loss = self._train_one_epoch(epoch)
            valid_loss, valid_l1_loss = self._valid_one_epoch()
            self.writer.add_scalars("TPSEstimator_RMSLE", {"train": train_loss, "valid": valid_loss}, epoch)
            self.writer.add_scalars("TPSEstimator_MAE", {"train": train_l1_loss, "valid": valid_l1_loss}, epoch)
            if valid_loss <= min_valid_loss:
                min_valid_loss = valid_loss
                not_improved_cnt = 0
                best_state_dict = deepcopy(self.tps_estimator.state_dict())
            else:
                not_improved_cnt += 1
            if not_improved_cnt > patience:
                print(f"Training TPSEstimator has been early-stopped at {epoch}/{self.module_config.epochs}epochs")
                break
            self.save_checkpoint(epoch, valid_loss)
        self.save_best_state_dict(best_state_dict)
        self.writer.close()
        print("\nTraining TPSEstimator has done!\n")

    def save_checkpoint(self, epoch: int, valid_loss: float) -> None:
        # checkpointごとに保存することにより，性能が低下するよりも前の状態のモデルが取り出せる
        save_path = self.save_dir / f"{self.module_name}_epoch_{epoch}.pth"
        save_dict = {
            "model_state_dict": self.tps_estimator.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "valid_loss": valid_loss,
        }
        torch.save(save_dict, save_path)
