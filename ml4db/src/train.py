from pathlib import Path

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from trainer import TPSEstimatorTrainer, WorkloadEmbedderTrainer
from utils.dataset import TPSEstimatorDataset, WorkloadEmbedderDataset, get_drop_metric_names
from utils.general import convert_config_str_to_path, fix_seed, make_dirs, split_dataset


@hydra.main(version_base=None, config_path=Path("./config"), config_name="config")
def main(config: DictConfig) -> None:
    config = convert_config_str_to_path(config)
    fix_seed(config.seed)
    make_dirs(config)

    # read csv data
    dataset_dir = config.dataset_dir
    df_train_path = dataset_dir / "train.csv"
    df = pd.read_csv(df_train_path)

    # encode label column
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["label"].values)
    target_col = df[config.target_name]
    # DataFrame is composed of metrics and knobs
    df.drop(["label", config.target_name], axis=1, inplace=True)
    # save label encoder

    # decide metrics for removing after excluding knobs
    drop_metric_names = get_drop_metric_names(df.drop(config.knob_names, axis=1))
    drop_metric_names_path = config.output_dir / "drop_metric_names.txt"
    ## save drop metric filenames for testing
    with open(drop_metric_names_path, "w") as f:
        for metric_name in drop_metric_names:
            f.write(metric_name + "\n")
    df.drop(drop_metric_names, axis=1, inplace=True)

    # normalize data
    scaler_path = config.output_dir / "scaler.pkl"
    scaler = MinMaxScaler()
    df[df.columns.to_list()] = scaler.fit_transform(df)
    joblib.dump(scaler, scaler_path)

    # make workload_embedder dataset
    if not config.load_trained_workload_embedder:
        workload_embedder_dataset = WorkloadEmbedderDataset(metric_knob_df=df, labels=labels)
        workload_embedder_train_dataset, workload_embedder_valid_dataset = split_dataset(
            valid_split=config.workload_embedder.valid_split, dataset=workload_embedder_dataset, seed=config.seed
        )

        # train workload_embedder
        workload_embedder_trainer = WorkloadEmbedderTrainer(config, workload_embedder_train_dataset, workload_embedder_valid_dataset)
        workload_embedder_trainer.train()

    # make tps_estimator dataset
    knob_df = df[config.knob_names]
    metric_df = df.drop(config.knob_names, axis=1)
    tps_estimator_dataset = TPSEstimatorDataset(metric_df=metric_df, knob_df=knob_df, target_col=target_col)
    tps_estimator_train_dataset, tps_estimator_valid_dataset = split_dataset(
        valid_split=config.tps_estimator.valid_split, dataset=tps_estimator_dataset, seed=config.seed
    )

    # train tps_estimator
    tps_estimator_trainer = TPSEstimatorTrainer(config, tps_estimator_train_dataset, tps_estimator_valid_dataset)
    tps_estimator_trainer.train()


if __name__ == "__main__":
    main()
