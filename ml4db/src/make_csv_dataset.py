import json
import re
from pathlib import Path

import pandas as pd


def make_csv_dataset(raw_dataset_dir: Path, dataset_type: str, save_dir: Path) -> None:
    df_lst = []
    regex_ptn = r"(.*)_(\d*)c_(\d*)g"
    raw_dataset_dir = raw_dataset_dir / dataset_type
    for data_dir in raw_dataset_dir.iterdir():
        data_dict = {}
        json_paths = data_dir.glob("*.json")
        for path in json_paths:
            with open(path, "r") as f:
                metric = path.name[:-24]
                results = json.load(f)["results"]
                metric_set = set()
                for src_data in results:
                    for data in src_data["data"]:
                        metric_set.add(data["metric"])
                for src_data in results:
                    if src_data["source"][:10] != "summarizer":
                        for data in src_data["data"]:
                            # jsonのキーは順番保証されていないので，念の為ソート
                            tags = [data["tags"][key] for key in sorted(data["tags"].keys()) if key != "origin"]
                            key_name = metric
                            # metricフィールドが全て同じならキー名に含めない
                            if len(metric_set) > 1:
                                # metricフィールドの冗長な名前を簡潔化
                                metric_attr = re.findall("^.*\.(.*)$", data["metric"])[0]
                                key_name += f"-{metric_attr}"
                            if len(tags) > 0:
                                key_name += f"-{'-'.join(tags)}"
                            data_dict[key_name] = data["NumericType"]
        df = pd.DataFrame(data_dict, dtype="float32")
        df.dropna(inplace=True)
        workload_name, core, memory = re.findall(regex_ptn, data_dir.name)[0]
        df["core"] = int(core)
        df["memory"] = int(memory)
        if dataset_type == "train":
            df["label"] = workload_name
        else:
            df["label"] = data_dir.name
        # 測定の開始時は外れ値であることが多いので，1つ目のレコードを消す
        if 1 in df.index:
            df.drop(1, axis=0, inplace=True)
        # tpsが0のレコードを削除する
        df = df[df["tps"] != 0.0]
        df_lst.append(df)
    whole_df = pd.concat(df_lst)
    # make the order of columns same between train and test dataset
    whole_df = whole_df.reindex(sorted(whole_df.columns), axis=1)
    if not save_dir.exists():
        save_dir.mkdir()
    dataset_path = save_dir / f"{dataset_type}.csv"
    whole_df.to_csv(dataset_path, index=False)


if __name__ == "__main__":
    raw_dataset_dir = Path("../raw_dataset")
    save_dir = Path("../dataset")
    for dataset_type in ["train", "test"]:
        make_csv_dataset(raw_dataset_dir, dataset_type, save_dir)
