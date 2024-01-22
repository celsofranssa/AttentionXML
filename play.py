import pickle
from pathlib import Path

import pandas as pd


def load_ids(dataset, fold_idx, split):
    with open(f"data/folded_datasets/{dataset}/fold_{fold_idx}/{split}.pkl", "rb") as ids_file:
        return pickle.load(ids_file)


def prepare_data(dataset, fold_idx, split):
    ids = load_ids(dataset, fold_idx, split)
    with open(f"data/folded_datasets/{dataset}/samples.pkl", "rb") as samples_file:
        samples_df = pd.DataFrame(pickle.load(samples_file))
    samples_df = samples_df[samples_df["idx"].isin(ids)]
    checkpoint_samples(samples_df, dataset, split)


def checkpoint_samples(samples_df, dataset, split):
    dataset_dir = f"data/{dataset}/"
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    samples_df["text"] = samples_df["text"].apply(lambda text: text.replace("\n", " "))
    samples_df["text"].to_csv(f"data/{dataset}/{split}_texts.txt", header=False, index=False)

    samples_df["labels"] = samples_df["labels"].apply(lambda labels: " ".join(labels))
    samples_df["labels"].to_csv(f"data/{dataset}/{split}_labels.txt", header=False, index=False)



if __name__ == '__main__':
    dataset = "Wiki10-31k"
    fold_idx = 0
    prepare_data(dataset, fold_idx, "train")
    prepare_data(dataset, fold_idx, "test")
