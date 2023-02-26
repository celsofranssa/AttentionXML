import pickle

import numpy as np
import pandas as pd
from ranx import evaluate, Run, Qrels


def _get_metrics(metrics, thresholds):
    threshold_metrics = []
    for metric in metrics:
        for threshold in thresholds:
            threshold_metrics.append(f"{metric}@{threshold}")
    return threshold_metrics


def _load_labels_cls(dataset):
    with open(f"resource/dataset/{dataset}/label_cls.pkl", "rb") as label_cls_file:
        return pickle.load(label_cls_file)


def _load_texts_cls(dataset):
    with open(f"resource/dataset/{dataset}/text_cls.pkl", "rb") as text_cls_file:
        return pickle.load(text_cls_file)


def _load_relevance_map(dataset):
    with open(f"resource/dataset/{dataset}/relevance_map.pkl", "rb") as relevances_file:
        data = pickle.load(relevances_file)
    relevance_map = {}
    for text_idx, labels_ids in data.items():
        d = {}
        for label_idx in labels_ids:
            d[f"label_{label_idx}"] = 1.0
        relevance_map[f"text_{text_idx}"] = d
    return relevance_map


def load_ids(dataset, fold_idx, split):
    with open(f"resource/dataset/{dataset}/fold_{fold_idx}/{split}.pkl", "rb") as ids_file:
        return pickle.load(ids_file)


def load_samples(dataset, fold_idx, split):
    split_ids = load_ids(dataset, fold_idx, split)
    with open(f"resource/dataset/{dataset}/samples.pkl", "rb") as samples_file:
        samples = pickle.load(samples_file)
    samples_df = pd.DataFrame(samples)
    return samples_df[samples_df["idx"].isin(split_ids)]


def get_texts_map(dataset, fold_idx, split):
    samples_df = load_samples(dataset, fold_idx, split).reset_index(drop=True)
    return pd.Series(
        samples_df["text_idx"].values,
        index=samples_df.index
    ).to_dict()


def get_labels_map(dataset):
    with open(f"resource/dataset/{dataset}/labels_map.pkl", "rb") as labels_map_file:
        return pickle.load(labels_map_file)


def get_ranking(prediction, texts_map, texts_cls, labels_cls, labels_map, cls):
    ranking = {}
    for idx, (labels, scores) in enumerate(prediction):
        text_idx = texts_map[idx]
        if cls in texts_cls[text_idx]:
            labels_scores = {}
            for label, score in zip(labels, scores):
                label_idx = labels_map.get(label, -1)  # labels_map[label]
                if cls in labels_cls.get(label_idx, []):
                    labels_scores[f"label_{label_idx}"] = score
            if len(labels_scores) > 0:
                ranking[f"text_{text_idx}"] = labels_scores
    return ranking


def load_prediction(dataset, model, fold_idx):
    return zip(
        np.load(f"resource/prediction/fold_{fold_idx}/{model}-{dataset}-labels.npy", allow_pickle=True),
        np.load(f"resource/prediction/fold_{fold_idx}/{model}-{dataset}-scores.npy", allow_pickle=True)
    )


if __name__ == '__main__':
    dataset = "Wiki10-31k"
    model = "AttentionXML"
    metrics = _get_metrics(["mrr", "recall", "hit_rate"], [1, 5, 10, 100])
    relevance_map = _load_relevance_map(dataset)

    # cls
    texts_cls = _load_texts_cls(dataset)
    labels_cls = _load_labels_cls(dataset)

    # maps
    labels_map = {v: k for k, v in get_labels_map(dataset).items()}

    results = []
    rankings = {}
    for fold_idx in [0]:
        prediction = load_prediction(dataset, model, fold_idx)
        texts_map = get_texts_map(dataset, fold_idx, split="test")
        for cls in ["tail"]:
            ranking = get_ranking(prediction, texts_map, texts_cls, labels_cls, labels_map, cls)
            result = evaluate(
                Qrels(
                    {key: value for key, value in relevance_map.items() if key in ranking.keys()}
                ),
                Run(ranking),
                metrics
            )
            result = {k: round(v, 3) for k, v in result.items()}
            result["fold"] = fold_idx
            result["cls"] = cls

            results.append(result)
            # rankings[fold_idx][cls] = ranking
    print(pd.DataFrame(results))
