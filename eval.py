import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from ranx import evaluate, Run, Qrels


def _get_metrics(metrics, thresholds):
    threshold_metrics = []
    for metric in metrics:
        for threshold in thresholds:
            threshold_metrics.append(f"{metric}@{threshold}")
    return threshold_metrics


def load_label_cls(dataset):
    with open(f"resource/dataset/{dataset}/label_cls.pkl", "rb") as label_cls_file:
        return pickle.load(label_cls_file)


def load_text_cls(dataset):
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


def get_ranking(labels_ids, scores, texts_map, text_cls, label_cls, cls):
    ranking = {}
    print("Ranking")
    for idx, (labels_ids, scores) in enumerate(zip(labels_ids, scores)):
        text_idx = texts_map[idx]
        if cls in text_cls[text_idx]:
            labels_scores = {}
            for label_idx, score in zip(labels_ids, scores):
                if cls in label_cls[int(label_idx)]:
                    labels_scores[f"label_{label_idx}"] = score
            if len(labels_scores) > 0:
                ranking[f"text_{text_idx}"] = labels_scores
            else:
                ranking[f"text_{text_idx}"] = {"label_-1": 0.0}
    return ranking


def load_prediction(dataset, model, fold_idx):
    print("Loading prediction")
    return np.load(f"resource/prediction/fold_{fold_idx}/{model}-{dataset}-Ensemble-labels.npy", allow_pickle=True), \
        np.load(f"resource/prediction/fold_{fold_idx}/{model}-{dataset}-Ensemble-scores.npy", allow_pickle=True)
    # return zip(
    #     np.load(f"resource/prediction/fold_{fold_idx}/{model}-{dataset}-Ensemble-labels.npy", allow_pickle=True),
    #     np.load(f"resource/prediction/fold_{fold_idx}/{model}-{dataset}-Ensemble-scores.npy", allow_pickle=True)
    # )


def checkpoint_results(results, dataset, model):
    pd.DataFrame(results).to_csv(
        f"resource/result/{model}_{dataset}.rts",
        sep='\t', index=False, header=True)


def checkpoint_rankings(rankings, dataset, model):
    with open(
            f"resource/ranking/{model}_{dataset}.rnk",
            "wb") as rankings_file:
        pickle.dump(rankings, rankings_file)


def checkpoint_ranking(ranking, model, dataset, fold_idx):
    ranking_dir = f"resource/ranking/{model}_{dataset}/"
    Path(ranking_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving ranking {fold_idx} on {ranking_dir}")
    with open(f"{ranking_dir}{model}_{dataset}_{fold_idx}.rnk", "wb") as ranking_file:
        pickle.dump(ranking, ranking_file)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return f"{round(100 * m, 1)}({round(100 * h, 1)})"


def get_ic(model, dataset, metrics, thresholds):
    print(f"Eval results for {model} in {dataset}:")
    result_df = pd.read_csv(f"resource/result/{model}_{dataset}.rts", header=0, sep="\t")
    s = ""
    for metric in metrics:
        for cls in ["tail", "head"]:
            cls_df = result_df[result_df["cls"] == cls]
            print([f"{metric}@{t}" for t in thresholds])
            for m in [f"{metric}@{t}" for t in thresholds]: #['ndcg@1', 'ndcg@5', 'ndcg@10', 'ndcg@19']:
                v = mean_confidence_interval(cls_df[m])
                s = s + f"{v}\t"
    print(s)


def get_result(dataset, model, folds, metrics, thresholds):
    metrics = _get_metrics(metrics, thresholds)
    print(f"Metrics: {metrics}")
    relevance_map = _load_relevance_map(dataset)

    # cls
    text_cls = load_text_cls(dataset)
    label_cls = load_label_cls(dataset)

    # maps
    # labels_map = {v: k for k, v in get_labels_map(dataset).items()}

    results = []
    rankings = {}
    for fold_idx in folds:
        rankings[fold_idx] = {}

        labels, scores = load_prediction(dataset, model, fold_idx)
        texts_map = get_texts_map(dataset, fold_idx, split="test")

        for cls in ["head", "tail"]:
            print(f"Evaluating on {cls} labels on fold {fold_idx}")
            ranking = get_ranking(labels, scores, texts_map, text_cls, label_cls, cls)
            print(f"Ranking size: {len(ranking)}")
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
            rankings[fold_idx][cls] = ranking
            checkpoint_ranking(rankings[fold_idx], model, dataset, fold_idx)

    print(pd.DataFrame(results))

    checkpoint_results(results, dataset, model)



if __name__ == '__main__':
    dataset = "AmazonCat-13k"
    model = "AttentionXML"
    metrics = ["ndcg", "precision"]
    thresholds = [1, 5, 10, 6]
    get_result(dataset, model, folds=[0, 1, 2], metrics=metrics, thresholds=thresholds)
    get_ic(model, dataset, metrics, thresholds)
