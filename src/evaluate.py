"""
Core experiment: compare within-cohort vs. cross-cohort generalization for
the CNN baseline and the interpretable feature model, with subject-level
splits (no subject appears in both train and test).

Two cohort-split modes are supported:
  --mode stratum   : train on one Sleep-EDF subset (e.g. cassette-young),
                     test on another (e.g. cassette-old), using age bands
                     from the Sleep-EDF SC-subjects metadata. Works with a
                     single downloaded subset.
  --mode dataset    : train on one subset (e.g. cassette), test on another
                     (e.g. telemetry). Requires both subsets downloaded.

Usage:
    python src/evaluate.py --mode dataset \
        --train data/processed/cassette.npz \
        --test data/processed/telemetry.npz \
        --out results/generalization_dataset.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.utils import resample

from models import CNNBaseline, make_feature_model, normalize_epochs
from preprocessing import STAGE_NAMES

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def _bootstrap_ci(y_true, y_pred, n_boot=500, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def train_eval_cnn(X_train, y_train, X_test, y_test, n_epochs=10, seed=0):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNBaseline().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    class_counts = np.bincount(y_train, minlength=5)
    weights = torch.tensor(class_counts.sum() / (class_counts + 1e-6), dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    Xt = torch.tensor(normalize_epochs(X_train), dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=128, shuffle=True
    )

    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    Xe = torch.tensor(normalize_epochs(X_test), dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(Xe).argmax(dim=1).cpu().numpy()
    return preds, model


def train_eval_feature_model(X_train, y_train, X_test, y_test):
    model = make_feature_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, model


def summarize(y_true, y_pred, label):
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(5)))
    lo, hi = _bootstrap_ci(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5))).tolist()
    print(f"[{label}] macro-F1 = {macro_f1:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    return {
        "label": label,
        "macro_f1": float(macro_f1),
        "macro_f1_ci": [lo, hi],
        "per_class_f1": {STAGE_NAMES[i]: float(per_class_f1[i]) for i in range(5)},
        "confusion_matrix": cm,
        "n_test_epochs": int(len(y_true)),
    }


def load_npz(path: pathlib.Path):
    d = np.load(path, allow_pickle=True)
    return d["epochs"], d["labels"], d["features"], d["subjects"]


def within_cohort_cv(epochs, labels, features, subjects, n_splits=5, seed=0):
    """Subject-grouped CV within a single cohort, as the 'in-domain' reference point."""
    gkf = GroupKFold(n_splits=n_splits)
    cnn_preds, feat_preds, y_all = [], [], []
    for train_idx, test_idx in gkf.split(epochs, labels, groups=subjects):
        cp, _ = train_eval_cnn(epochs[train_idx], labels[train_idx], epochs[test_idx], labels[test_idx], seed=seed)
        fp, _ = train_eval_feature_model(features[train_idx], labels[train_idx], features[test_idx], labels[test_idx])
        cnn_preds.append(cp)
        feat_preds.append(fp)
        y_all.append(labels[test_idx])
    return np.concatenate(y_all), np.concatenate(cnn_preds), np.concatenate(feat_preds)


def cross_dataset(train_npz, test_npz, seed=0):
    tr_epochs, tr_labels, tr_features, _ = load_npz(train_npz)
    te_epochs, te_labels, te_features, _ = load_npz(test_npz)
    cnn_preds, cnn_model = train_eval_cnn(tr_epochs, tr_labels, te_epochs, te_labels, seed=seed)
    feat_preds, feat_model = train_eval_feature_model(tr_features, tr_labels, te_features, te_labels)
    return te_labels, cnn_preds, feat_preds, cnn_model, feat_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["within", "dataset"], required=True)
    parser.add_argument("--train", type=pathlib.Path, required=True)
    parser.add_argument("--test", type=pathlib.Path, help="required for --mode dataset")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    results = {"mode": args.mode}

    if args.mode == "within":
        epochs, labels, features, subjects = load_npz(args.train)
        y, cnn_preds, feat_preds = within_cohort_cv(epochs, labels, features, subjects)
        results["cnn"] = summarize(y, cnn_preds, "within-cohort CNN")
        results["feature_model"] = summarize(y, feat_preds, "within-cohort feature model")
    else:
        if args.test is None:
            raise ValueError("--test is required for --mode dataset")
        y, cnn_preds, feat_preds, _, _ = cross_dataset(args.train, args.test)
        results["cnn"] = summarize(y, cnn_preds, "cross-dataset CNN")
        results["feature_model"] = summarize(y, feat_preds, "cross-dataset feature model")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
