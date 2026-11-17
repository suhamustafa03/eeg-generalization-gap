"""
Interpretability analysis comparing what the feature model relies on
(SHAP values over interpretable features) in two different cohorts, to see
whether the *reasons* for a prediction stay stable when the data distribution
shifts — the central question of this project.

Usage:
    python src/interpret.py --a data/processed/cassette.npz \
        --b data/processed/telemetry.npz \
        --out results/feature_stability.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import shap
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from features import FEATURE_NAMES
from models import make_feature_model

# Exact TreeExplainer SHAP cost grows with the number of samples explained;
# at tens/hundreds of thousands of epochs this becomes impractical (we
# measured >5 minutes and still running on just 26k epochs). Computing SHAP
# on a large, class-stratified random subsample instead of the full dataset
# is standard practice for TreeExplainer at scale and does not change what
# the feature-stability comparison measures (mean |SHAP value| per feature).
SHAP_MAX_SAMPLES = 5000


def mean_abs_shap(features: np.ndarray, labels: np.ndarray, max_samples: int = SHAP_MAX_SAMPLES) -> np.ndarray:
    model = make_feature_model()
    model.fit(features, labels)

    if len(features) > max_samples:
        shap_features, _ = train_test_split(
            features, train_size=max_samples, random_state=0, stratify=labels
        )
    else:
        shap_features = features

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_features)
    # Multiclass TreeExplainer output varies by SHAP version:
    #   - list of (n_samples, n_features) arrays, one per class, or
    #   - a single (n_samples, n_features, n_classes) ndarray.
    # Either way, reduce to one importance value per feature by averaging
    # |SHAP value| over samples and classes.
    if isinstance(shap_values, list):
        stacked = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values])
        return stacked.mean(axis=0)
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        return np.abs(shap_values).mean(axis=(0, 2))
    return np.abs(shap_values).mean(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=pathlib.Path, required=True, help="cohort A .npz")
    parser.add_argument("--b", type=pathlib.Path, required=True, help="cohort B .npz")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    da = np.load(args.a, allow_pickle=True)
    db = np.load(args.b, allow_pickle=True)

    imp_a = mean_abs_shap(da["features"], da["labels"])
    imp_b = mean_abs_shap(db["features"], db["labels"])

    rho, pval = spearmanr(imp_a, imp_b)

    out = {
        "cohort_a": str(da["subset"]),
        "cohort_b": str(db["subset"]),
        "feature_importance_a": dict(zip(FEATURE_NAMES, imp_a.tolist())),
        "feature_importance_b": dict(zip(FEATURE_NAMES, imp_b.tolist())),
        "spearman_rank_correlation": float(rho),
        "spearman_pvalue": float(pval),
        "interpretation": (
            "A high, significant rank correlation means the same features "
            "matter for prediction in both cohorts (stable reasoning). "
            "A low or non-significant correlation means the model relies on "
            "different cues per cohort, which helps explain any accuracy "
            "drop seen in evaluate.py's cross-dataset mode."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Spearman rho = {rho:.3f} (p={pval:.4f})")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
