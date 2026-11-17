"""
End-to-end driver: build processed datasets from downloaded raw data, run
within-cohort and cross-cohort evaluation, run the interpretability
comparison, and write a single results.json the website reads from.

Usage:
    python src/run_pipeline.py --max-subjects 10
"""
from __future__ import annotations

import argparse
import json
import pathlib

from dataset_build import build
from evaluate import cross_dataset, load_npz, summarize, within_cohort_cv
from interpret import mean_abs_shap
from features import FEATURE_NAMES
from scipy.stats import spearmanr
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-subjects", type=int, default=10)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    cassette_npz = PROCESSED / "cassette.npz"
    telemetry_npz = PROCESSED / "telemetry.npz"

    if not args.skip_build:
        build("cassette", cassette_npz, max_subjects=args.max_subjects)
        build("telemetry", telemetry_npz, max_subjects=args.max_subjects)

    results = {}

    # 1. Within-cohort reference (cassette only, subject-grouped CV).
    epochs, labels, features, subjects = load_npz(cassette_npz)
    y, cnn_preds, feat_preds = within_cohort_cv(epochs, labels, features, subjects, n_splits=min(5, len(set(subjects))))
    results["within_cassette_cnn"] = summarize(y, cnn_preds, "within-cassette CNN")
    results["within_cassette_feature"] = summarize(y, feat_preds, "within-cassette feature model")

    # 2. Cross-dataset generalization: train cassette, test telemetry.
    y2, cnn_preds2, feat_preds2, _, _ = cross_dataset(cassette_npz, telemetry_npz)
    results["cross_cnn"] = summarize(y2, cnn_preds2, "cassette->telemetry CNN")
    results["cross_feature"] = summarize(y2, feat_preds2, "cassette->telemetry feature model")

    # 3. Interpretability stability: SHAP importances, cassette vs telemetry.
    dc = np.load(cassette_npz, allow_pickle=True)
    dt = np.load(telemetry_npz, allow_pickle=True)
    imp_a = mean_abs_shap(dc["features"], dc["labels"])
    imp_b = mean_abs_shap(dt["features"], dt["labels"])
    rho, pval = spearmanr(imp_a, imp_b)
    results["feature_stability"] = {
        "cohort_a": "cassette",
        "cohort_b": "telemetry",
        "feature_importance_a": dict(zip(FEATURE_NAMES, imp_a.tolist())),
        "feature_importance_b": dict(zip(FEATURE_NAMES, imp_b.tolist())),
        "spearman_rank_correlation": float(rho),
        "spearman_pvalue": float(pval),
    }

    results["generalization_gap"] = {
        "cnn_gap": results["within_cassette_cnn"]["macro_f1"] - results["cross_cnn"]["macro_f1"],
        "feature_gap": results["within_cassette_feature"]["macro_f1"] - results["cross_feature"]["macro_f1"],
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {out_path}")
    print(json.dumps(results["generalization_gap"], indent=2))


if __name__ == "__main__":
    main()
