"""Sanity checks for feature extraction — run with: pytest tests/"""
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from features import extract_features, extract_features_batch, FEATURE_NAMES  # noqa: E402


def test_feature_vector_shape():
    rng = np.random.RandomState(0)
    epoch = rng.randn(3000).astype(np.float32)  # 30s @ 100Hz
    feats = extract_features(epoch, sfreq=100.0)
    assert feats.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(feats))


def test_batch_matches_single():
    rng = np.random.RandomState(1)
    epochs = rng.randn(5, 3000).astype(np.float32)
    batch_feats = extract_features_batch(epochs, sfreq=100.0)
    single_feats = np.stack([extract_features(e, sfreq=100.0) for e in epochs])
    np.testing.assert_allclose(batch_feats, single_feats)


def test_relative_band_powers_sum_near_one():
    rng = np.random.RandomState(2)
    epoch = rng.randn(3000).astype(np.float32)
    feats = extract_features(epoch, sfreq=100.0)
    band_power_idx = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("relpow_")]
    total = feats[band_power_idx].sum()
    # Bands don't cover the full spectrum (0.5-30Hz of presumably wider PSD),
    # so this should be <= 1, and not degenerate (zero or absurdly large).
    assert 0 < total <= 1.01


def test_constant_signal_does_not_crash():
    epoch = np.ones(3000, dtype=np.float32)
    feats = extract_features(epoch, sfreq=100.0)
    assert np.all(np.isfinite(feats))


if __name__ == "__main__":
    test_feature_vector_shape()
    test_batch_matches_single()
    test_relative_band_powers_sum_near_one()
    test_constant_signal_does_not_crash()
    print("All feature tests passed.")
