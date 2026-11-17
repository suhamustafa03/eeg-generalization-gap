"""
Hand-engineered, interpretable per-epoch EEG features for the feature-based
sleep-staging model. These are standard, well-documented quantities (not a
novel feature set) chosen specifically because each one has a clear
physiological meaning, so SHAP importances over them are interpretable by a
non-specialist reader.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "sigma": (12, 16),  # spindle band
    "beta": (16, 30),
}

FEATURE_NAMES = (
    [f"relpow_{b}" for b in BANDS]
    + ["spectral_edge_95", "spectral_entropy"]
    + ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"]
    + ["mean_abs_amplitude", "zero_crossing_rate"]
)


def _band_powers(freqs: np.ndarray, psd: np.ndarray) -> dict[str, float]:
    total = np.trapezoid(psd, freqs) + 1e-12
    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[f"relpow_{name}"] = np.trapezoid(psd[mask], freqs[mask]) / total
    return out


def _spectral_edge(freqs: np.ndarray, psd: np.ndarray, pct: float = 0.95) -> float:
    cumulative = np.cumsum(psd)
    cumulative /= cumulative[-1] + 1e-12
    idx = np.searchsorted(cumulative, pct)
    return float(freqs[min(idx, len(freqs) - 1)])


def _spectral_entropy(psd: np.ndarray) -> float:
    p = psd / (psd.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / np.log2(len(p) + 1e-12))


def _hjorth(x: np.ndarray) -> tuple[float, float, float]:
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x, var_dx, var_ddx = np.var(x), np.var(dx), np.var(ddx)
    activity = var_x
    mobility = np.sqrt(var_dx / (var_x + 1e-12))
    complexity = np.sqrt(var_ddx / (var_dx + 1e-12)) / (mobility + 1e-12)
    return float(activity), float(mobility), float(complexity)


def extract_features(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """Compute one feature vector for a single (n_samples,) epoch."""
    freqs, psd = welch(epoch, fs=sfreq, nperseg=min(len(epoch), int(sfreq * 4)))
    band_feats = _band_powers(freqs, psd)
    edge95 = _spectral_edge(freqs, psd)
    entropy = _spectral_entropy(psd)
    activity, mobility, complexity = _hjorth(epoch)
    mean_abs_amp = float(np.mean(np.abs(epoch)))
    zero_cross = float(np.mean(np.diff(np.sign(epoch)) != 0))

    vec = (
        [band_feats[f"relpow_{b}"] for b in BANDS]
        + [edge95, entropy, activity, mobility, complexity, mean_abs_amp, zero_cross]
    )
    return np.array(vec, dtype=np.float32)


def extract_features_batch(epochs: np.ndarray, sfreq: float) -> np.ndarray:
    """epochs: (n_epochs, n_samples) -> (n_epochs, n_features)."""
    return np.stack([extract_features(e, sfreq) for e in epochs])
