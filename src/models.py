"""
Two sleep-stage classifiers compared throughout this project:

1. CNNBaseline  - a small 1D-CNN over raw EEG epochs (standard deep-learning
   approach, e.g. similar in spirit to DeepSleepNet/TinySleepNet but much
   smaller, since this is a feasibility/interpretability study, not a bid
   for SOTA accuracy).
2. FeatureModel - a LightGBM classifier over the hand-engineered features in
   features.py (fully interpretable via SHAP and feature importances).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier

N_CLASSES = 5


class CNNBaseline(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=50, stride=6, padding=25),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_samples) -> (batch, 1, n_samples)
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)


def make_feature_model(class_weight: dict | str | None = "balanced") -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        class_weight=class_weight,
        random_state=0,
        verbosity=-1,
    )


def normalize_epochs(epochs: np.ndarray) -> np.ndarray:
    """Per-epoch z-score normalization, standard for raw-EEG CNN input."""
    mean = epochs.mean(axis=1, keepdims=True)
    std = epochs.std(axis=1, keepdims=True) + 1e-8
    return (epochs - mean) / std
