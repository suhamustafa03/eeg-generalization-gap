"""
Build a processed dataset (features + raw epochs + labels + subject/stratum
metadata) from downloaded Sleep-EDF raw files, and cache it to disk as .npz.

Usage:
    python src/dataset_build.py --subset cassette --out data/processed/cassette.npz
    python src/dataset_build.py --subset telemetry --out data/processed/telemetry.npz
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
from tqdm import tqdm

from features import extract_features_batch
from preprocessing import discover_pairs, load_recording

RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"


def build(subset: str, out_path: pathlib.Path, max_subjects: int | None = None) -> None:
    raw_dir = RAW_DIR / subset
    pairs = discover_pairs(raw_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No PSG/Hypnogram pairs found in {raw_dir}. "
            "Run src/download_data.py first."
        )
    if max_subjects:
        pairs = pairs[:max_subjects]

    all_epochs, all_labels, all_features, all_subjects = [], [], [], []
    for subj_idx, (psg, hyp) in enumerate(tqdm(pairs, desc=f"loading {subset}")):
        try:
            epochs, labels, sfreq = load_recording(psg, hyp)
        except Exception as e:
            print(f"Skipping {psg.name}: {e}")
            continue
        feats = extract_features_batch(epochs, sfreq)
        all_epochs.append(epochs)
        all_labels.append(labels)
        all_features.append(feats)
        all_subjects.append(np.full(len(labels), subj_idx, dtype=np.int32))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        epochs=np.concatenate(all_epochs),
        labels=np.concatenate(all_labels),
        features=np.concatenate(all_features),
        subjects=np.concatenate(all_subjects),
        subset=subset,
    )
    print(f"Saved {out_path} ({sum(len(l) for l in all_labels)} epochs, {len(pairs)} subjects)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["cassette", "telemetry"], required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args()
    build(args.subset, args.out, args.max_subjects)
