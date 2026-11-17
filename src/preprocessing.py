"""
Load Sleep-EDF PSG + hypnogram pairs into 30-second epochs with AASM-style
5-class labels (W, N1, N2, N3, REM), using MNE's built-in Sleep-EDF support.

Channels: we use a single EEG channel (Fpz-Cz, present in both cassette and
telemetry subsets) so the two strata are directly comparable, and so results
transfer to the single-channel, low-cost EEG setting most relevant outside
a sleep lab.
"""
from __future__ import annotations

import pathlib

import mne
import numpy as np

EPOCH_SECONDS = 30
EEG_CHANNEL = "EEG Fpz-Cz"

# Collapse AASM stages to the standard 5-class scheme used across the
# literature cited in docs/PROPOSAL.md.
STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # merge N3/N4 per AASM 2007 recommendation
    "Sleep stage R": 4,
}
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def load_recording(psg_path: pathlib.Path, hyp_path: pathlib.Path):
    """Return (epochs_array, labels_array) for one subject recording.

    epochs_array: (n_epochs, n_samples_per_epoch) float32, raw EEG in volts.
    labels_array: (n_epochs,) int, index into STAGE_NAMES.
    """
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
    annot = mne.read_annotations(hyp_path)
    raw.set_annotations(annot, emit_warning=False)

    raw.pick([EEG_CHANNEL])
    sfreq = raw.info["sfreq"]

    events, event_id = mne.events_from_annotations(
        raw, chunk_duration=EPOCH_SECONDS, verbose="ERROR"
    )
    # Keep only annotation labels we recognize; map back to our scheme.
    id_to_name = {v: k for k, v in event_id.items()}
    valid_mask = np.array([id_to_name[e] in STAGE_MAP for e in events[:, 2]])
    events = events[valid_mask]

    mne_epochs = mne.Epochs(
        raw,
        events,
        event_id={k: v for k, v in event_id.items() if k in STAGE_MAP},
        tmin=0,
        tmax=EPOCH_SECONDS - 1.0 / sfreq,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )

    data = mne_epochs.get_data(picks=EEG_CHANNEL)[:, 0, :].astype(np.float32)
    raw_labels = [id_to_name[e] for e in mne_epochs.events[:, 2]]
    labels = np.array([STAGE_MAP[name] for name in raw_labels], dtype=np.int64)

    return data, labels, sfreq


def discover_pairs(raw_dir: pathlib.Path) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Find matching (-PSG.edf, -Hypnogram.edf) pairs in a directory."""
    psg_files = sorted(raw_dir.glob("*PSG.edf"))
    pairs = []
    for psg in psg_files:
        subject_id = psg.name.split("-")[0][:6]  # e.g. SC4001 from SC4001E0-PSG.edf
        hyp_candidates = sorted(raw_dir.glob(f"{subject_id}*Hypnogram.edf"))
        if hyp_candidates:
            pairs.append((psg, hyp_candidates[0]))
    return pairs
