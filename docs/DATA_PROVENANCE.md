# Data Provenance

**Source:** Sleep-EDF Database Expanded, PhysioNet
https://physionet.org/content/sleep-edfx/1.0.0/

**Original studies:**
- Sleep Cassette: B. Kemp et al., healthy subjects aged 25–101, recorded
  1987–1991, studying age effects on sleep in healthy humans without
  sleep-related medication.
- Sleep Telemetry: subjects with mild difficulty falling asleep, recorded
  1994, studying temazepam's effect on sleep in a placebo-controlled trial.

**License:** Open Data Commons Attribution License v1.0 (ODC-By). Permits
reuse with attribution; see the PhysioNet project page for the exact terms
and the citation PhysioNet requests.

**How this project uses it:**
- Single EEG channel (Fpz-Cz) per recording, present in both subsets.
- 30-second epochs with AASM-mapped sleep-stage labels (from the
  accompanying Hypnogram annotation files).
- A small, fixed number of subjects per subset (configurable via
  `--max-subjects` / `--n`) to keep the pipeline fast and reproducible on a
  laptop; this is explicitly stated as a limitation, not hidden.

**What this project does not do:**
- No new data is collected from any person.
- No attempt to re-identify subjects.
- No redistribution of the raw data files in this repository (`data/raw/`
  and `data/processed/` are gitignored) — anyone reproducing results
  downloads directly from PhysioNet via `src/download_data.py`.

**Citation (for the original dataset):**
Kemp, B., Zwinderman, A.H., Tuk, B., Kamphuisen, H.A.C., & Oberye, J.J.L.
(2000). Analysis of a sleep-dependent neuronal feedback loop: the
HRV-focused Smoothed Nonlinear Energy Operator. IEEE-BME 47(9):1185-1194.

Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet:
Components of a new research resource for complex physiologic signals.
Circulation 101(23):e215-e220.
