# Where Does Sleep-Staging Generalization Break?

A controlled, reproducible study of how EEG sleep-stage classifiers
generalize (or fail to) across cohorts, comparing a small deep-learning
baseline against a fully interpretable feature-based model. Built as a
research-skills demonstration project, not a production medical tool.

**Full proposal:** [docs/PROPOSAL.md](docs/PROPOSAL.md)
**Results website:** [site/index.html](site/index.html) (open locally after running the pipeline — see below)

## Research question

When a classifier trained on one EEG cohort is applied to a different
cohort (different age distribution, health status, recording hardware),
which specific signal features drive the performance drop — and is a
simple, interpretable model more robust to that shift than a deep one, even
if less accurate in-domain?

## Why this project, not a generic classifier demo

This directly engages a recent finding — ["AI Generalisation Gap In
Comorbid Sleep Disorder Staging"](https://www.alphaxiv.org/abs/2603.23582)
(arXiv:2603.23582, March 2026) — that deep sleep-staging models trained on
healthy cohorts fail to generalize to clinical populations. Rather than
repeating that result, this project decomposes it at the feature level: it
trains two very different kinds of models on the same data and asks whether
their *reasoning* (not just their accuracy) survives a cohort shift, using
SHAP feature-importance stability as the evidence. See
[docs/PROPOSAL.md](docs/PROPOSAL.md) for full related-work discussion,
data/method details, risks, and milestones.

## Data

[Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)
(PhysioNet, Open Data Commons Attribution License). Two cohorts:

- **sleep-cassette** — healthy subjects, 1987–1991 study.
- **sleep-telemetry** — subjects with mild sleep difficulty, 1994 study.

Both are public, de-identified, and require no account or data-use
agreement — only attribution, which this README and `docs/PROPOSAL.md`
provide.

## Repository structure

```
docs/PROPOSAL.md       Full research proposal: question, related work, methods, risks
src/
  download_data.py     Pull Sleep-EDF recordings from PhysioNet
  preprocessing.py      30s epoching + AASM 5-class labels via MNE
  features.py            Interpretable per-epoch EEG features (band power, Hjorth, etc.)
  dataset_build.py      Build cached .npz datasets from raw recordings
  models.py                 CNN baseline + LightGBM feature model
  train_cnn.py             Standalone CNN training script
  evaluate.py              Within-cohort vs. cross-cohort evaluation (subject-grouped)
  interpret.py              SHAP feature-importance stability across cohorts
  run_pipeline.py        End-to-end driver -> results/results.json
notebooks/colab_pipeline.ipynb   Free-compute fallback (Google Colab) mirroring run_pipeline.py
site/                    Static results website (built from results/results.json)
results/                  Pipeline outputs (JSON), not checked in by default
```

## Reproducing the results

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download data (public, no account needed):

```bash
python src/download_data.py --subset cassette --n 10
python src/download_data.py --subset telemetry --n 10
```

Run the full pipeline (builds processed datasets, trains/evaluates both
models within-cohort and cross-cohort, runs the SHAP stability analysis):

```bash
python src/run_pipeline.py --max-subjects 10
```

This writes `results/results.json`. Build the website from it:

```bash
python site/build_site.py
open site/index.html   # or just open the file in a browser
```

No GPU is required; the CNN is intentionally small enough to train on CPU
in a few minutes for this subject count. For a larger run with more
subjects, see `notebooks/colab_pipeline.ipynb` for a free-compute (Google
Colab) alternative — check current Colab free-tier limits before relying on
it, since they change over time.

## Honesty notes

- Every number on the results site comes from `results/results.json`,
  produced by actually running `src/run_pipeline.py` — nothing is
  hand-typed or illustrative.
- Confidence intervals are bootstrap-based and reported alongside every
  point estimate; sample sizes are small for a fast reproducible demo, and
  this is stated directly on the site and in the limitations section.
- This is an educational replication/extension of published findings, not
  a novel clinical system, and is not validated for any diagnostic use.

## License and attribution

Code in this repository: for the student's own educational/application
use. Sleep-EDF data: Open Data Commons Attribution License — see
[PhysioNet's citation requirements](https://physionet.org/content/sleep-edfx/1.0.0/)
before any redistribution.
