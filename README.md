# Where Does Sleep-Staging Generalization Break?

A controlled, reproducible study of how EEG sleep-stage classifiers
generalize (or fail to) across cohorts, comparing a small deep-learning
baseline against a fully interpretable feature-based model. Built as a
research-skills demonstration project, not a production medical tool.

**Full proposal:** [docs/PROPOSAL.md](docs/PROPOSAL.md)
**Results website:** [site/index.html](site/index.html) — built from the full-cohort run below; open it directly in a browser.

## Headline result (full cohort: 153 cassette + 44 telemetry subjects)

| Model | Within-cohort macro-F1 | Cross-cohort macro-F1 | Generalization gap |
|---|---|---|---|
| CNN | 0.683 (95% CI 0.681–0.685) | 0.557 (95% CI 0.552–0.562) | **0.126** |
| Feature model (RandomForest) | 0.629 (95% CI 0.626–0.630) | 0.627 (95% CI 0.622–0.633) | **0.002** |

At full scale, both models' cross-cohort gaps are much smaller than the
10-subject preliminary run below suggested — and the feature model's gap is
essentially zero: its macro-F1 on the unseen telemetry cohort (0.627) is
statistically indistinguishable from its within-cassette macro-F1 (0.629).
The CNN still shows a real gap (0.126), driven mostly by Wake (W), whose F1
collapses from 0.95 within-cohort to 0.20 cross-cohort — the CNN relies on
cues that do not transfer for recognizing wakefulness in the telemetry
cohort, while the feature model's hand-engineered band-power/Hjorth features
hold up. The feature-importance rank correlation between cohorts is now 0.62
(Spearman, p = 0.031) — a significant, moderately strong correlation,
consistent with the feature model's near-zero generalization gap: it really
is relying on similar cues in both cohorts.

This reverses the preliminary 10-subject finding below, where the feature
model appeared to generalize better but with a non-trivial gap, and
underscores why the small-sample run was explicitly labeled preliminary:
sample size changed not just the precision of the estimates but, for the
feature model, the qualitative conclusion.

### Preliminary result (10 subjects per cohort — superseded by the full-cohort run above)

| Model | Within-cohort macro-F1 | Cross-cohort macro-F1 | Generalization gap |
|---|---|---|---|
| CNN | 0.702 (95% CI 0.694–0.710) | 0.270 (95% CI 0.262–0.279) | **0.432** |
| Feature model (RandomForest) | 0.735 (95% CI 0.727–0.743) | 0.470 (95% CI 0.459–0.479) | **0.266** |

This smaller run showed both models losing substantial accuracy
cross-cohort, with the CNN's gap roughly 1.6x the feature model's, and a
feature-importance rank correlation of 0.57 (Spearman, p = 0.055, not
significant at this sample size). It is kept here, clearly marked
preliminary, as an honest record of how the estimates and even the ranking
between models changed once the full cohort was used — not as a result to
build conclusions on.

Both runs are small relative to clinical sleep-staging literature (one EEG
channel, two specific PhysioNet cohorts) — not a claim to match published,
larger-scale systems. See [Limitations](#honesty-notes) below and in the
site.

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
  models.py                 CNN baseline + RandomForest feature model
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

Download data (public, no account needed). For the full cohort used in the
headline result above:

```bash
python src/download_data.py --subset cassette --n 153
python src/download_data.py --subset telemetry --n 44
```

(For a quick, smaller reproduction, use a lower `--n`, e.g. 10, matching the
preliminary result above.)

Run the full pipeline (builds processed datasets, trains/evaluates both
models within-cohort and cross-cohort, runs the SHAP stability analysis):

```bash
python src/run_pipeline.py --max-subjects 999
```

(The full cassette download is ~7GB and the full run takes roughly 15–20
minutes on a modern CPU — dataset loading dominates; model training and SHAP
are fast thanks to subsampling. Use `--max-subjects 10` for a quick
preliminary run instead.)

This writes `results/results.json`. Build the website from it:

```bash
python site/build_site.py
open site/index.html   # or just open the file in a browser
```

No GPU is required; the CNN and RandomForest both train on CPU in a few
minutes even at full scale. See `notebooks/colab_pipeline.ipynb` for a
free-compute (Google Colab) alternative if local compute or storage isn't
available — check current Colab free-tier limits before relying on it,
since they change over time.

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
