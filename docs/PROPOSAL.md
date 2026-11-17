# Project Proposal: Where Does Sleep-Staging Generalization Break?

## Research question
When an EEG sleep-stage classifier trained on one cohort is applied to a
different cohort (different age distribution, health status, or recording
hardware), which specific signal features drive the performance drop — and
can a simple, interpretable feature-based model be *more robust* to this
shift than a deep model, even if less accurate in-domain?

## Closest existing work
- Cross-cohort generalization failure in sleep staging, with Grad-CAM-based
  explanation: "AI Generalisation Gap In Comorbid Sleep Disorder Staging"
  (arXiv:2603.23582, March 2026). https://www.alphaxiv.org/abs/2603.23582
- Domain-generalization training across multiple sleep datasets:
  SleepDIFFormer (2025 preprint).
- Interpretable single-channel staging: AISleep (Patterns, Sept 2025),
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12745993/
- Interpretable embedding space for cross-dataset performance: NormIntSleep
  (arXiv:2211.03282, 2022). https://ar5iv.labs.arxiv.org/html/2211.03282

These papers either report a generalization gap exists, or propose
architectures that aim to close it. None (as surfaced by my search, abstracts
only — not verified against full text) does a controlled, feature-level
decomposition of *which specific spectral/temporal features* carry over
across cohorts versus which do not, comparing a deep model's learned
representations against hand-engineered sleep features side by side.

## Proposed contribution
A controlled replication-and-extension, not a SOTA claim:
1. Reproduce a standard CNN sleep-stage classifier on Sleep-EDF.
2. Build a fully interpretable baseline (gradient-boosted trees over
   hand-engineered band-power / spindle / K-complex / spectral-slope
   features per epoch).
3. Evaluate both within-dataset and cross-cohort (age-stratified split of
   Sleep-EDF as a first pass; a second public cohort if access terms permit).
4. Decompose the generalization gap: which features/channels have importance
   that is stable across cohorts vs. which collapse, using SHAP values for
   the interpretable model and Grad-CAM for the CNN.
5. Report where the two models agree and disagree about *why* performance
   drops — a result about interpretability-robustness tradeoffs, not just
   accuracy.

## Data
- **Sleep-EDF Database Expanded** (PhysioNet, physionet.org/content/sleep-edfx).
  Open, ODC-By-style license, no application process, contains both healthy
  (Sleep Cassette) and age/health-varied (Sleep Telemetry) recordings —
  itself enough for an age-stratified cross-cohort experiment.
- A second cohort (e.g. MESA or SHHS via NSRR) would strengthen the cross-
  dataset claim but requires a data use agreement; access terms will be
  checked before any attempt to use it, and the project does not depend on
  obtaining it.

## Methods
- Preprocessing: resample to common rate, 30s epochs, AASM 5-class labels
  (W, N1, N2, N3, REM), standard artifact rejection.
- Models: (a) 1D-CNN baseline on raw/filtered epochs; (b) LightGBM on
  hand-engineered features (relative band power per band, spectral edge
  frequency, Hjorth parameters, spindle/K-complex counts via existing
  detectors in `yasa`/`mne`).
- Cross-cohort evaluation: train on one stratum (e.g. younger/healthy),
  test on another (older / Sleep Telemetry), compared to within-stratum
  cross-validation.
- Interpretability: SHAP for the feature model; Grad-CAM/saliency for the CNN.

## Evaluation
- Per-stage F1 and macro-F1, confusion matrices.
- Generalization gap = within-stratum accuracy − cross-stratum accuracy,
  with bootstrap confidence intervals (not point estimates alone).
- Feature-importance stability across strata (e.g. rank correlation of SHAP
  importances between the two training strata).

## Risks and mitigations
- Second-cohort access may be denied or slow → entire analysis is designed
  to run on Sleep-EDF alone via age/cohort stratification.
- Deep model may underperform due to limited compute/tuning → the research
  question is about *generalization*, not absolute SOTA accuracy, so a
  modest but honestly-evaluated baseline is sufficient.
- Risk of overclaiming → every result will state sample sizes, confidence
  intervals, and explicitly flag this as a student replication, not a novel
  clinical tool.

## Milestones
1. Repo scaffold, data pipeline, preprocessing.
2. CNN baseline on Sleep-EDF (within-cohort).
3. Interpretable feature model on Sleep-EDF (within-cohort).
4. Cross-cohort/age-stratified generalization experiment, both models.
5. Interpretability analysis (SHAP + Grad-CAM) and feature-stability comparison.
6. Final report: results, limitations, figures, citations.

## Ethical note
All data is public, de-identified, retrospective, and already collected
under the original studies' own consent/ethics processes. No new human
subjects data is collected. No results from this project should be read as
diagnostic or clinically validated — it is an educational research replication.
