"""
Build the static results website from results/results.json.

The site is a small multi-page static site (Overview / Results / Methods /
Interpretation / Reproducibility), sharing one layout and one stylesheet.
Results data is embedded directly into the pages that need it (no
fetch/CORS issues when opened via file://, no server needed). Run this
after src/run_pipeline.py produces new results.

Usage:
    python site/build_site.py
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITE_DIR = pathlib.Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "results.json"
LAYOUT_PATH = SITE_DIR / "layout.html"
CONTENT_DIR = SITE_DIR / "content"

# (filename, nav label, page title, meta description, dek shown under the
# page heading, hero-style header instead of a plain page-head, is_welcome:
# index.html supplies its own full hero in content/overview.html, so no
# separate __HEADER__ block is built for it)
PAGES = [
    (
        "index.html",
        "Overview",
        "Overview",
        "Overview of a controlled cross-cohort study of EEG sleep-stage classifier generalization.",
        None,
        True,
        True,
    ),
    (
        "results.html",
        "Results",
        "Results",
        "Full model comparison, generalization gaps, stage-level performance, and SHAP feature stability.",
        "Full-cohort model comparison, generalization gaps, per-stage performance, and feature-importance stability across cohorts.",
        False,
        False,
    ),
    (
        "methods.html",
        "Methods",
        "Methods",
        "Datasets, preprocessing, models, and evaluation protocol.",
        "Datasets, preprocessing, the two models compared, and the evaluation protocol.",
        False,
        False,
    ),
    (
        "interpretation.html",
        "Interpretation",
        "Interpretation",
        "What the findings suggest, related work, uncertainty, and limitations.",
        "What the results suggest and don't suggest, why the preliminary finding reversed, related work, and limitations.",
        False,
        False,
    ),
    (
        "reproducibility.html",
        "Reproducibility",
        "Reproducibility",
        "Setup, data sources, and how to rerun the full analysis.",
        "Setup, data sources, and exact commands to rerun the full analysis yourself.",
        False,
        False,
    ),
]


def build_nav(active_filename: str) -> str:
    links = []
    for filename, nav_label, *_ in PAGES:
        if filename == active_filename:
            links.append(f'<a href="{filename}" class="active" aria-current="page">{nav_label}</a>')
        else:
            links.append(f'<a href="{filename}">{nav_label}</a>')
    return "\n    ".join(links)


def build_header(title: str, dek: str | None, is_hero: bool, is_welcome: bool = False) -> str:
    if is_welcome:
        # index.html supplies its own full welcome hero as the first
        # section of its content fragment; no separate header markup here.
        return ""
    if is_hero:
        return """<header class="hero">
  <h1>Where Does Sleep-Staging Generalization Break?</h1>
  <p class="sub">A controlled cross-cohort study comparing a deep CNN against an interpretable feature-based model on public EEG sleep data &mdash; not a claim of state-of-the-art accuracy, but a decomposition of <em>where and why</em> generalization fails.</p>
  <div class="badges">
    <span class="badge">Public data only (PhysioNet Sleep-EDF)</span>
    <span class="badge">Fully reproducible</span>
    <span class="badge">Educational research replication</span>
  </div>
</header>"""
    return f"""<div class="page-head">
  <div class="eyebrow">Sleep-staging generalization gap</div>
  <h1>{title}</h1>
  <p class="dek">{dek or ""}</p>
</div>"""


def build_footer_nav(index: int) -> str:
    prev_link = ""
    next_link = ""
    if index > 0:
        pfile, plabel = PAGES[index - 1][0], PAGES[index - 1][1]
        prev_link = f'<a href="{pfile}"><span class="dir">&larr; Previous</span>{plabel}</a>'
    if index < len(PAGES) - 1:
        nfile, nlabel = PAGES[index + 1][0], PAGES[index + 1][1]
        next_link = f'<a href="{nfile}" style="text-align:right;"><span class="dir">Next &rarr;</span>{nlabel}</a>'
    if not prev_link and not next_link:
        return ""
    return f'<div class="page-nav-footer">{prev_link or "<span></span>"}{next_link}</div>'


RENDER_SCRIPT = """
<script>
const RESULTS = __RESULTS_JSON__;

function fmt(x, d=3) { return (x === undefined || x === null) ? "–" : x.toFixed(d); }

function renderHeadline() {
  const el = document.getElementById("headline-card");
  if (!el) return;
  if (!RESULTS || !RESULTS.generalization_gap) {
    el.innerHTML = '<p class="no-data">No results yet.</p>';
    return;
  }
  const gap = RESULTS.generalization_gap;
  const stab = RESULTS.feature_stability;
  el.innerHTML = `
    <div class="grid-2">
      <div>
        <div class="metric-label">CNN generalization gap (within − cross)</div>
        <div class="metric-value ${gap.cnn_gap > 0 ? 'gap-bad' : 'gap-good'}">${fmt(gap.cnn_gap)}</div>
        <div class="metric-ci">macro-F1 points lost moving to the unseen cohort</div>
      </div>
      <div>
        <div class="metric-label">Feature model generalization gap</div>
        <div class="metric-value ${gap.feature_gap > 0 ? 'gap-bad' : 'gap-good'}">${fmt(gap.feature_gap)}</div>
        <div class="metric-ci">macro-F1 points lost moving to the unseen cohort</div>
      </div>
    </div>
    <p style="margin-top:18px;margin-bottom:0;">Feature-importance rank correlation across cohorts (Spearman): <strong>${fmt(stab ? stab.spearman_rank_correlation : null, 2)}</strong> (p = ${stab ? fmt(stab.spearman_pvalue, 4) : '–'}). ${stab && stab.spearman_pvalue < 0.05 ? 'The model relies on a statistically stable set of features across cohorts.' : 'There is no statistically significant evidence of stable feature reliance across cohorts at this sample size.'}</p>
  `;
}

function bar(label, value, max, color) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return `<div class="bar-row">
    <div class="bar-label">${label}</div>
    <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
    <div class="bar-val">${fmt(value, 2)}</div>
  </div>`;
}

function renderComparison() {
  const el = document.getElementById("comparison-card");
  if (!el) return;
  const r = RESULTS;
  if (!r || !r.within_cassette_cnn) { el.innerHTML = '<p class="no-data">No results yet.</p>'; return; }
  el.innerHTML = `
    <p style="color:var(--text-muted);margin-top:0;">Macro-F1, higher is better. "Within" = recording-grouped cross-validation on the cassette cohort (see Methods for a caveat on this grouping). "Cross" = trained on cassette, tested on the unseen telemetry cohort.</p>
    ${bar("CNN (within)", r.within_cassette_cnn.macro_f1, 1, "var(--accent)")}
    ${bar("CNN (cross)", r.cross_cnn.macro_f1, 1, "var(--accent-2)")}
    ${bar("Feature model (within)", r.within_cassette_feature.macro_f1, 1, "var(--accent)")}
    ${bar("Feature model (cross)", r.cross_feature.macro_f1, 1, "var(--accent-2)")}
    <div class="table-scroll">
    <table style="margin-top:20px;">
      <thead><tr><th scope="col">Model</th><th scope="col">Setting</th><th scope="col">Macro-F1</th><th scope="col">95% CI</th><th scope="col">N epochs</th></tr></thead>
      <tbody>
        <tr><td>CNN</td><td>Within-cohort</td><td>${fmt(r.within_cassette_cnn.macro_f1)}</td><td>${fmt(r.within_cassette_cnn.macro_f1_ci[0])}–${fmt(r.within_cassette_cnn.macro_f1_ci[1])}</td><td>${r.within_cassette_cnn.n_test_epochs}</td></tr>
        <tr><td>CNN</td><td>Cross-cohort</td><td>${fmt(r.cross_cnn.macro_f1)}</td><td>${fmt(r.cross_cnn.macro_f1_ci[0])}–${fmt(r.cross_cnn.macro_f1_ci[1])}</td><td>${r.cross_cnn.n_test_epochs}</td></tr>
        <tr><td>Feature model</td><td>Within-cohort</td><td>${fmt(r.within_cassette_feature.macro_f1)}</td><td>${fmt(r.within_cassette_feature.macro_f1_ci[0])}–${fmt(r.within_cassette_feature.macro_f1_ci[1])}</td><td>${r.within_cassette_feature.n_test_epochs}</td></tr>
        <tr><td>Feature model</td><td>Cross-cohort</td><td>${fmt(r.cross_feature.macro_f1)}</td><td>${fmt(r.cross_feature.macro_f1_ci[0])}–${fmt(r.cross_feature.macro_f1_ci[1])}</td><td>${r.cross_feature.n_test_epochs}</td></tr>
      </tbody>
    </table>
    </div>
  `;
}

function renderPreview() {
  var card = document.getElementById("preview-card");
  var note = document.getElementById("preview-note");
  if (!card) return;
  var r = RESULTS;
  if (!r || !r.within_cassette_cnn) {
    card.innerHTML = '<p class="no-data">No results yet.</p>';
    if (note) note.textContent = "";
    return;
  }
  var gap = r.generalization_gap;
  function row(label, value) {
    return '<div class="preview-row"><span>' + label + '</span><strong>' + fmt(value, 3) + '</strong></div>';
  }
  function gapRow(value) {
    var cls = value <= 0.02 ? "preview-gap-good" : "preview-gap-bad";
    return '<div class="preview-row"><span>Generalization gap</span><strong class="' + cls + '">' + fmt(value, 3) + '</strong></div>';
  }
  card.innerHTML =
    '<div class="preview-grid">' +
      '<div class="preview-model-card">' +
        '<p class="preview-model-name">CNN</p>' +
        row("Within-cohort macro-F1", r.within_cassette_cnn.macro_f1) +
        row("Cross-cohort macro-F1", r.cross_cnn.macro_f1) +
        gapRow(gap.cnn_gap) +
      '</div>' +
      '<div class="preview-model-card">' +
        '<p class="preview-model-name">Feature model (RandomForest)</p>' +
        row("Within-cohort macro-F1", r.within_cassette_feature.macro_f1) +
        row("Cross-cohort macro-F1", r.cross_feature.macro_f1) +
        gapRow(gap.feature_gap) +
      '</div>' +
    '</div>';
  if (note) {
    note.textContent = "The CNN still loses accuracy on the unseen cohort; the RandomForest's cross-cohort score is close enough to its within-cohort score that the full Results and Interpretation pages treat it as no detectable difference, not proof of flawless generalization.";
  }
}

function renderStageLevel() {
  const el = document.getElementById("stage-card");
  if (!el) return;
  const r = RESULTS;
  if (!r || !r.within_cassette_cnn) { el.innerHTML = '<p class="no-data">No results yet.</p>'; return; }
  const stages = ["W", "N1", "N2", "N3", "REM"];
  const rows = stages.map(s => `
    <tr>
      <td>${s}</td>
      <td>${fmt(r.within_cassette_cnn.per_class_f1[s], 2)}</td>
      <td>${fmt(r.cross_cnn.per_class_f1[s], 2)}</td>
      <td>${fmt(r.within_cassette_feature.per_class_f1[s], 2)}</td>
      <td>${fmt(r.cross_feature.per_class_f1[s], 2)}</td>
    </tr>
  `).join("");
  el.innerHTML = `
    <p style="color:var(--text-muted);margin-top:0;">Per-class F1 by sleep stage. The CNN's Wake (W) score collapses cross-cohort while the feature model's holds steady &mdash; the clearest single signal behind the overall generalization gap.</p>
    <div class="table-scroll">
    <table>
      <thead><tr><th scope="col">Stage</th><th scope="col">CNN (within)</th><th scope="col">CNN (cross)</th><th scope="col">Feature (within)</th><th scope="col">Feature (cross)</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    </div>
  `;
}

function renderInterp() {
  const el = document.getElementById("interp-card");
  if (!el) return;
  const stab = RESULTS ? RESULTS.feature_stability : null;
  if (!stab) { el.innerHTML = '<p class="no-data">No results yet.</p>'; return; }
  const feats = Object.keys(stab.feature_importance_a);
  let rows = feats.map(f => `
    <tr><td>${f}</td><td>${fmt(stab.feature_importance_a[f], 3)}</td><td>${fmt(stab.feature_importance_b[f], 3)}</td></tr>
  `).join("");
  el.innerHTML = `
    <p style="color:var(--text-muted);margin-top:0;">Mean |SHAP value| per feature, computed independently on each cohort (cassette = A, telemetry = B). Similar rankings across columns mean the model's reasoning transfers; divergent rankings point to which cues are cohort-specific.</p>
    <div class="table-scroll">
    <table>
      <thead><tr><th scope="col">Feature</th><th scope="col">Importance (cassette)</th><th scope="col">Importance (telemetry)</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    </div>
  `;
}

renderHeadline();
renderComparison();
renderPreview();
renderStageLevel();
renderInterp();
</script>
"""


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run `python src/run_pipeline.py` first."
        )
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    results_json = json.dumps(results)

    layout = LAYOUT_PATH.read_text()

    content_filenames = {
        "index.html": "overview.html",
    }

    for i, (filename, nav_label, title, description, dek, is_hero, is_welcome) in enumerate(PAGES):
        content_path = CONTENT_DIR / content_filenames.get(filename, filename)
        content = content_path.read_text()

        # Only pages with a data-driven card need the results script.
        if any(marker in content for marker in ("headline-card", "comparison-card", "stage-card", "interp-card", "preview-card")):
            content += RENDER_SCRIPT.replace("__RESULTS_JSON__", results_json)

        page = layout
        page = page.replace("__PAGE_TITLE__", title)
        page = page.replace("__PAGE_DESCRIPTION__", description)
        page = page.replace("__BODY_CLASS__", "page-welcome" if is_welcome else "")
        page = page.replace("__NAV__", build_nav(filename))
        page = page.replace("__HEADER__", build_header(title, dek, is_hero, is_welcome))
        page = page.replace("__CONTENT__", content)
        page = page.replace("__PAGE_FOOTER_NAV__", build_footer_nav(i))

        out_path = SITE_DIR / filename
        out_path.write_text(page)
        print(f"Built {out_path}")


if __name__ == "__main__":
    main()
