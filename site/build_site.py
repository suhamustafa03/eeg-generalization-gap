"""
Build the static results website from results/results.json.

The site is a single self-contained index.html with the results data
embedded directly (no fetch/CORS issues when opened via file://, and no
server needed). Run this after src/run_pipeline.py produces new results.

Usage:
    python site/build_site.py
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "results" / "results.json"
TEMPLATE_PATH = pathlib.Path(__file__).resolve().parent / "template.html"
OUT_PATH = pathlib.Path(__file__).resolve().parent / "index.html"


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run `python src/run_pipeline.py` first."
        )
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    template = TEMPLATE_PATH.read_text()
    html = template.replace("__RESULTS_JSON__", json.dumps(results))
    OUT_PATH.write_text(html)
    print(f"Built {OUT_PATH}")


if __name__ == "__main__":
    main()
