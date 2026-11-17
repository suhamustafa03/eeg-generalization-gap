"""
Download the Sleep-EDF Database Expanded from PhysioNet.

Dataset: https://physionet.org/content/sleep-edfx/1.0.0/
License: Open Data Commons Attribution License v1.0 (ODC-By).
Contains two sub-cohorts we use as our two "strata" for the cross-cohort
generalization experiment:
  - sleep-cassette: healthy 25-101 y/o subjects, 1987-1991 study.
  - sleep-telemetry: subjects with mild difficulty falling asleep, who took
    temazepam or placebo, 1994 study.

This script only downloads data; it does not require an account or DUA.
Run: python src/download_data.py --subset cassette --n 20
"""
import argparse
import pathlib
import re
import sys
import urllib.request

BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0"
RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"

SUBSET_DIRS = {
    "cassette": "sleep-cassette",
    "telemetry": "sleep-telemetry",
}

_LINK_RE = re.compile(r'href="([^"]+\.edf)"')


def fetch_record_list(subset: str) -> list[str]:
    """Fetch the directory listing for a subset (no RECORDS index exists for
    this dataset) and extract .edf filenames from the HTML."""
    url = f"{BASE_URL}/{SUBSET_DIRS[subset]}/"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            html = resp.read().decode()
        return _LINK_RE.findall(html)
    except Exception as e:
        print(f"Could not fetch directory listing for {subset}: {e}", file=sys.stderr)
        return []


def download_file(subset: str, relpath: str, dest_dir: pathlib.Path) -> None:
    url = f"{BASE_URL}/{SUBSET_DIRS[subset]}/{relpath}"
    dest = dest_dir / relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["cassette", "telemetry"], required=True)
    parser.add_argument("--n", type=int, default=20, help="number of subject recordings to pull")
    args = parser.parse_args()

    dest_dir = RAW_DIR / args.subset
    records = fetch_record_list(args.subset)
    if not records:
        print(
            "No records fetched. PhysioNet file listing may have changed format, "
            "or this environment has no outbound network access. "
            "You can also download manually from "
            f"{BASE_URL}/{SUBSET_DIRS[args.subset]}/ and place files under {dest_dir}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Each subject has a -PSG.edf (signals) and -Hypnogram.edf (labels) pair.
    psg_files = sorted(r for r in records if "PSG" in r)[: args.n]
    for psg in psg_files:
        subject_id = psg.split("-")[0]
        matching_hyp = [r for r in records if r.startswith(subject_id) and "Hypnogram" in r]
        download_file(args.subset, psg, dest_dir)
        for hyp in matching_hyp:
            download_file(args.subset, hyp, dest_dir)

    print(f"Done. Files saved under {dest_dir}")


if __name__ == "__main__":
    main()
