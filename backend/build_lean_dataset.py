"""
build_lean_dataset.py
======================
Run this LOCALLY (not on Render) against your full historical archive
to produce the small file the live API actually serves from.

Your full CSV (~256MB) has every match ever played, but
MatchFeatureBuilder only ever looks backward a bounded distance from
any candidate match date (see lean_dataset.py for exactly why this is
safe). This script keeps the union of each team's last N matches --
same feature-builder code, same formulas, ~10-20x smaller file --
which is what actually solves the Render out-of-memory problem
(the CSV, once loaded into a DataFrame, was almost certainly the
biggest single memory cost in the whole app).

After running this:
  1. Upload the OUTPUT file (data/recent_matches.csv by default) to
     wherever MATCHMIND_DATA_URL points (e.g. your Hugging Face
     dataset repo), replacing or alongside the full file.
  2. Keep your full archive somewhere safe off Render (for future
     model retraining) -- this script never modifies or deletes it.
  3. Update Render's MATCHMIND_DATA_URL (and MATCHMIND_DATA_PATH, if
     you rename the file) to point at the new lean file.

Usage:
    python build_lean_dataset.py \
        --input data/top5_leagues_features_full.csv \
        --output data/recent_matches.csv \
        --matches-per-team 60
"""
import argparse
import os

import pandas as pd

from dtype_utils import downcast_dtypes
from lean_dataset import trim_to_recent, DEFAULT_MATCHES_PER_TEAM


def build_lean_dataset(input_path: str, output_path: str, matches_per_team: int):
    print(f"Reading full dataset from {input_path} ...")
    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"Loaded {len(df):,} total matches, {df.shape[1]} columns.")

    lean = trim_to_recent(df, matches_per_team=matches_per_team)
    lean = downcast_dtypes(lean)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    lean.to_csv(output_path, index=False)

    orig_mb = os.path.getsize(input_path) / 1e6
    new_mb = os.path.getsize(output_path) / 1e6
    print(
        f"Lean dataset: {len(lean):,} matches "
        f"({len(lean) / len(df):.1%} of original row count), "
        f"last {matches_per_team} matches per team."
    )
    print(f"File size: {orig_mb:.1f} MB -> {new_mb:.1f} MB")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/top5_leagues_features_full.csv")
    parser.add_argument("--output", default="data/recent_matches.csv")
    parser.add_argument("--matches-per-team", type=int, default=DEFAULT_MATCHES_PER_TEAM)
    args = parser.parse_args()
    build_lean_dataset(args.input, args.output, args.matches_per_team)


if __name__ == "__main__":
    main()
