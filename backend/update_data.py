"""
update_data.py
================
Run periodically (cron / GitHub Action / Render cron job) to pull
newly completed matches into the LEAN serving dataset
(data/recent_matches.csv -- see build_lean_dataset.py for how that
file is derived from your full historical archive).

WHAT WAS WRONG BEFORE, AND WHY THIS VERSION FIXES IT
-----------------------------------------------------
The previous version replayed Elo from scratch over the whole
dataframe every time, starting every team at PROMOTED_ELO. That's
correct ONLY if the dataframe holds a team's complete history from
day one. Once we're serving from a LEAN, trimmed dataset (only each
team's last ~60 matches), a from-scratch replay would incorrectly
reset every established team (Real Madrid, Man City, etc.) down to a
brand-new-team rating, since their true multi-season history isn't in
this file anymore.

Fix: SEED each team's current Elo from the last row they already
appear in (their stored pre-match rating, replayed forward one
result), then only compute NEW rows going forward from that seeded
state. This is both correct (doesn't discard real rating history) and
cheap (no full replay needed).

SCOPE NOTE -- rolling/decay columns
-------------------------------------
This intentionally does NOT recompute roll5/roll10/decay15 columns.
MatchFeatureBuilder never reads those from the CSV at request time --
it derives rolling/decayed stats fresh from the raw per-match stat
columns on every call (see build_match_features.py). So those
precomputed columns are irrelevant to live serving; recomputing them
here would be wasted work. If you retrain the model later, do that
from your full archive (kept separately, off Render), not from this
trimmed file.

SCOPE NOTE -- missing match stats
------------------------------------
football-data.org doesn't provide shots/xG/PPDA/etc., so those columns
come through as NaN on newly appended rows until you backfill them
from your original stats source (e.g. understat). LightGBM and the
on-the-fly rolling calc both tolerate NaN, but backfilling improves
accuracy for future matches whose rolling windows include these rows.

Usage:
    python update_data.py --data-path data/recent_matches.csv
"""
import argparse

import pandas as pd

import fixtures
from dtype_utils import downcast_dtypes
from lean_dataset import trim_to_recent, DEFAULT_MATCHES_PER_TEAM

ELO_K = 20
ELO_HOME_ADV = 60
PROMOTED_ELO = 1400


def _elo_update(h_elo, a_elo, home_goals, away_goals):
    """One Elo step, same formula as training (k=20, home_advantage=60)."""
    expected_home = 1 / (1 + 10 ** (-((h_elo + ELO_HOME_ADV) - a_elo) / 400))
    if home_goals > away_goals:
        actual_home = 1.0
    elif home_goals == away_goals:
        actual_home = 0.5
    else:
        actual_home = 0.0
    new_h_elo = h_elo + ELO_K * (actual_home - expected_home)
    new_a_elo = a_elo + ELO_K * ((1 - actual_home) - (1 - expected_home))
    return new_h_elo, new_a_elo


def seed_current_elo(df: pd.DataFrame) -> dict:
    """
    Seed each (league, team)'s current Elo from the last row they
    appear in: take that row's stored PRE-match home_elo/away_elo and
    apply one more Elo update using that match's actual result, giving
    the POST-match ("entering the next match") rating -- matching
    MatchFeatureBuilder._current_elo's logic exactly.
    """
    current_elo = {}
    for _, row in df.sort_values("Date").iterrows():
        league, h, a = row["league"], row["HomeTeam"], row["AwayTeam"]
        if pd.isna(row["FTHG"]) or pd.isna(row["FTAG"]):
            continue
        new_h_elo, new_a_elo = _elo_update(row["home_elo"], row["away_elo"], row["FTHG"], row["FTAG"])
        current_elo[(league, h)] = new_h_elo
        current_elo[(league, a)] = new_a_elo
    return current_elo


def append_new_results(df: pd.DataFrame, current_elo: dict, days_back: int = 8):
    """
    Fetches recently finished results, skips ones already present, and
    appends the rest with correctly-seeded pre-match Elo ratings --
    updating `current_elo` in place as it goes so later new matches in
    the same batch see the effect of earlier ones.
    """
    new_results = []
    for league in fixtures.COMPETITION_CODES:
        for r in fixtures.get_recent_results(league, days_back=days_back):
            exists = (
                (df["Date"] == pd.Timestamp(r["Date"])) &
                (df["HomeTeam"] == r["HomeTeam"]) &
                (df["AwayTeam"] == r["AwayTeam"])
            ).any()
            if not exists:
                new_results.append(r)

    if not new_results:
        print("No new results found.")
        return df, 0

    new_results.sort(key=lambda r: r["Date"])
    print(f"Appending {len(new_results)} new results.")

    built_rows = []
    for r in new_results:
        league, h, a = r["league"], r["HomeTeam"], r["AwayTeam"]
        h_elo = current_elo.get((league, h), PROMOTED_ELO)
        a_elo = current_elo.get((league, a), PROMOTED_ELO)

        row = dict(r)
        row["Date"] = pd.Timestamp(row["Date"])
        row["home_elo"] = h_elo
        row["away_elo"] = a_elo
        row["elo_diff"] = h_elo - a_elo
        built_rows.append(row)

        if pd.notna(row.get("FTHG")) and pd.notna(row.get("FTAG")):
            new_h_elo, new_a_elo = _elo_update(h_elo, a_elo, row["FTHG"], row["FTAG"])
            current_elo[(league, h)] = new_h_elo
            current_elo[(league, a)] = new_a_elo

    # NOTE: football-data.org's payload has no shots/xG/etc columns, so
    # those come through as NaN here -- concat below leaves them missing,
    # which downstream code (LightGBM, rolling calc) tolerates natively.
    new_df = pd.DataFrame(built_rows)
    combined = pd.concat([df, new_df], ignore_index=True, sort=False)
    return combined.sort_values("Date").reset_index(drop=True), len(new_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/recent_matches.csv")
    parser.add_argument("--days-back", type=int, default=8)
    parser.add_argument("--matches-per-team", type=int, default=DEFAULT_MATCHES_PER_TEAM)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    df["Date"] = pd.to_datetime(df["Date"])

    current_elo = seed_current_elo(df)
    df, n_added = append_new_results(df, current_elo, days_back=args.days_back)

    if n_added == 0:
        print("Dataset unchanged.")
        return

    # Re-trim so the file stays bounded in size as new matches keep
    # arriving over a season, instead of growing without limit.
    df = trim_to_recent(df, matches_per_team=args.matches_per_team)
    df = downcast_dtypes(df)
    df.to_csv(args.data_path, index=False)
    print(f"Saved updated dataset: {len(df)} matches -> {args.data_path}")


if __name__ == "__main__":
    main()
