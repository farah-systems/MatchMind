"""
lean_dataset.py
================
Shared logic for shrinking the match history down to only what
MatchFeatureBuilder actually needs to serve live predictions.

Why this works, mechanically -- everything build_match_features.py
computes only ever looks BACKWARD from a candidate match date, per team:
  - Elo: only needs the team's single most recent match (its stored
    pre-match home_elo/away_elo, replayed forward one result).
  - Rolling windows [5, 10] and 15-match EWM decay: converge using
    the team's last ~15-40 matches; anything older barely moves the
    EWM average and isn't touched by the fixed 5/10-match windows at all.
  - Rest days, matches_last_14d: only need the single most recent match.
  - Streaks: only look at the last 20 matches.
  - elo_momentum5: only the last 5 matches.
  - H2H last-4: only the last 4 meetings between the two specific teams
    -- in a round-robin top-5 league, any two teams meet every season,
    so this is always well within a team's recent-N-matches window.
  - Standings (current season replay): needs every match played so far
    THIS season for every team in the league -- a full season is at
    most 38 matches, so keeping each team's last ~60 matches naturally
    covers the full current season plus a buffer from last season.

So: keep the UNION of each team's last N matches (default 60), across
all leagues/seasons in one pass, and every one of the above stays
correct -- using the exact same MatchFeatureBuilder code, unmodified.
"""
import pandas as pd

DEFAULT_MATCHES_PER_TEAM = 60


def trim_to_recent(df: pd.DataFrame, matches_per_team: int = DEFAULT_MATCHES_PER_TEAM) -> pd.DataFrame:
    """
    Returns a new DataFrame containing the union of each team's most
    recent `matches_per_team` rows (by Date), deduplicated.

    Expects `df["Date"]` to already be a datetime dtype.
    """
    df = df.sort_values("Date").reset_index(drop=True)

    all_teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]], ignore_index=True))

    keep_indices = set()
    for team in all_teams:
        team_mask = (df["HomeTeam"] == team) | (df["AwayTeam"] == team)
        team_rows = df[team_mask].tail(matches_per_team)
        keep_indices.update(team_rows.index)

    lean = df.loc[sorted(keep_indices)].reset_index(drop=True)
    return lean
