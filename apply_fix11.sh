#!/bin/bash
set -e
cd "$(dirname "$0")"

cat > 'backend/season_simulator.py' << 'MATCHMIND_EOF'
"""
season_simulator.py
====================
Simulates a full season's remaining fixtures using Monte Carlo sampling.

DESIGN CHOICE (documented deliberately, not an oversight): each remaining
fixture's win/draw/loss probability is computed ONCE, using team form/Elo
as of today. Those probabilities are then sampled many times (e.g. 10,000
trials) to produce a distribution over final standings.

This means simulated in-season form changes (a team going on an actual
mid-simulation win streak, updating Elo trial-by-trial) are NOT modeled —
that would require re-running the full model per fixture per trial, which
is thousands of times more expensive (feature-rebuild + 60-model ensemble
per fixture per trial) and not practical for a live web request. This is
the same simplification widely used by public season-simulator tools
(e.g. FiveThirtyEight's), and is a reasonable one: it treats team quality
as fixed over the simulation horizon rather than fully dynamic.

Match margins for goal-difference tie-breaking are sampled with a simple
heuristic (not a full goals model) — see _sample_margin().
"""

import numpy as np
import pandas as pd


def _sample_margin(outcome: str, rng: np.random.Generator) -> tuple[int, int]:
    """
    Approximate scoreline for a sampled W/D/L outcome, used only for
    goal-difference tie-breaking in standings — not a precise scoreline
    model. Draws lean toward low-scoring (0-0, 1-1); wins lean toward
    small margins, matching typical football scoreline distributions.
    """
    if outcome == "D":
        g = rng.choice([0, 1, 2], p=[0.35, 0.40, 0.25])
        return g, g
    margin = rng.choice([1, 2, 3, 4], p=[0.50, 0.30, 0.14, 0.06])
    base = rng.choice([0, 1], p=[0.55, 0.45])
    if outcome == "H":
        return base + margin, base
    else:
        return base, base + margin


def simulate_season(fixtures_with_probs: list[dict], n_trials: int = 10000, seed: int = 42,
                     progress_callback=None, progress_every: int = 50):
    """
    fixtures_with_probs: list of dicts, each with keys
        home_team, away_team, p_away, p_draw, p_home
    (already-played matches should be excluded — see main.py for how
    current standings from real results get combined with this.)

    progress_callback: optional callable(completed: int, total: int),
    invoked every `progress_every` trials — lets the caller (e.g. a
    background job in main.py) report live progress to the frontend
    without changing the simulation's actual behavior or results.

    Returns per-team distributions: title %, top-4 %, relegation %,
    average final points, average final position.
    """
    rng = np.random.default_rng(seed)
    teams = sorted(set(f["home_team"] for f in fixtures_with_probs) |
                    set(f["away_team"] for f in fixtures_with_probs))

    final_positions = {t: [] for t in teams}
    final_points = {t: [] for t in teams}

    outcomes_arr = np.array([[f["p_away"], f["p_draw"], f["p_home"]] for f in fixtures_with_probs])
    home_teams = [f["home_team"] for f in fixtures_with_probs]
    away_teams = [f["away_team"] for f in fixtures_with_probs]

    # Starting points/GD, if the caller passes current real standings in
    # via each team's existing accumulated values (see main.py) — default 0
    start_points = {t: fixtures_with_probs[0].get("_start_points", {}).get(t, 0) for t in teams} \
        if fixtures_with_probs and "_start_points" in fixtures_with_probs[0] else {t: 0 for t in teams}
    start_gd = {t: fixtures_with_probs[0].get("_start_gd", {}).get(t, 0) for t in teams} \
        if fixtures_with_probs and "_start_gd" in fixtures_with_probs[0] else {t: 0 for t in teams}

    for trial in range(n_trials):
        points = dict(start_points)
        gd = dict(start_gd)

        sampled = [rng.choice(["A", "D", "H"], p=probs) for probs in outcomes_arr]

        for i, outcome in enumerate(sampled):
            h, a = home_teams[i], away_teams[i]
            hg, ag = _sample_margin(outcome, rng)
            gd[h] = gd.get(h, 0) + (hg - ag)
            gd[a] = gd.get(a, 0) + (ag - hg)
            if outcome == "H":
                points[h] = points.get(h, 0) + 3
            elif outcome == "A":
                points[a] = points.get(a, 0) + 3
            else:
                points[h] = points.get(h, 0) + 1
                points[a] = points.get(a, 0) + 1

        standings = sorted(teams, key=lambda t: (-points.get(t, 0), -gd.get(t, 0)))
        for pos, team in enumerate(standings, start=1):
            final_positions[team].append(pos)
            final_points[team].append(points.get(team, 0))

        if progress_callback and (trial + 1) % progress_every == 0:
            progress_callback(trial + 1, n_trials)

    if progress_callback:
        progress_callback(n_trials, n_trials)

    results = []
    for team in teams:
        positions = np.array(final_positions[team])
        pts = np.array(final_points[team])
        results.append({
            "team": team,
            "avg_position": round(float(positions.mean()), 1),
            "avg_points": round(float(pts.mean()), 1),
            "title_pct": round(float((positions == 1).mean() * 100), 1),
            "top4_pct": round(float((positions <= 4).mean() * 100), 1),
            "relegation_pct": round(float((positions >= len(teams) - 2).mean() * 100), 1),
        })

    return sorted(results, key=lambda r: r["avg_position"])
MATCHMIND_EOF
echo 'wrote backend/season_simulator.py'

cat > 'backend/build_match_features.py' << 'MATCHMIND_EOF'
"""
build_match_features.py
========================
Builds a single-row feature vector for a hypothetical/upcoming match,
using ONLY data available before match_date (no leakage).

This reuses the EXACT formulas from the original feature-engineering
pipeline (Elo with k=20/home_advantage=60, rolling windows [5,10] +
15-match EWM decay, last-4 H2H, and live standings replay) — not
approximations.

Usage:
    from build_match_features import MatchFeatureBuilder

    builder = MatchFeatureBuilder("top5_leagues_features_full.csv")
    X = builder.build(
        home_team="Manchester United",
        away_team="Liverpool",
        match_date="2026-08-15",
        league="epl",
    )
    # X is a 1-row DataFrame ready for predict.py's ModelA.predict(X)

NOTE: this covers Elo, rolling/decayed team-form stats, rest days,
H2H, streaks, matches_last_14d, elo_momentum5, and standings/point-gap
features. Compare its output against a known historical match (see
validate_against_known_match() at the bottom) before trusting it on
genuinely hypothetical fixtures.
"""

import pandas as pd
import numpy as np

from dtype_utils import downcast_dtypes


# =========================================================
# Same stat definitions as the training pipeline (cell 3/4)
# =========================================================
STAT_MAP = {
    "goals": ("FTHG", "FTAG"),
    "ht_goals": ("HTHG", "HTAG"),
    "xg": ("home_xg", "away_xg"),
    "npxg": ("home_np_xg", "away_np_xg"),
    "shots": ("HS", "AS"),
    "sot": ("HST", "AST"),
    "corners": ("HC", "AC"),
    "deep_completions": ("home_deep_completions", "away_deep_completions"),
    "sot_pct": ("home_sot_pct", "away_sot_pct"),
    "xg_per_shot": ("home_xg_per_shot", "away_xg_per_shot"),
    "npxg_per_shot": ("home_npxg_per_shot", "away_npxg_per_shot"),
}
STAT_FOR_ONLY = {
    "ppda": ("home_ppda", "away_ppda"),
    "points": ("home_points", "away_points"),
}
ROLL_WINDOWS = [5, 10]
DECAY_SPAN = 15

EURO_SPOTS = {
    "epl": {"ucl": 4, "europa_total": 6},
    "spa": {"ucl": 4, "europa_total": 6},
    "ita": {"ucl": 4, "europa_total": 6},
    "ger": {"ucl": 4, "europa_total": 6},
    "fra": {"ucl": 3, "europa_total": 5},
}
RELEGATION_SPOTS = 3


class MatchFeatureBuilder:
    def __init__(self, history_csv_path, elo_k=20, elo_home_advantage=60,
                 first_season_elo=1600, promoted_elo=1400):
        self.df = pd.read_csv(history_csv_path)
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.df.sort_values("Date").reset_index(drop=True)
        # Halves-to-thirds the DataFrame's memory footprint (float64->float32,
        # low-cardinality strings->category) with no effect on model output --
        # see dtype_utils.py for why this is safe. Matters most on
        # memory-constrained hosts (e.g. Render's free 512MB tier).
        self.df = downcast_dtypes(self.df)
        self.elo_k = elo_k
        self.elo_home_advantage = elo_home_advantage
        self.first_season_elo = first_season_elo
        self.promoted_elo = promoted_elo

        # ---------------------------------------------------------------
        # PERFORMANCE: for season simulation, every one of a team's ~19-38
        # remaining fixtures gets scored with a different before_date, but
        # since none of those dates have any REAL match results yet (the
        # whole season is unplayed), _team_history(team, before_date, ...)
        # returns the EXACT SAME rows for every one of them -- there's no
        # real data between "today" and any future date to differentiate
        # them. That means Elo, rolling/decayed stats, streaks, H2H, and
        # standings were all being recomputed from scratch dozens of times
        # per team for no reason (measured: ~2.2s/fixture, ~14 minutes for
        # a full season). This cache canonicalizes any before_date beyond
        # the dataset's real max date down to one shared key, so those
        # expensive computations run once per team instead of once per
        # fixture. rest_days and matches_last_14d are NOT cached this way
        # since they genuinely depend on the exact before_date (they
        # measure a gap/window relative to it, not just which rows exist).
        # ---------------------------------------------------------------
        self._cache = {}
        self._max_real_date = self.df["Date"].max()

    def _canonical_date(self, before_date):
        ts = pd.Timestamp(before_date)
        return ts if ts <= self._max_real_date else self._max_real_date + pd.Timedelta(days=1)

    # -----------------------------------------------------------------
    # Elo: a team's current rating = its rating AFTER its most recent
    # real match before match_date. The stored home_elo/away_elo columns
    # are PRE-match values (leakage-safe design from training), so we
    # apply that match's actual result once more to get the POST-match
    # (i.e. "current, entering the next match") rating.
    # -----------------------------------------------------------------
    def _current_elo(self, team, league, before_date):
        cache_key = ("elo", team, league, self._canonical_date(before_date))
        if cache_key in self._cache:
            return self._cache[cache_key]

        team_matches = self.df[
            (self.df["league"] == league)
            & ((self.df["HomeTeam"] == team) | (self.df["AwayTeam"] == team))
            & (self.df["Date"] < before_date)
        ].sort_values("Date")

        if len(team_matches) == 0:
            result = self.promoted_elo  # unseen team -> same rule as newly promoted
            self._cache[cache_key] = result
            return result

        last = team_matches.iloc[-1]
        was_home = last["HomeTeam"] == team

        if pd.isna(last["FTHG"]) or pd.isna(last["FTAG"]):
            result = last["home_elo"] if was_home else last["away_elo"]
            self._cache[cache_key] = result
            return result

        h_elo, a_elo = last["home_elo"], last["away_elo"]
        expected_home = 1 / (1 + 10 ** (-((h_elo + self.elo_home_advantage) - a_elo) / 400))
        if last["FTHG"] > last["FTAG"]:
            actual_home = 1.0
        elif last["FTHG"] == last["FTAG"]:
            actual_home = 0.5
        else:
            actual_home = 0.0

        if was_home:
            result = h_elo + self.elo_k * (actual_home - expected_home)
        else:
            result = a_elo + self.elo_k * ((1 - actual_home) - (1 - expected_home))
        self._cache[cache_key] = result
        return result

    # -----------------------------------------------------------------
    # Rolling / decayed stats — mirrors add_rolling()'s shift(1) logic:
    # the value "entering" a new match equals the rolling/decayed stat
    # computed over the team's matches strictly BEFORE that match — which
    # for a hypothetical new row is just the plain rolling/EWM mean over
    # the team's most recent real matches.
    # -----------------------------------------------------------------
    def _team_history(self, team, before_date, venue="both"):
        if venue == "home":
            mask = (self.df["HomeTeam"] == team) & (self.df["Date"] < before_date)
        elif venue == "away":
            mask = (self.df["AwayTeam"] == team) & (self.df["Date"] < before_date)
        else:
            mask = ((self.df["HomeTeam"] == team) | (self.df["AwayTeam"] == team)) & (self.df["Date"] < before_date)
        return self.df[mask].sort_values("Date")

    def _stat_series_for_team(self, team, matches, home_col, away_col):
        """
        For each match in `matches` (already sorted by Date), pick home_col
        if `team` was the home side that match, else away_col.

        Vectorized with np.where instead of a Python-level iterrows() loop --
        this function is called dozens of times per team per fixture (once
        per stat, per venue), so on a season simulation building features
        for ~300+ remaining fixtures, an iterrows()-based version compounds
        into real, noticeable slowness (potentially minutes). np.where does
        the same row-by-row selection in one vectorized pass.
        """
        if len(matches) == 0:
            return pd.Series([], dtype=float)
        is_home = (matches["HomeTeam"] == team).to_numpy()
        vals = np.where(is_home, matches[home_col].to_numpy(), matches[away_col].to_numpy())
        return pd.Series(vals, dtype=float)

    def _rolling_and_decay(self, team, before_date, venue):
        """
        Returns {prefix}_roll{w}_{stat}_for/against and
        {prefix}_decay15_{stat}_for/against, matching training column names.
        """
        cache_key = ("rolling", team, venue, self._canonical_date(before_date))
        if cache_key in self._cache:
            return self._cache[cache_key]

        prefix = venue  # "both", "home", "away"
        matches = self._team_history(team, before_date, venue=venue)
        out = {}

        for stat, (home_c, away_c) in STAT_MAP.items():
            s_for = self._stat_series_for_team(team, matches, home_c, away_c)
            s_against = self._stat_series_for_team(team, matches, away_c, home_c)

            for w in ROLL_WINDOWS:
                out[f"{prefix}_roll{w}_{stat}_for"] = s_for.tail(w).mean() if len(s_for) else np.nan
                out[f"{prefix}_roll{w}_{stat}_against"] = s_against.tail(w).mean() if len(s_against) else np.nan
            out[f"{prefix}_decay{DECAY_SPAN}_{stat}_for"] = (
                s_for.ewm(span=DECAY_SPAN, adjust=False).mean().iloc[-1] if len(s_for) else np.nan
            )
            out[f"{prefix}_decay{DECAY_SPAN}_{stat}_against"] = (
                s_against.ewm(span=DECAY_SPAN, adjust=False).mean().iloc[-1] if len(s_against) else np.nan
            )

        for stat, (home_c, away_c) in STAT_FOR_ONLY.items():
            s_for = self._stat_series_for_team(team, matches, home_c, away_c)
            for w in ROLL_WINDOWS:
                out[f"{prefix}_roll{w}_{stat}_for"] = s_for.tail(w).mean() if len(s_for) else np.nan
            out[f"{prefix}_decay{DECAY_SPAN}_{stat}_for"] = (
                s_for.ewm(span=DECAY_SPAN, adjust=False).mean().iloc[-1] if len(s_for) else np.nan
            )

        # -----------------------------------------------------------------
        # Derived stats the training notebook computes on top of the raw
        # STAT_MAP columns (cell 4): saves, save_pct, clean_sheet, and
        # elo-as-a-rolled-stat. These were missing from this file entirely,
        # which is why /predict and /calendar were failing with a KeyError
        # on columns like hometeam_both_roll5_saves_for.
        # -----------------------------------------------------------------
        goals_for = self._stat_series_for_team(team, matches, "FTHG", "FTAG")
        goals_against = self._stat_series_for_team(team, matches, "FTAG", "FTHG")
        sot_for = self._stat_series_for_team(team, matches, "HST", "AST")
        sot_against = self._stat_series_for_team(team, matches, "AST", "HST")

        # Saves = shots on target FACED minus goals CONCEDED (keeper stops) —
        # same formula as the notebook's long_all["saves_for"] derivation.
        saves_for = sot_against - goals_against
        saves_against = sot_for - goals_for
        save_pct_for = pd.Series(
            np.where(sot_against > 0, saves_for / sot_against, 0), dtype=float
        )
        save_pct_against = pd.Series(
            np.where(sot_for > 0, saves_against / sot_for, 0), dtype=float
        )
        clean_sheet_for = (goals_against == 0).astype(float)

        for name, series in [
            ("saves_for", saves_for), ("saves_against", saves_against),
            ("save_pct_for", save_pct_for), ("save_pct_against", save_pct_against),
        ]:
            for w in ROLL_WINDOWS:
                out[f"{prefix}_roll{w}_{name}"] = series.tail(w).mean() if len(series) else np.nan
            out[f"{prefix}_decay{DECAY_SPAN}_{name}"] = (
                series.ewm(span=DECAY_SPAN, adjust=False).mean().iloc[-1] if len(series) else np.nan
            )

        for w in ROLL_WINDOWS:
            out[f"{prefix}_roll{w}_clean_sheet_for"] = (
                clean_sheet_for.tail(w).mean() if len(clean_sheet_for) else np.nan
            )
        out[f"{prefix}_decay{DECAY_SPAN}_clean_sheet_for"] = (
            clean_sheet_for.ewm(span=DECAY_SPAN, adjust=False).mean().iloc[-1]
            if len(clean_sheet_for) else np.nan
        )

        # elo_for: the team's own pre-match Elo entering each historical
        # match in this venue window — rolled the same way as any other
        # stat (this is what feeds hometeam_both_roll5_elo_for,
        # hometeam_home_decay15_elo_for, etc.)
        elo_for = self._stat_series_for_team(team, matches, "home_elo", "away_elo")
        for w in ROLL_WINDOWS:
            out[f"{prefix}_roll{w}_elo_for"] = elo_for.tail(w).mean() if len(elo_for) else np.nan
        out[f"{prefix}_decay{DECAY_SPAN}_elo_for"] = (
            elo_for.ewm(span=DECAY_SPAN, adjust=False).mean().iloc[-1] if len(elo_for) else np.nan
        )

        # std_goals / std_points / std_xg over roll5/roll10 (both-venue only in feature list)
        for stat in ["goals", "points", "xg"]:
            home_c, away_c = STAT_MAP.get(stat) or STAT_FOR_ONLY.get(stat)
            s_for = self._stat_series_for_team(team, matches, home_c, away_c)
            for w in ROLL_WINDOWS:
                out[f"{prefix}_roll{w}_std_{stat}_for"] = s_for.tail(w).std() if len(s_for) >= 2 else np.nan

        self._cache[cache_key] = out
        return out

    def _rest_days(self, team, before_date):
        matches = self._team_history(team, before_date, venue="both")
        if len(matches) == 0:
            return np.nan
        last_date = matches.iloc[-1]["Date"]
        days = (pd.Timestamp(before_date) - last_date).days
        return min(days, 8)  # same cap used in training

    def _streaks(self, team, before_date):
        cache_key = ("streaks", team, self._canonical_date(before_date))
        if cache_key in self._cache:
            return self._cache[cache_key]

        matches = self._team_history(team, before_date, venue="both").tail(20)
        results = []
        for _, m in matches.iterrows():
            if pd.isna(m["FTHG"]) or pd.isna(m["FTAG"]):
                continue
            is_home = m["HomeTeam"] == team
            gd = (m["FTHG"] - m["FTAG"]) if is_home else (m["FTAG"] - m["FTHG"])
            results.append("W" if gd > 0 else ("D" if gd == 0 else "L"))

        if not results:
            result = {"win_streak": 0, "loss_streak": 0, "unbeaten_streak": 0, "nowin_streak": 0}
            self._cache[cache_key] = result
            return result

        def count_from_end(pred):
            c = 0
            for r in reversed(results):
                if pred(r):
                    c += 1
                else:
                    break
            return c

        result = {
            "win_streak": count_from_end(lambda r: r == "W"),
            "loss_streak": count_from_end(lambda r: r == "L"),
            "unbeaten_streak": count_from_end(lambda r: r in ("W", "D")),
            "nowin_streak": count_from_end(lambda r: r in ("L", "D")),
        }
        self._cache[cache_key] = result
        return result

    def _matches_last_14d(self, team, before_date):
        matches = self._team_history(team, before_date, venue="both")
        cutoff = pd.Timestamp(before_date) - pd.Timedelta(days=14)
        return int((matches["Date"] >= cutoff).sum())

    def _elo_momentum5(self, team, league, before_date):
        cache_key = ("momentum5", team, league, self._canonical_date(before_date))
        if cache_key in self._cache:
            return self._cache[cache_key]

        matches = self.df[
            (self.df["league"] == league)
            & ((self.df["HomeTeam"] == team) | (self.df["AwayTeam"] == team))
            & (self.df["Date"] < before_date)
        ].sort_values("Date").tail(5)
        if len(matches) < 2:
            result = np.nan
        else:
            elos = [(m["home_elo"] if m["HomeTeam"] == team else m["away_elo"]) for _, m in matches.iterrows()]
            result = elos[-1] - elos[0]
        self._cache[cache_key] = result
        return result

    # -----------------------------------------------------------------
    # Head-to-head — same last-4-meetings logic as training
    # -----------------------------------------------------------------
    def _h2h(self, home_team, away_team, before_date, n=4):
        cache_key = ("h2h", home_team, away_team, self._canonical_date(before_date), n)
        if cache_key in self._cache:
            return self._cache[cache_key]

        pair_matches = self.df[
            (((self.df["HomeTeam"] == home_team) & (self.df["AwayTeam"] == away_team)) |
             ((self.df["HomeTeam"] == away_team) & (self.df["AwayTeam"] == home_team)))
            & (self.df["Date"] < before_date)
        ].sort_values("Date").tail(n)

        if len(pair_matches) == 0:
            result = (np.nan, np.nan, 0)
            self._cache[cache_key] = result
            return result

        gd_list, pts_list = [], []
        for _, m in pair_matches.iterrows():
            if pd.isna(m["FTHG"]) or pd.isna(m["FTAG"]):
                continue
            if m["HomeTeam"] == home_team:
                gd = m["FTHG"] - m["FTAG"]
            else:
                gd = m["FTAG"] - m["FTHG"]
            pts = 3 if gd > 0 else (1 if gd == 0 else 0)
            gd_list.append(gd)
            pts_list.append(pts)

        if not gd_list:
            result = (np.nan, np.nan, len(pair_matches))
        else:
            result = (float(np.mean(gd_list)), float(np.mean(pts_list)), len(pair_matches))
        self._cache[cache_key] = result
        return result

    # -----------------------------------------------------------------
    # Standings: replay this league-season's results up to match_date
    # -----------------------------------------------------------------
    def _standings_table(self, league, season, before_date):
        """
        The shared, expensive part of standings: points/goal-difference
        per team and the resulting sort order. Cached because this table
        is IDENTICAL for every fixture sharing the same
        (league, season, canonical before_date) -- only which two teams'
        values get extracted from it differs per fixture (see _standings
        below), so there's no reason to redo this groupby 380 times over.
        """
        cache_key = ("standings_table", league, season, self._canonical_date(before_date))
        if cache_key in self._cache:
            return self._cache[cache_key]

        season_matches = self.df[
            (self.df["league"] == league)
            & (self.df["season"] == season)
            & (self.df["Date"] < before_date)
        ]

        valid = season_matches.dropna(subset=["FTHG", "FTAG"])
        if len(valid) == 0:
            points, gd = {}, {}
        else:
            match_gd = (valid["FTHG"] - valid["FTAG"]).to_numpy()
            h_pts = np.select([match_gd > 0, match_gd == 0], [3, 1], default=0)
            a_pts = np.select([match_gd < 0, match_gd == 0], [3, 1], default=0)

            combined = pd.concat([
                pd.DataFrame({"team": valid["HomeTeam"].to_numpy(), "pts": h_pts, "gd": match_gd}),
                pd.DataFrame({"team": valid["AwayTeam"].to_numpy(), "pts": a_pts, "gd": -match_gd}),
            ], ignore_index=True)
            grouped = combined.groupby("team", observed=True)[["pts", "gd"]].sum()
            points = grouped["pts"].to_dict()
            gd = grouped["gd"].to_dict()

        standings = sorted(points.keys(), key=lambda t: (-points.get(t, 0), -gd.get(t, 0)))
        result = (points, gd, standings, len(season_matches))
        self._cache[cache_key] = result
        return result

    def _standings(self, home_team, away_team, league, season, before_date):
        points, gd, standings, n_season_matches = self._standings_table(league, season, before_date)
        n_teams = len(standings)

        def pos(team):
            return standings.index(team) + 1 if team in standings else np.nan

        def pts_at(rank):
            if 1 <= rank <= len(standings):
                return points.get(standings[rank - 1], 0)
            return np.nan

        spots = EURO_SPOTS.get(league, {"ucl": 4, "europa_total": 6})
        pts_1st, pts_ucl, pts_europe = pts_at(1), pts_at(spots["ucl"]), pts_at(spots["europa_total"])
        pts_releg = pts_at(n_teams - RELEGATION_SPOTS + 1) if n_teams >= RELEGATION_SPOTS else np.nan

        h_pts, a_pts = points.get(home_team, 0), points.get(away_team, 0)
        h_pos, a_pos = pos(home_team), pos(away_team)

        def gap(team_pts, cutoff):
            return team_pts - cutoff if not pd.isna(cutoff) else np.nan

        result = {
            "home_position": h_pos, "away_position": a_pos,
            "position_diff": (h_pos - a_pos) if (pd.notna(h_pos) and pd.notna(a_pos)) else np.nan,
            "home_pts_to_1st": gap(h_pts, pts_1st), "away_pts_to_1st": gap(a_pts, pts_1st),
            "home_pts_to_ucl": gap(h_pts, pts_ucl), "away_pts_to_ucl": gap(a_pts, pts_ucl),
            "home_pts_to_europe": gap(h_pts, pts_europe), "away_pts_to_europe": gap(a_pts, pts_europe),
            "home_pts_to_relegation": gap(h_pts, pts_releg), "away_pts_to_relegation": gap(a_pts, pts_releg),
        }
        return result, n_season_matches

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------
    def build(self, home_team, away_team, match_date, league, season=None):
        match_date = pd.Timestamp(match_date)

        if season is None:
            # Infer from date — adjust the month cutoff if a league's
            # season boundary differs from an August start
            year = match_date.year if match_date.month >= 7 else match_date.year - 1
            season = f"{str(year)[-2:]}-{str(year + 1)[-2:]}"

        home_elo = self._current_elo(home_team, league, match_date)
        away_elo = self._current_elo(away_team, league, match_date)

        home_rest = self._rest_days(home_team, match_date)
        away_rest = self._rest_days(away_team, match_date)

        h2h_gd, h2h_pts, h2h_n = self._h2h(home_team, away_team, match_date)

        home_streaks = self._streaks(home_team, match_date)
        away_streaks = self._streaks(away_team, match_date)

        home_m14 = self._matches_last_14d(home_team, match_date)
        away_m14 = self._matches_last_14d(away_team, match_date)

        home_mom = self._elo_momentum5(home_team, league, match_date)
        away_mom = self._elo_momentum5(away_team, league, match_date)

        standings, matches_played = self._standings(home_team, away_team, league, season, match_date)

        row = {
            "league": league,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "hometeam_rest_days": home_rest,
            "awayteam_rest_days": away_rest,
            "rest_days_diff": home_rest - away_rest,
            "h2h_goal_diff_last4": h2h_gd,
            "h2h_points_last4": h2h_pts,
            "h2h_matches_available": h2h_n,
            "matchday": matches_played + 1,
            "season_progress": np.nan,  # needs a total-rounds estimate; fill in if available
            "hometeam_win_streak": home_streaks["win_streak"],
            "hometeam_loss_streak": home_streaks["loss_streak"],
            "hometeam_unbeaten_streak": home_streaks["unbeaten_streak"],
            "hometeam_nowin_streak": home_streaks["nowin_streak"],
            "awayteam_win_streak": away_streaks["win_streak"],
            "awayteam_loss_streak": away_streaks["loss_streak"],
            "awayteam_unbeaten_streak": away_streaks["unbeaten_streak"],
            "awayteam_nowin_streak": away_streaks["nowin_streak"],
            "win_streak_diff": home_streaks["win_streak"] - away_streaks["win_streak"],
            "loss_streak_diff": home_streaks["loss_streak"] - away_streaks["loss_streak"],
            "unbeaten_streak_diff": home_streaks["unbeaten_streak"] - away_streaks["unbeaten_streak"],
            "nowin_streak_diff": home_streaks["nowin_streak"] - away_streaks["nowin_streak"],
            "hometeam_matches_last_14d": home_m14,
            "awayteam_matches_last_14d": away_m14,
            "matches_last_14d_diff": home_m14 - away_m14,
            "hometeam_elo_momentum5": home_mom,
            "awayteam_elo_momentum5": away_mom,
            "elo_momentum5_diff": (home_mom - away_mom) if (pd.notna(home_mom) and pd.notna(away_mom)) else np.nan,
            **standings,
        }

        for venue in ["both", "home"]:
            for k, v in self._rolling_and_decay(home_team, match_date, venue).items():
                row[f"hometeam_{k}"] = v
        for venue in ["both", "away"]:
            for k, v in self._rolling_and_decay(away_team, match_date, venue).items():
                row[f"awayteam_{k}"] = v

        # both-venue *_diff columns (home minus away), as used in feature_cols
        diff_stat_groups = (
            [(s, ["for", "against"]) for s in STAT_MAP.keys()]
            + [(s, ["for"]) for s in STAT_FOR_ONLY.keys()]
            + [("saves", ["for", "against"]), ("save_pct", ["for", "against"])]
            + [("clean_sheet", ["for"]), ("elo", ["for"])]
        )
        for stat, suffixes in diff_stat_groups:
            for w in ROLL_WINDOWS:
                for suffix in suffixes:
                    hk, ak = f"hometeam_both_roll{w}_{stat}_{suffix}", f"awayteam_both_roll{w}_{stat}_{suffix}"
                    if hk in row and ak in row:
                        row[f"both_roll{w}_{stat}_{suffix}_diff"] = row[hk] - row[ak]
            for suffix in suffixes:
                hk = f"hometeam_both_decay{DECAY_SPAN}_{stat}_{suffix}"
                ak = f"awayteam_both_decay{DECAY_SPAN}_{stat}_{suffix}"
                if hk in row and ak in row:
                    row[f"both_decay{DECAY_SPAN}_{stat}_{suffix}_diff"] = row[hk] - row[ak]

        # std_ (variance) diffs: roll5/roll10 only, "for" only, no decay variant
        for stat in ["std_goals", "std_points", "std_xg"]:
            for w in ROLL_WINDOWS:
                hk, ak = f"hometeam_both_roll{w}_{stat}_for", f"awayteam_both_roll{w}_{stat}_for"
                if hk in row and ak in row:
                    row[f"both_roll{w}_{stat}_for_diff"] = row[hk] - row[ak]

        return pd.DataFrame([row])


# =========================================================
# Validation helper: rebuild features for a KNOWN historical match
# and compare against the actual stored row, to sanity-check the
# builder before trusting it on hypothetical fixtures.
# =========================================================
def validate_against_known_match(builder: "MatchFeatureBuilder", full_df: pd.DataFrame,
                                   match_id=None, atol=1e-3):
    if match_id is None:
        row = full_df[full_df["is_training_season"] == True].sample(1, random_state=1).iloc[0]
    else:
        row = full_df[full_df["match_id"] == match_id].iloc[0]

    rebuilt = builder.build(
        home_team=row["HomeTeam"], away_team=row["AwayTeam"],
        match_date=row["Date"], league=row["league"], season=row["season"],
    ).iloc[0]

    mismatches = []
    for col in rebuilt.index:
        if col not in row.index:
            continue
        actual, rebuilt_val = row[col], rebuilt[col]
        if pd.isna(actual) and pd.isna(rebuilt_val):
            continue
        if pd.isna(actual) or pd.isna(rebuilt_val):
            mismatches.append((col, actual, rebuilt_val))
            continue
        if isinstance(actual, (int, float)) and isinstance(rebuilt_val, (int, float)):
            if abs(actual - rebuilt_val) > atol:
                mismatches.append((col, actual, rebuilt_val))
        elif actual != rebuilt_val:
            mismatches.append((col, actual, rebuilt_val))

    print(f"Checked match: {row['HomeTeam']} vs {row['AwayTeam']} on {row['Date'].date()}")
    print(f"Mismatches: {len(mismatches)} / {len(rebuilt.index)} columns compared")
    for col, a, r in mismatches[:20]:
        print(f"  {col}: actual={a} vs rebuilt={r}")
    return mismatches
MATCHMIND_EOF
echo 'wrote backend/build_match_features.py'

