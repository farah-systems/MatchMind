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
        self.elo_k = elo_k
        self.elo_home_advantage = elo_home_advantage
        self.first_season_elo = first_season_elo
        self.promoted_elo = promoted_elo

    # -----------------------------------------------------------------
    # Elo: a team's current rating = its rating AFTER its most recent
    # real match before match_date. The stored home_elo/away_elo columns
    # are PRE-match values (leakage-safe design from training), so we
    # apply that match's actual result once more to get the POST-match
    # (i.e. "current, entering the next match") rating.
    # -----------------------------------------------------------------
    def _current_elo(self, team, league, before_date):
        team_matches = self.df[
            (self.df["league"] == league)
            & ((self.df["HomeTeam"] == team) | (self.df["AwayTeam"] == team))
            & (self.df["Date"] < before_date)
        ].sort_values("Date")

        if len(team_matches) == 0:
            return self.promoted_elo  # unseen team -> same rule as newly promoted

        last = team_matches.iloc[-1]
        was_home = last["HomeTeam"] == team

        if pd.isna(last["FTHG"]) or pd.isna(last["FTAG"]):
            return last["home_elo"] if was_home else last["away_elo"]

        h_elo, a_elo = last["home_elo"], last["away_elo"]
        expected_home = 1 / (1 + 10 ** (-((h_elo + self.elo_home_advantage) - a_elo) / 400))
        if last["FTHG"] > last["FTAG"]:
            actual_home = 1.0
        elif last["FTHG"] == last["FTAG"]:
            actual_home = 0.5
        else:
            actual_home = 0.0

        if was_home:
            return h_elo + self.elo_k * (actual_home - expected_home)
        else:
            return a_elo + self.elo_k * ((1 - actual_home) - (1 - expected_home))

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
        vals = []
        for _, m in matches.iterrows():
            vals.append(m[home_col] if m["HomeTeam"] == team else m[away_col])
        return pd.Series(vals, dtype=float)

    def _rolling_and_decay(self, team, before_date, venue):
        """
        Returns {prefix}_roll{w}_{stat}_for/against and
        {prefix}_decay15_{stat}_for/against, matching training column names.
        """
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

        # std_goals / std_points / std_xg over roll5/roll10 (both-venue only in feature list)
        for stat in ["goals", "points", "xg"]:
            home_c, away_c = STAT_MAP.get(stat) or STAT_FOR_ONLY.get(stat)
            s_for = self._stat_series_for_team(team, matches, home_c, away_c)
            for w in ROLL_WINDOWS:
                out[f"{prefix}_roll{w}_std_{stat}_for"] = s_for.tail(w).std() if len(s_for) >= 2 else np.nan

        return out

    def _rest_days(self, team, before_date):
        matches = self._team_history(team, before_date, venue="both")
        if len(matches) == 0:
            return np.nan
        last_date = matches.iloc[-1]["Date"]
        days = (pd.Timestamp(before_date) - last_date).days
        return min(days, 8)  # same cap used in training

    def _streaks(self, team, before_date):
        matches = self._team_history(team, before_date, venue="both").tail(20)
        results = []
        for _, m in matches.iterrows():
            if pd.isna(m["FTHG"]) or pd.isna(m["FTAG"]):
                continue
            is_home = m["HomeTeam"] == team
            gd = (m["FTHG"] - m["FTAG"]) if is_home else (m["FTAG"] - m["FTHG"])
            results.append("W" if gd > 0 else ("D" if gd == 0 else "L"))

        if not results:
            return {"win_streak": 0, "loss_streak": 0, "unbeaten_streak": 0, "nowin_streak": 0}

        def count_from_end(pred):
            c = 0
            for r in reversed(results):
                if pred(r):
                    c += 1
                else:
                    break
            return c

        return {
            "win_streak": count_from_end(lambda r: r == "W"),
            "loss_streak": count_from_end(lambda r: r == "L"),
            "unbeaten_streak": count_from_end(lambda r: r in ("W", "D")),
            "nowin_streak": count_from_end(lambda r: r in ("L", "D")),
        }

    def _matches_last_14d(self, team, before_date):
        matches = self._team_history(team, before_date, venue="both")
        cutoff = pd.Timestamp(before_date) - pd.Timedelta(days=14)
        return int((matches["Date"] >= cutoff).sum())

    def _elo_momentum5(self, team, league, before_date):
        matches = self.df[
            (self.df["league"] == league)
            & ((self.df["HomeTeam"] == team) | (self.df["AwayTeam"] == team))
            & (self.df["Date"] < before_date)
        ].sort_values("Date").tail(5)
        if len(matches) < 2:
            return np.nan
        elos = [(m["home_elo"] if m["HomeTeam"] == team else m["away_elo"]) for _, m in matches.iterrows()]
        return elos[-1] - elos[0]

    # -----------------------------------------------------------------
    # Head-to-head — same last-4-meetings logic as training
    # -----------------------------------------------------------------
    def _h2h(self, home_team, away_team, before_date, n=4):
        pair_matches = self.df[
            (((self.df["HomeTeam"] == home_team) & (self.df["AwayTeam"] == away_team)) |
             ((self.df["HomeTeam"] == away_team) & (self.df["AwayTeam"] == home_team)))
            & (self.df["Date"] < before_date)
        ].sort_values("Date").tail(n)

        if len(pair_matches) == 0:
            return np.nan, np.nan, 0

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
            return np.nan, np.nan, len(pair_matches)
        return float(np.mean(gd_list)), float(np.mean(pts_list)), len(pair_matches)

    # -----------------------------------------------------------------
    # Standings: replay this league-season's results up to match_date
    # -----------------------------------------------------------------
    def _standings(self, home_team, away_team, league, season, before_date):
        season_matches = self.df[
            (self.df["league"] == league)
            & (self.df["season"] == season)
            & (self.df["Date"] < before_date)
        ].sort_values("Date")

        points, gd = {}, {}
        for _, m in season_matches.iterrows():
            if pd.isna(m["FTHG"]) or pd.isna(m["FTAG"]):
                continue
            h, a = m["HomeTeam"], m["AwayTeam"]
            match_gd = m["FTHG"] - m["FTAG"]
            h_pts = 3 if match_gd > 0 else (1 if match_gd == 0 else 0)
            a_pts = 3 if match_gd < 0 else (1 if match_gd == 0 else 0)
            points[h] = points.get(h, 0) + h_pts
            points[a] = points.get(a, 0) + a_pts
            gd[h] = gd.get(h, 0) + match_gd
            gd[a] = gd.get(a, 0) - match_gd

        standings = sorted(points.keys(), key=lambda t: (-points.get(t, 0), -gd.get(t, 0)))
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
        return result, len(season_matches)

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
        for stat in list(STAT_MAP.keys()) + list(STAT_FOR_ONLY.keys()):
            suffixes = ["for", "against"] if stat in STAT_MAP else ["for"]
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
