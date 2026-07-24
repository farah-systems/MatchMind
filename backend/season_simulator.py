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
