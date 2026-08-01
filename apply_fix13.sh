#!/bin/bash
set -e
cd "$(dirname "$0")"

cat > 'backend/main.py' << 'MATCHMIND_EOF'
"""
main.py — MatchMind API
========================
FastAPI backend serving:
  GET  /leagues                       -> league list + teams per league
  GET  /calendar?league=epl           -> upcoming fixtures + predictions
  POST /predict                       -> hypothetical match prediction
  POST /simulate-season                -> Monte Carlo full-season simulation
  POST /admin/update-data             -> pulls recent results, appends to DB

Run locally:  uvicorn main:app --reload
Deploy: see DEPLOYMENT.md
"""

import os
import sys
import threading
import uuid
from datetime import date

# Force stdout to flush on every line instead of full-buffering, which is
# the default when a process's output isn't connected to a real terminal
# (exactly Render's setup). Without this, print() calls can sit in a
# buffer indefinitely and never reach the visible logs -- which is very
# likely why the season-simulation diagnostic prints below appeared to
# not exist at all, even after being correctly deployed.
sys.stdout.reconfigure(line_buffering=True)

import requests
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predict import ModelA
from build_match_features import MatchFeatureBuilder
import fixtures
from season_simulator import simulate_season
from team_names import normalize_team_name

# In-memory job store for season-simulation progress. Fine for a single
# worker process (Render's free tier runs WEB_CONCURRENCY=1) -- jobs are
# lost on restart, which is acceptable since they're short-lived
# (seconds to tens of seconds) and re-runnable by the client.
SEASON_SIM_JOBS = {}

DATA_PATH = os.environ.get("MATCHMIND_DATA_PATH", "data/recent_matches.csv")
DATA_DOWNLOAD_URL = os.environ.get("MATCHMIND_DATA_URL")  # e.g. Hugging Face dataset URL
MODEL_DIR = os.environ.get("MATCHMIND_MODEL_DIR", "model_a_ensemble")


def _ensure_data_present():
    """
    The historical CSV (~256MB) is too large for a normal GitHub push
    (100MB limit). Rather than using Git LFS, this downloads it from an
    external host (e.g. Hugging Face Datasets, S3, etc.) at container
    startup if it isn't already present on disk.

    Streams the download to disk in chunks instead of loading the whole
    response into memory (resp.content), which matters on memory-limited
    hosts like Render's free tier (512MB) where buffering a 256MB file
    fully in RAM before writing it can tip you into an OOM kill before
    the app even starts serving requests.
    """
    if os.path.exists(DATA_PATH):
        return
    if not DATA_DOWNLOAD_URL:
        raise RuntimeError(
            f"{DATA_PATH} not found and MATCHMIND_DATA_URL is not set. "
            "Either commit the file via Git LFS, or set MATCHMIND_DATA_URL "
            "to a direct download link (see DEPLOYMENT.md)."
        )
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    print(f"Downloading dataset from {DATA_DOWNLOAD_URL} ...")
    with requests.get(DATA_DOWNLOAD_URL, timeout=300, stream=True) as resp:
        resp.raise_for_status()
        bytes_written = 0
        with open(DATA_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                f.write(chunk)
                bytes_written += len(chunk)
    print(f"Saved dataset to {DATA_PATH} ({bytes_written / 1e6:.1f} MB)")


_ensure_data_present()

app = FastAPI(title="MatchMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend's actual domain once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ModelA(MODEL_DIR)
builder = MatchFeatureBuilder(DATA_PATH)

LEAGUE_NAMES = {"epl": "Premier League", "spa": "LaLiga", "ger": "Bundesliga",
                "ita": "Serie A", "fra": "Ligue 1"}

_known_teams_cache = {}


def _known_teams_for_league(league: str) -> set:
    """
    The set of team names actually present in our historical dataset
    for this league -- used to normalize football-data.org's official
    long-form names (e.g. "Arsenal FC") down to what our dataset uses
    (e.g. "Arsenal"). See team_names.py for why this matters: without
    it, every /calendar and season-simulation prediction was silently
    wrong (identical generic output) instead of visibly erroring.
    """
    if league not in _known_teams_cache:
        rows = builder.df[builder.df["league"] == league]
        _known_teams_cache[league] = set(rows["HomeTeam"]) | set(rows["AwayTeam"])
    return _known_teams_cache[league]


# =========================================================
# Request/response models
# =========================================================
class PredictRequest(BaseModel):
    league: str
    home_team: str
    away_team: str
    match_date: str  # "YYYY-MM-DD"


class SeasonSimRequest(BaseModel):
    league: str
    season: str  # e.g. "26-27"


# =========================================================
# GET /leagues — for populating the cascading dropdowns
# =========================================================
@app.get("/leagues")
def get_leagues():
    out = []
    for code, name in LEAGUE_NAMES.items():
        recent = builder.df[builder.df["league"] == code]
        if recent.empty:
            teams = []
        else:
            latest_season = sorted(recent["season"].unique())[-1]
            season_rows = recent[recent["season"] == latest_season]
            teams = sorted(set(season_rows["HomeTeam"]) | set(season_rows["AwayTeam"]))
        out.append({"code": code, "name": name, "teams": teams})
    return out


# =========================================================
# GET /seasons — real, queryable seasons for a league, so the frontend
# can offer a dropdown of values that actually work instead of a
# free-text field the user has to guess (this is what was causing
# repeated "season not found" errors in Simulate a Season).
# =========================================================
@app.get("/seasons")
def get_seasons(league: str):
    if league not in fixtures.COMPETITION_CODES:
        raise HTTPException(400, f"Unknown league '{league}'")
    try:
        start_years = fixtures.get_available_seasons(league)
    except Exception as e:
        raise HTTPException(502, f"Couldn't fetch seasons from football-data.org: {e}")

    seasons = [f"{y[-2:]}-{str(int(y) + 1)[-2:]}" for y in start_years]
    return {"league": league, "seasons": seasons}


# =========================================================
# GET /calendar — upcoming real fixtures (FAST: no per-fixture
# prediction here anymore)
#
# This used to call builder.build() + model.predict() for every single
# fixture in the date range before returning anything -- for a 60-day
# window across a whole league that's dozens of full feature-building
# passes done sequentially, which is what was making /calendar slow
# (and, combined with any one slow/failing fixture, made it look like
# it "wasn't working" even when it eventually would have returned).
#
# Now /calendar only fetches the fixture list + normalizes team names
# (cheap: no model inference at all) and returns immediately. When a
# specific game is selected, call POST /predict with that fixture's
# league/home_team/away_team/date to get its probabilities -- same
# endpoint already used by "Simulate a match", just triggered per-click
# instead of eagerly for every fixture up front.
# =========================================================
@app.get("/calendar")
def get_calendar(league: str, days_ahead: int = 60):
    if league not in fixtures.COMPETITION_CODES:
        raise HTTPException(400, f"Unknown league '{league}'")

    known_teams = _known_teams_for_league(league)
    upcoming = fixtures.get_upcoming_fixtures(league, days_ahead=days_ahead)
    out = []
    for f in upcoming:
        # Same fallback as season simulation: use the normalized (matched)
        # name when we have real history for this team, otherwise fall
        # back to the raw football-data.org name -- build_match_features.py
        # already handles a never-seen team gracefully (defaults to a
        # "newly promoted" profile, Elo 1400), so there's no reason to
        # block these fixtures from being predicted at all.
        home = normalize_team_name(f["home_team"], known_teams) or f["home_team"]
        away = normalize_team_name(f["away_team"], known_teams) or f["away_team"]
        out.append({
            **f,
            "home_team": home,
            "away_team": away,
            "predictable": True,
        })
    return out


# =========================================================
# POST /predict — hypothetical match (league -> team -> date -> team)
# =========================================================
@app.post("/predict")
def predict_match(req: PredictRequest):
    if req.league not in fixtures.COMPETITION_CODES:
        raise HTTPException(400, f"Unknown league '{req.league}'")
    if req.home_team == req.away_team:
        raise HTTPException(400, "Home and away team must be different")

    # No longer requiring both teams to already be in our dataset's latest
    # season -- build_match_features.py already handles a team it's never
    # seen by falling back to a "newly promoted" profile (Elo 1400, other
    # stats treated as missing), same fallback used by /calendar and season
    # simulation. Blocking here just meant clicking a fixture involving a
    # promoted team silently failed even though a reasonable prediction
    # was possible.
    X = builder.build(req.home_team, req.away_team, req.match_date, req.league)
    probs = model.predict(X)[0]
    return {
        "home_team": req.home_team, "away_team": req.away_team,
        "date": req.match_date, "league": req.league,
        "p_away": round(float(probs[0]), 4),
        "p_draw": round(float(probs[1]), 4),
        "p_home": round(float(probs[2]), 4),
    }


# =========================================================
# POST /simulate-season — Monte Carlo full-season simulation
#
# Runs as a background job so the frontend can poll live progress
# (e.g. "1,340 / 5,000 simulations") instead of the request blocking
# silently for however long 5,000 trials take.
#   POST /simulate-season/start  -> {job_id}
#   GET  /simulate-season/status/{job_id} -> {status, completed, total, result?}
# =========================================================
def _prepare_season_fixtures(league: str, season: str, prep_progress_callback=None):
    """Shared setup: fetch fixtures, replay real standings so far, and
    compute each remaining fixture's win/draw/loss probability once."""
    print(f"[season-sim] Fetching full season fixture list for {league} {season}...")
    season_year = 2000 + int(season.split("-")[0])
    all_fixtures = fixtures.get_full_season_fixtures(league, season_year)
    print(f"[season-sim] Got {len(all_fixtures)} total fixtures from football-data.org.")

    played = [f for f in all_fixtures if f["status"] == "FINISHED"]
    remaining = [f for f in all_fixtures if f["status"] != "FINISHED"]
    print(f"[season-sim] {len(played)} played, {len(remaining)} remaining to score.")

    start_points, start_gd = {}, {}
    hist = builder.df[(builder.df["league"] == league) & (builder.df["season"] == season)]
    for _, m in hist.iterrows():
        if pd.isna(m["FTHG"]) or pd.isna(m["FTAG"]):
            continue
        h, a, hg, ag = m["HomeTeam"], m["AwayTeam"], m["FTHG"], m["FTAG"]
        gd = hg - ag
        h_pts = 3 if gd > 0 else (1 if gd == 0 else 0)
        a_pts = 3 if gd < 0 else (1 if gd == 0 else 0)
        start_points[h] = start_points.get(h, 0) + h_pts
        start_points[a] = start_points.get(a, 0) + a_pts
        start_gd[h] = start_gd.get(h, 0) + gd
        start_gd[a] = start_gd.get(a, 0) - gd
    print(f"[season-sim] Replayed {len(hist)} historical matches for current standings.")

    known_teams = _known_teams_for_league(league)
    fixtures_with_probs = []
    skipped_unknown = 0
    skipped_error = 0
    import time as _time
    t0 = _time.time()
    for i, f in enumerate(remaining):
        # Prefer the normalized (matched-to-our-dataset) name when we have
        # real history for this team. When we don't (genuinely new/promoted
        # team, name matches nothing under any spelling), fall back to the
        # raw football-data.org name rather than skipping the fixture --
        # build_match_features.py already handles a team with zero rows by
        # defaulting to a "newly promoted" profile (Elo 1400, other stats
        # treated as missing, which the model handles natively). Skipping
        # these fixtures instead was a real bug: it doesn't just drop the
        # new team's own games, it drops every ESTABLISHED opponent's
        # fixture against them too -- shrinking everyone's simulated season
        # below the real 38 games and understating their points totals.
        home = normalize_team_name(f["home_team"], known_teams) or f["home_team"]
        away = normalize_team_name(f["away_team"], known_teams) or f["away_team"]
        if f["home_team"] not in known_teams and home == f["home_team"]:
            skipped_unknown += 1  # tracked for logging only, not skipped
        try:
            X = builder.build(home, away, f["date"], league, season=season)
            probs = model.predict(X)[0]
            fixtures_with_probs.append({
                "home_team": home, "away_team": away,
                "p_away": float(probs[0]), "p_draw": float(probs[1]), "p_home": float(probs[2]),
                "_start_points": start_points, "_start_gd": start_gd,
            })
        except Exception as e:
            skipped_error += 1
            if skipped_error <= 3:  # don't flood the log if every fixture fails the same way
                print(f"[season-sim] Fixture {i} ({home} vs {away}) failed: {e}")
            continue

        if (i + 1) % 25 == 0 or (i + 1) == len(remaining):
            elapsed = _time.time() - t0
            print(f"[season-sim] Scored {i + 1}/{len(remaining)} fixtures "
                  f"({elapsed:.1f}s elapsed, {elapsed / (i + 1) * 1000:.0f}ms/fixture avg)")
            if prep_progress_callback:
                prep_progress_callback(i + 1, len(remaining))

    print(f"[season-sim] Done: {len(fixtures_with_probs)} scored, "
          f"{skipped_unknown} skipped (unknown team), {skipped_error} skipped (build error).")

    return fixtures_with_probs, len(played), len(remaining)


def _run_season_sim_job(job_id: str, league: str, season: str, n_trials: int):
    try:
        def on_prep_progress(completed, total):
            SEASON_SIM_JOBS[job_id].update({
                "phase": "preparing", "prep_completed": completed, "prep_total": total,
            })

        fixtures_with_probs, n_played, n_remaining = _prepare_season_fixtures(
            league, season, prep_progress_callback=on_prep_progress,
        )

        if not fixtures_with_probs:
            SEASON_SIM_JOBS[job_id] = {
                "status": "error", "completed": 0, "total": n_trials,
                "error": "No remaining fixtures could be simulated for this season",
            }
            return

        def on_progress(completed, total):
            SEASON_SIM_JOBS[job_id].update({"phase": "simulating", "completed": completed, "total": total})

        standings = simulate_season(fixtures_with_probs, n_trials=n_trials, progress_callback=on_progress)

        SEASON_SIM_JOBS[job_id] = {
            "status": "done", "completed": n_trials, "total": n_trials,
            "result": {
                "league": league, "season": season,
                "matches_played": n_played, "matches_remaining": n_remaining,
                "standings": standings,
            },
        }
    except Exception as e:
        SEASON_SIM_JOBS[job_id] = {"status": "error", "completed": 0, "total": n_trials, "error": str(e)}


@app.post("/simulate-season/start")
def start_season_simulation(req: SeasonSimRequest):
    job_id = str(uuid.uuid4())
    n_trials = 5000
    SEASON_SIM_JOBS[job_id] = {
        "status": "running", "phase": "preparing",
        "completed": 0, "total": n_trials,
        "prep_completed": 0, "prep_total": None,
    }

    thread = threading.Thread(
        target=_run_season_sim_job, args=(job_id, req.league, req.season, n_trials), daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "total": n_trials}


@app.get("/simulate-season/status/{job_id}")
def get_season_simulation_status(job_id: str):
    job = SEASON_SIM_JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id")
    return job


# =========================================================
# POST /admin/update-data — weekly data refresh
# =========================================================
@app.post("/admin/update-data")
def update_data(secret: str, days_back: int = 8):
    expected = os.environ.get("MATCHMIND_ADMIN_SECRET")
    if not expected or secret != expected:
        raise HTTPException(403, "Invalid admin secret")

    added = 0
    for league in fixtures.COMPETITION_CODES:
        results = fixtures.get_recent_results(league, days_back=days_back)
        for r in results:
            already_exists = (
                (builder.df["Date"] == pd.Timestamp(r["Date"])) &
                (builder.df["HomeTeam"] == r["HomeTeam"]) &
                (builder.df["AwayTeam"] == r["AwayTeam"])
            ).any()
            if not already_exists:
                added += 1

    return {
        "message": f"Found {added} new results across all leagues. "
                   f"Run update_data.py to rebuild features and persist them.",
        "new_results_found": added,
    }


@app.get("/")
def root():
    return {"status": "MatchMind API running", "date": str(date.today())}
MATCHMIND_EOF
echo 'wrote backend/main.py'

