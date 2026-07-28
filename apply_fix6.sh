#!/bin/bash
set -e
cd "$(dirname "$0")"

cat > 'backend/team_names.py' << 'MATCHMIND_EOF'
"""
team_names.py
=============
Maps football-data.org's official long-form team names (e.g. "Arsenal
FC", "Brighton & Hove Albion FC", "FC Bayern München") to the short
names used in our historical dataset (e.g. "Arsenal", "Brighton",
"Bayern Munich").

WHY THIS MATTERS -- THE BUG THIS FIXES:
Without this mapping, build_match_features.py looks up e.g. "Arsenal
FC" in a dataset that only has "Arsenal". No match is found, so the
team is silently treated as brand-new/unknown (promoted_elo default,
empty history everywhere) -- the model then produces the same generic
prediction for every fixture regardless of who's actually playing.
This is much worse than an error: it looks like it's working while
being confidently wrong for every /calendar and season-simulation
prediction.

Two layers of defense:
1. Explicit OVERRIDES for names that don't reduce to the dataset name
   via simple suffix-stripping (accents, translated club nicknames,
   "1. FC" prefixes, "&" in club names, etc).
2. A generic normalizer (strip common club-name suffixes/prefixes) as
   a fallback for anything not explicitly listed.
3. normalize_team_name() returns None if nothing matches -- callers
   MUST treat None as "unknown team, don't silently predict" rather
   than passing it through, or this defeats the whole point.
"""
import re

# Explicit overrides: football-data.org name -> our dataset's name.
# Only needed where suffix-stripping alone wouldn't produce a match
# (different language/spelling, ampersands, historical club prefixes).
OVERRIDES = {
    # Premier League
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion FC": "Brighton",
    "Leeds United FC": "Leeds",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",

    # LaLiga
    "Deportivo Alavés": "Alaves",
    "Atlético de Madrid": "Atletico Madrid",
    "FC Barcelona": "Barcelona",
    "RC Celta de Vigo": "Celta Vigo",
    "RCD Espanyol de Barcelona": "Espanyol",
    "Girona FC": "Girona",
    "Levante UD": "Levante",
    "RCD Mallorca": "Mallorca",
    "CA Osasuna": "Osasuna",
    "Rayo Vallecano de Madrid": "Rayo Vallecano",
    "Real Betis Balompié": "Real Betis",
    "Real Sociedad de Fútbol": "Real Sociedad",

    # Bundesliga
    "FC Bayern München": "Bayern Munich",
    "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "Borussia Mönchengladbach": "Borussia M.Gladbach",
    "1. FC Köln": "FC Cologne",
    "1. FC Heidenheim 1846": "FC Heidenheim",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "1. FSV Mainz 05": "Mainz 05",
    "RB Leipzig": "RasenBallsport Leipzig",
    "FC St. Pauli 1910": "St. Pauli",
    "1. FC Union Berlin": "Union Berlin",
    "SV Werder Bremen": "Werder Bremen",
    "VfL Wolfsburg": "Wolfsburg",

    # Serie A
    "Atalanta BC": "Atalanta",
    "Bologna FC 1909": "Bologna",
    "Cagliari Calcio": "Cagliari",
    "Como 1907": "Como",
    "US Cremonese": "Cremonese",
    "ACF Fiorentina": "Fiorentina",
    "Genoa CFC": "Genoa",
    "FC Internazionale Milano": "Inter",
    "SS Lazio": "Lazio",
    "US Lecce": "Lecce",
    "SSC Napoli": "Napoli",
    "Pisa SC": "Pisa",
    "AS Roma": "Roma",
    "US Sassuolo Calcio": "Sassuolo",
    "Torino FC": "Torino",
    "Udinese Calcio": "Udinese",
    "Hellas Verona FC": "Verona",

    # Ligue 1
    "Angers SCO": "Angers",
    "AJ Auxerre": "Auxerre",
    "Stade Brestois 29": "Brest",
    "Le Havre AC": "Le Havre",
    "RC Lens": "Lens",
    "LOSC Lille": "Lille",
    "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille",
    "AS Monaco FC": "Monaco",
    "OGC Nice": "Nice",
    "Paris Saint-Germain FC": "Paris Saint Germain",
    "Stade Rennais FC 1901": "Rennes",
    "RC Strasbourg Alsace": "Strasbourg",
    "FC Toulouse": "Toulouse",
}

# Generic fallback: strip common club-name suffixes/prefixes that
# football-data.org appends/prepends but our dataset doesn't use.
_STRIP_PATTERNS = [
    r"^\d+\.\s*",       # "1. FC ..." leading numeral
    r"\s+FC$", r"^FC\s+", r"\s+CFC$", r"\s+AFC$", r"^AFC\s+",
    r"\s+CF$", r"^CF\s+", r"\s+SC$", r"^SC\s+",
    r"\s+Calcio$", r"^Calcio\s+",
]


def _strip_generic(name: str) -> str:
    cleaned = name
    for pattern in _STRIP_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    return cleaned.strip()


def normalize_team_name(fd_name: str, dataset_teams: set) -> str | None:
    """
    Maps a football-data.org team name to the matching name in
    `dataset_teams` (the set of team names actually present in our
    historical CSV for this league). Returns None if no match is
    found by any method -- callers MUST NOT silently proceed with an
    unmapped name, since that's exactly the bug this function exists
    to prevent (see module docstring).
    """
    if fd_name in dataset_teams:
        return fd_name
    if fd_name in OVERRIDES and OVERRIDES[fd_name] in dataset_teams:
        return OVERRIDES[fd_name]
    stripped = _strip_generic(fd_name)
    if stripped in dataset_teams:
        return stripped
    return None
MATCHMIND_EOF
echo 'wrote backend/team_names.py'

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
import threading
import uuid
from datetime import date

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
# GET /calendar — upcoming real fixtures + model predictions
# =========================================================
@app.get("/calendar")
def get_calendar(league: str, days_ahead: int = 60):
    if league not in fixtures.COMPETITION_CODES:
        raise HTTPException(400, f"Unknown league '{league}'")

    known_teams = _known_teams_for_league(league)
    upcoming = fixtures.get_upcoming_fixtures(league, days_ahead=days_ahead)
    out = []
    for f in upcoming:
        home = normalize_team_name(f["home_team"], known_teams)
        away = normalize_team_name(f["away_team"], known_teams)
        if home is None or away is None:
            unknown = f["home_team"] if home is None else f["away_team"]
            out.append({**f, "error": f"No historical data for '{unknown}' -- can't predict this fixture"})
            continue
        try:
            X = builder.build(home, away, f["date"], league)
            probs = model.predict(X)[0]
            out.append({
                **f,
                "p_away": round(float(probs[0]), 4),
                "p_draw": round(float(probs[1]), 4),
                "p_home": round(float(probs[2]), 4),
            })
        except Exception as e:
            out.append({**f, "error": str(e)})
    return out


# =========================================================
# POST /predict — hypothetical match (league -> team -> date -> team)
# =========================================================
@app.post("/predict")
def predict_match(req: PredictRequest):
    league_teams = get_leagues()
    league_entry = next((l for l in league_teams if l["code"] == req.league), None)
    if league_entry is None:
        raise HTTPException(400, f"Unknown league '{req.league}'")
    if req.home_team not in league_entry["teams"] or req.away_team not in league_entry["teams"]:
        raise HTTPException(400, "Both teams must belong to the selected league")
    if req.home_team == req.away_team:
        raise HTTPException(400, "Home and away team must be different")

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
def _prepare_season_fixtures(league: str, season: str):
    """Shared setup: fetch fixtures, replay real standings so far, and
    compute each remaining fixture's win/draw/loss probability once."""
    season_year = 2000 + int(season.split("-")[0])
    all_fixtures = fixtures.get_full_season_fixtures(league, season_year)

    played = [f for f in all_fixtures if f["status"] == "FINISHED"]
    remaining = [f for f in all_fixtures if f["status"] != "FINISHED"]

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

    known_teams = _known_teams_for_league(league)
    fixtures_with_probs = []
    for f in remaining:
        home = normalize_team_name(f["home_team"], known_teams)
        away = normalize_team_name(f["away_team"], known_teams)
        if home is None or away is None:
            continue  # no historical data for this team -- can't simulate it
        try:
            X = builder.build(home, away, f["date"], league, season=season)
            probs = model.predict(X)[0]
            fixtures_with_probs.append({
                "home_team": home, "away_team": away,
                "p_away": float(probs[0]), "p_draw": float(probs[1]), "p_home": float(probs[2]),
                "_start_points": start_points, "_start_gd": start_gd,
            })
        except Exception:
            continue

    return fixtures_with_probs, len(played), len(remaining)


def _run_season_sim_job(job_id: str, league: str, season: str, n_trials: int):
    try:
        fixtures_with_probs, n_played, n_remaining = _prepare_season_fixtures(league, season)

        if not fixtures_with_probs:
            SEASON_SIM_JOBS[job_id] = {
                "status": "error", "completed": 0, "total": n_trials,
                "error": "No remaining fixtures could be simulated for this season",
            }
            return

        def on_progress(completed, total):
            SEASON_SIM_JOBS[job_id].update({"completed": completed, "total": total})

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
    SEASON_SIM_JOBS[job_id] = {"status": "running", "completed": 0, "total": n_trials}

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

