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

LEAGUE_NAMES = {"epl": "Premier League", "spa": "La Liga", "ger": "Bundesliga",
                "ita": "Serie A", "fra": "Ligue 1"}


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
# GET /calendar — upcoming real fixtures + model predictions
# =========================================================
@app.get("/calendar")
def get_calendar(league: str, days_ahead: int = 60):
    if league not in fixtures.COMPETITION_CODES:
        raise HTTPException(400, f"Unknown league '{league}'")

    upcoming = fixtures.get_upcoming_fixtures(league, days_ahead=days_ahead)
    out = []
    for f in upcoming:
        try:
            X = builder.build(f["home_team"], f["away_team"], f["date"], league)
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

    fixtures_with_probs = []
    for f in remaining:
        try:
            X = builder.build(f["home_team"], f["away_team"], f["date"], league, season=season)
            probs = model.predict(X)[0]
            fixtures_with_probs.append({
                "home_team": f["home_team"], "away_team": f["away_team"],
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
