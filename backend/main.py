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
Deploy: see README_DEPLOYMENT.md
"""

import os
from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predict import ModelA
from build_match_features import MatchFeatureBuilder
import fixtures
from season_simulator import simulate_season

DATA_PATH = os.environ.get("MATCHMIND_DATA_PATH", "data/top5_leagues_features_full.csv")
MODEL_DIR = os.environ.get("MATCHMIND_MODEL_DIR", "model_a_ensemble")

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
def get_calendar(league: str, days_ahead: int = 14):
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
            # Team name mismatch between football-data.org and historical
            # naming is the most likely cause — surface it rather than
            # silently dropping the fixture
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
# =========================================================
@app.post("/simulate-season")
def simulate_season_endpoint(req: SeasonSimRequest):
    season_year = 2000 + int(req.season.split("-")[0])
    all_fixtures = fixtures.get_full_season_fixtures(req.league, season_year)

    played = [f for f in all_fixtures if f["status"] == "FINISHED"]
    remaining = [f for f in all_fixtures if f["status"] != "FINISHED"]

    # Real standings so far, from already-played matches this season
    start_points, start_gd = {}, {}
    hist = builder.df[(builder.df["league"] == req.league) & (builder.df["season"] == req.season)]
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
            X = builder.build(f["home_team"], f["away_team"], f["date"], req.league, season=req.season)
            probs = model.predict(X)[0]
            fixtures_with_probs.append({
                "home_team": f["home_team"], "away_team": f["away_team"],
                "p_away": float(probs[0]), "p_draw": float(probs[1]), "p_home": float(probs[2]),
                "_start_points": start_points, "_start_gd": start_gd,
            })
        except Exception:
            continue  # skip fixtures we can't build features for (e.g. brand-new promoted team)

    if not fixtures_with_probs:
        raise HTTPException(400, "No remaining fixtures could be simulated for this season")

    standings = simulate_season(fixtures_with_probs, n_trials=5000)
    return {
        "league": req.league, "season": req.season,
        "matches_played": len(played), "matches_remaining": len(remaining),
        "standings": standings,
    }


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
                # NOTE: appending here only adds the raw result row.
                # Elo/rolling/standings features need the full pipeline
                # re-run (see update_data.py) for this new row and every
                # subsequent match to reflect it — that's a heavier,
                # offline job, not done inline in this request.

    return {
        "message": f"Found {added} new results across all leagues. "
                   f"Run update_data.py to rebuild features and persist them.",
        "new_results_found": added,
    }


@app.get("/")
def root():
    return {"status": "MatchMind API running", "date": str(date.today())}
