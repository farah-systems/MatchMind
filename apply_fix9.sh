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
        home = normalize_team_name(f["home_team"], known_teams)
        away = normalize_team_name(f["away_team"], known_teams)
        out.append({
            **f,
            "home_team": home or f["home_team"],
            "away_team": away or f["away_team"],
            # False means: don't bother calling /predict for this one,
            # we have no historical data for at least one of these teams
            # (e.g. a newly-promoted club) -- surface this in the UI
            # instead of letting a click silently fail.
            "predictable": home is not None and away is not None,
        })
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
        home = normalize_team_name(f["home_team"], known_teams)
        away = normalize_team_name(f["away_team"], known_teams)
        if home is None or away is None:
            skipped_unknown += 1
            continue  # no historical data for this team -- can't simulate it
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

cat > 'frontend/src/components/CalendarView.jsx' << 'MATCHMIND_EOF'
import { useEffect, useState } from "react";
import { CalendarDays, AlertCircle, Loader2, ChevronDown } from "lucide-react";
import { api } from "../api";
import ProbabilityBar from "./ProbabilityBar";

const LEAGUES = [
  { code: "epl", name: "Premier League" },
  { code: "spa", name: "LaLiga" },
  { code: "ger", name: "Bundesliga" },
  { code: "ita", name: "Serie A" },
  { code: "fra", name: "Ligue 1" },
];

const DAYS_AHEAD = 60;

function formatDateHeading(dateStr) {
  const d = new Date(dateStr);
  return d.toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" });
}

export default function CalendarView() {
  const [league, setLeague] = useState("epl");
  const [fixtures, setFixtures] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Per-fixture prediction state, keyed by match_id_external. Each entry:
  // { status: "loading" | "done" | "error", data?: {...}, error?: string }
  // Predictions are fetched lazily -- only when a fixture is clicked --
  // since /calendar itself no longer computes them eagerly for speed.
  const [predictions, setPredictions] = useState({});
  const [expanded, setExpanded] = useState(new Set());

  useEffect(() => {
    setLoading(true);
    setError(null);
    setPredictions({});
    setExpanded(new Set());
    api
      .getCalendar(league, DAYS_AHEAD)
      .then(setFixtures)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [league]);

  const handleSelectFixture = (f) => {
    const key = f.match_id_external;
    const isExpanded = expanded.has(key);
    const next = new Set(expanded);
    if (isExpanded) {
      next.delete(key);
      setExpanded(next);
      return;
    }
    next.add(key);
    setExpanded(next);

    // Already fetched (or currently fetching) -- don't re-request
    if (predictions[key]) return;

    setPredictions((prev) => ({ ...prev, [key]: { status: "loading" } }));
    api
      .predictMatch({
        league,
        home_team: f.home_team,
        away_team: f.away_team,
        match_date: f.date,
      })
      .then((data) => {
        setPredictions((prev) => ({ ...prev, [key]: { status: "done", data } }));
      })
      .catch((e) => {
        setPredictions((prev) => ({ ...prev, [key]: { status: "error", error: e.message } }));
      });
  };

  const groups = fixtures.reduce((acc, f) => {
    (acc[f.date] = acc[f.date] || []).push(f);
    return acc;
  }, {});
  const orderedDates = Object.keys(groups).sort();

  return (
    <div>
      {/* Hero */}
      <div className="relative pitch-grid pt-16 pb-10 px-6 overflow-hidden">
        <div className="max-w-3xl mx-auto relative">
          <div className="flex items-center gap-2 mb-3">
            <div className="pulse-ring w-1.5 h-1.5 rounded-full bg-floodlight" />
            <span className="text-xs uppercase tracking-widest text-ink-dim font-mono">
              Live model · top 5 leagues
            </span>
          </div>
          <h1 className="font-display font-700 text-4xl sm:text-5xl tracking-tight mb-3">
            What the model sees<br className="hidden sm:block" /> for the next two months
          </h1>
          <p className="text-ink-dim text-sm max-w-lg">
            Every scheduled fixture for the next 2 months. Tap a game to simulate it.
          </p>
        </div>
      </div>

      <div className="max-w-3xl mx-auto px-6 pb-16">
        <div className="flex gap-2 mb-8 flex-wrap">
          {LEAGUES.map((l) => (
            <button
              key={l.code}
              onClick={() => setLeague(l.code)}
              className={`px-3 py-1.5 text-sm rounded-full border transition-all duration-200 ${
                league === l.code
                  ? "border-floodlight text-floodlight bg-floodlight/10 shadow-glow-amber"
                  : "border-night-700 text-ink-dim hover:text-ink hover:border-ink-dim"
              }`}
            >
              {l.name}
            </button>
          ))}
        </div>

        {loading && (
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-24 bg-night-900 border border-night-700 rounded-md animate-pulse"
                style={{ animationDelay: `${i * 100}ms` }}
              />
            ))}
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 text-sm text-red-400 bg-red-400/5 border border-red-400/20 rounded-md px-4 py-3">
            <AlertCircle size={16} />
            Couldn't load fixtures: {error}
          </div>
        )}

        {!loading && !error && (
          <div className="space-y-6">
            {orderedDates.map((dateStr, gi) => (
              <div key={dateStr} className="stagger-item" style={{ animationDelay: `${gi * 60}ms` }}>
                <p className="text-xs uppercase tracking-wide text-ink-dim mb-2 font-mono">
                  {formatDateHeading(dateStr)}
                </p>
                <div className="space-y-3">
                  {groups[dateStr].map((f, i) => {
                    const key = f.match_id_external;
                    const isExpanded = expanded.has(key);
                    const pred = predictions[key];

                    return (
                      <div
                        key={i}
                        className={`bg-night-900 border rounded-md transition-all duration-200 ${
                          isExpanded ? "border-pulse/50 shadow-glow" : "border-night-700 hover:border-pulse/30"
                        } ${f.predictable === false ? "opacity-50" : ""}`}
                      >
                        <button
                          onClick={() => f.predictable !== false && handleSelectFixture(f)}
                          disabled={f.predictable === false}
                          className="w-full flex items-center justify-between p-4 text-left disabled:cursor-not-allowed"
                        >
                          <span className="font-medium text-sm">
                            {f.home_team} <span className="text-ink-dim">vs</span> {f.away_team}
                          </span>
                          {f.predictable === false ? (
                            <span className="text-xs text-ink-dim">No data</span>
                          ) : (
                            <ChevronDown
                              size={16}
                              className={`text-ink-dim transition-transform duration-200 ${
                                isExpanded ? "rotate-180" : ""
                              }`}
                            />
                          )}
                        </button>

                        {isExpanded && (
                          <div className="px-4 pb-4">
                            {(!pred || pred.status === "loading") && (
                              <div className="pt-1">
                                <div className="h-2 bg-night-800 rounded-full overflow-hidden mb-2 relative">
                                  <div className="absolute inset-0 w-1/3 bg-gradient-to-r from-pulse to-floodlight rounded-full animate-[loadingSlide_1.2s_ease-in-out_infinite]" />
                                </div>
                                <p className="text-xs font-mono text-ink-dim flex items-center gap-1.5">
                                  <Loader2 size={12} className="animate-spin" />
                                  Simulating this match…
                                </p>
                              </div>
                            )}
                            {pred?.status === "error" && (
                              <p className="text-xs text-red-400">Couldn't simulate: {pred.error}</p>
                            )}
                            {pred?.status === "done" && (
                              <ProbabilityBar
                                pAway={pred.data.p_away}
                                pDraw={pred.data.p_draw}
                                pHome={pred.data.p_home}
                                homeTeam={f.home_team}
                                awayTeam={f.away_team}
                              />
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
            {fixtures.length === 0 && (
              <p className="text-ink-dim text-sm">No fixtures found in the next {DAYS_AHEAD} days.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
MATCHMIND_EOF
echo 'wrote frontend/src/components/CalendarView.jsx'

