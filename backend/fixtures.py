"""
fixtures.py
===========
Pulls upcoming and recently-completed fixtures from football-data.org
(free tier: https://www.football-data.org/client/register).

Why football-data.org: it's the most reliable free, structured (JSON,
not scraped HTML) source covering exactly the top-5 leagues, with a
stable competition-code scheme and no legal ambiguity around scraping
a site's HTML. Free tier is rate-limited (10 requests/minute) but that's
plenty for a calendar view and weekly update job.

Set your API key as an environment variable: FOOTBALL_DATA_API_KEY
"""

import os
import requests
from datetime import date, timedelta

FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"

# football-data.org competition codes for the top 5 leagues
COMPETITION_CODES = {
    "epl": "PL",
    "spa": "PD",
    "ger": "BL1",
    "ita": "SA",
    "fra": "FL1",
}


def _headers():
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        raise RuntimeError("Set FOOTBALL_DATA_API_KEY environment variable")
    return {"X-Auth-Token": api_key}


def get_upcoming_fixtures(league: str, days_ahead: int = 14):
    """Returns a list of scheduled (not yet played) fixtures for a league."""
    code = COMPETITION_CODES[league]
    date_from = date.today().isoformat()
    date_to = (date.today() + timedelta(days=days_ahead)).isoformat()

    url = f"{FOOTBALL_DATA_BASE}/competitions/{code}/matches"
    params = {"dateFrom": date_from, "dateTo": date_to, "status": "SCHEDULED"}
    resp = requests.get(url, headers=_headers(), params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    return [
        {
            "match_id_external": m["id"],
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
            "date": m["utcDate"][:10],
            "matchday": m.get("matchday"),
        }
        for m in data.get("matches", [])
    ]


def get_recent_results(league: str, days_back: int = 8):
    """Returns recently FINISHED matches, for the weekly data-update job."""
    code = COMPETITION_CODES[league]
    date_from = (date.today() - timedelta(days=days_back)).isoformat()
    date_to = date.today().isoformat()

    url = f"{FOOTBALL_DATA_BASE}/competitions/{code}/matches"
    params = {"dateFrom": date_from, "dateTo": date_to, "status": "FINISHED"}
    resp = requests.get(url, headers=_headers(), params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for m in data.get("matches", []):
        score = m["score"]["fullTime"]
        results.append({
            "Date": m["utcDate"][:10],
            "HomeTeam": m["homeTeam"]["name"],
            "AwayTeam": m["awayTeam"]["name"],
            "FTHG": score["home"],
            "FTAG": score["away"],
            "FTR": "H" if score["home"] > score["away"] else ("A" if score["home"] < score["away"] else "D"),
            "league": league,
        })
    return results


def get_full_season_fixtures(league: str, season_year: int):
    """
    All fixtures for a given season (for the season simulator).
    season_year is the START year, e.g. 2026 for the 2026-27 season.
    """
    code = COMPETITION_CODES[league]
    url = f"{FOOTBALL_DATA_BASE}/competitions/{code}/matches"
    params = {"season": season_year}
    resp = requests.get(url, headers=_headers(), params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    fixtures = []
    for m in data.get("matches", []):
        fixtures.append({
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
            "date": m["utcDate"][:10],
            "matchday": m.get("matchday"),
            "status": m["status"],  # SCHEDULED, FINISHED, etc.
        })
    return sorted(fixtures, key=lambda f: f["date"])
