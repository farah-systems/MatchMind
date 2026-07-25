"""
fixtures.py
===========
Pulls upcoming and recently-completed fixtures from football-data.org
(free tier: https://www.football-data.org/client/register).

Why football-data.org: it's the most reliable free, structured (JSON,
not scraped HTML) source covering exactly the top-5 leagues, with a
stable competition-code scheme and no legal ambiguity around scraping
a site's HTML.

FREE TIER CONSTRAINTS (both matter a lot for this file):
  - 10 requests/minute rate limit.
  - Date-range filters (dateFrom/dateTo) on the matches endpoint
    cannot span more than 10 days -- a wider range is rejected, which
    is why get_upcoming_fixtures used to silently show nothing beyond
    ~14 days out even after asking for 60. Fixed below by splitting a
    wide window into consecutive <=10-day chunks and combining results
    (6 calls for a 60-day window, well under the 10/min cap).

Set your API key as an environment variable: FOOTBALL_DATA_API_KEY
"""

import os
import time
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

MAX_DATE_RANGE_DAYS = 10  # football-data.org free-tier hard limit


def _headers():
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        raise RuntimeError("Set FOOTBALL_DATA_API_KEY environment variable")
    return {"X-Auth-Token": api_key}


def _get(url, params):
    """
    GET with football-data.org's actual error message surfaced, instead
    of requests' generic "400 Client Error" (which hides what's actually
    wrong -- e.g. an invalid/unavailable season, or a too-wide date range).
    """
    resp = requests.get(url, headers=_headers(), params=params, timeout=15)
    if not resp.ok:
        try:
            detail = resp.json().get("message", resp.text)
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"football-data.org error ({resp.status_code}): {detail}")
    return resp.json()


def _date_chunks(date_from: date, date_to: date, max_days: int = MAX_DATE_RANGE_DAYS):
    """Split [date_from, date_to] into consecutive chunks, each <= max_days wide."""
    chunks = []
    cursor = date_from
    while cursor <= date_to:
        chunk_end = min(cursor + timedelta(days=max_days - 1), date_to)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def get_upcoming_fixtures(league: str, days_ahead: int = 60):
    """
    Returns scheduled (not yet played) fixtures for a league, up to
    `days_ahead` days out -- transparently chunked into multiple
    <=10-day requests since that's the free tier's hard per-request
    limit, not something a single wider request can bypass.
    """
    code = COMPETITION_CODES[league]
    url = f"{FOOTBALL_DATA_BASE}/competitions/{code}/matches"

    date_from = date.today()
    date_to = date.today() + timedelta(days=days_ahead)

    all_fixtures = []
    for i, (chunk_from, chunk_to) in enumerate(_date_chunks(date_from, date_to)):
        params = {
            "dateFrom": chunk_from.isoformat(),
            "dateTo": chunk_to.isoformat(),
            "status": "SCHEDULED",
        }
        data = _get(url, params)
        all_fixtures.extend(
            {
                "match_id_external": m["id"],
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "date": m["utcDate"][:10],
                "matchday": m.get("matchday"),
            }
            for m in data.get("matches", [])
        )
        # Stay comfortably under the 10 req/min free-tier limit when
        # chunking pushes us to several calls in a row.
        if i > 0 and (i + 1) % 8 == 0:
            time.sleep(60)

    return sorted(all_fixtures, key=lambda f: f["date"])


def get_recent_results(league: str, days_back: int = 8):
    """Returns recently FINISHED matches, for the weekly data-update job."""
    code = COMPETITION_CODES[league]
    date_from = (date.today() - timedelta(days=days_back)).isoformat()
    date_to = date.today().isoformat()

    url = f"{FOOTBALL_DATA_BASE}/competitions/{code}/matches"
    params = {"dateFrom": date_from, "dateTo": date_to, "status": "FINISHED"}
    data = _get(url, params)

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


def get_available_seasons(league: str):
    """
    Returns the list of season start-years football-data.org actually
    has data for, in this competition -- used to validate a requested
    season before querying matches, so a bad/unavailable season gives
    a clear message instead of a cryptic API error.
    """
    code = COMPETITION_CODES[league]
    url = f"{FOOTBALL_DATA_BASE}/competitions/{code}"
    data = _get(url, {})
    return sorted(s["startDate"][:4] for s in data.get("seasons", []))


def get_full_season_fixtures(league: str, season_year: int):
    """
    All fixtures for a given season (for the season simulator).
    season_year is the START year, e.g. 2026 for the 2026-27 season.
    """
    code = COMPETITION_CODES[league]
    url = f"{FOOTBALL_DATA_BASE}/competitions/{code}/matches"
    params = {"season": season_year}
    try:
        data = _get(url, params)
    except RuntimeError as e:
        # Give a specific, actionable message instead of the raw API error
        available = get_available_seasons(league)
        raise RuntimeError(
            f"Season {season_year}-{str(season_year + 1)[-2:]} isn't available for "
            f"{league} on football-data.org's free tier. Seasons with data: "
            f"{', '.join(available)}. (Original error: {e})"
        )

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
