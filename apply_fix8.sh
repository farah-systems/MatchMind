#!/bin/bash
set -e
cd "$(dirname "$0")"

cat > 'backend/fixtures.py' << 'MATCHMIND_EOF'
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
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
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

# Short-lived in-memory cache for get_upcoming_fixtures: avoids re-doing
# 6-7 chunked external API calls on every page load/re-render, and
# reduces the chance of the whole request chain being slow enough to
# trip a client or proxy timeout (which shows up client-side as a
# generic "Failed to fetch", not a clean HTTP error).
_CACHE = {}
_CACHE_TTL_SECONDS = 300
_cache_lock = threading.Lock()


def _cache_get(key, ttl_seconds=_CACHE_TTL_SECONDS):
    with _cache_lock:
        entry = _CACHE.get(key)
        if entry and (time.time() - entry[0]) < ttl_seconds:
            return entry[1]
        return None


def _cache_set(key, value):
    with _cache_lock:
        _CACHE[key] = (time.time(), value)


def _headers():
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        raise RuntimeError("Set FOOTBALL_DATA_API_KEY environment variable")
    return {"X-Auth-Token": api_key}


def _get(url, params, max_retries=3):
    """
    GET with football-data.org's actual error message surfaced, instead
    of requests' generic "400 Client Error" (which hides what's actually
    wrong -- e.g. an invalid/unavailable season, or a too-wide date range).

    Also automatically retries on 429 (rate limit) -- football-data.org's
    free tier is 10 requests/minute, and with multiple endpoints
    (/calendar, /seasons, season simulation) potentially firing close
    together, especially right after a page load hits several of them,
    it's easy to briefly exceed that. football-data.org tells us exactly
    how long to wait via the Retry-After header, so we just wait that
    long and retry instead of surfacing a raw 429 to the user.
    """
    for attempt in range(max_retries + 1):
        resp = requests.get(url, headers=_headers(), params=params, timeout=15)
        if resp.status_code == 429 and attempt < max_retries:
            wait_seconds = int(resp.headers.get("Retry-After", 15))
            time.sleep(wait_seconds)
            continue
        if not resp.ok:
            try:
                detail = resp.json().get("message", resp.text)
            except ValueError:
                detail = resp.text
            raise RuntimeError(f"football-data.org error ({resp.status_code}): {detail}")
        return resp.json()

    raise RuntimeError("football-data.org error (429): still rate-limited after retries")
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

    Chunks are fetched CONCURRENTLY (not one-by-one) to keep total
    wall-clock time low -- 7 sequential requests at ~1-2s each can add
    up to 10-15s, which risks tripping a client/proxy timeout upstream;
    running them in parallel brings it down to roughly one request's
    worth of latency. Results are also cached for 5 minutes so repeat
    page loads don't re-pay this cost at all.
    """
    cache_key = (league, days_ahead)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    code = COMPETITION_CODES[league]
    url = f"{FOOTBALL_DATA_BASE}/competitions/{code}/matches"

    date_from = date.today()
    date_to = date.today() + timedelta(days=days_ahead)
    chunks = _date_chunks(date_from, date_to)

    def fetch_chunk(chunk):
        chunk_from, chunk_to = chunk
        # NOTE: deliberately no status filter here. football-data.org marks
        # most confirmed upcoming fixtures as "TIMED" (once kickoff time is
        # set), not "SCHEDULED" -- filtering server-side to status=SCHEDULED
        # was silently excluding almost every real upcoming match, which is
        # why /calendar was returning an empty list instead of erroring.
        # We fetch everything in the date range and filter out only the
        # statuses that mean "not actually upcoming" ourselves, below.
        params = {
            "dateFrom": chunk_from.isoformat(),
            "dateTo": chunk_to.isoformat(),
        }
        data = _get(url, params)
        NOT_UPCOMING = {"FINISHED", "POSTPONED", "CANCELLED", "SUSPENDED", "AWARDED"}
        return [
            {
                "match_id_external": m["id"],
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "date": m["utcDate"][:10],
                "matchday": m.get("matchday"),
            }
            for m in data.get("matches", [])
            if m.get("status") not in NOT_UPCOMING
        ]

    all_fixtures = []
    # len(chunks) for a 60-day window is ~7, safely under the 10 req/min
    # free-tier limit even fired all at once.
    with ThreadPoolExecutor(max_workers=min(len(chunks), 10)) as pool:
        for result in pool.map(fetch_chunk, chunks):
            all_fixtures.extend(result)

    all_fixtures = sorted(all_fixtures, key=lambda f: f["date"])
    _cache_set(cache_key, all_fixtures)
    return all_fixtures


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

    football-data.org's competition metadata includes the ENTIRE
    historical archive (some leagues going back to the 1880s) and can
    include duplicate entries for the same start year. That's useless
    for this app (predictions only make sense for seasons covered by
    our own feature dataset), so this dedupes and keeps only recent
    years -- from 10 years ago through next year, generously covering
    "current season" and "just-announced next season" cases.

    Cached for an hour -- this was the one fixture-fetching function
    with no caching at all, so every dropdown load hit football-data.org
    fresh. Combined with /calendar's own several chunked requests, that
    made it easy to briefly exceed the free tier's 10 requests/minute
    limit (this is what a 429 error means).
    """
    cache_key = ("seasons", league)
    cached = _cache_get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    code = COMPETITION_CODES[league]
    url = f"{FOOTBALL_DATA_BASE}/competitions/{code}"
    data = _get(url, {})

    cutoff_year = date.today().year - 10
    years = {s["startDate"][:4] for s in data.get("seasons", [])}
    recent_years = sorted(y for y in years if int(y) >= cutoff_year)
    _cache_set(cache_key, recent_years)
    return recent_years


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
MATCHMIND_EOF
echo 'wrote backend/fixtures.py'

