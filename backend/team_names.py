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
