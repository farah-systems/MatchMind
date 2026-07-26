#!/bin/bash
set -e
cd "$(dirname "$0")"

cat > 'frontend/src/components/Nav.jsx' << 'MATCHMIND_EOF'
import { CalendarDays, Swords, Trophy, Info } from "lucide-react";

export default function Nav({ view, setView }) {
  const tabs = [
    { id: "calendar", label: "Calendar", icon: CalendarDays },
    { id: "simulate", label: "Simulate a match", icon: Swords },
    { id: "season", label: "Simulate a season", icon: Trophy },
    { id: "about", label: "About", icon: Info },
  ];

  return (
    <nav className="border-b border-night-700 sticky top-0 bg-night-950/85 backdrop-blur-md z-20">
      <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="pulse-ring w-2.5 h-2.5 rounded-full bg-floodlight" />
          <div className="font-display font-600 text-xl tracking-tight">
            Match<span className="text-floodlight">Mind</span>
          </div>
        </div>
        <div className="flex gap-1">
          {tabs.map((t) => {
            const Icon = t.icon;
            const active = view === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setView(t.id)}
                className={`relative flex items-center gap-1.5 px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
                  active
                    ? "bg-night-700 text-ink shadow-glow"
                    : "text-ink-dim hover:text-ink hover:bg-night-800"
                }`}
              >
                <Icon size={15} className={active ? "text-pulse-bright" : ""} />
                <span className="hidden sm:inline">{t.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
MATCHMIND_EOF
echo 'wrote frontend/src/components/Nav.jsx'

cat > 'frontend/src/components/ProbabilityBar.jsx' << 'MATCHMIND_EOF'
import { useEffect, useState } from "react";

// Animates a number counting up from 0 on mount/change — gives the
// prediction a sense of the model "arriving" at its answer.
function useCountUp(target, duration = 700) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    let raf;
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(Math.round(target * eased));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);
  return value;
}

export default function ProbabilityBar({ pAway, pDraw, pHome, homeTeam, awayTeam }) {
  const away = Math.round(pAway * 100);
  const draw = Math.round(pDraw * 100);
  const home = Math.round(pHome * 100);

  const awayAnim = useCountUp(away);
  const drawAnim = useCountUp(draw);
  const homeAnim = useCountUp(home);

  const favorite = home >= away && home >= draw ? "home" : away >= draw ? "away" : "draw";

  return (
    <div className="w-full">
      <div className="flex justify-between text-xs font-mono text-ink-dim mb-1.5 uppercase tracking-wide">
        <span className={favorite === "home" ? "text-floodlight" : ""}>{homeTeam}</span>
        <span className={favorite === "draw" ? "text-ink" : ""}>Draw</span>
        <span className={favorite === "away" ? "text-pulse-bright" : ""}>{awayTeam}</span>
      </div>
      <div className="flex h-9 rounded-md overflow-hidden border border-night-700">
        <div
          className="relative bg-floodlight flex items-center justify-center text-xs font-mono text-night-950 font-semibold transition-[width] duration-700 ease-out overflow-hidden"
          style={{ width: `${home}%` }}
        >
          {favorite === "home" && <div className="absolute inset-0 shimmer-bar" />}
          <span className="relative z-10">{home >= 10 && `${homeAnim}%`}</span>
        </div>
        <div
          className="bg-night-700 flex items-center justify-center text-xs font-mono text-ink-dim transition-[width] duration-700 ease-out"
          style={{ width: `${draw}%` }}
        >
          {draw >= 10 && `${drawAnim}%`}
        </div>
        <div
          className="relative bg-pulse-dim flex items-center justify-center text-xs font-mono text-ink transition-[width] duration-700 ease-out overflow-hidden"
          style={{ width: `${away}%` }}
        >
          {favorite === "away" && <div className="absolute inset-0 shimmer-bar" />}
          <span className="relative z-10">{away >= 10 && `${awayAnim}%`}</span>
        </div>
      </div>
    </div>
  );
}
MATCHMIND_EOF
echo 'wrote frontend/src/components/ProbabilityBar.jsx'

cat > 'frontend/src/components/CalendarView.jsx' << 'MATCHMIND_EOF'
import { useEffect, useState } from "react";
import { CalendarDays, AlertCircle } from "lucide-react";
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

  useEffect(() => {
    setLoading(true);
    setError(null);
    api
      .getCalendar(league, DAYS_AHEAD)
      .then(setFixtures)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [league]);

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
            Every scheduled fixture, scored the moment it's requested — win, draw,
            and loss probabilities from Model A's ensemble.
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
                  {groups[dateStr].map((f, i) => (
                    <div
                      key={i}
                      className="group bg-night-900 border border-night-700 rounded-md p-4 hover:border-pulse/40 hover:shadow-glow transition-all duration-200"
                    >
                      <div className="flex justify-between items-baseline mb-3">
                        <span className="font-medium text-sm">
                          {f.home_team} <span className="text-ink-dim">vs</span> {f.away_team}
                        </span>
                      </div>
                      {f.error ? (
                        <p className="text-xs text-red-400">Prediction unavailable: {f.error}</p>
                      ) : (
                        <ProbabilityBar
                          pAway={f.p_away}
                          pDraw={f.p_draw}
                          pHome={f.p_home}
                          homeTeam={f.home_team}
                          awayTeam={f.away_team}
                        />
                      )}
                    </div>
                  ))}
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

cat > 'frontend/src/components/SimulateMatch.jsx' << 'MATCHMIND_EOF'
import { useEffect, useState } from "react";
import { Swords, CalendarIcon, ShieldCheck, AlertCircle } from "lucide-react";
import { api } from "../api";
import ProbabilityBar from "./ProbabilityBar";

const LEAGUE_NAMES = {
  epl: "Premier League",
  spa: "LaLiga",
  ger: "Bundesliga",
  ita: "Serie A",
  fra: "Ligue 1",
};

function initials(name) {
  return name
    .split(" ")
    .map((w) => w[0])
    .join("")
    .slice(0, 3)
    .toUpperCase();
}

function StepLabel({ n, text }) {
  return (
    <div className="flex items-center gap-2 mb-2">
      <span className="w-5 h-5 flex items-center justify-center rounded-full bg-night-700 text-[11px] font-mono text-ink-dim">
        {n}
      </span>
      <p className="text-xs uppercase tracking-wide text-ink-dim">{text}</p>
    </div>
  );
}

function ChoiceGrid({ options, value, onChange, columns = 3 }) {
  return (
    <div className={`grid grid-cols-${columns} gap-2`}>
      {options.map((opt) => (
        <button
          key={opt}
          onClick={() => onChange(opt)}
          className={`px-3 py-2.5 text-sm rounded-md border text-left transition-all duration-150 ${
            value === opt
              ? "border-pulse text-pulse-bright bg-pulse/10 shadow-glow"
              : "border-night-700 text-ink-dim hover:text-ink hover:border-ink-dim"
          }`}
        >
          {opt}
        </button>
      ))}
    </div>
  );
}

// Once a team is chosen, collapse the grid down to just that team's pill
// (less clutter once you've decided) -- clicking the pill again reopens
// the full grid so you can change your mind.
function CollapsibleChoice({ options, value, onChange, columns = 3 }) {
  if (value) {
    return (
      <button
        onClick={() => onChange(null)}
        className="px-3 py-2.5 text-sm rounded-md border border-pulse text-pulse-bright bg-pulse/10 shadow-glow text-left w-full sm:w-auto transition-all duration-150"
      >
        {value} <span className="text-ink-dim text-xs ml-1">(change)</span>
      </button>
    );
  }
  return <ChoiceGrid options={options} value={value} onChange={onChange} columns={columns} />;
}

export default function SimulateMatch() {
  const [leagues, setLeagues] = useState([]);
  const [league, setLeague] = useState(null);
  const [matchDate, setMatchDate] = useState("");
  const [homeTeam, setHomeTeam] = useState(null);
  const [awayTeam, setAwayTeam] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getLeagues().then(setLeagues).catch((e) => setError(e.message));
  }, []);

  const teams = leagues.find((l) => l.code === league)?.teams || [];

  const resetBelow = (level) => {
    if (level <= 1) setMatchDate("");
    if (level <= 2) setHomeTeam(null);
    if (level <= 3) setAwayTeam(null);
    setResult(null);
  };

  const canSubmit = league && matchDate && homeTeam && awayTeam && homeTeam !== awayTeam;

  const handleSimulate = () => {
    setLoading(true);
    setError(null);
    api
      .predictMatch({ league, home_team: homeTeam, away_team: awayTeam, match_date: matchDate })
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  return (
    <div>
      <div className="relative pitch-grid pt-16 pb-10 px-6">
        <div className="max-w-2xl mx-auto relative">
          <div className="flex items-center gap-2 mb-3">
            <Swords size={16} className="text-pulse-bright" />
            <span className="text-xs uppercase tracking-widest text-ink-dim font-mono">
              Head-to-head simulator
            </span>
          </div>
          <h1 className="font-display font-700 text-4xl tracking-tight mb-3">
            Any two teams.<br />Any matchday.
          </h1>
          <p className="text-ink-dim text-sm max-w-md">
            Pick a league, a date, then two teams — the model builds fresh features
            and scores the matchup in real time.
          </p>
        </div>
      </div>

      <div className="max-w-2xl mx-auto px-6 pb-16">
        <div className="space-y-8">
          <div>
            <StepLabel n={1} text="League" />
            <ChoiceGrid
              options={leagues.map((l) => l.code)}
              value={league}
              onChange={(v) => {
                setLeague(v);
                resetBelow(1);
              }}
              columns={2}
            />
            {league && <p className="mt-2 text-xs text-ink-dim">→ {LEAGUE_NAMES[league]}</p>}
          </div>

          {league && (
            <div className="stagger-item">
              <StepLabel n={2} text="Date" />
              <div className="relative w-fit">
                <CalendarIcon
                  size={15}
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-dim pointer-events-none"
                />
                <input
                  type="date"
                  value={matchDate}
                  onChange={(e) => {
                    setMatchDate(e.target.value);
                    resetBelow(2);
                  }}
                  className="bg-night-900 border border-night-700 rounded-md pl-9 pr-3 py-2 text-sm text-ink focus:border-pulse outline-none"
                />
              </div>
            </div>
          )}

          {league && matchDate && (
            <div className="stagger-item">
              <StepLabel n={3} text="Home team" />
              <CollapsibleChoice
                options={teams}
                value={homeTeam}
                onChange={(v) => {
                  setHomeTeam(v);
                  resetBelow(3);
                }}
              />
            </div>
          )}

          {league && matchDate && homeTeam && (
            <div className="stagger-item">
              <StepLabel n={4} text="Away team" />
              <CollapsibleChoice
                options={teams.filter((t) => t !== homeTeam)}
                value={awayTeam}
                onChange={setAwayTeam}
              />
            </div>
          )}

          {canSubmit && (
            <button
              onClick={handleSimulate}
              disabled={loading}
              className="flex items-center gap-2 bg-floodlight text-night-950 font-medium px-5 py-2.5 rounded-md text-sm hover:bg-floodlight-bright hover:shadow-glow-amber transition-all duration-200 disabled:opacity-50"
            >
              <ShieldCheck size={16} />
              {loading ? "Simulating…" : "Simulate match"}
            </button>
          )}

          {error && (
            <div className="flex items-center gap-2 text-sm text-red-400 bg-red-400/5 border border-red-400/20 rounded-md px-4 py-3">
              <AlertCircle size={16} />
              {error}
            </div>
          )}

          {result && (
            <div
              className="bg-night-900 border border-night-700 rounded-lg p-6 mt-4"
              style={{ animation: "fadeIn 0.4s ease-out" }}
            >
              <div className="flex items-center justify-center gap-6 mb-6">
                <div className="flex flex-col items-center gap-2">
                  <div className="w-14 h-14 rounded-full bg-night-700 border border-floodlight/40 flex items-center justify-center font-display font-600 text-sm">
                    {initials(result.home_team)}
                  </div>
                  <span className="text-xs text-ink-dim max-w-[80px] text-center truncate">
                    {result.home_team}
                  </span>
                </div>
                <span className="font-display text-ink-dim text-lg">vs</span>
                <div className="flex flex-col items-center gap-2">
                  <div className="w-14 h-14 rounded-full bg-night-700 border border-pulse/40 flex items-center justify-center font-display font-600 text-sm">
                    {initials(result.away_team)}
                  </div>
                  <span className="text-xs text-ink-dim max-w-[80px] text-center truncate">
                    {result.away_team}
                  </span>
                </div>
              </div>
              <p className="text-center text-xs font-mono text-ink-dim mb-5">{result.date}</p>
              <ProbabilityBar
                pAway={result.p_away}
                pDraw={result.p_draw}
                pHome={result.p_home}
                homeTeam={result.home_team}
                awayTeam={result.away_team}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
MATCHMIND_EOF
echo 'wrote frontend/src/components/SimulateMatch.jsx'

cat > 'frontend/src/components/SimulateSeason.jsx' << 'MATCHMIND_EOF'
import { useEffect, useState } from "react";
import { Trophy, AlertCircle, Loader2 } from "lucide-react";
import { api } from "../api";

const LEAGUES = [
  { code: "epl", name: "Premier League" },
  { code: "spa", name: "LaLiga" },
  { code: "ger", name: "Bundesliga" },
  { code: "ita", name: "Serie A" },
  { code: "fra", name: "Ligue 1" },
];

function ProgressBar({ completed, total }) {
  const pct = total > 0 ? Math.round((completed / total) * 100) : 0;
  return (
    <div className="bg-night-900 border border-night-700 rounded-lg p-6">
      <div className="flex items-center gap-2 mb-4 text-sm text-ink-dim">
        <Loader2 size={16} className="animate-spin text-pulse-bright" />
        Running {total.toLocaleString()} Monte Carlo trials…
      </div>
      <div className="h-3 bg-night-800 rounded-full overflow-hidden mb-2 relative">
        <div
          className="h-full bg-gradient-to-r from-pulse to-floodlight transition-[width] duration-200 ease-out rounded-full relative overflow-hidden"
          style={{ width: `${pct}%` }}
        >
          <div className="absolute inset-0 shimmer-bar" />
        </div>
      </div>
      <p className="text-xs font-mono text-ink-dim">
        {completed.toLocaleString()} / {total.toLocaleString()} simulations ({pct}%)
      </p>
    </div>
  );
}

function zoneColor(pos, nTeams) {
  if (pos === 1) return "text-floodlight";
  if (pos <= 4) return "text-pulse-bright";
  if (pos >= nTeams - 2) return "text-red-400";
  return "text-ink";
}

export default function SimulateSeason() {
  const [league, setLeague] = useState("epl");
  const [season, setSeason] = useState("");
  const [availableSeasons, setAvailableSeasons] = useState([]);
  const [seasonsLoading, setSeasonsLoading] = useState(false);
  const [seasonsError, setSeasonsError] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ completed: 0, total: 5000 });
  const [error, setError] = useState(null);

  // Real, queryable seasons for the selected league -- avoids the user
  // guessing a season string that football-data.org's free tier
  // doesn't actually have data for.
  useEffect(() => {
    setSeasonsLoading(true);
    setSeasonsError(null);
    setSeason("");
    api
      .getSeasons(league)
      .then((res) => {
        setAvailableSeasons(res.seasons);
        if (res.seasons.length) setSeason(res.seasons[res.seasons.length - 1]);
      })
      .catch((e) => setSeasonsError(e.message))
      .finally(() => setSeasonsLoading(false));
  }, [league]);

  const handleSimulate = () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setProgress({ completed: 0, total: 5000 });

    api
      .simulateSeason({ league, season }, (completed, total) => setProgress({ completed, total }))
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  const nTeams = result?.standings?.length || 0;

  return (
    <div>
      <div className="relative pitch-grid pt-16 pb-10 px-6">
        <div className="max-w-3xl mx-auto relative">
          <div className="flex items-center gap-2 mb-3">
            <Trophy size={16} className="text-floodlight" />
            <span className="text-xs uppercase tracking-widest text-ink-dim font-mono">
              Monte Carlo · 5,000 trials
            </span>
          </div>
          <h1 className="font-display font-700 text-4xl tracking-tight mb-3">
            Play out the rest<br />of the season.
          </h1>
          <p className="text-ink-dim text-sm max-w-lg">
            Every remaining fixture is scored once, then the season is replayed
            thousands of times to see how the table could realistically settle.
          </p>
        </div>
      </div>

      <div className="max-w-3xl mx-auto px-6 pb-16">
        <div className="flex gap-3 items-end mb-8 flex-wrap">
          <div>
            <label className="text-xs uppercase tracking-wide text-ink-dim block mb-1.5">
              League
            </label>
            <select
              value={league}
              onChange={(e) => setLeague(e.target.value)}
              className="bg-night-900 border border-night-700 rounded-md px-3 py-2 text-sm focus:border-pulse outline-none"
            >
              {LEAGUES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs uppercase tracking-wide text-ink-dim block mb-1.5">
              Season
            </label>
            {seasonsLoading ? (
              <div className="bg-night-900 border border-night-700 rounded-md px-3 py-2 text-sm text-ink-dim w-32">
                Loading…
              </div>
            ) : (
              <select
                value={season}
                onChange={(e) => setSeason(e.target.value)}
                disabled={availableSeasons.length === 0}
                className="bg-night-900 border border-night-700 rounded-md px-3 py-2 text-sm w-32 focus:border-pulse outline-none disabled:opacity-50"
              >
                {availableSeasons.length === 0 && <option>No data</option>}
                {availableSeasons.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            )}
          </div>
          <button
            onClick={handleSimulate}
            disabled={loading || !season}
            className="bg-floodlight text-night-950 font-medium px-5 py-2.5 rounded-md text-sm hover:bg-floodlight-bright hover:shadow-glow-amber transition-all duration-200 disabled:opacity-50"
          >
            {loading ? "Simulating…" : "Simulate season"}
          </button>
        </div>

        {seasonsError && (
          <div className="flex items-center gap-2 text-sm text-red-400 bg-red-400/5 border border-red-400/20 rounded-md px-4 py-3 mb-6">
            <AlertCircle size={16} />
            Couldn't load available seasons: {seasonsError}
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 text-sm text-red-400 bg-red-400/5 border border-red-400/20 rounded-md px-4 py-3 mb-6">
            <AlertCircle size={16} />
            {error}
          </div>
        )}

        {loading && <ProgressBar completed={progress.completed} total={progress.total} />}

        {result && !loading && (
          <div
            className="bg-night-900 border border-night-700 rounded-lg overflow-hidden"
            style={{ animation: "fadeIn 0.4s ease-out" }}
          >
            <div className="px-4 py-3 border-b border-night-700 text-xs text-ink-dim font-mono flex items-center justify-between flex-wrap gap-2">
              <span>
                {result.matches_played} played · {result.matches_remaining} remaining
              </span>
              <span className="flex items-center gap-3">
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-floodlight" /> Title
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-pulse" /> Continental
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-red-400" /> Relegation
                </span>
              </span>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs uppercase text-ink-dim border-b border-night-700">
                  <th className="text-left px-4 py-2 font-normal">#</th>
                  <th className="text-left px-4 py-2 font-normal">Team</th>
                  <th className="text-right px-4 py-2 font-normal">Avg pts</th>
                  <th className="text-right px-4 py-2 font-normal">Title %</th>
                  <th className="text-right px-4 py-2 font-normal">Top 4 %</th>
                  <th className="text-right px-4 py-2 font-normal">Releg. %</th>
                </tr>
              </thead>
              <tbody>
                {result.standings.map((s, i) => (
                  <tr
                    key={s.team}
                    className="border-b border-night-700/50 last:border-0 hover:bg-night-800/60 transition-colors stagger-item"
                    style={{ animationDelay: `${i * 25}ms` }}
                  >
                    <td className={`px-4 py-2.5 font-mono ${zoneColor(i + 1, nTeams)}`}>{i + 1}</td>
                    <td className="px-4 py-2.5">{s.team}</td>
                    <td className="px-4 py-2.5 text-right font-mono">{s.avg_points}</td>
                    <td className="px-4 py-2.5 text-right font-mono text-floodlight">
                      {s.title_pct > 0 ? `${s.title_pct}%` : "—"}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-pulse-bright">
                      {s.top4_pct > 0 ? `${s.top4_pct}%` : "—"}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-red-400">
                      {s.relegation_pct > 0 ? `${s.relegation_pct}%` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
MATCHMIND_EOF
echo 'wrote frontend/src/components/SimulateSeason.jsx'

cat > 'frontend/src/components/About.jsx' << 'MATCHMIND_EOF'
import { Info, Github, Layers, Cpu, LineChart } from "lucide-react";

function Section({ icon: Icon, title, children }) {
  return (
    <div className="mb-8">
      <div className="flex items-center gap-2 mb-2">
        <Icon size={16} className="text-floodlight" />
        <h2 className="font-display font-600 text-lg">{title}</h2>
      </div>
      <div className="text-sm text-ink-dim leading-relaxed space-y-3">{children}</div>
    </div>
  );
}

export default function About() {
  return (
    <div>
      <div className="relative pitch-grid pt-16 pb-10 px-6">
        <div className="max-w-2xl mx-auto relative">
          <div className="flex items-center gap-2 mb-3">
            <Info size={16} className="text-pulse-bright" />
            <span className="text-xs uppercase tracking-widest text-ink-dim font-mono">
              About this project
            </span>
          </div>
          <h1 className="font-display font-700 text-4xl tracking-tight mb-3">
            How MatchMind works.
          </h1>
        </div>
      </div>

      <div className="max-w-2xl mx-auto px-6 pb-16">
        <Section icon={Layers} title="What this is">
          <p>
            MatchMind is a football match outcome predictor covering the top 5
            European leagues — Premier League, LaLiga, Bundesliga, Serie A, and
            Ligue 1. It scores upcoming fixtures, lets you simulate a hypothetical
            matchup between any two teams, and runs Monte Carlo simulations to
            project how the rest of a season could realistically unfold.
          </p>
        </Section>

        <Section icon={Cpu} title="Under the hood">
          <p>
            Every prediction comes from a blended LightGBM ensemble trained on
            roughly 560 engineered features per match — rolling and
            exponentially-decayed form (goals, shots, xG, corners, saves, clean
            sheets), Elo ratings, head-to-head history, rest days, and
            schedule congestion, computed separately for home and away form.
          </p>
          <p>
            The season simulator replays all remaining fixtures thousands of
            times using each match's predicted probabilities, sampling outcomes
            to build a distribution over final standings — title chances,
            continental qualification, and relegation risk, not just a single
            predicted table.
          </p>
        </Section>

        <Section icon={LineChart} title="Stack">
          <p>
            Backend: FastAPI + LightGBM + pandas, deployed on Render. Frontend:
            React + Vite + Tailwind, deployed on Vercel. Historical match data is
            hosted on Hugging Face and streamed in at startup. Fixture and
            results data comes from football-data.org.
          </p>
        </Section>

        <Section icon={Github} title="Source">
          <p>
            This project is open on GitHub — feel free to look through the code,
            file an issue, or fork it.
          </p>
        </Section>

        <div className="border-t border-night-700 pt-6 mt-10 text-xs text-ink-dim">
          <p>
            Built by Angelo, an Electrical &amp; Computer Engineering student,
            as a personal project exploring applied ML end-to-end — from
            feature engineering through to a deployed, working product.
          </p>
        </div>
      </div>
    </div>
  );
}
MATCHMIND_EOF
echo 'wrote frontend/src/components/About.jsx'

cat > 'frontend/src/App.jsx' << 'MATCHMIND_EOF'
import { useState } from "react";
import Nav from "./components/Nav";
import CalendarView from "./components/CalendarView";
import SimulateMatch from "./components/SimulateMatch";
import SimulateSeason from "./components/SimulateSeason";
import About from "./components/About";

export default function App() {
  const [view, setView] = useState("calendar");

  return (
    <div className="min-h-screen">
      <Nav view={view} setView={setView} />
      {view === "calendar" && <CalendarView />}
      {view === "simulate" && <SimulateMatch />}
      {view === "season" && <SimulateSeason />}
      {view === "about" && <About />}
    </div>
  );
}
MATCHMIND_EOF
echo 'wrote frontend/src/App.jsx'

cat > 'frontend/src/api.js' << 'MATCHMIND_EOF'
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  getLeagues: () => request("/leagues"),
  getSeasons: (league) => request(`/seasons?league=${league}`),
  getCalendar: (league, daysAhead = 60) =>
    request(`/calendar?league=${league}&days_ahead=${daysAhead}`),
  predictMatch: (payload) =>
    request("/predict", { method: "POST", body: JSON.stringify(payload) }),

  // Season simulation is a background job on the backend (so the
  // frontend can show live "N / 5,000 simulations" progress instead of
  // one long blocking request). startSeasonSimulation kicks it off and
  // returns a job_id; poll getSeasonSimulationStatus with that id.
  startSeasonSimulation: (payload) =>
    request("/simulate-season/start", { method: "POST", body: JSON.stringify(payload) }),
  getSeasonSimulationStatus: (jobId) =>
    request(`/simulate-season/status/${jobId}`),

  // Convenience wrapper: starts the job and polls until done/error,
  // calling onProgress(completed, total) along the way.
  simulateSeason: (payload, onProgress) => {
    return api.startSeasonSimulation(payload).then(({ job_id, total }) => {
      onProgress?.(0, total);
      return new Promise((resolve, reject) => {
        const poll = () => {
          api
            .getSeasonSimulationStatus(job_id)
            .then((status) => {
              onProgress?.(status.completed, status.total);
              if (status.status === "done") {
                resolve(status.result);
              } else if (status.status === "error") {
                reject(new Error(status.error || "Simulation failed"));
              } else {
                setTimeout(poll, 400);
              }
            })
            .catch(reject);
        };
        poll();
      });
    });
  },
};
MATCHMIND_EOF
echo 'wrote frontend/src/api.js'

cat > 'frontend/src/index.css' << 'MATCHMIND_EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;

html {
  scroll-behavior: smooth;
}

body {
  @apply bg-night-950 text-ink font-body;
  background-image:
    radial-gradient(ellipse 900px 500px at 50% -10%, rgba(255, 176, 32, 0.07), transparent 60%),
    radial-gradient(ellipse 700px 400px at 90% 10%, rgba(139, 124, 255, 0.06), transparent 60%);
  background-attachment: fixed;
}

::selection {
  background: theme("colors.pulse.DEFAULT");
  color: theme("colors.night.950");
}

:focus-visible {
  outline: 2px solid theme("colors.pulse.bright");
  outline-offset: 2px;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* ---------------------------------------------------------------
   Signature motif: a floodlight "pulse sweep" — a slow-rotating
   conic gradient beam, like a stadium floodlight scanning the pitch,
   paired with a soft breathing glow. Used once, prominently, behind
   the logo mark and page heroes — not scattered through the UI.
--------------------------------------------------------------- */
@keyframes sweep {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes breathe {
  0%, 100% { opacity: 0.5; transform: scale(1); }
  50% { opacity: 0.9; transform: scale(1.06); }
}

.pulse-ring {
  position: relative;
}
.pulse-ring::before {
  content: "";
  position: absolute;
  inset: -40%;
  border-radius: 9999px;
  background: conic-gradient(from 0deg, transparent 0%, rgba(255, 176, 32, 0.5) 8%, transparent 20%);
  animation: sweep 4s linear infinite;
  pointer-events: none;
}
.pulse-ring::after {
  content: "";
  position: absolute;
  inset: -18%;
  border-radius: 9999px;
  background: radial-gradient(circle, rgba(139, 124, 255, 0.35), transparent 70%);
  animation: breathe 3s ease-in-out infinite;
  pointer-events: none;
}

/* Subtle perspective pitch-line grid, used behind hero sections only */
.pitch-grid {
  background-image:
    linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px);
  background-size: 40px 40px;
  mask-image: linear-gradient(to bottom, black, transparent 85%);
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInStagger {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.shimmer-bar {
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(255, 255, 255, 0.25) 50%,
    transparent 100%
  );
  background-size: 200% 100%;
  animation: shimmer 2s ease-in-out infinite;
}

.stagger-item {
  animation: fadeInStagger 0.5s ease-out backwards;
}
MATCHMIND_EOF
echo 'wrote frontend/src/index.css'

cat > 'frontend/tailwind.config.js' << 'MATCHMIND_EOF'
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // "Floodlit pitch at night" — deep navy stadium dark, not pure
        // black, so panels read as lit surfaces rather than voids.
        night: {
          950: "#080B14",
          900: "#0D1220",
          800: "#131A2C",
          700: "#1D2740",
        },
        // Floodlight amber — the football/matchday energy accent.
        floodlight: {
          DEFAULT: "#FFB020",
          bright: "#FFCB66",
          dim: "#8A5D18",
        },
        // Electric violet — the AI / model-confidence accent.
        pulse: {
          DEFAULT: "#8B7CFF",
          bright: "#AEA3FF",
          dim: "#453D8A",
        },
        // Reserved for live/positive signals only (in-play, goals, form).
        pitch: {
          DEFAULT: "#00D68F",
          dim: "#0A6647",
        },
        ink: {
          DEFAULT: "#F3F5FA",
          dim: "#8790A8",
        },
      },
      fontFamily: {
        display: ["'Bricolage Grotesque'", "sans-serif"],
        body: ["'Inter'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
      },
      boxShadow: {
        glow: "0 0 24px -4px rgba(139, 124, 255, 0.35)",
        "glow-amber": "0 0 24px -4px rgba(255, 176, 32, 0.35)",
      },
    },
  },
  plugins: [],
};
MATCHMIND_EOF
echo 'wrote frontend/tailwind.config.js'

cat > 'frontend/index.html' << 'MATCHMIND_EOF'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>MatchMind — Football Outcome Predictor</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link
      href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap"
      rel="stylesheet"
    />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
MATCHMIND_EOF
echo 'wrote frontend/index.html'

cat > 'frontend/package.json' << 'MATCHMIND_EOF'
{
  "name": "matchmind-frontend",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "lucide-react": "^0.383.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.1",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.47",
    "tailwindcss": "^3.4.13",
    "vite": "^5.4.8"
  }
}
MATCHMIND_EOF
echo 'wrote frontend/package.json'

