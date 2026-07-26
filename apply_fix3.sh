#!/bin/bash
set -e
cd "$(dirname "$0")"

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

/* Subtle perspective pitch-line grid, used behind hero sections only.
   IMPORTANT: this must live on a ::before pseudo-element, not be
   applied as a mask directly on the element -- mask-image affects
   the ENTIRE element's rendered output including real text content,
   which is what was making hero paragraphs/cards fade to invisible. */
.pitch-grid {
  position: relative;
}
.pitch-grid::before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px);
  background-size: 40px 40px;
  mask-image: linear-gradient(to bottom, black, transparent 85%);
  pointer-events: none;
  z-index: 0;
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

@keyframes loadingSlide {
  0% { transform: translateX(-100%); }
  50% { transform: translateX(150%); }
  100% { transform: translateX(150%); }
}
MATCHMIND_EOF
echo 'wrote frontend/src/index.css'

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
  // options can be plain strings (team names, where value === label) or
  // {value, label} objects (leagues, where the code and display name differ)
  const normalized = options.map((o) => (typeof o === "string" ? { value: o, label: o } : o));
  return (
    <div className={`grid grid-cols-${columns} gap-2`}>
      {normalized.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={`px-3 py-2.5 text-sm rounded-md border text-left transition-all duration-150 ${
            value === opt.value
              ? "border-pulse text-pulse-bright bg-pulse/10 shadow-glow"
              : "border-night-700 text-ink-dim hover:text-ink hover:border-ink-dim"
          }`}
        >
          {opt.label}
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
  const [loadingMessage, setLoadingMessage] = useState("");
  const [error, setError] = useState(null);

  const LOADING_MESSAGES = [
    "Pulling recent form for both teams…",
    "Weighing head-to-head history…",
    "Factoring in Elo and rest days…",
    "Running the ensemble…",
  ];

  useEffect(() => {
    api.getLeagues().then(setLeagues).catch((e) => setError(e.message));
  }, []);

  // Rotate through status messages while the request is in flight, so a
  // few-second wait doesn't feel like nothing is happening.
  useEffect(() => {
    if (!loading) return;
    let i = 0;
    setLoadingMessage(LOADING_MESSAGES[0]);
    const interval = setInterval(() => {
      i = (i + 1) % LOADING_MESSAGES.length;
      setLoadingMessage(LOADING_MESSAGES[i]);
    }, 1200);
    return () => clearInterval(interval);
  }, [loading]);

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
              options={leagues.map((l) => ({ value: l.code, label: l.name }))}
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

          {loading && (
            <div className="bg-night-900 border border-night-700 rounded-lg p-5">
              <div className="h-2 bg-night-800 rounded-full overflow-hidden mb-3 relative">
                <div className="absolute inset-0 w-1/3 bg-gradient-to-r from-pulse to-floodlight rounded-full animate-[loadingSlide_1.2s_ease-in-out_infinite]" />
              </div>
              <p className="text-xs font-mono text-ink-dim">{loadingMessage}</p>
            </div>
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

