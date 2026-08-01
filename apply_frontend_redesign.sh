#!/bin/bash
set -e
cd "$(dirname "$0")"

cat > 'frontend/index.html' << 'MATCHMIND_EOF'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>MatchMind — Football Outcome Predictor</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;700;800;900&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap"
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

cat > 'frontend/tailwind.config.js' << 'MATCHMIND_EOF'
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Graphite-black terminal surface, not pure black -- panels
        // read as backlit glass, not voids.
        void: {
          DEFAULT: "#06070A",
          panel: "#0E1015",
          raised: "#161922",
        },
        line: {
          DEFAULT: "#23262F",
          bright: "#383C48",
        },
        ink: {
          DEFAULT: "#F2F0E9",
          dim: "#84868F",
          faint: "#4A4C56",
        },
        // "Magma" ramp -- the same low-to-high colormap analysts use
        // for xG heatmaps and model-confidence surfaces. This IS the
        // model's voice throughout the product: cold indigo (unlikely)
        // through magenta to hot amber (likely).
        magma: {
          cold: "#3C2079",
          mid: "#D6336C",
          hot: "#F2A93C",
          "hot-bright": "#FFCC70",
        },
        risk: {
          DEFAULT: "#E5484D",
          dim: "#5C2224",
        },
      },
      fontFamily: {
        // Condensed, heavy-shouldered display face -- the register of
        // a broadcast scoreboard or a manager's team-sheet graphic,
        // not a generic startup sans.
        display: ["'Big Shoulders Display'", "sans-serif"],
        body: ["'IBM Plex Sans'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
      },
      boxShadow: {
        // Hard 1px edge-lit outline instead of a soft blurred glow --
        // reads as an illuminated panel edge, not a diffuse aura.
        "edge-hot": "inset 0 0 0 1px rgba(242,169,60,0.55), 0 0 0 1px rgba(242,169,60,0.12)",
        "edge-cold": "inset 0 0 0 1px rgba(214,51,108,0.5), 0 0 0 1px rgba(214,51,108,0.1)",
      },
    },
  },
  plugins: [],
};
MATCHMIND_EOF
echo 'wrote frontend/tailwind.config.js'

cat > 'frontend/vite.config.js' << 'MATCHMIND_EOF'
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
});
MATCHMIND_EOF
echo 'wrote frontend/vite.config.js'

cat > 'frontend/postcss.config.js' << 'MATCHMIND_EOF'
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
MATCHMIND_EOF
echo 'wrote frontend/postcss.config.js'

cat > 'frontend/src/main.jsx' << 'MATCHMIND_EOF'
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
MATCHMIND_EOF
echo 'wrote frontend/src/main.jsx'

cat > 'frontend/src/App.jsx' << 'MATCHMIND_EOF'
import { useState } from "react";
import Nav from "./components/Nav";
import CalendarView from "./components/CalendarView";
import SimulateMatch from "./components/SimulateMatch";
import SimulateSeason from "./components/SimulateSeason";
import About from "./components/About";
import CornerBadge from "./components/CornerBadge";

export default function App() {
  const [view, setView] = useState("calendar");

  return (
    <div className="min-h-screen">
      <Nav view={view} setView={setView} />
      {view === "calendar" && <CalendarView />}
      {view === "simulate" && <SimulateMatch />}
      {view === "season" && <SimulateSeason />}
      {view === "about" && <About />}
      <CornerBadge />
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
  @apply bg-void text-ink font-body;
  background-image:
    linear-gradient(rgba(255, 255, 255, 0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.025) 1px, transparent 1px);
  background-size: 48px 48px;
  background-attachment: fixed;
}

::selection {
  background: theme("colors.magma.hot");
  color: theme("colors.void.DEFAULT");
}

:focus-visible {
  outline: 2px solid theme("colors.magma.hot");
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
   Cut-corner: every structural panel, button and tag in this
   product has its top-right corner clipped at an angle -- a single
   consistent tell borrowed from broadcast sports-graphics packages
   (team-sheet cards, score bugs, lower-thirds). Two sizes so small
   controls and large panels stay proportionate.
--------------------------------------------------------------- */
.cut-corner {
  clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
}
.cut-corner-lg {
  clip-path: polygon(0 0, calc(100% - 22px) 0, 100% 22px, 100% 100%, 0 100%);
}
.cut-corner-br {
  clip-path: polygon(0 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%);
}

/* Hairline data-grid, used behind hero sections only -- a faint
   analytics-plot backdrop, not decoration scattered everywhere. */
.data-grid {
  position: relative;
}
.data-grid::before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(242, 169, 60, 0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(214, 51, 108, 0.05) 1px, transparent 1px);
  background-size: 64px 64px;
  mask-image: linear-gradient(to bottom, black, transparent 88%);
  pointer-events: none;
  z-index: 0;
}

/* Live signal dot -- a hard pulse, not a soft breathing blur. */
@keyframes hardPulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}
.signal-dot {
  animation: hardPulse 1.6s steps(2, jump-none) infinite;
}

/* Ticker marquee -- the signature broadcast lower-third motif, used
   once, directly under the nav, persistent across every view. */
@keyframes marquee {
  from { transform: translateX(0); }
  to { transform: translateX(-50%); }
}
.ticker-track {
  animation: marquee 26s linear infinite;
  width: max-content;
}
.ticker-row:hover .ticker-track {
  animation-play-state: paused;
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
    rgba(255, 255, 255, 0.22) 50%,
    transparent 100%
  );
  background-size: 200% 100%;
  animation: shimmer 1.8s ease-in-out infinite;
}

.stagger-item {
  animation: fadeInStagger 0.5s ease-out backwards;
}

@keyframes loadingSlide {
  0% { transform: translateX(-100%); }
  50% { transform: translateX(150%); }
  100% { transform: translateX(150%); }
}

/* Corner credit bug -- fixed, always visible, styled like a channel
   watermark rather than a footer line. */
@keyframes badgeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
.corner-badge {
  animation: badgeIn 0.6s ease-out 0.3s backwards;
}
MATCHMIND_EOF
echo 'wrote frontend/src/index.css'

cat > 'frontend/src/components/Nav.jsx' << 'MATCHMIND_EOF'
import { CalendarDays, Swords, Trophy, Info } from "lucide-react";

const TICKER_ITEMS = [
  "5 LEAGUES TRACKED",
  "560+ ENGINEERED FEATURES",
  "LIGHTGBM ENSEMBLE",
  "5,000-TRIAL MONTE CARLO",
  "LIVE FIXTURE FEED",
];

export default function Nav({ view, setView }) {
  const tabs = [
    { id: "calendar", label: "Calendar", icon: CalendarDays },
    { id: "simulate", label: "Simulate a match", icon: Swords },
    { id: "season", label: "Simulate a season", icon: Trophy },
    { id: "about", label: "About", icon: Info },
  ];

  const tickerContent = TICKER_ITEMS.join("   ◆   ");

  return (
    <div className="sticky top-0 z-20 bg-void/90 backdrop-blur-md border-b border-line">
      <nav className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="cut-corner w-7 h-7 bg-gradient-to-br from-magma-hot to-magma-mid flex items-center justify-center">
            <span className="font-display font-800 text-void text-sm leading-none">M</span>
          </div>
          <div className="font-display font-700 text-xl tracking-tight uppercase leading-none">
            Match<span className="text-magma-hot">Mind</span>
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
                className={`relative flex items-center gap-1.5 px-3.5 py-2 text-xs font-mono uppercase tracking-wide transition-colors duration-150 border-b-2 ${
                  active
                    ? "text-ink border-magma-hot"
                    : "text-ink-dim border-transparent hover:text-ink hover:border-line-bright"
                }`}
              >
                <Icon size={14} className={active ? "text-magma-hot" : ""} />
                <span className="hidden sm:inline">{t.label}</span>
              </button>
            );
          })}
        </div>
      </nav>

      {/* Signature ticker strip -- the one recurring broadcast motif,
          scrolling model vitals under the nav on every view. */}
      <div className="ticker-row overflow-hidden border-t border-line bg-void-panel/60">
        <div className="ticker-track flex whitespace-nowrap py-1.5">
          <span className="font-mono text-[10px] tracking-[0.18em] text-ink-faint px-4">
            {tickerContent}
          </span>
          <span className="font-mono text-[10px] tracking-[0.18em] text-ink-faint px-4">
            {tickerContent}
          </span>
        </div>
      </div>
    </div>
  );
}
MATCHMIND_EOF
echo 'wrote frontend/src/components/Nav.jsx'

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
      <div className="relative data-grid pt-16 pb-10 px-6 overflow-hidden">
        <div className="max-w-3xl mx-auto relative">
          <div className="flex items-center gap-2 mb-4">
            <span className="signal-dot w-1.5 h-1.5 rounded-full bg-magma-hot" />
            <span className="text-[11px] uppercase tracking-[0.18em] text-ink-dim font-mono">
              Live model · top 5 leagues
            </span>
          </div>
          <h1 className="font-display font-800 uppercase text-5xl sm:text-6xl leading-[0.95] tracking-tight mb-4">
            What the model sees<br className="hidden sm:block" /> for the next two months
          </h1>
          <p className="text-ink-dim text-sm max-w-lg font-body">
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
              className={`px-3 py-1.5 text-xs font-mono uppercase tracking-wide border transition-colors duration-150 ${
                league === l.code
                  ? "border-magma-hot text-magma-hot bg-magma-hot/10"
                  : "border-line text-ink-dim hover:text-ink hover:border-line-bright"
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
                className="h-24 bg-void-panel border border-line animate-pulse"
                style={{ animationDelay: `${i * 100}ms` }}
              />
            ))}
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 text-sm text-risk bg-risk-dim/20 border border-risk/30 px-4 py-3">
            <AlertCircle size={16} />
            Couldn't load fixtures: {error}
          </div>
        )}

        {!loading && !error && (
          <div className="space-y-6">
            {orderedDates.map((dateStr, gi) => (
              <div key={dateStr} className="stagger-item" style={{ animationDelay: `${gi * 60}ms` }}>
                <p className="text-[11px] uppercase tracking-[0.14em] text-ink-dim mb-2 font-mono flex items-center gap-2">
                  <span className="w-3 h-px bg-line-bright" />
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
                        className={`cut-corner bg-void-panel border transition-colors duration-150 ${
                          isExpanded ? "border-magma-hot/50" : "border-line hover:border-line-bright"
                        } ${f.predictable === false ? "opacity-50" : ""}`}
                      >
                        <button
                          onClick={() => f.predictable !== false && handleSelectFixture(f)}
                          disabled={f.predictable === false}
                          className="w-full flex items-center justify-between p-4 text-left disabled:cursor-not-allowed"
                        >
                          <span className="font-medium text-sm font-body">
                            {f.home_team} <span className="text-ink-dim font-mono text-xs">VS</span> {f.away_team}
                          </span>
                          {f.predictable === false ? (
                            <span className="text-xs text-ink-dim font-mono">No data</span>
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
                                <div className="h-2 bg-void-raised overflow-hidden mb-2 relative">
                                  <div className="absolute inset-0 w-1/3 bg-gradient-to-r from-magma-mid to-magma-hot animate-[loadingSlide_1.2s_ease-in-out_infinite]" />
                                </div>
                                <p className="text-xs font-mono text-ink-dim flex items-center gap-1.5">
                                  <Loader2 size={12} className="animate-spin" />
                                  Simulating this match…
                                </p>
                              </div>
                            )}
                            {pred?.status === "error" && (
                              <p className="text-xs text-risk">Couldn't simulate: {pred.error}</p>
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
              <p className="text-ink-dim text-sm font-mono">No fixtures found in the next {DAYS_AHEAD} days.</p>
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
    <div className="flex items-center gap-2 mb-2.5">
      <span className="w-5 h-5 flex items-center justify-center bg-void-raised border border-line text-[10px] font-mono text-ink-dim">
        {n}
      </span>
      <p className="text-[11px] uppercase tracking-[0.14em] text-ink-dim font-mono">{text}</p>
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
          className={`px-3 py-2.5 text-sm text-left border transition-colors duration-150 font-body ${
            value === opt.value
              ? "border-magma-hot text-magma-hot-bright bg-magma-hot/10"
              : "border-line text-ink-dim hover:text-ink hover:border-line-bright"
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
        className="px-3 py-2.5 text-sm border border-magma-hot text-magma-hot-bright bg-magma-hot/10 text-left w-full sm:w-auto transition-colors duration-150"
      >
        {value} <span className="text-ink-dim text-xs ml-1 font-mono">(change)</span>
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
      <div className="relative data-grid pt-16 pb-10 px-6">
        <div className="max-w-2xl mx-auto relative">
          <div className="flex items-center gap-2 mb-4">
            <Swords size={15} className="text-magma-hot" />
            <span className="text-[11px] uppercase tracking-[0.18em] text-ink-dim font-mono">
              Head-to-head simulator
            </span>
          </div>
          <h1 className="font-display font-800 uppercase text-5xl leading-[0.95] tracking-tight mb-4">
            Any two teams.<br />Any matchday.
          </h1>
          <p className="text-ink-dim text-sm max-w-md font-body">
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
            {league && <p className="mt-2 text-xs text-ink-dim font-mono">→ {LEAGUE_NAMES[league]}</p>}
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
                  className="bg-void-panel border border-line pl-9 pr-3 py-2 text-sm text-ink focus:border-magma-hot outline-none font-mono"
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
              className="cut-corner flex items-center gap-2 bg-magma-hot text-void font-semibold px-5 py-2.5 text-sm hover:bg-magma-hot-bright transition-colors duration-150 disabled:opacity-50 font-mono uppercase tracking-wide"
            >
              <ShieldCheck size={16} />
              {loading ? "Simulating…" : "Simulate match"}
            </button>
          )}

          {loading && (
            <div className="cut-corner-lg bg-void-panel border border-line p-5">
              <div className="h-2 bg-void-raised overflow-hidden mb-3 relative">
                <div className="absolute inset-0 w-1/3 bg-gradient-to-r from-magma-mid to-magma-hot animate-[loadingSlide_1.2s_ease-in-out_infinite]" />
              </div>
              <p className="text-xs font-mono text-ink-dim">{loadingMessage}</p>
            </div>
          )}

          {error && (
            <div className="flex items-center gap-2 text-sm text-risk bg-risk-dim/20 border border-risk/30 px-4 py-3">
              <AlertCircle size={16} />
              {error}
            </div>
          )}

          {result && (
            <div
              className="cut-corner-lg bg-void-panel border border-line p-6 mt-4"
              style={{ animation: "fadeIn 0.4s ease-out" }}
            >
              <div className="flex items-center justify-center gap-6 mb-6">
                <div className="flex flex-col items-center gap-2">
                  <div className="cut-corner w-14 h-14 bg-void-raised border border-magma-hot/40 flex items-center justify-center font-display font-700 text-sm">
                    {initials(result.home_team)}
                  </div>
                  <span className="text-xs text-ink-dim max-w-[80px] text-center truncate font-mono">
                    {result.home_team}
                  </span>
                </div>
                <span className="font-display font-700 text-ink-dim text-lg uppercase">vs</span>
                <div className="flex flex-col items-center gap-2">
                  <div className="cut-corner w-14 h-14 bg-void-raised border border-magma-mid/40 flex items-center justify-center font-display font-700 text-sm">
                    {initials(result.away_team)}
                  </div>
                  <span className="text-xs text-ink-dim max-w-[80px] text-center truncate font-mono">
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
    <div className="cut-corner-lg bg-void-panel border border-line p-6">
      <div className="flex items-center gap-2 mb-4 text-sm text-ink-dim font-mono">
        <Loader2 size={16} className="animate-spin text-magma-hot" />
        Running {total.toLocaleString()} Monte Carlo trials…
      </div>
      <div className="h-3 bg-void-raised overflow-hidden mb-2 relative">
        <div
          className="h-full bg-gradient-to-r from-magma-mid to-magma-hot transition-[width] duration-200 ease-out relative overflow-hidden"
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
  if (pos === 1) return "text-magma-hot";
  if (pos <= 4) return "text-magma-hot-bright";
  if (pos >= nTeams - 2) return "text-risk";
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
      <div className="relative data-grid pt-16 pb-10 px-6">
        <div className="max-w-3xl mx-auto relative">
          <div className="flex items-center gap-2 mb-4">
            <Trophy size={15} className="text-magma-hot" />
            <span className="text-[11px] uppercase tracking-[0.18em] text-ink-dim font-mono">
              Monte Carlo · 5,000 trials
            </span>
          </div>
          <h1 className="font-display font-800 uppercase text-5xl leading-[0.95] tracking-tight mb-4">
            Play out the rest<br />of the season.
          </h1>
          <p className="text-ink-dim text-sm max-w-lg font-body">
            Every remaining fixture is scored once, then the season is replayed
            thousands of times to see how the table could realistically settle.
          </p>
        </div>
      </div>

      <div className="max-w-3xl mx-auto px-6 pb-16">
        <div className="flex gap-3 items-end mb-8 flex-wrap">
          <div>
            <label className="text-[11px] uppercase tracking-[0.14em] text-ink-dim font-mono block mb-1.5">
              League
            </label>
            <select
              value={league}
              onChange={(e) => setLeague(e.target.value)}
              className="bg-void-panel border border-line px-3 py-2 text-sm focus:border-magma-hot outline-none font-mono"
            >
              {LEAGUES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-[11px] uppercase tracking-[0.14em] text-ink-dim font-mono block mb-1.5">
              Season
            </label>
            {seasonsLoading ? (
              <div className="bg-void-panel border border-line px-3 py-2 text-sm text-ink-dim w-32 font-mono">
                Loading…
              </div>
            ) : (
              <select
                value={season}
                onChange={(e) => setSeason(e.target.value)}
                disabled={availableSeasons.length === 0}
                className="bg-void-panel border border-line px-3 py-2 text-sm w-32 focus:border-magma-hot outline-none disabled:opacity-50 font-mono"
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
            className="cut-corner bg-magma-hot text-void font-semibold px-5 py-2.5 text-sm hover:bg-magma-hot-bright transition-colors duration-150 disabled:opacity-50 font-mono uppercase tracking-wide"
          >
            {loading ? "Simulating…" : "Simulate season"}
          </button>
        </div>

        {seasonsError && (
          <div className="flex items-center gap-2 text-sm text-risk bg-risk-dim/20 border border-risk/30 px-4 py-3 mb-6">
            <AlertCircle size={16} />
            Couldn't load available seasons: {seasonsError}
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 text-sm text-risk bg-risk-dim/20 border border-risk/30 px-4 py-3 mb-6">
            <AlertCircle size={16} />
            {error}
          </div>
        )}

        {loading && <ProgressBar completed={progress.completed} total={progress.total} />}

        {result && !loading && (
          <div
            className="cut-corner-lg bg-void-panel border border-line overflow-hidden"
            style={{ animation: "fadeIn 0.4s ease-out" }}
          >
            <div className="px-4 py-3 border-b border-line text-xs text-ink-dim font-mono flex items-center justify-between flex-wrap gap-2">
              <span>
                {result.matches_played} played · {result.matches_remaining} remaining
              </span>
              <span className="flex items-center gap-3 uppercase tracking-wide text-[10px]">
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 bg-magma-hot" /> Title
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 bg-magma-hot-bright" /> Continental
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 bg-risk" /> Relegation
                </span>
              </span>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-[11px] uppercase tracking-wide text-ink-dim border-b border-line font-mono">
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
                    className="border-b border-line/50 last:border-0 hover:bg-void-raised/60 transition-colors stagger-item font-body"
                    style={{ animationDelay: `${i * 25}ms` }}
                  >
                    <td className={`px-4 py-2.5 font-mono ${zoneColor(i + 1, nTeams)}`}>{i + 1}</td>
                    <td className="px-4 py-2.5">{s.team}</td>
                    <td className="px-4 py-2.5 text-right font-mono">{s.avg_points}</td>
                    <td className="px-4 py-2.5 text-right font-mono text-magma-hot">
                      {s.title_pct > 0 ? `${s.title_pct}%` : "—"}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-magma-hot-bright">
                      {s.top4_pct > 0 ? `${s.top4_pct}%` : "—"}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-risk">
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

cat > 'frontend/src/components/ProbabilityBar.jsx' << 'MATCHMIND_EOF'
import { useEffect, useState } from "react";

// Animates a number counting up from 0 on mount/change -- gives the
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
      <div className="flex justify-between text-[11px] font-mono text-ink-dim mb-1.5 uppercase tracking-wide">
        <span className={favorite === "home" ? "text-magma-hot" : ""}>{homeTeam}</span>
        <span className={favorite === "draw" ? "text-ink" : ""}>Draw</span>
        <span className={favorite === "away" ? "text-magma-mid" : ""}>{awayTeam}</span>
      </div>
      {/* Confidence gauge -- a single continuous ramp from home (hot
          amber) through draw (neutral) to away (cold magenta/indigo),
          the same colormap logic used for the model's xG heatmaps. */}
      <div className="flex h-9 overflow-hidden border border-line">
        <div
          className="relative flex items-center justify-center text-xs font-mono text-void font-semibold transition-[width] duration-700 ease-out overflow-hidden"
          style={{ width: `${home}%`, background: "linear-gradient(90deg, #C98426, #F2A93C)" }}
        >
          {favorite === "home" && <div className="absolute inset-0 shimmer-bar" />}
          <span className="relative z-10">{home >= 10 && `${homeAnim}%`}</span>
        </div>
        <div
          className="bg-void-raised flex items-center justify-center text-xs font-mono text-ink-dim transition-[width] duration-700 ease-out border-x border-line"
          style={{ width: `${draw}%` }}
        >
          {draw >= 10 && `${drawAnim}%`}
        </div>
        <div
          className="relative flex items-center justify-center text-xs font-mono text-ink font-semibold transition-[width] duration-700 ease-out overflow-hidden"
          style={{ width: `${away}%`, background: "linear-gradient(90deg, #D6336C, #3C2079)" }}
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

cat > 'frontend/src/components/About.jsx' << 'MATCHMIND_EOF'
import { Info, Github, Layers, Cpu, LineChart } from "lucide-react";

function Section({ icon: Icon, title, children }) {
  return (
    <div className="mb-9">
      <div className="flex items-center gap-2 mb-2.5">
        <Icon size={15} className="text-magma-hot" />
        <h2 className="font-display font-700 uppercase text-xl tracking-tight">{title}</h2>
      </div>
      <div className="text-sm text-ink-dim leading-relaxed space-y-3 font-body">{children}</div>
    </div>
  );
}

export default function About() {
  return (
    <div>
      <div className="relative data-grid pt-16 pb-10 px-6">
        <div className="max-w-2xl mx-auto relative">
          <div className="flex items-center gap-2 mb-4">
            <Info size={15} className="text-magma-hot" />
            <span className="text-[11px] uppercase tracking-[0.18em] text-ink-dim font-mono">
              About this project
            </span>
          </div>
          <h1 className="font-display font-800 uppercase text-5xl leading-[0.95] tracking-tight mb-4">
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

        <div className="border-t border-line pt-6 mt-10 text-xs text-ink-dim font-mono">
          <p>
            Built by <span className="text-ink">Angelo</span>, an Electrical &amp; Computer Engineering
            student, as a personal project exploring applied ML end-to-end — from
            feature engineering through to a deployed, working product.
          </p>
        </div>
      </div>
    </div>
  );
}
MATCHMIND_EOF
echo 'wrote frontend/src/components/About.jsx'

cat > 'frontend/src/components/CornerBadge.jsx' << 'MATCHMIND_EOF'
export default function CornerBadge() {
  return (
    <div className="corner-badge fixed bottom-4 left-4 z-30 hidden sm:block select-none">
      <div className="cut-corner-br flex items-center gap-2 bg-void-panel/80 backdrop-blur-md border border-line px-3 py-2">
        <span className="signal-dot w-1.5 h-1.5 rounded-full bg-magma-hot shrink-0" />
        <span className="font-mono text-[10px] tracking-[0.18em] text-ink-dim uppercase leading-none">
          Built by <span className="text-ink">Angelo</span>
        </span>
      </div>
    </div>
  );
}
MATCHMIND_EOF
echo 'wrote frontend/src/components/CornerBadge.jsx'

