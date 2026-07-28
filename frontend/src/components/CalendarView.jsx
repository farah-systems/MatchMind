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
