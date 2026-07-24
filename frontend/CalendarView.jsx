import { useEffect, useState } from "react";
import { CalendarDays, AlertCircle } from "lucide-react";
import { api } from "../api";
import ProbabilityBar from "./ProbabilityBar";

const LEAGUES = [
  { code: "epl", name: "Premier League" },
  { code: "spa", name: "La Liga" },
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

  // Group fixtures by date for a cleaner, scannable layout
  const groups = fixtures.reduce((acc, f) => {
    (acc[f.date] = acc[f.date] || []).push(f);
    return acc;
  }, {});
  const orderedDates = Object.keys(groups).sort();

  return (
    <div className="max-w-3xl mx-auto px-6 py-10">
      <div className="flex items-center gap-2 mb-1">
        <CalendarDays size={20} className="text-signal" />
        <h1 className="font-display font-600 text-2xl">Upcoming fixtures</h1>
      </div>
      <p className="text-ink-dim text-sm mb-6">
        Next 2 months, with win / draw / loss probabilities from Model A.
      </p>

      <div className="flex gap-2 mb-8 flex-wrap">
        {LEAGUES.map((l) => (
          <button
            key={l.code}
            onClick={() => setLeague(l.code)}
            className={`px-3 py-1.5 text-sm rounded-full border transition-colors ${
              league === l.code
                ? "border-signal text-signal bg-signal/10"
                : "border-pitch-700 text-ink-dim hover:text-ink hover:border-ink-dim"
            }`}
          >
            {l.name}
          </button>
        ))}
      </div>

      {loading && (
        <div className="space-y-4 animate-pulse">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-24 bg-pitch-900 border border-pitch-700 rounded-md" />
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
          {orderedDates.map((dateStr) => (
            <div key={dateStr}>
              <p className="text-xs uppercase tracking-wide text-ink-dim mb-2 font-mono">
                {formatDateHeading(dateStr)}
              </p>
              <div className="space-y-3">
                {groups[dateStr].map((f, i) => (
                  <div
                    key={i}
                    className="bg-pitch-900 border border-pitch-700 rounded-md p-4 hover:border-pitch-700/80 transition-colors"
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
  );
}
