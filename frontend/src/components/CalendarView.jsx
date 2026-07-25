import { useEffect, useState } from "react";
import { api } from "../api";
import ProbabilityBar from "./ProbabilityBar";

const LEAGUES = [
  { code: "epl", name: "Premier League" },
  { code: "spa", name: "La Liga" },
  { code: "ger", name: "Bundesliga" },
  { code: "ita", name: "Serie A" },
  { code: "fra", name: "Ligue 1" },
];

export default function CalendarView() {
  const [league, setLeague] = useState("epl");
  const [fixtures, setFixtures] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api
      .getCalendar(league)
      .then(setFixtures)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [league]);

  return (
    <div className="max-w-3xl mx-auto px-6 py-10">
      <h1 className="font-display font-600 text-2xl mb-1">Upcoming fixtures</h1>
      <p className="text-ink-dim text-sm mb-6">
        Next 14 days, with win / draw / loss probabilities from Model A.
      </p>

      <div className="flex gap-2 mb-8 flex-wrap">
        {LEAGUES.map((l) => (
          <button
            key={l.code}
            onClick={() => setLeague(l.code)}
            className={`px-3 py-1.5 text-sm rounded-sm border transition-colors ${
              league === l.code
                ? "border-signal text-signal"
                : "border-pitch-700 text-ink-dim hover:text-ink"
            }`}
          >
            {l.name}
          </button>
        ))}
      </div>

      {loading && <p className="text-ink-dim text-sm">Loading fixtures…</p>}
      {error && (
        <p className="text-sm text-red-400">
          Couldn't load fixtures: {error}
        </p>
      )}

      <div className="space-y-4">
        {fixtures.map((f, i) => (
          <div key={i} className="bg-pitch-900 border border-pitch-700 rounded-sm p-4">
            <div className="flex justify-between items-baseline mb-3">
              <span className="font-medium text-sm">
                {f.home_team} <span className="text-ink-dim">vs</span> {f.away_team}
              </span>
              <span className="text-xs font-mono text-ink-dim">{f.date}</span>
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
        {!loading && !error && fixtures.length === 0 && (
          <p className="text-ink-dim text-sm">No fixtures found in the next 14 days.</p>
        )}
      </div>
    </div>
  );
}
