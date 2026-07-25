import { useEffect, useState } from "react";
import { api } from "../api";
import ProbabilityBar from "./ProbabilityBar";

const LEAGUE_NAMES = {
  epl: "Premier League",
  spa: "La Liga",
  ger: "Bundesliga",
  ita: "Serie A",
  fra: "Ligue 1",
};

function ChoiceGrid({ options, value, onChange, columns = 3 }) {
  return (
    <div className={`grid grid-cols-${columns} gap-2`}>
      {options.map((opt) => (
        <button
          key={opt}
          onClick={() => onChange(opt)}
          className={`px-3 py-2.5 text-sm rounded-sm border text-left transition-colors ${
            value === opt
              ? "border-signal text-signal bg-signal/5"
              : "border-pitch-700 text-ink-dim hover:text-ink hover:border-pitch-700"
          }`}
        >
          {opt}
        </button>
      ))}
    </div>
  );
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
    <div className="max-w-2xl mx-auto px-6 py-10">
      <h1 className="font-display font-600 text-2xl mb-1">Simulate a hypothetical match</h1>
      <p className="text-ink-dim text-sm mb-8">
        Pick a league, a date, then two teams from that league.
      </p>

      <div className="space-y-8">
        <div>
          <p className="text-xs uppercase tracking-wide text-ink-dim mb-2">1. League</p>
          <ChoiceGrid
            options={leagues.map((l) => l.code)}
            value={league}
            onChange={(v) => {
              setLeague(v);
              resetBelow(1);
            }}
            columns={2}
          />
          {leagues.length > 0 && (
            <div className="mt-2 flex gap-2 flex-wrap text-xs text-ink-dim">
              {leagues.map((l) => (
                <span key={l.code}>{l.code === league ? `→ ${LEAGUE_NAMES[l.code]}` : ""}</span>
              ))}
            </div>
          )}
        </div>

        {league && (
          <div>
            <p className="text-xs uppercase tracking-wide text-ink-dim mb-2">2. Date</p>
            <input
              type="date"
              value={matchDate}
              onChange={(e) => {
                setMatchDate(e.target.value);
                resetBelow(2);
              }}
              className="bg-pitch-900 border border-pitch-700 rounded-sm px-3 py-2 text-sm text-ink focus:border-signal outline-none"
            />
          </div>
        )}

        {league && matchDate && (
          <div>
            <p className="text-xs uppercase tracking-wide text-ink-dim mb-2">3. Home team</p>
            <ChoiceGrid
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
          <div>
            <p className="text-xs uppercase tracking-wide text-ink-dim mb-2">4. Away team</p>
            <ChoiceGrid
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
            className="bg-signal text-pitch-950 font-medium px-5 py-2.5 rounded-sm text-sm hover:bg-signal-bright transition-colors disabled:opacity-50"
          >
            {loading ? "Simulating…" : "Simulate match"}
          </button>
        )}

        {error && <p className="text-sm text-red-400">{error}</p>}

        {result && (
          <div className="bg-pitch-900 border border-pitch-700 rounded-sm p-5 mt-4">
            <p className="text-sm font-medium mb-4">
              {result.home_team} vs {result.away_team} — {result.date}
            </p>
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
  );
}
