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
