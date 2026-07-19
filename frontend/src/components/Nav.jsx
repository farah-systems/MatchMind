export default function Nav({ view, setView }) {
  const tabs = [
    { id: "calendar", label: "Calendar" },
    { id: "simulate", label: "Simulate a match" },
    { id: "season", label: "Simulate a season" },
  ];

  return (
    <nav className="border-b border-pitch-700 sticky top-0 bg-pitch-950/95 backdrop-blur z-10">
      <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="font-display font-700 text-xl tracking-tight">
          Match<span className="text-signal">Mind</span>
        </div>
        <div className="flex gap-1">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setView(t.id)}
              className={`px-4 py-2 text-sm font-medium rounded-sm transition-colors ${
                view === t.id
                  ? "bg-pitch-700 text-ink"
                  : "text-ink-dim hover:text-ink"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>
    </nav>
  );
}
