import { CalendarDays, Swords, Trophy } from "lucide-react";

export default function Nav({ view, setView }) {
  const tabs = [
    { id: "calendar", label: "Calendar", icon: CalendarDays },
    { id: "simulate", label: "Simulate a match", icon: Swords },
    { id: "season", label: "Simulate a season", icon: Trophy },
  ];

  return (
    <nav className="border-b border-pitch-700 sticky top-0 bg-pitch-950/90 backdrop-blur-md z-10">
      <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-signal shadow-[0_0_8px_theme(colors.signal.DEFAULT)]" />
          <div className="font-display font-700 text-xl tracking-tight">
            Match<span className="text-signal">Mind</span>
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
                className={`flex items-center gap-1.5 px-4 py-2 text-sm font-medium rounded-sm transition-all ${
                  active
                    ? "bg-pitch-700 text-ink"
                    : "text-ink-dim hover:text-ink hover:bg-pitch-800"
                }`}
              >
                <Icon size={15} className={active ? "text-signal" : ""} />
                <span className="hidden sm:inline">{t.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
