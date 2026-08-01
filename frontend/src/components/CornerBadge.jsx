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
