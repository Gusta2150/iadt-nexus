import type { HistoryPoint } from "@/lib/types";

export function Convergence({ history }: { history: HistoryPoint[] }) {
  if (history.length < 2) {
    return <p className="text-sm text-muted-foreground">A curva aparece depois da primeira busca.</p>;
  }
  const width = 560;
  const height = 168;
  const pad = 18;
  const values = history.flatMap((point) => [point.best, point.mean]);
  const min = Math.min(...values) - 0.008;
  const max = Math.max(...values) + 0.008;
  const x = (index: number) => pad + (index / (history.length - 1)) * (width - pad * 2);
  const y = (value: number) => height - pad - ((value - min) / (max - min)) * (height - pad * 2);
  const line = (key: "best" | "mean") =>
    history
      .map((point, index) => `${index === 0 ? "M" : "L"} ${x(index).toFixed(1)} ${y(point[key]).toFixed(1)}`)
      .join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-40 w-full" role="img" aria-label="Curva de fitness por geração">
      <path d={line("mean")} fill="none" stroke="#c4b8a8" strokeWidth="2" />
      <path d={line("best")} fill="none" stroke="#0e4f45" strokeWidth="2.5" />
    </svg>
  );
}

export function Meter({
  label,
  original,
  evolved,
}: {
  label: string;
  original: number;
  evolved: number;
}) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-3 text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium text-foreground">
          {(evolved * 100).toLocaleString("pt-BR", { maximumFractionDigits: 1 })}%
        </span>
      </div>
      <div className="space-y-1">
        <div className="h-1.5 overflow-hidden rounded-full bg-[#ebe4da]">
          <div className="h-full rounded-full bg-[#b7aea3]" style={{ width: `${Math.max(2, original * 100)}%` }} />
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-[#ebe4da]">
          <div className="h-full rounded-full bg-[#0e4f45]" style={{ width: `${Math.max(2, evolved * 100)}%` }} />
        </div>
      </div>
    </div>
  );
}
