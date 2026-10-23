import type {
  CaseRow,
  Crossover,
  LogEvent,
  Metrics,
  ModelId,
  Narrative,
  OptimizeResponse,
  Overview,
} from "@/lib/types";

async function read<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(body?.detail || "O serviço de modelos não respondeu.");
  }
  return response.json() as Promise<T>;
}

export function getOverview(source = "data") {
  const query = new URLSearchParams({ source });
  return fetch(`/solver/api/overview?${query}`, { cache: "no-store" }).then((response) => read<Overview>(response));
}

export function optimize(input: {
  model: ModelId;
  population_size: number;
  generations: number;
  mutation_rate: number;
  crossover: Crossover;
  seed: number;
  experiments: boolean;
  source: string;
}) {
  return fetch("/solver/api/optimize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...input, crossover_rate: 0.9, elite_count: 2, tournament_size: 3 }),
  }).then((response) => read<OptimizeResponse>(response));
}

export function getCases(runId: string) {
  return fetch(`/solver/api/runs/${runId}/cases`, { cache: "no-store" }).then((response) =>
    read<{ cases: CaseRow[] }>(response),
  );
}

export function explainCase(runId: string, caseIndex: number) {
  return fetch(`/solver/api/runs/${runId}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ case_index: caseIndex }),
  }).then((response) => read<Narrative>(response));
}

export function explainReport(runId: string) {
  return fetch(`/solver/api/runs/${runId}/report`, { method: "POST" }).then((response) =>
    read<Narrative>(response),
  );
}

export function getMetrics() {
  return fetch("/solver/api/metrics", { cache: "no-store" }).then((response) => read<Metrics>(response));
}

export function getLogs() {
  return fetch("/solver/api/logs", { cache: "no-store" }).then((response) =>
    read<{ events: LogEvent[] }>(response),
  );
}
