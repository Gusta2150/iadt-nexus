export type ModelId = "logistica" | "arvore" | "floresta" | "knn";
export type Crossover = "mistura" | "uniforme" | "um_ponto";

export type Scores = {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  fitness: number;
};

export type Param = { label: string; value: string };

export type Baseline = {
  model: ModelId;
  label: string;
  blurb: string;
  validation: Scores;
  test: Scores;
  params: Param[];
};

export type Overview = {
  dataset: string;
  source: string;
  samples: number;
  features: number;
  malignant: number;
  benign: number;
  positive_label: string;
  note: string;
  split: {
    train: number;
    validation: number;
    test: number;
    malignant_train: number;
    malignant_validation: number;
    malignant_test: number;
  };
  baselines: Baseline[];
  fitness: string;
};

export type HistoryPoint = { generation: number; best: number; mean: number };

export type RunSummary = {
  label: string;
  validation: Scores;
  test: Scores;
  params: Param[];
  threshold: number;
  evaluations: number;
  elapsed_ms: number;
  history: HistoryPoint[];
  population_size: number | null;
  generations: number | null;
  mutation_rate: number | null;
  crossover: string | null;
};

export type OptimizeResponse = {
  run_id: string;
  model: ModelId;
  model_label: string;
  chosen_label: string;
  runs: RunSummary[];
  importance: { label: string; share: number }[];
  evaluations: number;
  elapsed_ms: number;
  seed: number;
};

export type CaseRow = {
  index: number;
  actual: "maligno" | "benigno";
  predicted: "maligno" | "benigno";
  probability: number;
  correct: boolean;
  features: { label: string; group: string; value: number }[];
};

export type Narrative = {
  text: string;
  source: "llm" | "local";
  warning: string | null;
  prompt: string;
  quality: {
    score: number;
    summary: string;
    checks: { name: string; passed: boolean; detail: string }[];
  };
};

export type Metrics = {
  started_at: string;
  requests: number;
  optimizations: number;
  rejected_busy: number;
  failures: number;
  evaluations: number;
  last_model: string | null;
  last_elapsed_ms: number | null;
  last_fitness: number | null;
};

export type LogEvent = { at: string; message: string };
