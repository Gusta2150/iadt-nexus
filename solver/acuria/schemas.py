"""Contratos da API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ScoresOut(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    fitness: float


class ParamOut(BaseModel):
    label: str
    value: str


class HistoryOut(BaseModel):
    generation: int
    best: float
    mean: float


class ImportanceOut(BaseModel):
    label: str
    share: float


class RunSummary(BaseModel):
    label: str
    validation: ScoresOut
    test: ScoresOut
    params: list[ParamOut]
    threshold: float
    evaluations: int
    elapsed_ms: float
    history: list[HistoryOut] = Field(default_factory=list)
    population_size: int | None = None
    generations: int | None = None
    mutation_rate: float | None = None
    crossover: str | None = None


class OptimizeRequest(BaseModel):
    model: Literal["logistica", "arvore", "floresta", "knn"] = "logistica"
    population_size: int = Field(18, ge=6, le=28)
    generations: int = Field(12, ge=2, le=20)
    mutation_rate: float = Field(0.18, ge=0, le=1)
    crossover_rate: float = Field(0.9, ge=0, le=1)
    elite_count: int = Field(2, ge=1, le=6)
    tournament_size: int = Field(3, ge=2, le=6)
    crossover: Literal["mistura", "uniforme", "um_ponto"] = "mistura"
    seed: int = Field(42, ge=0, le=10_000_000)
    experiments: bool = False
    source: str = Field("data", min_length=1, max_length=40)


class OptimizeResponse(BaseModel):
    run_id: str
    model: str
    model_label: str
    chosen_label: str
    runs: list[RunSummary]
    importance: list[ImportanceOut]
    evaluations: int
    elapsed_ms: float
    seed: int


class SplitOut(BaseModel):
    train: int
    validation: int
    test: int
    malignant_train: int
    malignant_validation: int
    malignant_test: int


class BaselineOut(BaseModel):
    model: str
    label: str
    blurb: str
    validation: ScoresOut
    test: ScoresOut
    params: list[ParamOut]


class SourceOut(BaseModel):
    id: str
    filename: str
    title: str
    human: bool
    samples: int
    detail: str


class OverviewResponse(BaseModel):
    dataset: str
    source: str
    active_source: str
    accepted: bool
    rejection: str | None = None
    samples: int
    features: int
    malignant: int
    benign: int
    positive_label: str
    note: str
    split: SplitOut
    baselines: list[BaselineOut]
    fitness: str
    sources: list[SourceOut]


class FeatureOut(BaseModel):
    label: str
    group: str
    value: float


class CaseOut(BaseModel):
    index: int
    actual: Literal["maligno", "benigno"]
    predicted: Literal["maligno", "benigno"]
    probability: float
    correct: bool
    features: list[FeatureOut]


class CasesResponse(BaseModel):
    run_id: str
    model_label: str
    cases: list[CaseOut]


class ExplainRequest(BaseModel):
    case_index: int = Field(ge=0)


class CheckOut(BaseModel):
    name: str
    passed: bool
    detail: str


class QualityOut(BaseModel):
    score: int
    summary: str
    checks: list[CheckOut]


class NarrativeResponse(BaseModel):
    text: str
    source: Literal["llm", "local"]
    warning: str | None = None
    prompt: str
    quality: QualityOut


class MetricsResponse(BaseModel):
    started_at: str
    requests: int
    optimizations: int
    rejected_busy: int
    failures: int
    evaluations: int
    last_model: str | None
    last_elapsed_ms: float | None
    last_fitness: float | None


class LogEvent(BaseModel):
    at: str
    message: str


class LogsResponse(BaseModel):
    events: list[LogEvent]


def scores_out(scores: Any) -> ScoresOut:
    return ScoresOut(
        accuracy=round(float(scores.accuracy), 6),
        precision=round(float(scores.precision), 6),
        recall=round(float(scores.recall), 6),
        f1=round(float(scores.f1), 6),
        fitness=round(float(scores.fitness), 6),
    )
