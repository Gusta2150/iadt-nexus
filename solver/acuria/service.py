"""Orquestra a busca, guarda o último modelo em memória e registra o desempenho."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4
import logging

import numpy as np

from acuria.dataset import Bundle, OriginError, get_source, list_sources, load_bundle
from acuria.encoding import LABELS, BLURBS, Decoded
from acuria.genetic import GAConfig, GARun, experiment_configs, run_ga
from acuria.language import narrate
from acuria.schemas import (
    BaselineOut,
    CasesResponse,
    CaseOut,
    FeatureOut,
    HistoryOut,
    ImportanceOut,
    MetricsResponse,
    NarrativeResponse,
    OptimizeRequest,
    OptimizeResponse,
    OverviewResponse,
    ParamOut,
    RunSummary,
    SourceOut,
    SplitOut,
    scores_out,
)
from acuria.training import global_importance, local_signals, refit_and_test

logger = logging.getLogger("acuria")

FITNESS_TEXT = (
    "50% recall de malignidade + 35% F1 + 15% precisão, medida só no conjunto de validação. "
    "O teste fica de fora da busca."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%".replace(".", ",")


def _params(decoded: Decoded) -> list[ParamOut]:
    return [ParamOut(label=label, value=value) for label, value in decoded.readable]


@dataclass
class StoredRun:
    model: str
    chosen_label: str
    estimator: object
    decoded: Decoded
    test_pred: np.ndarray
    test_proba: np.ndarray
    runs: list[RunSummary]
    importance: list[ImportanceOut]
    seed: int
    bundle: Bundle


@dataclass
class Monitor:
    started_at: str = field(default_factory=_now)
    requests: int = 0
    optimizations: int = 0
    rejected_busy: int = 0
    failures: int = 0
    evaluations: int = 0
    last_model: str | None = None
    last_elapsed_ms: float | None = None
    last_fitness: float | None = None
    events: list[dict[str, str]] = field(default_factory=list)

    def event(self, message: str) -> None:
        logger.info(message)
        self.events.append({"at": _now(), "message": message})
        self.events = self.events[-40:]


class Lab:
    def __init__(self) -> None:
        self.monitor = Monitor()
        self._bundles: dict[str, Bundle] = {}
        self._overviews: dict[str, OverviewResponse] = {}
        self._runs: dict[str, StoredRun] = {}
        self._guard = Lock()
        self._busy = Lock()

    def overview(self, source: str = "data") -> OverviewResponse:
        cached = self._overviews.get(source)
        if cached is not None:
            return cached
        with self._guard:
            cached = self._overviews.get(source)
            if cached is None:
                cached = self._build_overview(source)
                self._overviews[source] = cached
        return cached

    def _build_overview(self, source: str) -> OverviewResponse:
        spec = self._spec(source)
        sources = _source_list()
        if not spec.human:
            self.monitor.event(f"Triagem recusou {spec.filename}: base falsa do Módulo 1.")
            return OverviewResponse(
                dataset=spec.title,
                source=spec.filename,
                active_source=spec.id,
                accepted=False,
                rejection=spec.detail,
                samples=spec.samples,
                features=spec.features,
                malignant=spec.malignant,
                benign=spec.benign,
                positive_label="maligno",
                note=spec.detail,
                split=SplitOut(
                    train=0,
                    validation=0,
                    test=0,
                    malignant_train=0,
                    malignant_validation=0,
                    malignant_test=0,
                ),
                baselines=[],
                fitness=FITNESS_TEXT,
                sources=sources,
            )
        bundle = self._bundle(spec.id)
        split = bundle.split
        baselines: list[BaselineOut] = []
        for model in ("logistica", "arvore", "floresta", "knn"):
            _estimator, decoded, test, validation, _pred, _proba = refit_and_test(
                model, baseline_genes(model), bundle, 42
            )
            baselines.append(
                BaselineOut(
                    model=model,
                    label=LABELS[model],
                    blurb=BLURBS[model],
                    validation=scores_out(validation),
                    test=scores_out(test),
                    params=_params(decoded),
                )
            )
        self.monitor.event(f"Baselines calculadas em {spec.filename}.")
        cohort = "data.csv, a coorte humana" if spec.id == "data" else spec.filename
        return OverviewResponse(
            dataset=spec.title,
            source=spec.filename,
            active_source=spec.id,
            accepted=True,
            rejection=None,
            samples=int(split.y_train.size + split.y_val.size + split.y_test.size),
            features=len(bundle.features),
            malignant=bundle.malignant,
            benign=bundle.benign,
            positive_label="maligno",
            note=(
                f"Triagem do Módulo 1 aceitou {cohort}. Os modelos treinados do Módulo 1 "
                "não vieram nos arquivos: as baselines são os quatro classificadores de fábrica "
                "neste recorte. Animais e os arquivos de câncer de mama marcados como base falsa "
                "não entram na busca."
            ),
            split=SplitOut(
                train=int(split.y_train.size),
                validation=int(split.y_val.size),
                test=int(split.y_test.size),
                malignant_train=int(split.y_train.sum()),
                malignant_validation=int(split.y_val.sum()),
                malignant_test=int(split.y_test.sum()),
            ),
            baselines=baselines,
            fitness=FITNESS_TEXT,
            sources=sources,
        )

    def _spec(self, source: str):
        try:
            return get_source(source)
        except KeyError as exc:
            raise MissingError("Essa base não está no Módulo 1.") from exc

    def _bundle(self, source: str) -> Bundle:
        cached = self._bundles.get(source)
        if cached is None:
            cached = load_bundle(42, source)
            self._bundles[source] = cached
        return cached

    def optimize(self, request: OptimizeRequest) -> OptimizeResponse:
        if not self._busy.acquire(blocking=False):
            self.monitor.rejected_busy += 1
            self.monitor.event("Otimização recusada: já havia uma busca em andamento.")
            raise BusyError("Já existe uma otimização em andamento. Espere ela terminar.")
        try:
            return self._optimize(request)
        finally:
            self._busy.release()

    def _optimize(self, request: OptimizeRequest) -> OptimizeResponse:
        spec = self._spec(request.source)
        if not spec.human:
            self.monitor.event(f"Busca recusada em {spec.filename}: a triagem não aceita base falsa.")
            raise OriginError(spec.detail)
        bundle = self._bundle(spec.id)
        started_evals = 0
        runs: list[tuple[GARun, object, Decoded, object, object]] = []
        configs = (
            experiment_configs(request.model, request.seed)
            if request.experiments
            else [("Algoritmo genético", _config_from(request))]
        )
        self.monitor.event(
            f"Busca iniciada no modelo {LABELS[request.model]} "
            f"({'3 configurações' if request.experiments else '1 configuração'}, semente {request.seed})."
        )
        for label, config in configs:
            ga_run = run_ga(request.model, bundle, config, label=label)
            estimator, decoded, test, validation, pred, proba = refit_and_test(
                request.model, ga_run.genes, bundle, request.seed
            )
            # A validação publicada é a do próprio AG, no ajuste só com o treino.
            ga_run.validation = validation
            runs.append((ga_run, test, decoded, pred, proba, estimator))
            started_evals += ga_run.evaluations
            self.monitor.event(
                f"{label}: fitness de validação {validation.fitness:.4f} em {ga_run.evaluations} avaliações."
            )

        original_estimator, original_decoded, original_test, original_val, _op, _ob = refit_and_test(
            request.model, baseline_genes(request.model), bundle, request.seed
        )
        _ = original_estimator
        summaries = [
            RunSummary(
                label="Modelo original",
                validation=scores_out(original_val),
                test=scores_out(original_test),
                params=_params(original_decoded),
                threshold=original_decoded.threshold,
                evaluations=0,
                elapsed_ms=0,
            )
        ]
        packed = []
        for ga_run, test, decoded, pred, proba, estimator in runs:
            summary = RunSummary(
                label=ga_run.label,
                validation=scores_out(ga_run.validation),
                test=scores_out(test),
                params=_params(decoded),
                threshold=decoded.threshold,
                evaluations=ga_run.evaluations,
                elapsed_ms=round(ga_run.elapsed_ms, 1),
                history=[
                    HistoryOut(generation=point.generation, best=round(point.best, 6), mean=round(point.mean, 6))
                    for point in ga_run.history
                ],
                population_size=ga_run.config.population_size,
                generations=ga_run.config.generations,
                mutation_rate=ga_run.config.mutation_rate,
                crossover=ga_run.config.crossover,
            )
            summaries.append(summary)
            packed.append((summary, decoded, pred, proba, estimator))

        chosen = max(
            packed,
            key=lambda item: (item[0].validation.fitness, item[0].test.f1, item[0].test.recall),
        )
        summary, decoded, pred, proba, estimator = chosen
        importance = [
            ImportanceOut(label=label, share=round(share, 6))
            for label, share in global_importance(request.model, estimator, bundle)[:8]
        ]
        run_id = uuid4().hex[:8]
        elapsed = round(sum(item.elapsed_ms for item in summaries), 1)
        with self._guard:
            self._runs[run_id] = StoredRun(
                model=request.model,
                chosen_label=summary.label,
                estimator=estimator,
                decoded=decoded,
                test_pred=pred,
                test_proba=proba,
                runs=summaries,
                importance=importance,
                seed=request.seed,
                bundle=bundle,
            )
            if len(self._runs) > 8:
                oldest = next(iter(self._runs))
                del self._runs[oldest]
        self.monitor.optimizations += 1
        self.monitor.evaluations += started_evals
        self.monitor.last_model = request.model
        self.monitor.last_elapsed_ms = elapsed
        self.monitor.last_fitness = summary.validation.fitness
        self.monitor.event(
            f"Busca {run_id} concluída. Configuração destacada: {summary.label}, "
            f"fitness {summary.validation.fitness:.4f}."
        )
        return OptimizeResponse(
            run_id=run_id,
            model=request.model,
            model_label=LABELS[request.model],
            chosen_label=summary.label,
            runs=summaries,
            importance=importance,
            evaluations=started_evals,
            elapsed_ms=elapsed,
            seed=request.seed,
        )

    def cases(self, run_id: str) -> CasesResponse:
        stored = self._get(run_id)
        split = stored.bundle.split
        rows: list[CaseOut] = []
        for index in range(split.y_test.size):
            actual = "maligno" if int(split.y_test[index]) == 1 else "benigno"
            predicted = "maligno" if int(stored.test_pred[index]) == 1 else "benigno"
            features = [
                FeatureOut(label=feature.label, group=feature.group, value=float(split.X_test[index, column]))
                for column, feature in enumerate(stored.bundle.features)
            ]
            rows.append(
                CaseOut(
                    index=index,
                    actual=actual,
                    predicted=predicted,
                    probability=round(float(stored.test_proba[index]), 6),
                    correct=actual == predicted,
                    features=features,
                )
            )
        return CasesResponse(run_id=run_id, model_label=LABELS[stored.model], cases=rows)

    def explain(self, run_id: str, case_index: int) -> NarrativeResponse:
        stored = self._get(run_id)
        context = self._case_context(stored, case_index)
        payload = narrate("caso", context)
        self.monitor.event(
            f"Leitura do caso {case_index} na busca {run_id}: origem {payload['source']}, "
            f"qualidade {payload['quality']['score']}."
        )
        return NarrativeResponse(**payload)

    def report(self, run_id: str) -> NarrativeResponse:
        stored = self._get(run_id)
        context = self._report_context(stored)
        payload = narrate("relatorio", context)
        self.monitor.event(
            f"Relatório da busca {run_id}: origem {payload['source']}, qualidade {payload['quality']['score']}."
        )
        return NarrativeResponse(**payload)

    def _case_context(self, stored: StoredRun, case_index: int) -> dict:
        split = stored.bundle.split
        if case_index < 0 or case_index >= split.y_test.size:
            raise MissingError("Esse caso não está no conjunto de teste.")
        row = split.X_test[case_index]
        signed, neighbor = local_signals(stored.model, stored.estimator, stored.bundle, row)
        order = np.argsort(np.abs(signed))[::-1][:4]
        top = []
        for column in order:
            feature = stored.bundle.features[int(column)]
            value = float(row[int(column)])
            top.append(
                {
                    "label": feature.label,
                    "value": value,
                    "value_label": _value_label(value),
                    "direction": "maligno" if signed[int(column)] >= 0 else "benigno",
                }
            )
        chosen = next(item for item in stored.runs if item.label == stored.chosen_label)
        probability = float(stored.test_proba[case_index])
        predicted = "Maligno" if int(stored.test_pred[case_index]) == 1 else "Benigno"
        return {
            "model_label": LABELS[stored.model],
            "predicted_label": predicted,
            "probability": probability,
            "probability_label": _pct(probability),
            "threshold_label": _pct(stored.decoded.threshold),
            "recall_label": _pct(chosen.test.recall),
            "precision_label": _pct(chosen.test.precision),
            "f1_label": _pct(chosen.test.f1),
            "params": [item.model_dump() for item in chosen.params],
            "top_features": top,
            "neighbor_note": neighbor,
        }

    def _report_context(self, stored: StoredRun) -> dict:
        rows = []
        for item in stored.runs:
            rows.append(
                {
                    "label": item.label,
                    "validation_fitness": f"{item.validation.fitness:.3f}".replace(".", ","),
                    "test_recall": _pct(item.test.recall),
                    "test_precision": _pct(item.test.precision),
                    "test_f1": _pct(item.test.f1),
                    "test_recall_raw": item.test.recall,
                    "test_precision_raw": item.test.precision,
                }
            )
        chosen = next(item for item in stored.runs if item.label == stored.chosen_label)
        return {
            "model_label": LABELS[stored.model],
            "chosen_label": stored.chosen_label,
            "seed": stored.seed,
            "evaluations": sum(item.evaluations for item in stored.runs),
            "elapsed_label": f"{sum(item.elapsed_ms for item in stored.runs) / 1000:.1f} s".replace(".", ","),
            "rows": rows,
            "params": [item.model_dump() for item in chosen.params],
            "importance": [
                {"label": item.label, "share_label": _pct(item.share)} for item in stored.importance
            ],
        }

    def _get(self, run_id: str) -> StoredRun:
        stored = self._runs.get(run_id)
        if stored is None:
            raise MissingError("Essa busca não está mais em memória. Rode a otimização de novo.")
        return stored

    def metrics(self) -> MetricsResponse:
        monitor = self.monitor
        return MetricsResponse(
            started_at=monitor.started_at,
            requests=monitor.requests,
            optimizations=monitor.optimizations,
            rejected_busy=monitor.rejected_busy,
            failures=monitor.failures,
            evaluations=monitor.evaluations,
            last_model=monitor.last_model,
            last_elapsed_ms=monitor.last_elapsed_ms,
            last_fitness=monitor.last_fitness,
        )

    def logs(self) -> list[dict[str, str]]:
        return list(self.monitor.events)


class BusyError(RuntimeError):
    pass


class MissingError(RuntimeError):
    pass


def _source_list() -> list[SourceOut]:
    return [
        SourceOut(
            id=spec.id,
            filename=spec.filename,
            title=spec.title,
            human=spec.human,
            samples=spec.samples,
            detail=spec.detail,
        )
        for spec in list_sources()
    ]


def baseline_genes(model: str) -> list[float]:
    from acuria.encoding import default_genes

    return default_genes(model)


def _config_from(request: OptimizeRequest) -> GAConfig:
    return GAConfig(
        population_size=request.population_size,
        generations=request.generations,
        mutation_rate=request.mutation_rate,
        crossover_rate=request.crossover_rate,
        elite_count=request.elite_count,
        tournament_size=request.tournament_size,
        crossover=request.crossover,
        seed=request.seed,
    )


def _value_label(value: float) -> str:
    if abs(value) >= 100:
        text = f"{value:.1f}"
    elif abs(value) >= 10:
        text = f"{value:.2f}"
    else:
        text = f"{value:.3f}"
    return text.replace(".", ",")
