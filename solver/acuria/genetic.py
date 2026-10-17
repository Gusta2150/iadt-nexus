"""Algoritmo genético real para maximizar a função fitness na validação.

Cada indivíduo é um vetor de genes em [0, 1]. A decodificação em
`encoding.py` transforma esse vetor nos hiperparâmetros do classificador
e no limiar de malignidade. Seleção por torneio, cruzamento e mutação
gaussiana com elitismo. O conjunto de teste não entra aqui.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import time

from acuria.dataset import Bundle
from acuria.encoding import gene_length, default_genes
from acuria.training import Scores, evaluate_genes

CROSSOVERS = ("mistura", "uniforme", "um_ponto")


@dataclass
class GAConfig:
    population_size: int = 18
    generations: int = 12
    mutation_rate: float = 0.18
    crossover_rate: float = 0.9
    elite_count: int = 2
    tournament_size: int = 3
    crossover: str = "mistura"
    seed: int = 42


@dataclass
class HistoryPoint:
    generation: int
    best: float
    mean: float


@dataclass
class GARun:
    genes: list[float]
    validation: Scores
    history: list[HistoryPoint]
    evaluations: int
    elapsed_ms: float
    config: GAConfig
    label: str = "Algoritmo genético"
    cache_hits: int = 0


@dataclass
class EvaluationCache:
    store: dict[tuple[float, ...], Scores] = field(default_factory=dict)
    misses: int = 0
    hits: int = 0

    def get(self, model: str, genes: list[float], bundle: Bundle, seed: int) -> Scores:
        key = tuple(round(gene, 4) for gene in genes)
        found = self.store.get(key)
        if found is not None:
            self.hits += 1
            return found
        scores, _, _ = evaluate_genes(model, genes, bundle, seed)
        self.store[key] = scores
        self.misses += 1
        return scores


def experiment_configs(model: str, seed: int) -> list[tuple[str, GAConfig]]:
    """Três configurações pedidas pelo desafio. A floresta usa populações menores."""
    configs = [
        (
            "Exploratória",
            GAConfig(
                population_size=14,
                generations=8,
                mutation_rate=0.45,
                crossover_rate=0.7,
                elite_count=1,
                tournament_size=2,
                crossover="uniforme",
                seed=seed,
            ),
        ),
        (
            "Balanceada",
            GAConfig(
                population_size=18,
                generations=12,
                mutation_rate=0.18,
                crossover_rate=0.9,
                elite_count=2,
                tournament_size=3,
                crossover="mistura",
                seed=seed,
            ),
        ),
        (
            "Intensiva",
            GAConfig(
                population_size=24,
                generations=16,
                mutation_rate=0.08,
                crossover_rate=0.95,
                elite_count=3,
                tournament_size=4,
                crossover="um_ponto",
                seed=seed,
            ),
        ),
    ]
    if model == "floresta":
        reduced: list[tuple[str, GAConfig]] = []
        for name, config in configs:
            config.population_size = max(8, int(config.population_size * 0.55))
            config.generations = max(4, int(config.generations * 0.5))
            reduced.append((name, config))
        return reduced
    return configs


def run_ga(model: str, bundle: Bundle, config: GAConfig, label: str = "Algoritmo genético") -> GARun:
    started = time.perf_counter()
    rng = random.Random(config.seed)
    cache = EvaluationCache()
    population = _initial(model, config.population_size, rng)
    scored = _rank(model, population, bundle, config.seed, cache)
    history = [_point(0, scored)]
    elite_count = max(1, min(config.elite_count, config.population_size - 1))
    tournament = max(2, min(config.tournament_size, config.population_size))

    for generation in range(1, config.generations + 1):
        elites = [genes[:] for _, genes in scored[:elite_count]]
        children: list[list[float]] = elites[:]
        while len(children) < config.population_size:
            parent_a = _tournament(scored, rng, tournament)
            parent_b = _tournament(scored, rng, tournament)
            if rng.random() < config.crossover_rate:
                child = _crossover(parent_a, parent_b, rng, config.crossover)
            else:
                child = parent_a[:]
            children.append(_mutate(child, config.mutation_rate, rng))
        population = children
        scored = _rank(model, population, bundle, config.seed, cache)
        history.append(_point(generation, scored))

    best_fitness, best_genes = scored[0]
    validation = cache.get(model, best_genes, bundle, config.seed)
    # cache.get no melhor indivíduo é hit; a fitness já está em best_fitness.
    _ = best_fitness
    return GARun(
        genes=best_genes,
        validation=validation,
        history=history,
        evaluations=cache.misses,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        config=config,
        label=label,
        cache_hits=cache.hits,
    )


def _initial(model: str, size: int, rng: random.Random) -> list[list[float]]:
    length = gene_length(model)
    base = _quantize(default_genes(model))
    population = [base[:], _quantize([gene + rng.uniform(-0.08, 0.08) for gene in base])]
    population.append([0.5] * length)
    while len(population) < size:
        population.append(_quantize([rng.random() for _ in range(length)]))
    return population[:size]


def _rank(model: str, population: list[list[float]], bundle: Bundle, seed: int, cache: EvaluationCache):
    scored = [(cache.get(model, genes, bundle, seed).fitness, genes) for genes in population]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def _point(generation: int, scored) -> HistoryPoint:
    values = [fitness for fitness, _ in scored]
    return HistoryPoint(generation=generation, best=values[0], mean=sum(values) / len(values))


def _tournament(scored, rng: random.Random, k: int) -> list[float]:
    picks = [rng.choice(scored) for _ in range(k)]
    return max(picks, key=lambda item: item[0])[1][:]


def _crossover(a: list[float], b: list[float], rng: random.Random, kind: str) -> list[float]:
    if kind == "uniforme":
        return _quantize([x if rng.random() < 0.5 else y for x, y in zip(a, b)])
    if kind == "um_ponto":
        if len(a) < 2:
            return a[:]
        cut = rng.randrange(1, len(a))
        return _quantize(a[:cut] + b[cut:])
    child: list[float] = []
    for x, y in zip(a, b):
        lo, hi = min(x, y), max(x, y)
        span = hi - lo
        if span < 1e-9:
            child.append(x + rng.uniform(-0.05, 0.05))
        else:
            child.append(rng.uniform(lo - 0.3 * span, hi + 0.3 * span))
    return _quantize(child)


def _mutate(genes: list[float], rate: float, rng: random.Random) -> list[float]:
    mutated = genes[:]
    for index, gene in enumerate(mutated):
        if rng.random() >= rate:
            continue
        if rng.random() < 0.2:
            mutated[index] = rng.random()
        else:
            mutated[index] = gene + rng.gauss(0, 0.12)
    return _quantize(mutated)


def _quantize(genes: list[float]) -> list[float]:
    return [round(_clamp(gene), 4) for gene in genes]


def _clamp(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
