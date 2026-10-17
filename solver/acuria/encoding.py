"""Codificação dos hiperparâmetros como genes no intervalo [0, 1]."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MODELS = ("logistica", "arvore", "floresta", "knn")

LABELS = {
    "logistica": "Regressão logística",
    "arvore": "Árvore de decisão",
    "floresta": "Floresta aleatória",
    "knn": "K vizinhos mais próximos",
}

BLURBS = {
    "logistica": "Coeficientes legíveis. A busca mexe na regularização, no peso das classes e no limiar.",
    "arvore": "Uma árvore só, rápida de treinar. A busca mexe na profundidade e no critério de divisão.",
    "floresta": "Várias árvores. Costuma acertar mais e demora mais para cada indivíduo da população.",
    "knn": "Compara o caso com os exames mais parecidos do treino. A busca escolhe quantos vizinhos olhar.",
}


@dataclass
class Decoded:
    params: dict[str, Any]
    threshold: float
    readable: list[tuple[str, str]] = field(default_factory=list)


def gene_length(model: str) -> int:
    return {"logistica": 3, "arvore": 6, "floresta": 6, "knn": 4}[model]


def default_genes(model: str) -> list[float]:
    """Ponto do espaço que reproduz os hiperparâmetros de fábrica, com limiar 0,50."""
    if model == "logistica":
        return [0.6, 0.0, 0.5]
    if model == "arvore":
        return [0.95, 0.0, 0.0, 0.0, 0.0, 0.5]
    if model == "floresta":
        return [1.0, 0.95, 0.0, 0.0, 0.0, 0.5]
    if model == "knn":
        return [4 / 30, 0.0, 0.0, 0.5]
    raise KeyError(model)


def decode(model: str, genes: list[float]) -> Decoded:
    values = _fit(genes, gene_length(model))
    if model == "logistica":
        return _logistic(values)
    if model == "arvore":
        return _tree(values)
    if model == "floresta":
        return _forest(values)
    if model == "knn":
        return _knn(values)
    raise KeyError(model)


def _logistic(genes: list[float]) -> Decoded:
    c_value = float(10 ** (-3 + genes[0] * 5))
    class_weight = "balanced" if genes[1] >= 0.5 else None
    threshold = _threshold(genes[2])
    weight_label = "balanceado" if class_weight else "frequência natural"
    return Decoded(
        params={"C": c_value, "class_weight": class_weight, "solver": "lbfgs", "max_iter": 800},
        threshold=threshold,
        readable=[
            ("Regularização C", _pt(c_value, 3)),
            ("Peso das classes", weight_label),
            ("Limiar de malignidade", _pt(threshold, 2)),
        ],
    )


def _tree(genes: list[float]) -> Decoded:
    depth = _depth(genes[0])
    min_split = int(round(2 + genes[1] * 18))
    min_leaf = int(round(1 + genes[2] * 9))
    criterion = "entropy" if genes[3] >= 0.5 else "gini"
    class_weight = "balanced" if genes[4] >= 0.5 else None
    threshold = _threshold(genes[5])
    return Decoded(
        params={
            "max_depth": depth,
            "min_samples_split": min_split,
            "min_samples_leaf": min_leaf,
            "criterion": criterion,
            "class_weight": class_weight,
        },
        threshold=threshold,
        readable=[
            ("Profundidade", "livre" if depth is None else str(depth)),
            ("Mínimo para dividir", str(min_split)),
            ("Mínimo na folha", str(min_leaf)),
            ("Critério", "entropia" if criterion == "entropy" else "Gini"),
            ("Peso das classes", "balanceado" if class_weight else "frequência natural"),
            ("Limiar de malignidade", _pt(threshold, 2)),
        ],
    )


def _forest(genes: list[float]) -> Decoded:
    trees = int(round(30 + genes[0] * 70))
    depth = _depth(genes[1])
    min_split = int(round(2 + genes[2] * 8))
    if genes[3] < 0.34:
        max_features: str | float = "sqrt"
        features_label = "raiz do número de variáveis"
    elif genes[3] < 0.67:
        max_features = "log2"
        features_label = "log2 do número de variáveis"
    else:
        max_features = 0.5
        features_label = "metade das variáveis"
    class_weight = "balanced" if genes[4] >= 0.5 else None
    threshold = _threshold(genes[5])
    return Decoded(
        params={
            "n_estimators": trees,
            "max_depth": depth,
            "min_samples_split": min_split,
            "max_features": max_features,
            "class_weight": class_weight,
        },
        threshold=threshold,
        readable=[
            ("Árvores", str(trees)),
            ("Profundidade", "livre" if depth is None else str(depth)),
            ("Mínimo para dividir", str(min_split)),
            ("Variáveis por divisão", features_label),
            ("Peso das classes", "balanceado" if class_weight else "frequência natural"),
            ("Limiar de malignidade", _pt(threshold, 2)),
        ],
    )


def _knn(genes: list[float]) -> Decoded:
    neighbors = int(round(1 + genes[0] * 30))
    weights = "distance" if genes[1] >= 0.5 else "uniform"
    minkowski = 1 if genes[2] >= 0.5 else 2
    threshold = _threshold(genes[3])
    return Decoded(
        params={"n_neighbors": neighbors, "weights": weights, "p": minkowski},
        threshold=threshold,
        readable=[
            ("Vizinhos", str(neighbors)),
            ("Peso", "pela distância" if weights == "distance" else "uniforme"),
            ("Distância", "Manhattan" if minkowski == 1 else "euclidiana"),
            ("Limiar de malignidade", _pt(threshold, 2)),
        ],
    )


def _depth(gene: float) -> int | None:
    if gene >= 0.85:
        return None
    depth = int(round(2 + (gene / 0.85) * 14))
    return max(2, min(16, depth))


def _threshold(gene: float) -> float:
    return float(0.30 + _clamp(gene) * 0.40)


def _fit(genes: list[float], size: int) -> list[float]:
    values = [_clamp(gene) for gene in genes[:size]]
    if len(values) < size:
        values.extend([0.5] * (size - len(values)))
    return values


def _clamp(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _pt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")
