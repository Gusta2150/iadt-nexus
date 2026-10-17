"""Treino, limiar e métricas. A busca nunca vê o conjunto de teste."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from acuria.dataset import Bundle
from acuria.encoding import Decoded, decode, default_genes

FITNESS_RECALL = 0.50
FITNESS_F1 = 0.35
FITNESS_PRECISION = 0.15


@dataclass
class Scores:
    accuracy: float
    precision: float
    recall: float
    f1: float
    fitness: float


def fitness_from(recall: float, precision: float, f1: float) -> float:
    return FITNESS_RECALL * recall + FITNESS_F1 * f1 + FITNESS_PRECISION * precision


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Scores:
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    return Scores(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        fitness=fitness_from(recall, precision, f1),
    )


def build_estimator(model: str, decoded: Decoded, seed: int):
    params = dict(decoded.params)
    if model == "logistica":
        clf = LogisticRegression(random_state=seed, **params)
        return Pipeline([("scaler", StandardScaler()), ("model", clf)])
    if model == "arvore":
        return DecisionTreeClassifier(random_state=seed, **params)
    if model == "floresta":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    if model == "knn":
        clf = KNeighborsClassifier(**params)
        return Pipeline([("scaler", StandardScaler()), ("model", clf)])
    raise KeyError(model)


def baseline_decoded(model: str) -> Decoded:
    return decode(model, default_genes(model))


def fit_estimator(model: str, decoded: Decoded, X: np.ndarray, y: np.ndarray, seed: int):
    estimator = build_estimator(model, decoded, seed)
    estimator.fit(X, y)
    return estimator


def positive_proba(estimator, X: np.ndarray) -> np.ndarray:
    model = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
    proba = estimator.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=np.float64)
    return np.asarray(proba[:, classes.index(1)], dtype=np.float64)


def predict_labels(estimator, X: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    proba = positive_proba(estimator, X)
    return (proba >= threshold).astype(np.int64), proba


def evaluate_genes(model: str, genes: list[float], bundle: Bundle, seed: int) -> tuple[Scores, Decoded, object]:
    decoded = decode(model, genes)
    split = bundle.split
    estimator = fit_estimator(model, decoded, split.X_train, split.y_train, seed)
    pred, _ = predict_labels(estimator, split.X_val, decoded.threshold)
    return score_predictions(split.y_val, pred), decoded, estimator


def refit_and_test(model: str, genes: list[float], bundle: Bundle, seed: int):
    """Reajusta no treino+validação e mede no teste, que ficou fora da busca."""
    decoded = decode(model, genes)
    split = bundle.split
    estimator = fit_estimator(model, decoded, split.X_trainval, split.y_trainval, seed)
    pred, proba = predict_labels(estimator, split.X_test, decoded.threshold)
    scores = score_predictions(split.y_test, pred)
    val_est = fit_estimator(model, decoded, split.X_train, split.y_train, seed)
    val_pred, _ = predict_labels(val_est, split.X_val, decoded.threshold)
    validation = score_predictions(split.y_val, val_pred)
    return estimator, decoded, scores, validation, pred, proba


def global_importance(model: str, estimator, bundle: Bundle) -> list[tuple[str, float]]:
    labels = bundle.feature_labels
    if model in {"arvore", "floresta"}:
        raw = np.asarray(estimator.feature_importances_, dtype=np.float64)
    elif model == "logistica":
        raw = np.abs(np.asarray(estimator.named_steps["model"].coef_[0], dtype=np.float64))
    else:
        split = bundle.split
        std = split.X_trainval.std(axis=0) + 1e-9
        mean_pos = split.X_trainval[split.y_trainval == 1].mean(axis=0)
        mean_neg = split.X_trainval[split.y_trainval == 0].mean(axis=0)
        raw = np.abs(mean_pos - mean_neg) / std
    total = float(raw.sum()) or 1.0
    order = np.argsort(raw)[::-1]
    return [(labels[int(index)], float(raw[int(index)] / total)) for index in order]


def local_signals(model: str, estimator, bundle: Bundle, row: np.ndarray, k_note: str | None = None):
    """Sinais locais: valor positivo empurra a leitura para maligno."""
    split = bundle.split
    std = split.X_trainval.std(axis=0) + 1e-9
    mean_pos = split.X_trainval[split.y_trainval == 1].mean(axis=0)
    mean_neg = split.X_trainval[split.y_trainval == 0].mean(axis=0)
    signed = np.zeros(row.shape[0], dtype=np.float64)
    neighbor_note = k_note

    if model == "logistica":
        scaler = estimator.named_steps["scaler"]
        coef = np.asarray(estimator.named_steps["model"].coef_[0], dtype=np.float64)
        z = scaler.transform(row.reshape(1, -1))[0]
        signed = coef * z
    elif model == "knn":
        scaler = estimator.named_steps["scaler"]
        knn = estimator.named_steps["model"]
        z = scaler.transform(row.reshape(1, -1))
        distances, indices = knn.kneighbors(z, return_distance=True)
        votes = split.y_trainval[indices[0]]
        malignant_votes = int(votes.sum())
        neighbor_note = (
            f"Dos {len(votes)} vizinhos mais próximos no treino, {malignant_votes} são malignos "
            f"e {len(votes) - malignant_votes} são benignos."
        )
        midpoint = (mean_pos + mean_neg) / 2
        side = np.sign(row - midpoint) * np.sign(mean_pos - mean_neg)
        signed = side * (np.abs(row - midpoint) / std)
        _ = distances
    else:
        importance = np.asarray(estimator.feature_importances_, dtype=np.float64)
        side = np.where(np.abs(row - mean_pos) <= np.abs(row - mean_neg), 1.0, -1.0)
        signed = importance * side * (np.abs(row - ((mean_pos + mean_neg) / 2)) / std)

    return signed, neighbor_note

