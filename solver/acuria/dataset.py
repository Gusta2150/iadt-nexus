"""Conjunto público usado no lugar dos modelos do Módulo 1.

O repositório da fase anterior não veio junto com este desafio. As baselines
são classificadores do scikit-learn com hiperparâmetros de fábrica, treinados
no Wisconsin Diagnostic Breast Cancer (WDBC). A classe positiva é maligno.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

STEMS = {
    "radius": ("raio", False),
    "texture": ("textura", True),
    "perimeter": ("perímetro", False),
    "area": ("área", True),
    "smoothness": ("suavidade", True),
    "compactness": ("compacidade", True),
    "concavity": ("concavidade", True),
    "concave points": ("pontos côncavos", False),
    "symmetry": ("simetria", True),
    "fractal dimension": ("dimensão fractal", True),
}


@dataclass(frozen=True)
class FeatureMeta:
    key: str
    label: str
    group: str


@dataclass(frozen=True)
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_trainval: np.ndarray
    y_trainval: np.ndarray


def feature_catalog(keys: list[str]) -> list[FeatureMeta]:
    catalog: list[FeatureMeta] = []
    for key in keys:
        catalog.append(FeatureMeta(key=key, label=_label(key), group=_group(key)))
    return catalog


def _label(key: str) -> str:
    kind, stem = _split_key(key)
    if stem == "concave points":
        if kind == "mean":
            return "pontos côncavos médios"
        if kind == "error":
            return "erro padrão dos pontos côncavos"
        return "maiores pontos côncavos"
    noun, feminine = STEMS[stem]
    if kind == "mean":
        return f"{noun} média" if feminine else f"{noun} médio"
    if kind == "error":
        article = "da" if feminine else "do"
        return f"erro padrão {article} {noun}"
    return f"maior {noun}"


def _group(key: str) -> str:
    kind, _ = _split_key(key)
    return {"mean": "Médias", "error": "Erro padrão", "worst": "Maiores valores"}[kind]


def _split_key(key: str) -> tuple[str, str]:
    if key.startswith("mean "):
        return "mean", key[5:]
    if key.startswith("worst "):
        return "worst", key[6:]
    if key.endswith(" error"):
        return "error", key[: -len(" error")]
    raise KeyError(key)


@dataclass
class Bundle:
    features: list[FeatureMeta]
    split: Split
    malignant: int
    benign: int

    @property
    def feature_labels(self) -> list[str]:
        return [feature.label for feature in self.features]


def load_bundle(seed: int = 42) -> Bundle:
    bunch = load_breast_cancer()
    keys = [str(name) for name in bunch.feature_names]
    # No sklearn, 0 é maligno e 1 é benigno. Invertido para a classe positiva
    # ser a que a clínica não quer perder.
    y = (bunch.target == 0).astype(np.int64)
    X = np.asarray(bunch.data, dtype=np.float64)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=seed
    )
    return Bundle(
        features=feature_catalog(keys),
        split=Split(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            X_trainval=np.asarray(X_trainval),
            y_trainval=np.asarray(y_trainval),
        ),
        malignant=int(y.sum()),
        benign=int((y == 0).sum()),
    )
