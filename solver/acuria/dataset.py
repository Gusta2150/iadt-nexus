"""Bases do Módulo 1 e o recorte usado pela busca.

data.csv é a coorte humana (569 exames de núcleo celular). cancer_prostata_data.csv
é outra coorte humana, menor. Animais e os dois arquivos de câncer de mama eram as
bases falsas: a triagem recusa esses arquivos e o diagnóstico não roda neles.
A classe positiva, quando a base é aceita, é maligno.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

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


@dataclass(frozen=True)
class SourceSpec:
    id: str
    filename: str
    title: str
    human: bool
    detail: str
    samples: int
    features: int
    malignant: int
    benign: int


class OriginError(RuntimeError):
    """A base escolhida não passa na triagem do Módulo 1."""


_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        id="data",
        filename="data.csv",
        title="data.csv",
        human=True,
        detail=(
            "Coorte humana do Módulo 1. 569 exames, 30 medidas de núcleo celular, "
            "com maligno e benigno. É neste arquivo que o diagnóstico segue."
        ),
        samples=569,
        features=30,
        malignant=212,
        benign=357,
    ),
    SourceSpec(
        id="prostata",
        filename="cancer_prostata_data.csv",
        title="Câncer de próstata",
        human=True,
        detail=(
            "Outra coorte humana do Módulo 1: 100 casos de próstata, 8 medidas e "
            "diagnóstico benigno ou maligno. A busca pode rodar aqui."
        ),
        samples=100,
        features=8,
        malignant=40,
        benign=60,
    ),
    SourceSpec(
        id="animais",
        filename="animais_data.csv",
        title="Animais",
        human=False,
        detail=(
            "Base falsa do Módulo 1. As colunas são de animal (pelo, penas, ovos, leite, patas), "
            "não de exame humano. O diagnóstico não roda neste arquivo."
        ),
        samples=101,
        features=16,
        malignant=0,
        benign=0,
    ),
    SourceSpec(
        id="mama",
        filename="cancer_mama_1_diagnostico.csv",
        title="Câncer de mama",
        human=False,
        detail=(
            "Base falsa do Módulo 1. São 212 linhas e só a classe maligna. "
            "No módulo anterior esse arquivo servia para a triagem recusar a origem. "
            "A coorte humana é o data.csv."
        ),
        samples=212,
        features=30,
        malignant=212,
        benign=0,
    ),
    SourceSpec(
        id="amostra",
        filename="breast_cancer_sample.csv",
        title="Amostra de mama",
        human=False,
        detail=(
            "Base falsa do Módulo 1. Amostra de 5 exames de mama, com colunas incompletas. "
            "Não é a coorte humana."
        ),
        samples=5,
        features=12,
        malignant=3,
        benign=2,
    ),
)


def list_sources() -> tuple[SourceSpec, ...]:
    return _SPECS


def get_source(source_id: str) -> SourceSpec:
    for spec in _SPECS:
        if spec.id == source_id:
            return spec
    raise KeyError(source_id)


def load_bundle(seed: int = 42, source: str = "data") -> Bundle:
    spec = get_source(source)
    if not spec.human:
        raise OriginError(spec.detail)
    header, body = _read_table(spec.filename)
    if spec.id == "prostata":
        return _bundle_prostate(header, body, seed)
    return _bundle_nucleus(header, body, seed)


def _bundle_nucleus(header: list[str], body: list[list[str]], seed: int) -> Bundle:
    label_at = header.index("diagnosis")
    feature_at = [index for index, name in enumerate(header) if name not in {"id", "diagnosis"}]
    keys = [_nucleus_key(header[index]) for index in feature_at]
    X = _matrix(body, feature_at)
    y = np.array([1 if row[label_at] == "M" else 0 for row in body], dtype=np.int64)
    return _pack(feature_catalog(keys), X, y, seed)


def _bundle_prostate(header: list[str], body: list[list[str]], seed: int) -> Bundle:
    label_at = header.index("diagnosis_result")
    names = {
        "radius": "raio",
        "texture": "textura",
        "perimeter": "perímetro",
        "area": "área",
        "smoothness": "suavidade",
        "compactness": "compacidade",
        "symmetry": "simetria",
        "fractal_dimension": "dimensão fractal",
    }
    feature_at = [index for index, name in enumerate(header) if name not in {"id", "diagnosis_result"}]
    features = [
        FeatureMeta(key=header[index], label=names[header[index]], group="Medidas") for index in feature_at
    ]
    X = _matrix(body, feature_at)
    y = np.array([1 if row[label_at] == "M" else 0 for row in body], dtype=np.int64)
    return _pack(features, X, y, seed)


def _pack(features: list[FeatureMeta], X: np.ndarray, y: np.ndarray, seed: int) -> Bundle:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=seed
    )
    return Bundle(
        features=features,
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


def _matrix(body: list[list[str]], columns: list[int]) -> np.ndarray:
    return np.array([[float(row[index]) for index in columns] for row in body], dtype=np.float64)


def _nucleus_key(column: str) -> str:
    if column.endswith("_mean"):
        return "mean " + column[: -len("_mean")].replace("_", " ")
    if column.endswith("_se"):
        return column[: -len("_se")].replace("_", " ") + " error"
    if column.endswith("_worst"):
        return "worst " + column[: -len("_worst")].replace("_", " ")
    raise KeyError(column)


def _read_table(filename: str) -> tuple[list[str], list[list[str]]]:
    path = DATA_DIR / filename
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    header = [cell.strip() for cell in rows[0]]
    while header and header[-1] == "":
        header.pop()
    body: list[list[str]] = []
    width = len(header)
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        body.append(row[:width])
    return header, body
