"""Demonstração da busca. Rode a partir da raiz do repositório."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solver"))

from acuria.schemas import OptimizeRequest
from acuria.service import Lab


def line(run) -> str:
    return (
        f"{run.label:22}  val {run.validation.fitness:.3f}  "
        f"teste rec {run.test.recall:.3f}  prec {run.test.precision:.3f}  "
        f"f1 {run.test.f1:.3f}  aval {run.evaluations}"
    )


def main() -> None:
    lab = Lab()
    overview = lab.overview()
    print(f"WDBC  treino {overview.split.train}  validação {overview.split.validation}  teste {overview.split.test}")
    print("Modelos originais no teste")
    for baseline in overview.baselines:
        print(
            f"  {baseline.label:28} rec {baseline.test.recall:.3f}  "
            f"prec {baseline.test.precision:.3f}  f1 {baseline.test.f1:.3f}"
        )
    for model, experiments in (("knn", False), ("logistica", True)):
        result = lab.optimize(OptimizeRequest(model=model, experiments=experiments, seed=42))
        print(f"\n{result.model_label} · destaque {result.chosen_label} · {result.evaluations} avaliações")
        for run in result.runs:
            print(" ", line(run))
        reading = lab.explain(result.run_id, 0)
        print(f"Leitura do caso 1 ({reading.source}, qualidade {reading.quality.score})")
        print(reading.text.split("\n\n")[0])


if __name__ == "__main__":
    main()
