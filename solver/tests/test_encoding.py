from acuria.encoding import decode, default_genes, gene_length
from acuria.genetic import _crossover, _mutate
import random


def test_tamanhos_e_limites_dos_genes():
    for model, size in {"logistica": 3, "arvore": 6, "floresta": 6, "knn": 4}.items():
        assert gene_length(model) == size
        genes = default_genes(model)
        assert len(genes) == size
        assert all(0 <= gene <= 1 for gene in genes)


def test_genes_de_fabrica_decodificam_o_modelo_original():
    logistic = decode("logistica", default_genes("logistica"))
    assert logistic.params["C"] == 1
    assert logistic.params["class_weight"] is None
    assert logistic.threshold == 0.5

    tree = decode("arvore", default_genes("arvore"))
    assert tree.params["max_depth"] is None
    assert tree.params["min_samples_split"] == 2
    assert tree.params["criterion"] == "gini"

    forest = decode("floresta", default_genes("floresta"))
    assert forest.params["n_estimators"] == 100
    assert forest.params["max_features"] == "sqrt"

    knn = decode("knn", default_genes("knn"))
    assert knn.params["n_neighbors"] == 5
    assert knn.params["weights"] == "uniform"
    assert knn.params["p"] == 2


def test_limiar_cobre_o_intervalo_clinico():
    low = decode("logistica", [0.6, 0.0, 0.0])
    high = decode("logistica", [0.6, 0.0, 1.0])
    assert abs(low.threshold - 0.30) < 1e-9
    assert abs(high.threshold - 0.70) < 1e-9


def test_operadores_preservam_vetor_valido():
    rng = random.Random(3)
    parent_a = [0.1, 0.2, 0.3, 0.4]
    parent_b = [0.9, 0.8, 0.7, 0.6]
    for kind in ("mistura", "uniforme", "um_ponto"):
        child = _crossover(parent_a, parent_b, rng, kind)
        assert len(child) == 4
        assert all(0 <= gene <= 1 for gene in child)
    mutated = _mutate(parent_a, 1, rng)
    assert len(mutated) == 4
    assert all(0 <= gene <= 1 for gene in mutated)
    assert set(mutated) != set()
