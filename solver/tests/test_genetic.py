from acuria.dataset import load_bundle
from acuria.encoding import default_genes
from acuria.genetic import GAConfig, run_ga
from acuria.training import evaluate_genes, fitness_from


def test_formula_da_fitness_prioriza_recall():
    assert fitness_from(1, 0, 0) == 0.5
    assert fitness_from(1, 1, 1) == 1
    assert fitness_from(1, 0.5, 0.5) > fitness_from(0.5, 1, 0.5)


def test_recorte_estratificado_soma_o_conjunto():
    bundle = load_bundle()
    split = bundle.split
    assert split.y_train.size + split.y_val.size + split.y_test.size == 569
    assert int(split.y_train.sum() + split.y_val.sum() + split.y_test.sum()) == 212
    assert bundle.benign == 357
    assert len(bundle.features) == 30
    assert len({feature.label for feature in bundle.features}) == 30


def test_ag_nao_piora_a_validacao_do_modelo_original():
    bundle = load_bundle()
    config = GAConfig(population_size=8, generations=3, seed=7, crossover="mistura")
    result = run_ga("arvore", bundle, config)
    baseline, _, _ = evaluate_genes("arvore", default_genes("arvore"), bundle, config.seed)
    assert result.validation.fitness + 1e-9 >= baseline.fitness
    assert result.history[0].generation == 0
    assert result.history[-1].generation == 3
    assert result.history[-1].best + 1e-9 >= result.history[0].best
    repeat = run_ga("arvore", bundle, config)
    assert repeat.validation.fitness == result.validation.fitness
    assert repeat.genes == result.genes
