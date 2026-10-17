from acuria.language import evaluate_text, fallback_case, fallback_report, render_case_prompt


CASE = {
    "model_label": "Regressão logística",
    "predicted_label": "Maligno",
    "probability": 0.91,
    "probability_label": "91,0%",
    "threshold_label": "42,0%",
    "recall_label": "96,0%",
    "precision_label": "94,0%",
    "f1_label": "95,0%",
    "params": [{"label": "Regularização C", "value": "1,000"}],
    "top_features": [
        {"label": "maior perímetro", "value": 180.2, "value_label": "180,2", "direction": "maligno"},
        {"label": "maior raio", "value": 25.1, "value_label": "25,10", "direction": "maligno"},
        {"label": "pontos côncavos médios", "value": 0.15, "value_label": "0,150", "direction": "maligno"},
    ],
    "neighbor_note": None,
}

REPORT = {
    "model_label": "Regressão logística",
    "chosen_label": "Balanceada",
    "seed": 42,
    "evaluations": 80,
    "elapsed_label": "1,2 s",
    "rows": [
        {
            "label": "Modelo original",
            "validation_fitness": "0,940",
            "test_recall": "95,0%",
            "test_precision": "93,0%",
            "test_f1": "94,0%",
            "test_recall_raw": 0.95,
            "test_precision_raw": 0.93,
        },
        {
            "label": "Balanceada",
            "validation_fitness": "0,970",
            "test_recall": "97,0%",
            "test_precision": "92,0%",
            "test_f1": "94,5%",
            "test_recall_raw": 0.97,
            "test_precision_raw": 0.92,
        },
    ],
    "params": [{"label": "Limiar de malignidade", "value": "0,42"}],
    "importance": [
        {"label": "maior perímetro", "share_label": "18,0%"},
        {"label": "maior raio", "share_label": "14,0%"},
        {"label": "pior concavidade", "share_label": "9,0%"},
    ],
}


def test_prompt_nao_recebe_gabarito_do_caso():
    _system, user = render_case_prompt(CASE)
    assert "gabarito" in user
    assert "actual" not in user
    assert "sintomas" in _system or "Não invente" in _system


def test_redator_local_passa_na_checagem():
    text = fallback_case(CASE)
    quality = evaluate_text(text, CASE, "caso")
    assert quality["score"] == 100
    uncertain = dict(CASE, probability=0.5, probability_label="50,0%")
    uncertain_text = fallback_case(uncertain)
    assert evaluate_text(uncertain_text, uncertain, "caso")["score"] == 100
    report = fallback_report(REPORT)
    assert evaluate_text(report, REPORT, "relatorio")["score"] == 100


def test_texto_vago_reprova():
    quality = evaluate_text("ok", CASE, "caso")
    assert quality["score"] < 60
