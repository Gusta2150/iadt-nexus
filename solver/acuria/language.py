"""Leitura em linguagem natural e a checagem de qualidade dessa leitura.

Sem chave de API o texto sai de um redator local, preso aos números do plano.
Com OPENAI_API_KEY, o mesmo prompt vai para o modelo configurado. Se a chamada
falha, o redator local assume e a origem fica registrada.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

SYSTEM_PROMPT = """Você é o apoio à leitura de modelos de diagnóstico do hospital universitário.
Escreva em português do Brasil, para a equipe médica, em tom sóbrio e direto.
Use somente os fatos do contexto. Não invente sintomas, história clínica, conduta ou exames que não estejam listados.
Não feche diagnóstico. Deixe claro que o resultado é apoio à leitura e não substitui a equipe.
Use os nomes das características exatamente como foram escritos.
Se a probabilidade de malignidade estiver entre 40% e 60%, diga que a leitura é incerta.
Não use as expressões "com certeza" nem "é certamente".
"""


def render_case_prompt(context: dict[str, Any]) -> tuple[str, str]:
    features = []
    for item in context["top_features"]:
        features.append(
            f"- {item['label']}: valor {item['value_label']}, empurra para {item['direction']}"
        )
    params = "\n".join(f"- {item['label']}: {item['value']}" for item in context["params"])
    neighbor = context.get("neighbor_note") or "Não se aplica."
    user = f"""Contexto do caso, sem o gabarito do conjunto de teste.

Modelo: {context["model_label"]}
Classe apontada: {context["predicted_label"]}
Probabilidade de malignidade: {context["probability_label"]}
Limiar de malignidade: {context["threshold_label"]}
Recall de malignidade no teste: {context["recall_label"]}
Precisão de malignidade no teste: {context["precision_label"]}
F1 no teste: {context["f1_label"]}

Hiperparâmetros desta busca:
{params}

Características que mais pesaram:
{chr(10).join(features)}

Vizinhos, quando o modelo for KNN:
{neighbor}

Escreva três parágrafos curtos: a leitura, o que pesou nas medidas e o limite do modelo.
"""
    return SYSTEM_PROMPT, user


def render_report_prompt(context: dict[str, Any]) -> tuple[str, str]:
    rows = []
    for row in context["rows"]:
        rows.append(
            f"- {row['label']}: fitness de validação {row['validation_fitness']}, "
            f"recall de teste {row['test_recall']}, precisão de teste {row['test_precision']}, "
            f"F1 de teste {row['test_f1']}"
        )
    user = f"""Relatório da busca de hiperparâmetros.

Modelo: {context["model_label"]}
Configuração destacada: {context["chosen_label"]}
Semente: {context["seed"]}
Avaliações de treino: {context["evaluations"]}
Tempo: {context["elapsed_label"]}

Comparativo, com o teste fora da função fitness:
{chr(10).join(rows)}

Hiperparâmetros encontrados:
{chr(10).join(f"- {item['label']}: {item['value']}" for item in context["params"])}

Variáveis de maior peso global:
{chr(10).join(f"- {item['label']}: {item['share_label']}" for item in context["importance"][:5])}

Escreva um relatório curto para a equipe: o que mudou em relação ao modelo original, se o ganho apareceu no teste e qual cuidado tomar com falso positivo ou falso negativo. Não invente números.
"""
    return SYSTEM_PROMPT, user


def fallback_case(context: dict[str, Any]) -> str:
    probability = float(context["probability"])
    names = "; ".join(
        f"{item['label']} (valor {item['value_label']}, empurra para {item['direction']})"
        for item in context["top_features"][:3]
    )
    uncertainty = ""
    if 0.4 <= probability <= 0.6:
        uncertainty = (
            " A probabilidade ficou na faixa intermediária, então a leitura é incerta "
            "e as medidas precisam ser revistas antes de qualquer conduta."
        )
    neighbor = ""
    if context.get("neighbor_note"):
        neighbor = " " + context["neighbor_note"]
    return (
        f"Leitura de apoio: o modelo de {context['model_label']} aponta "
        f"{context['predicted_label'].lower()} com probabilidade de malignidade de "
        f"{context['probability_label']}. O limiar usado nesta busca foi "
        f"{context['threshold_label']}.{uncertainty}{neighbor}\n\n"
        f"As características que mais pesaram foram: {names}. "
        "Confira essas medidas na morfometria do exame. O modelo não vê sintomas, "
        "história clínica nem exame físico.\n\n"
        f"No conjunto de teste, separado antes da busca, o recall de malignidade foi "
        f"{context['recall_label']} e a precisão foi {context['precision_label']}. "
        "Isso é apoio à leitura do exame, não um diagnóstico definitivo, e não substitui "
        "a avaliação da equipe médica."
    )


def fallback_report(context: dict[str, Any]) -> str:
    original = context["rows"][0]
    chosen = next(row for row in context["rows"] if row["label"] == context["chosen_label"])
    recall_delta = float(chosen["test_recall_raw"]) - float(original["test_recall_raw"])
    if recall_delta > 0.005:
        movement = (
            f"No teste, o recall de malignidade subiu de {original['test_recall']} "
            f"para {chosen['test_recall']}."
        )
    elif recall_delta < -0.005:
        movement = (
            f"No teste, o recall de malignidade caiu de {original['test_recall']} "
            f"para {chosen['test_recall']}. A busca otimizou a validação; o teste não confirmou o ganho."
        )
    else:
        movement = (
            f"No teste, o recall de malignidade ficou em {chosen['test_recall']}, "
            "perto do modelo original. O ganho da validação não se repetiu de forma clara."
        )
    params = ", ".join(f"{item['label']} {item['value']}" for item in context["params"])
    top = ", ".join(item["label"] for item in context["importance"][:3])
    return (
        f"A busca {context['chosen_label'].lower()} no modelo de {context['model_label']} "
        f"avaliou {context['evaluations']} indivíduos em {context['elapsed_label']}, "
        f"com a semente {context['seed']}. A função fitness, usada só na validação, "
        f"passou de {original['validation_fitness']} para {chosen['validation_fitness']}.\n\n"
        f"{movement} A precisão no teste foi de {original['test_precision']} para "
        f"{chosen['test_precision']}, e o F1 de {original['test_f1']} para {chosen['test_f1']}. "
        "Se o recall sobe e a precisão cai, a equipe passa a ver mais falsos positivos.\n\n"
        f"Hiperparâmetros desta configuração: {params}. "
        f"As variáveis de maior peso foram {top}. "
        "O relatório é apoio à leitura do experimento, não substitui uma validação externa "
        "nem autoriza uso clínico."
    )


def complete(system: str, user: str) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    try:
        response = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=40,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, IndexError, TypeError):
        return None


def evaluate_text(text: str, context: dict[str, Any], kind: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    folded = text.casefold()
    add("Extensão útil", 280 <= len(text) <= 4000, f"{len(text)} caracteres.")
    add(
        "Ressalva clínica",
        "não substitui" in folded or "apoio à leitura" in folded or "apoio a leitura" in folded,
        "O texto precisa dizer que é apoio e não substitui a equipe.",
    )
    add(
        "Sem certeza absoluta",
        "com certeza" not in folded and "é certamente" not in folded and "e certamente" not in folded,
        "Expressões de certeza absoluta reprovam a leitura.",
    )
    if kind == "caso":
        label = str(context["predicted_label"]).casefold()
        add("Classe apontada", label in folded, f"Deveria citar {context['predicted_label']}.")
        add(
            "Probabilidade",
            str(context["probability_label"]) in text or "%" in text,
            "Deveria citar a probabilidade.",
        )
        mentioned = [item["label"] for item in context["top_features"][:3] if item["label"].casefold() in folded]
        add(
            "Características reais",
            len(mentioned) >= 2,
            "Pelo menos dois nomes de características do contexto.",
        )
        probability = float(context["probability"])
        if 0.4 <= probability <= 0.6:
            add("Incerteza", "incert" in folded, "Probabilidade intermediária pede a palavra incerto.")
    else:
        add("Cita o teste", "teste" in folded, "O relatório precisa falar do conjunto de teste.")
        add("Cita recall ou F1", "recall" in folded or "f1" in folded, "Deveria citar recall ou F1.")

    passed = sum(1 for check in checks if check["passed"])
    score = round(100 * passed / len(checks))
    if score >= 85:
        summary = "A leitura cobre os fatos obrigatórios."
    elif score >= 60:
        summary = "A leitura está parcialmente ancorada nos fatos."
    else:
        summary = "A leitura deixou de fora fatos ou ressalvas obrigatórias."
    return {"score": score, "checks": checks, "summary": summary}


def narrate(kind: str, context: dict[str, Any]) -> dict[str, Any]:
    system, user = render_case_prompt(context) if kind == "caso" else render_report_prompt(context)
    generated = complete(system, user)
    warning = None
    if generated:
        source = "llm"
        text = generated
    else:
        source = "local"
        text = fallback_case(context) if kind == "caso" else fallback_report(context)
        if os.environ.get("OPENAI_API_KEY", "").strip():
            warning = "A chamada ao modelo falhou. O texto abaixo foi escrito pelo redator local."
    quality = evaluate_text(text, context, kind)
    return {
        "text": text,
        "source": source,
        "warning": warning,
        "prompt": system + "\n\n" + user,
        "quality": quality,
    }
