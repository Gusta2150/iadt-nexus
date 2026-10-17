# Acuria

Bancada do Projeto 1 da Fase 2. Um algoritmo genético ajusta hiperparâmetros de modelos de diagnóstico, e um redator transforma o resultado em texto para a equipe médica.

O conjunto é o Wisconsin Diagnostic Breast Cancer: 569 exames, 30 medidas de núcleo celular, classe positiva maligno. Os modelos do Módulo 1 não estavam neste repositório. As baselines são regressão logística, árvore de decisão, floresta aleatória e K vizinhos com os hiperparâmetros de fábrica do scikit-learn.

O teste fica fora da busca. A função fitness, só na validação, é 50% recall de malignidade, 35% F1 e 15% precisão.

## Subir na máquina

Dois processos: a API em Python e a bancada em Next.js.

```bash
python3 -m venv solver/.venv
solver/.venv/bin/pip install -r solver/requirements.txt
solver/.venv/bin/uvicorn acuria.api:app --app-dir solver --host 127.0.0.1 --port 8741
```

Em outro terminal:

```bash
npm install
npm run dev
```

A bancada abre em [http://127.0.0.1:43123](http://127.0.0.1:43123). A API interativa abre em [http://127.0.0.1:8741/docs](http://127.0.0.1:8741/docs).

## Sem chave de modelo de linguagem

As leituras saem de um redator local, com o mesmo prompt e a mesma checagem de qualidade. Para usar um modelo externo:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
```

`OPENAI_BASE_URL` troca o servidor, se a API for compatível com o formato de chat da OpenAI.

## Testes e demonstração

```bash
cd solver && .venv/bin/pytest
python scripts/demo.py
```

## Onde está cada coisa

- `solver/acuria/encoding.py` — genes e decodificação
- `solver/acuria/genetic.py` — seleção, cruzamento, mutação, elitismo e as três configurações
- `solver/acuria/training.py` — treino, limiar e fitness
- `solver/acuria/language.py` — prompts, redator e checagem
- `solver/acuria/api.py` — HTTP, métricas e registro
- `docs/RELATORIO_TECNICO.md` — números da semente 42
- `docs/ARQUITETURA.md` — recorte, fluxo e limites
- `docs/API.md` — rotas

Isto é material de ensino. Não é ferramenta de decisão clínica.
