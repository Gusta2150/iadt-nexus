# Relatório técnico — Acuria

Projeto 1 da Fase 2: otimização de modelos de diagnóstico com algoritmo genético e leitura em linguagem natural.

## O que foi construído

Uma bancada com quatro classificadores sobre o Wisconsin Diagnostic Breast Cancer. O algoritmo genético procura hiperparâmetros e o limiar de malignidade. O comparativo separa validação, que a busca enxerga, de teste, que ela não enxerga. Um redator transforma esses números em texto para a equipe.

Os arquivos do Módulo 1 estão em `solver/data`. A triagem aceita `data.csv` (coorte humana, 569 exames) e `cancer_prostata_data.csv` (100 casos). Recusa `animais_data.csv`, `cancer_mama_1_diagnostico.csv` e `breast_cancer_sample.csv`, que nesse módulo eram as bases falsas para o sistema reconhecer que a origem não era a coorte humana. Os modelos já treinados não vieram nesses arquivos. As baselines continuam sendo os modelos de fábrica do scikit-learn, no mesmo recorte, para o comparativo ter um original honesto. `data.csv` é o Wisconsin Diagnostic Breast Cancer na ordem clássica, então os números da semente 42 abaixo seguem iguais.

## Algoritmo genético

A codificação é um vetor unitário. Exemplos do ponto de fábrica:

- logística: C = 1, classes na frequência natural, limiar 0,50
- árvore: profundidade livre, Gini, limiar 0,50
- floresta: 100 árvores, raiz do número de variáveis, limiar 0,50
- KNN: 5 vizinhos, peso uniforme, distância euclidiana, limiar 0,50

A seleção é por torneio. O cruzamento é mistura (BLX-α com α = 0,3), uniforme ou de um ponto, conforme a configuração. A mutação é gaussiana por gene. O elitismo carrega os melhores indivíduos, e a população inicial já contém o gene de fábrica.

A fitness maximiza `0,50 recall + 0,35 F1 + 0,15 precisão` da classe maligna na validação. Recall pesa mais porque o custo de perder um maligno é o que a bancada trata como prioridade.

## Resultado da semente 42

Recorte: 341 treino, 114 validação, 114 teste. Malignos: 127, 43 e 42.

Modelos originais no teste:

| Modelo | Acurácia | Precisão | Recall | F1 | Fitness na validação |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regressão logística | 0,965 | 0,975 | 0,929 | 0,951 | 0,969 |
| Árvore de decisão | 0,930 | 0,905 | 0,905 | 0,905 | 0,887 |
| Floresta aleatória | 0,974 | 1,000 | 0,929 | 0,963 | 0,961 |
| K vizinhos | 0,956 | 0,974 | 0,905 | 0,938 | 0,952 |

Busca balanceada, população 18, 12 gerações, mutação 0,18, cruzamento por mistura:

| Modelo | Fitness val. original → busca | F1 teste original → busca | Recall teste | O que mudou |
| --- | --- | --- | --- | --- |
| K vizinhos | 0,952 → 0,984 | 0,938 → 0,951 | 0,905 → 0,929 | 8 vizinhos, peso pela distância, limiar 0,36 |
| Logística | 0,969 → 0,984 | 0,951 → 0,950 | 0,929 → 0,905 | limiar foi a 0,70 e a precisão de teste foi a 1,000 |
| Árvore | 0,887 → 0,940 | 0,905 → 0,875 | 0,905 → 0,833 | entropia, profundidade 13, classes balanceadas |
| Floresta | 0,961 → 0,977 | 0,963 → 0,940 | 0,929 → 0,929 | 45 árvores, profundidade 10, variáveis em log2 |

O KNN é o caso em que o ganho da validação apareceu no teste. A logística ilustra o outro lado: na validação, subir o limiar até o teto de 0,70 não custou recall, então a fitness subiu; no teste, alguns malignos ficaram entre 0,50 e 0,70 e o recall caiu, com a precisão indo a 100%. A árvore melhorou a validação e piorou o teste. A floresta original, com 100 árvores, já era a melhor no teste; a busca, limitada a no máximo 100 e premiada só pela validação, escolheu 45 árvores e perdeu F1 fora da amostra.

Nas três configurações da logística, exploratória, balanceada e intensiva, a fitness de validação estacionou em 0,984 e o teste ficou no mesmo ponto operacional, com limiar 0,70. A paisagem, nesse modelo e nessa semente, tem um platô. A configuração destacada é a de melhor fitness de validação e, no empate, a de melhor F1 de teste.

Tempos aproximados nesta máquina: cerca de 1,3 s para logística, árvore e KNN; cerca de 14 s para a floresta com população 18 e 12 gerações; cerca de 3 s para as três configurações da logística.

## Leitura em linguagem natural

O prompt de sistema pede português clínico, proíbe inventar sintomas e proíbe certeza absoluta. O prompt do caso manda usar os nomes das características como estão e omite o gabarito. O prompt do relatório pede o comparativo de validação contra teste.

Sem `OPENAI_API_KEY`, o redator local preenche o mesmo roteiro. A checagem marca extensão, ressalva de que o texto não substitui a equipe, ausência de "com certeza", classe, probabilidade e pelo menos duas características reais. No caso de probabilidade entre 40% e 60%, também exige a palavra incerto. Os textos do redator local fecham essa lista. O prompt usado aparece na bancada, aberto sob "Prompt usado".

## Desafios

Raio, perímetro e área andam juntos. O peso da logística se espalha entre elas, e a lista de importância global fica achatada. A leitura do caso usa o produto do coeficiente pelo valor padronizado, que é local, em vez de só o ranking global.

O teste tem 42 malignos. Um ponto percentual é menos de um caso. Por isso a bancada mostra o delta e diz, em frase, se o ganho viajou ou não.

A floresta é cara: cada indivíduo treina dezenas de árvores. As três configurações usam população menor nesse modelo, e a interface avisa o tempo.

## O que ficou de fora

Não há nuvem nem autoscaling. A API registra métricas, recusa busca concorrente e está descrita para um processo só. Um vídeo de demonstração continua sendo entrega à parte do grupo.
