# API

A documentação interativa sobe junto com o serviço em [http://127.0.0.1:8741/docs](http://127.0.0.1:8741/docs). Pela bancada, o mesmo endereço aparece em `/solver/docs`.

| Método | Caminho | Função |
| --- | --- | --- |
| GET | `/api/health` | Saúde do processo |
| GET | `/api/overview` | Recorte, baselines e fitness |
| POST | `/api/optimize` | Uma busca, ou as três configurações |
| GET | `/api/runs/{id}/cases` | Casos do teste com previsão e gabarito |
| POST | `/api/runs/{id}/explain` | Leitura clínica de um caso |
| POST | `/api/runs/{id}/report` | Relatório da busca |
| GET | `/api/metrics` | Contadores da sessão |
| GET | `/api/logs` | Últimos eventos |

`POST /api/optimize` recebe o modelo (`logistica`, `arvore`, `floresta` ou `knn`), o tamanho da população, as gerações, a taxa de mutação por gene, o cruzamento (`mistura`, `uniforme` ou `um_ponto`), a semente e a flag `experiments`. Com `experiments: true`, os controles numéricos dão lugar às três configurações, e a semente continua valendo.

Uma segunda busca simultânea recebe HTTP 429. Uma leitura de busca que já saiu da memória recebe HTTP 404.
