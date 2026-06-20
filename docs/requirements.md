# Documentação de Requisitos

## Requisitos Funcionais

| ID | Descrição |
|----|-----------|
| RF01 | O sistema deve carregar a base de dados de reviews em formato CSV. |
| RF02 | O sistema deve pré-processar os textos removendo pontuação, convertendo para minúsculas e colapsando espaços. |
| RF03 | O sistema deve converter ratings numéricos em labels binários de sentimento (positivo/negativo), descartando reviews neutros (rating = 3). |
| RF04 | O sistema deve vetorizar os textos usando TF-IDF e normalizar os vetores com NumPy. |
| RF05 | O sistema deve treinar um modelo de rede neural (MLP) em PyTorch para classificação binária de sentimento. |
| RF06 | O sistema deve avaliar o modelo com métricas de accuracy, F1-score, precision e recall. |
| RF07 | O sistema deve realizar inferência (predição de sentimento) sobre novos textos. |
| RF08 | O sistema deve salvar os resultados dos experimentos (métricas, figuras, modelo treinado). |

## Requisitos Não Funcionais

| ID | Descrição |
|----|-----------|
| RNF01 | O código deve ser modularizado em pacotes com responsabilidades separadas. |
| RNF02 | O sistema deve ser reproduzível: mesma semente aleatória (`RANDOM_SEED=42`) deve gerar os mesmos resultados. |
| RNF03 | O sistema deve possuir testes automatizados com `unittest` cobrindo carregamento, pré-processamento, modelo e treinamento. |
| RNF04 | O sistema deve utilizar tipagem estática em todas as funções públicas. |
| RNF05 | O sistema deve ser versionado com Git, com histórico de commits rastreável. |
| RNF06 | O sistema deve possuir documentação de execução no README. |
| RNF07 | O modelo deve apresentar desempenho mensurável por accuracy, F1, precision e recall, salvos em `results/metrics/metrics.json`. |
| RNF08 | O sistema deve funcionar em CPU, com suporte opcional a GPU via PyTorch. |

## Restrições

- Python >= 3.10
- PyTorch >= 2.0
- Dataset: Amazon Product Reviews (Kaggle: yasserh)

## Critérios de Aceitação

- `python main.py` executa sem erros e salva resultados em `results/`
- `python -m unittest discover tests` com todos os testes passando
- Modelo salvo em `results/models/model.pt`
- Métricas salvas em `results/metrics/metrics.json`
