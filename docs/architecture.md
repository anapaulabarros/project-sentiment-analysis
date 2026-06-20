# Documentação de Arquitetura

## Visão Geral

O sistema segue uma arquitetura em camadas, com separação clara entre dados, pré-processamento, modelo, treinamento, avaliação e inferência.

## Fluxo de Dados

```
data/raw/reviews.csv
        │
        ▼
src/data/loader.py              # Carregamento e validação do CSV
        │
        ▼
src/preprocessing/transform.py  # Limpeza de texto, normalização de labels
        │                        # Vetorização TF-IDF → NumPy
        │                        # Normalização L2 com NumPy
        ▼
src/training/train.py           # Divisão treino/teste
        │                        # NumPy → tensores PyTorch
        │                        # TensorDataset + DataLoader
        ▼
src/models/model.py             # SentimentMLP (nn.Module)
        │                        # Training loop + Validation loop
        ▼
src/evaluation/metrics.py       # Accuracy, F1, Precision, Recall
        │
        ▼
results/                        # metrics.json, confusion_matrix.png, model.pt
```

## Módulos

| Módulo | Responsabilidade |
|--------|-----------------|
| `src/data/loader.py` | Carregamento do CSV e validação das colunas obrigatórias |
| `src/preprocessing/transform.py` | Limpeza de texto, conversão de labels, vetorização TF-IDF, normalização NumPy |
| `src/models/model.py` | Definição do MLP PyTorch, funções de predição, save/load |
| `src/training/train.py` | Split, conversão para tensores, DataLoader, loops de treino e validação |
| `src/evaluation/metrics.py` | Cálculo e exibição de métricas de desempenho |
| `src/inference/predict.py` | Inferência sobre textos individuais ou em batch |
| `src/utils/config.py` | Constantes e hiperparâmetros globais |
| `main.py` | Orquestração do pipeline completo |

## Modelo

**SentimentMLP** (`nn.Module`):
- Entrada: vetor TF-IDF normalizado (dim = `MAX_FEATURES = 2000`)
- Camada oculta: `Linear(2000, 256)` + `ReLU` + `Dropout(0.3)`
- Saída: `Linear(256, 1)` (logit bruto)
- Loss: `BCEWithLogitsLoss`
- Otimizador: `Adam(lr=1e-3)`

## Decisões de Projeto

- **TF-IDF + MLP** em vez de embeddings: escolha didática para demonstrar o pipeline NumPy → PyTorch de forma explícita.
- **BCEWithLogitsLoss** com saída linear: mais numericamente estável que BCE com sigmoid separado.
- **`RANDOM_SEED=42`** em todas as operações aleatórias: garante reprodutibilidade.
- **Artefatos salvos juntos** em `results/models/`: `model.pt` e `vectorizer.pkl` devem ser carregados em par para inferência correta.
