# **Projeto Integrador da Disciplina**

## **Engenharia de Software para IA e Frameworks Profundos**

## **Título geral do projeto**

**Desenvolvimento de uma Aplicação de Inteligência Artificial com Boas Práticas de Engenharia de Software**

## **Ideia central**

Cada equipe deverá escolher livremente um problema de Inteligência Artificial para resolver ao longo da disciplina. O tema da aplicação ficará em aberto, permitindo que os estudantes escolham problemas que façam sentido em sua própria realidade profissional, acadêmica ou pessoal.

A proposta é que cada grupo (no máximo 6 pessoas) desenvolva um sistema de IA completo, utilizando Python, NumPy, PyTorch, testes, modularização, tipagem, requisitos, arquitetura de software e versionamento com Git.

O projeto deve evoluir junto com os assuntos ministrados na disciplina, de modo que cada novo conteúdo apresentado em aula seja aplicado diretamente no desenvolvimento do sistema.

## **Escolha do tema**

Cada equipe deverá propor seu próprio tema de aplicação.

Recomenda-se que os grupos escolham problemas relacionados a:

* sua área de trabalho;  
* sua pesquisa acadêmica;  
* sua empresa ou setor de atuação;  
* um projeto paralelo de interesse;  
* uma dificuldade real observada no cotidiano;  
* uma base de dados pública que desperte interesse;  
* uma aplicação social, educacional, científica, médica, financeira, ambiental ou industrial.

O mais importante é que o problema escolhido seja significativo para a equipe. Projetos que nascem de interesses reais tendem a gerar maior envolvimento, melhores decisões técnicas e apresentações finais mais interessantes.

## **Exemplos de temas possíveis**

As equipes podem escolher, por exemplo:

* classificador de imagens;  
* classificador de sentimentos em textos;  
* preditor de preços;  
* preditor de evasão escolar;  
* classificador de documentos;  
* detector de anomalias;  
* sistema de recomendação simples;  
* predição de demanda;  
* classificação de sinais;  
* análise de dados financeiros;  
* apoio à decisão em saúde;  
* análise de dados educacionais;  
* reconhecimento de padrões em dados científicos;  
* análise de produtividade;  
* sistema para apoiar alguma atividade profissional do grupo.

Esses exemplos são apenas sugestões. As equipes são incentivadas a propor temas próprios.

## **Requisitos gerais do projeto**

Independentemente do tema escolhido, todo projeto deverá obrigatoriamente conter:

1. carregamento de dados;  
2. pré-processamento;  
3. uso de NumPy;  
4. uso de PyTorch;  
5. treinamento de um modelo;  
6. avaliação experimental;  
7. modularização do código;  
8. uso de tipagem;  
9. testes automatizados com `unittest`;  
10. documentação de requisitos;  
11. definição de arquitetura;  
12. versionamento com Git;  
13. apresentação final.

## **Estrutura mínima esperada do repositório**

project/

 ├── data/

 ├── docs/

 ├── notebooks/

 ├── src/

 │   ├── data/

 │   ├── preprocessing/

 │   ├── models/

 │   ├── training/

 │   ├── evaluation/

 │   ├── inference/

 │   └── utils/

 ├── tests/

 ├── requirements.txt

 ├── README.md

 └── main.py

A estrutura pode ser adaptada conforme o problema escolhido, mas deve preservar a ideia de separação de responsabilidades.

# **ETAPAS de desenvolvimento do projeto**

## **1\. Funções, modularização inicial e criação do repositório no GitHub**

Nesta etapa inicial, a equipe deverá transformar o problema escolhido em pequenas funções e criar a primeira versão organizada do projeto em um repositório no GitHub.

Além de implementar as primeiras funções do sistema, cada equipe deverá criar uma conta no GitHub, caso ainda não possua, criar um repositório para o projeto e carregar os arquivos iniciais.

### **Entrega esperada**

A equipe deverá entregar:

* link do repositório no GitHub;  
* nome provisório do projeto;  
* descrição inicial do problema escolhido;  
* base de dados que pretende utilizar;  
* arquivos iniciais carregados no repositório;  
* primeiro script executável;  
* funções iniciais do sistema;  
* arquivo `README.md` com descrição básica do projeto;  
* arquivo `requirements.txt`, mesmo que ainda simples.

  ### **Estrutura mínima inicial**

* project/  
*  ├── src/  
*  │   ├── data\_loader.py  
*  │   ├── preprocess.py  
*  │   └── main.py  
*  ├── README.md  
   └── requirements.txt

  ### **Funções iniciais sugeridas**

* def load\_data(path: str):  
*     pass  
*   
* def clean\_data(data):  
*     pass  
*   
* def split\_data(data):  
*     pass  
*   
* def main():  
      pass

  ### **Conceitos avaliados**

* definição de funções;  
* separação inicial de responsabilidades;  
* organização básica do código;  
* criação de conta no GitHub;  
* criação do repositório;  
* envio dos primeiros arquivos;  
* documentação inicial do projeto.

---

## **2\. Modularização**

Nesta etapa, o código deverá ser reorganizado em módulos e pacotes.

Exemplo:

src/

 ├── data/

 │   └── loader.py

 ├── preprocessing/

 │   └── transform.py

 ├── models/

 │   └── model.py

 ├── training/

 │   └── train.py

 ├── evaluation/

 │   └── metrics.py

 └── utils/

     └── config.py

### **Entrega esperada**

A equipe deverá entregar:

* código dividido em módulos;  
* imports funcionando corretamente;  
* separação entre dados, modelo, treinamento, avaliação e utilidades;  
* execução centralizada em `main.py`.

### **Conceitos avaliados**

* módulos;  
* pacotes;  
* organização de projeto;  
* legibilidade;  
* manutenção.

---

## **3\. Tipagem**

Nesta etapa, o grupo deverá adicionar tipagem ao projeto.

Exemplos:

import numpy as np

import pandas as pd

from typing import Tuple

def load\_data(path: str) \-\> pd.DataFrame:

    pass

def split\_features\_target(

    data: pd.DataFrame,

    target\_column: str

) \-\> Tuple\[np.ndarray, np.ndarray\]:

    pass

### **Entrega esperada**

A equipe deverá entregar:

* funções principais com type hints;  
* tipos de entrada e saída explícitos;  
* código mais claro e documentado;  
* redução de ambiguidade nas funções.

### **Conceitos avaliados**

* type hints;  
* contratos de função;  
* legibilidade;  
* prevenção de erros.

---

## **4\. NumPy**

Nesta etapa, o grupo deverá utilizar NumPy para manipular os dados antes de levá-los ao PyTorch.

Exemplos de operações esperadas:

* normalização;  
* padronização;  
* vetorização;  
* divisão treino/teste;  
* manipulação de matrizes;  
* cálculo de estatísticas;  
* conversão de dados para arrays.

Exemplo:

def standardize(X: np.ndarray) \-\> np.ndarray:

    mean \= np.mean(X, axis=0)

    std \= np.std(X, axis=0)

    return (X \- mean) / std

### **Entrega esperada**

A equipe deverá entregar:

* pipeline de pré-processamento usando NumPy;  
* funções vetorizadas;  
* preparação dos dados para treinamento;  
* análise simples das dimensões dos dados.

### **Conceitos avaliados**

* arrays;  
* vetorização;  
* operações matriciais;  
* manipulação numérica eficiente.

---

## **5\. Introdução ao PyTorch — Parte 1**

Nesta etapa, o grupo deverá converter os dados processados para tensores PyTorch.

Exemplos:

import torch

X\_tensor \= torch.tensor(X\_train, dtype=torch.float32)

y\_tensor \= torch.tensor(y\_train, dtype=torch.float32)

Também deverá utilizar:

TensorDataset

DataLoader

### **Entrega esperada**

A equipe deverá entregar:

* conversão correta de dados NumPy para tensores;  
* criação de datasets;  
* criação de dataloaders;  
* verificação das dimensões dos tensores;  
* uso correto de CPU/GPU, quando aplicável.

### **Conceitos avaliados**

* tensores;  
* shapes;  
* dtypes;  
* Dataset;  
* DataLoader;  
* device.

---

## **6\. Introdução ao PyTorch — Parte 2**

Nesta etapa, o grupo deverá implementar o modelo neural em PyTorch.

Exemplo:

import torch.nn as nn

class NeuralModel(nn.Module):

    def \_\_init\_\_(self, input\_dim: int, hidden\_dim: int, output\_dim: int):

        super().\_\_init\_\_()

        self.network \= nn.Sequential(

            nn.Linear(input\_dim, hidden\_dim),

            nn.ReLU(),

            nn.Linear(hidden\_dim, output\_dim)

        )

    def forward(self, x):

        return self.network(x)

O grupo deverá implementar:

* modelo neural;  
* função de perda;  
* otimizador;  
* laço de treinamento;  
* laço de validação;  
* salvamento do modelo;  
* carregamento do modelo.

### **Entrega esperada**

A equipe deverá entregar:

* modelo implementado em PyTorch;  
* treinamento funcional;  
* avaliação inicial;  
* modelo salvo em arquivo;  
* script de inferência simples.

### **Conceitos avaliados**

* `nn.Module`;  
* `forward`;  
* loss function;  
* optimizer;  
* treinamento;  
* validação;  
* inferência.

---

## **7\. Exercícios intermediários**

Nesta etapa, o grupo deverá melhorar o projeto a partir de exercícios orientados.

Possíveis melhorias:

* testar diferentes hiperparâmetros;  
* comparar duas arquiteturas;  
* alterar função de ativação;  
* alterar número de camadas;  
* comparar batch sizes;  
* analisar overfitting;  
* gerar gráficos de perda;  
* melhorar o pré-processamento.

### **Entrega esperada**

A equipe deverá entregar:

* pelo menos duas configurações experimentais;  
* comparação entre resultados;  
* breve análise técnica;  
* gráfico ou tabela de desempenho.

### **Conceitos avaliados**

* experimentação;  
* análise crítica;  
* interpretação de resultados;  
* ajustes de modelo.

---

## **8\. Python — Testes com unittest**

Nesta etapa, o grupo deverá criar testes automatizados.

Estrutura esperada:

tests/

 ├── test\_data.py

 ├── test\_preprocessing.py

 ├── test\_model.py

 └── test\_training.py

Exemplos de testes:

import unittest

class TestPreprocessing(unittest.TestCase):

    def test\_standardize\_shape(self):

        pass

    def test\_standardize\_mean(self):

        pass

O grupo deverá testar pelo menos:

* carregamento dos dados;  
* saída do pré-processamento;  
* formato dos tensores;  
* saída do modelo;  
* salvamento e carregamento do modelo.

### **Entrega esperada**

A equipe deverá entregar:

* suíte de testes com `unittest`;  
* instrução de execução dos testes;  
* todos os testes passando;  
* evidência dos testes no README.

### **Conceitos avaliados**

* testes unitários;  
* assertivas;  
* qualidade de software;  
* confiabilidade;  
* prevenção de regressões.

---

## **9\. Engenharia de Software — Introdução**

Nesta etapa, o grupo deverá refletir sobre o projeto como um sistema de software, não apenas como um notebook de IA.

A equipe deverá responder:

* Qual problema o sistema resolve?  
* Quem são os usuários?  
* Qual valor o sistema entrega?  
* Qual é o contexto de uso?  
* Quais são as limitações do sistema?

### **Entrega esperada**

Documento inicial de visão do sistema contendo:

* nome do projeto;  
* problema;  
* público-alvo;  
* justificativa;  
* escopo;  
* limitações;  
* possíveis impactos.

### **Conceitos avaliados**

* visão de produto;  
* contexto do sistema;  
* papel da engenharia de software em IA;  
* distinção entre experimento e aplicação.

---

## **10\. Engenharia de Software — Requisitos**

Nesta etapa, o grupo deverá elaborar os requisitos do sistema.

### **Requisitos funcionais**

Exemplos:

RF01 — O sistema deve carregar uma base de dados.

RF02 — O sistema deve pré-processar os dados.

RF03 — O sistema deve treinar um modelo de IA.

RF04 — O sistema deve avaliar o modelo treinado.

RF05 — O sistema deve realizar inferência com novos dados.

RF06 — O sistema deve salvar os resultados dos experimentos.

### **Requisitos não funcionais**

Exemplos:

RNF01 — O código deve ser modularizado.

RNF02 — O sistema deve ser reproduzível.

RNF03 — O sistema deve possuir testes automatizados.

RNF04 — O sistema deve ser seguro.

RNF05 — O sistema deve ter disponibilidade.

RNF06 — O sistema deve ser versionado com Git.

RNF07 — O sistema deve possuir documentação de execução.

RNF08 — O modelo deve apresentar desempenho mensurável por métricas adequadas.

### **Entrega esperada**

Documento de requisitos contendo:

* requisitos funcionais;  
* requisitos não funcionais;  
* restrições;  
* critérios de aceitação.

### **Conceitos avaliados**

* requisitos funcionais;  
* requisitos não funcionais;  
* escopo;  
* critérios de aceitação;  
* rastreabilidade.

---

## **11\. Engenharia de Software — Design e Arquitetura**

Nesta etapa, o grupo deverá definir a arquitetura do sistema.

Exemplo de arquitetura em camadas:

Camada de Interface

        |

Camada de Aplicação

        |

Camada de Serviços de IA

        |

Camada de Dados

Ou, para projetos mais simples:

main.py

  |

  |-- data\_loader

  |-- preprocessing

  |-- model

  |-- train

  |-- evaluate

  |-- inference

A equipe deverá explicar:

* quais são os módulos;  
* qual é a responsabilidade de cada módulo;  
* como os dados fluem no sistema;  
* onde ocorre o treinamento;  
* onde ocorre a inferência;  
* como o modelo é salvo e carregado.

### **Entrega esperada**

Documento de arquitetura contendo:

* diagrama simples;  
* descrição dos módulos;  
* fluxo de dados;  
* decisões de projeto;  
* justificativa das decisões.

### **Conceitos avaliados**

* separação de responsabilidades;  
* arquitetura;  
* coesão;  
* acoplamento;  
* organização para manutenção.

---

## **12\. Versionamento de código com Git**

Nesta etapa, a equipe deve amadurecer a organização do repositório Git.

O grupo deverá utilizar:

* commits frequentes;  
* mensagens de commit claras;  
* branches;  
* merge ou pull request;  
* histórico de contribuições;  
* README atualizado.

Exemplo de fluxo:

main

 ├── feature/data-pipeline

 ├── feature/model-training

 ├── feature/tests

 ├── feature/docs

 └── feature/inference

### **Entrega esperada**

A equipe deverá entregar:

* link do repositório;  
* histórico de commits;  
* participação dos membros;  
* README com instruções;  
* organização das branches ou descrição do fluxo de trabalho.

### **Conceitos avaliados**

* versionamento;  
* colaboração;  
* rastreabilidade;  
* organização do desenvolvimento.

---

## **13\. Exercícios finais**

Nesta etapa, a equipe deverá consolidar o projeto.

Atividades recomendadas:

* revisar o código;  
* corrigir testes;  
* melhorar README;  
* melhorar requisitos;  
* ajustar arquitetura;  
* treinar modelo final;  
* gerar resultados finais;  
* preparar apresentação.

### **Entrega esperada**

Versão quase final do projeto contendo:

* código completo;  
* testes funcionando;  
* modelo treinado;  
* documentação;  
* resultados;  
* repositório organizado.

### **Conceitos avaliados**

* integração dos conteúdos;  
* maturidade do projeto;  
* qualidade da entrega;  
* capacidade de revisão.

---

## **14\. Apresentação do Projeto**

Na apresentação final, cada grupo deverá demonstrar o sistema desenvolvido.

A apresentação deve conter:

1. tema escolhido;  
2. motivação;  
3. usuários ou contexto de uso;  
4. requisitos;  
5. arquitetura;  
6. pipeline de dados;  
7. modelo em PyTorch;  
8. testes;  
9. uso de Git;  
10. resultados experimentais;  
11. demonstração do sistema;  
12. contribuições de cada membro;  
13. dificuldades encontradas;  
14. melhorias futuras.

## **Produto final esperado**

Ao final da disciplina, cada equipe deverá entregar:

1. repositório Git;  
2. código-fonte;  
3. README;  
4. documentação de requisitos;  
5. documentação de arquitetura;  
6. testes automatizados;  
7. modelo treinado ou instruções para treinamento;  
8. resultados experimentais;  
9. apresentação final.

## **Critérios de avaliação**

| Entregas | Critério | Peso |
| ----- | ----- | ----- |
| Entrega 1 | Escolha e contextualização do problema | 10% |
|  | Modularização, funções e organização do código | 15% |
|  | Tipagem e qualidade do código Python | 10% |
| Entrega 2 | Implementação em PyTorch \-1 | 20% |
|  | Uso adequado de NumPy | 10% |
| Entrega 3 | Implementação em PyTorch \-2 |  |
| Entrega 4 | Testes automatizados com unittest | 10% |
| Entrega 5 | Requisitos | 15% |
| Entrega 6 |  Design e arquitetura, Uso de Git e colaboração |  |
| Entrega Final | Apresentação final | 5% |

## **Orientação final para as equipes**

O projeto não deve ser apenas um notebook que treina um modelo. Ele deve ser tratado como um pequeno sistema de software baseado em IA.

A qualidade do projeto será avaliada não apenas pelo desempenho do modelo, mas também pela clareza do código, organização dos módulos, documentação, testes, requisitos, arquitetura e capacidade de reprodução.

As equipes devem escolher temas que tenham significado para seus integrantes. Um bom projeto pode nascer de um problema do trabalho, de uma pesquisa acadêmica, de uma atividade profissional, de uma necessidade social ou de uma curiosidade técnica bem definida.

