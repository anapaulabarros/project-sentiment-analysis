# Visão do Sistema

## Nome do Projeto
Sentiment Analysis — Amazon Product Reviews

## Problema
Plataformas de e-commerce acumulam milhares de reviews de produtos diariamente. A leitura manual é inviável, e decisões de negócio que dependem do sentimento dos clientes ficam atrasadas. O sistema automatiza a classificação de sentimento de reviews textuais.

## Público-alvo
- Equipes de produto e marketing que precisam monitorar a satisfação de clientes em escala.
- Desenvolvedores e pesquisadores que desejam integrar classificação de sentimento em pipelines de dados.
- Estudantes de IA e Engenharia de Software que buscam um exemplo completo de sistema de ML em produção.

## Justificativa
A classificação binária de sentimento (positivo/negativo) é um problema bem definido com dataset público disponível, permitindo validação objetiva via métricas de accuracy, F1, precision e recall. O uso de PyTorch garante extensibilidade para arquiteturas mais complexas em entregas futuras.

## Escopo

**Inclui:**
- Carregamento e validação do dataset Amazon Product Reviews
- Pré-processamento de texto (limpeza, normalização)
- Vetorização TF-IDF com NumPy
- Classificação binária com MLP PyTorch
- Avaliação com métricas padrão
- Inferência sobre textos novos
- Testes automatizados com unittest
- Documentação de requisitos e arquitetura

**Não inclui:**
- Interface gráfica ou API REST
- Suporte a idiomas além do inglês
- Análise de sentimento multi-classe (neutro não é classificado)
- Retreinamento online com novos dados

## Limitações
- O modelo é sensível ao tamanho do vocabulário TF-IDF (`MAX_FEATURES=2000`); reviews muito curtos podem ter representação pobre.
- Reviews com rating 3 (neutro) são descartados — o sistema não classifica sentimentos ambíguos.
- O desempenho foi avaliado apenas no dataset Amazon Product Reviews; generalização para outros domínios não foi validada.
- O modelo atual (MLP sobre TF-IDF) não captura contexto sequencial; arquiteturas como LSTM ou Transformers podem ser superiores.

## Possíveis Impactos
- **Positivo:** redução do tempo de análise de feedback de clientes, identificação rápida de problemas em produtos.
- **Positivo:** base para sistemas de recomendação ou alertas automáticos de qualidade de produto.
- **Atenção:** classificações incorretas podem influenciar decisões de negócio; o sistema deve ser usado como apoio, não como árbitro final.
- **Atenção:** o dataset de treino pode conter vieses de seleção (produtos com mais reviews tendem a ser mais populares).
