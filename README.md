# Especificação de Requisitos do Sistema  
## Sistema de Predição de Churn com K-modes

**Versão:** 1.1  
**Data:** 22/09/2025  
**Status:** Draft  

---

## 1. VISÃO GERAL DO PROJETO

### 1.1 Objetivo
Desenvolver um sistema de Machine Learning para predição de churn de clientes utilizando clustering k-modes para segmentação de clientes categóricos, seguido de modelos preditivos especializados por segmento.

### 1.2 Escopo
O sistema deve processar dados categóricos e numéricos de clientes, aplicar clustering k-modes para identificar perfis comportamentais, e treinar modelos de predição específicos para cada cluster, fornecendo insights acionáveis para equipes de marketing e retenção.

### 1.3 Abordagem Metodológica
- **Fase 1:** Clustering de clientes usando k-modes para dados categóricos
- **Fase 2:** Análise de padrões de churn por cluster identificado
- **Fase 3:** Desenvolvimento de modelos preditivos especializados por segmento
- **Fase 4:** Sistema de scoring integrado considerando cluster + probabilidade de churn

---

## 2. REQUISITOS FUNCIONAIS

### 2.1 Módulo de Ingestão de Dados (RF01-RF05)

**RF01 - Coleta de Dados Categóricos de Billing**  
- O sistema deve extrair dados categóricos de faturamento: faixa de valor mensal, status de pagamento, tipo de contestação, categoria de inadimplência.  
- Transformar dados numéricos em categorias quando necessário.  
- Frequência: Diária (batch) e tempo real para eventos críticos.  
- Fontes: Sistema de Billing, ERP Financeiro.  

**RF02 - Coleta de Dados Categóricos de Uso da Rede**  
- O sistema deve capturar dados categóricos: padrão de uso (baixo/médio/alto), tipo predominante de chamada, categoria de consumo de dados, qualidade de conexão.  
- Frequência: Tempo real com agregações categóricas diárias.  
- Fontes: CDR (Call Detail Records), sistemas de monitoramento de rede.  

**RF03 - Coleta de Dados Categóricos de Atendimento**  
- O sistema deve integrar dados categóricos: tipo de interação, categoria de reclamação, canal de atendimento, status de resolução.  
- Frequência: Tempo real.  
- Fontes: CRM, Sistema de Tickets, Base Anatel.  

**RF04 - Coleta de Dados Demográficos e Contratuais Categóricos**  
- O sistema deve acessar informações categóricas: região geográfica, categoria de plano, tipo de aparelho, perfil demográfico.  
- Frequência: Diária.  
- Fontes: CRM, Sistema de Vendas.  

**RF05 - Validação e Normalização de Dados Categóricos**  
- O sistema deve validar categorias válidas e consistentes.  
- Normalizar e padronizar valores categóricos.  
- Tratar valores missing e criar categoria "outros" quando apropriado.  
- Gerar relatórios de distribuição categórica.  

---

### 2.2 Módulo de Feature Engineering Categórico (RF06-RF10)

**RF06 - Criação de Features Categóricas Comportamentais**  
- Categorizar tendências de uso: "crescente", "decrescente", "estável", "irregular".  
- Criar categorias de sazonalidade: "matutino", "vespertino", "noturno", "fim_semana".  
- Classificar mudanças comportamentais: "abrupta", "gradual", "sem_mudança".  

**RF07 - Features Categóricas de Relacionamento**  
- Categorizar tempo de vida: "novo", "estabelecido", "veterano", "longo_prazo".  
- Classificar padrão de atendimento: "sem_contato", "eventual", "frequente", "intensivo".  
- Categorizar satisfação: "alta", "média", "baixa", "crítica".  

**RF08 - Features Categóricas Financeiras**  
- Categorizar perfil de pagamento: "pontual", "atraso_ocasional", "atraso_frequente", "inadimplente".  
- Classificar valor relativo: "baixo_valor", "médio_valor", "alto_valor", "premium".  
- Categorizar tendência financeira: "crescimento", "estabilidade", "declínio".  

**RF09 - Features Categóricas de Engagement**  
- Classificar uso digital: "não_usa", "básico", "moderado", "avançado".  
- Categorizar resposta a promoções: "nunca_adere", "seletivo", "aderente", "early_adopter".  
- Classificar inovação: "conservador", "moderado", "inovador".  

**RF10 - Feature Store Categórico**  
- Armazenar features categóricas com versionamento.  
- Manter dicionários de categorias consistentes.  
- Garantir encoding consistente entre treino e inferência.  

---

### 2.3 Módulo de Clustering K-modes (RF11-RF15)

**RF11 - Implementação do Algoritmo K-modes**  
- Implementar k-modes otimizado para dados categóricos.  
- Suportar diferentes métricas de distância categórica.  
- Permitir configuração flexível do número de clusters.  
- Implementar inicialização inteligente dos centroides.  

**RF12 - Otimização de Clusters**  
- Determinar número ótimo de clusters usando métricas adequadas (Cost function, Silhouette para dados categóricos).  
- Implementar validação cruzada para estabilidade dos clusters.  
- Analisar pureza e homogeneidade dos clusters formados.  

**RF13 - Análise e Profiling de Clusters**  
- Gerar perfis detalhados de cada cluster identificado.  
- Calcular frequências categóricas por cluster.  
- Identificar características distintivas de cada segmento.  
- Analisar distribuição de churn por cluster.  

**RF14 - Atribuição de Novos Clientes a Clusters**  
- Implementar algoritmo eficiente para classificar novos clientes.  
- API de classificação em tempo real com latência < 50ms.  
- Suportar classificação em batch para base completa.  

**RF15 - Monitoramento e Evolução de Clusters**  
- Detectar drift nos padrões de clusters.  
- Identificar necessidade de re-clustering.  
- Implementar evolução incremental dos clusters.  

---

### 2.4 Módulo de Campanhas e Ações (RF16)

**RF16 - Segmentação Baseada em Clusters**  
- Agrupar clientes por cluster k-modes + score de churn.  
- Gerar estratégias específicas por perfil de cluster.  
- Personalizar abordagens de retenção por segmento.  

---

### 2.5 Módulo de Monitoramento e Alertas (RF22-RF25)

**RF22 - Monitoramento de Clusters**  
- Detectar mudanças na distribuição de clusters.  
- Monitorar estabilidade dos centroides k-modes.  
- Alertar sobre drift comportamental por segmento.  

**RF23 - Dashboard de Clusters e Churn**  
- Visualizar distribuição de clientes por cluster.  
- Mostrar taxa de churn por segmento.  
- Heatmap de características por cluster.  
- Monitorar performance dos modelos especializados.  

**RF24 - Relatórios de Segmentação**  
- Análise evolutiva dos clusters ao longo do tempo.  
- Performance de campanhas por segmento.  
- ROI diferenciado por cluster.  
- Insights de migração entre clusters.  

**RF25 - Alertas Especializados**  
- Notificar sobre clientes migrando para clusters de alto risco.  
- Alertar sobre degradação de performance por segmento.  
- Avisar sobre necessidade de re-clustering.  

---

## 3. REQUISITOS NÃO FUNCIONAIS

### 3.1 Performance
- **RNF01:** Clustering k-modes da base completa em até 6 horas
- **RNF02:** Classificação de cluster em tempo real < 50ms (p95)
- **RNF03:** Predição completa (cluster + churn) < 100ms (p95)
- **RNF04:** Throughput de 10000+ classificações por segundo
- **RNF05:** Disponibilidade do sistema de 99.5%

### 3.2 Escalabilidade
- **RNF06:** Suportar até 50 clusters simultâneos
- **RNF07:** Processar base de até 10 milhões de clientes
- **RNF08:** Escalar processamento conforme número de clusters
- **RNF09:** Suportar até 1TB de dados categóricos por dia

### 3.3 Qualidade de Clustering
- **RNF10:** Score mínimo de 0.3 para clusters
- **RNF11:** Máximo 5% de clientes não classificáveis
- **RNF12:** Estabilidade de clusters > 90% entre execuções
- **RNF13:** Cobertura mínima de 95% das categorias principais

### 3.4-3.6 [Mantidos os mesmos requisitos de Segurança, Confiabilidade, Usabilidade e Manutenibilidade do documento original]

---

## 4. ARQUITETURA DO SISTEMA

### 4.1 Componentes Principais
- **Data Lake:** Armazenamento raw dos dados categóricos
- **Categorical Data Warehouse:** Dados processados e features categóricas
- **K-modes Clustering Engine:** Motor de clustering categórico
- **ML Platform:** Treinamento de modelos especializados por cluster
- **Hybrid Scoring API:** Interface unificada cluster + churn
- **Cluster Dashboard:** Visualização de segmentação
- **Message Queue:** Processamento assíncrono de clustering

### 4.2 Tecnologias Específicas para K-modes
- **Clustering:** kmodes (Python), pandas, numpy
- **Data Processing:** Apache Spark com suporte categórico
- **ML Framework:** Scikit-learn, XGBoost, MLflow
- **Categorical Analysis:** Category Encoders, Pandas-profiling
- **Visualization:** Plotly, Seaborn para dados categóricos

---

## 5. CRONOGRAMA E MARCOS

### Fase 1 - Fundação e Clustering
- Setup da infraestrutura para dados categóricos
- Pipeline de dados categóricos
- Implementação e otimização do k-modes
- Análise inicial de clusters

### Fase 2 - Modelos Especializados
- Feature engineering categórico completo
- Desenvolvimento de modelos por cluster
- Sistema de scoring híbrido
- Validação de performance por segmento

### Fase 3 - Integração e Dashboard
- Dashboard de clusters e churn
- API unificada de classificação + predição
- Integração com sistemas existentes
- Testes de carga para clustering

### Fase 4 - Produção e Monitoramento
- Deploy em produção com monitoramento de clusters
- Sistema de alertas especializados
- Treinamento de usuários em segmentação
- Otimização final de performance

---

## 6. ESTRUTURA DO PROJETO PYTHON COM K-MODES

### 6.1 Organização de Diretórios
```
churn_prediction_kmodes/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── database.py          # Conexões com BD
│   │   ├── extractors.py        # Extração de dados categóricos
│   │   ├── validators.py        # Validação de dados categóricos
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── kmodes_engine.py     # Implementação k-modes otimizada
│   │   ├── cluster_optimizer.py # Otimização de número de clusters
│   │   ├── cluster_analyzer.py  # Análise e profiling de clusters
│   │   └── cluster_predictor.py # Classificação de novos clientes
│   ├── features/
│   │   ├── __init__.py
│   │   ├── categorical_engineering.py # Feature engineering categórico
│   │   ├── store.py             # Feature store categórico
│   │   └── transformers.py      # Transformações categóricas
│   ├── models/
│   │   ├── training.py          # Treinamento coordenado
│   │   ├── evaluation.py        # Avaliação por cluster
│   │   └── prediction.py        # Predição multiestágio
│   ├── monitoring/
│   │   ├── cluster_drift.py     # Detecção de drift em clusters
│   │   ├── cluster_stability.py # Monitoramento de estabilidade
│   │   ├── performance_tracker.py # Performance por cluster
│   │   └── alerts.py            # Alertas especializados
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app
│   │   ├── cluster_endpoints.py # Endpoints de clustering
│   │   ├── prediction_endpoints.py # Endpoints de predição
│   │   └── models.py            # Modelos Pydantic
│   └── dashboard/
│       ├── __init__.py
│       ├── cluster_app.py       
│       ├── cluster_components.py 
│       ├── churn_app.py         
│       └── charts.py            
├── config/
│   ├── __init__.py
│   ├── settings.py              
│   ├── cluster_config.py       
│   └── database_config.py       
├── tests/
│   ├── __init__.py
│   ├── test_clustering/        
│   ├── test_data/
│   ├── test_features/
│   ├── test_models/
│   └── test_api/
├── scripts/
│   ├── run_clustering.py        
│   ├── train_cluster_models.py 
│   ├── batch_prediction.py      
│   └── deploy_models.py        
├── data/
│   ├── raw/                     
│   ├── processed/               
│   ├── clusters/                
│   └── features/                
├── models/
│   ├── clusters/               
│   ├── specialized/             
│   └── artifacts/              
├── requirements.txt
├── setup.py
├── README.md
└── .env
```
