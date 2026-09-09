# 🔍 Detecção de Uso Indevido de Créditos de ICMS via Análise de Notas Fiscais Eletrônicas

> **TCC — MBA em Data Science**  
> Tema: Análise de Fraudes Fiscais  
> Abordagem: Machine Learning + Análise de Grafos sobre dados públicos

---

## 📋 Descrição

Este projeto desenvolve um modelo de detecção de padrões suspeitos de
aproveitamento indevido de créditos de ICMS, utilizando dados públicos de
CNPJs, notas fiscais eletrônicas (NF-e) e cadastros de empresas sancionadas.

**Pergunta de pesquisa:**
> *É possível identificar automaticamente contribuintes com comportamento
> suspeito de aproveitamento indevido de crédito de ICMS a partir de
> padrões nas transações de NF-e?*

---

## 🏗️ Estrutura do Projeto

```
tcc_fraude_fiscal/
│
├── data/
│   ├── raw/                  # Dados brutos baixados (não versionados)
│   └── processed/            # Datasets limpos e anonimizados
│
├── src/
│   ├── 01_coleta.py          # Download e extração dos dados públicos
│   ├── 02_preprocessamento.py # Limpeza e padronização
│   ├── 03_feature_engineering.py # Criação de variáveis analíticas
│   ├── 04_modelagem.py       # Treinamento dos modelos (IF, XGBoost, DBSCAN)
│   ├── 05_avaliacao.py       # Métricas, curvas ROC, análise de resultados
│   └── utils.py              # Funções auxiliares compartilhadas
│
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_graficos_e_visualizacoes.ipynb
│   └── 03_experimentos_modelos.ipynb
│
├── outputs/
│   ├── graficos/             # Visualizações exportadas
│   ├── tabelas/              # Tabelas de resultados
│   └── modelos/              # Modelos treinados serializados (.pkl)
│
├── docs/
│   ├── referencias/          # Artigos e referências bibliográficas
│   └── texto_tcc/            # Rascunhos e versões do texto
│
├── venv/                     # Ambiente virtual (não versionado)
├── requirements.txt          # Dependências do projeto
├── .gitignore
└── README.md
```

---

## 🗂️ Fontes de Dados

| Fonte | Conteúdo | URL |
|---|---|---|
| Receita Federal | Cadastro CNPJ completo | dadosabertos.rfb.gov.br/CNPJ |
| SEFAZ Nacional | Estatísticas NF-e | nfe.fazenda.gov.br |
| Portal Transparência | Empresas sancionadas (CEIS/CNEP) | portaldatransparencia.gov.br |
| CONFAZ | Arrecadação ICMS por estado | fazenda.gov.br/confaz |
| STN / SICONFI | Receitas orçamentárias | siconfi.tesouro.gov.br |

> **Conformidade legal:** todos os dados utilizados são públicos e abertos.
> Identificadores foram anonimizados em conformidade com a LGPD (Lei 13.709/2018)
> e o Art. 198 do Código Tributário Nacional.

---

## ⚙️ Como Executar

### 1. Clonar e configurar o ambiente

```bash
# Ativar o ambiente virtual (já existente)
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 2. Executar o pipeline completo

```bash
# Etapa 1 — Coleta de dados
python src/01_coleta.py

# Etapa 2 — Pré-processamento
python src/02_preprocessamento.py

# Etapa 3 — Feature Engineering
python src/03_feature_engineering.py

# Etapa 4 — Modelagem
python src/04_modelagem.py

# Etapa 5 — Avaliação
python src/05_avaliacao.py
```

### 3. Exploração interativa

```bash
jupyter notebook notebooks/01_analise_exploratoria.ipynb
```

---

## 🤖 Modelos Utilizados

| Modelo | Tipo | Aplicação |
|---|---|---|
| Isolation Forest | Não supervisionado | Detecção de anomalias cadastrais |
| DBSCAN | Não supervisionado | Clustering de perfis suspeitos |
| XGBoost | Supervisionado | Score de risco (com rótulos do CEIS) |
| NetworkX (Grafos) | Análise de redes | Mapeamento de esquemas entre CNPJs |

---

## 📊 Features Principais

- Idade da empresa na data de análise
- Capital social declarado
- Quantidade e perfil de sócios (PF vs PJ)
- Número de empresas por sócio (indicador de "laranja")
- Número de filiais e UFs de atuação
- Situação cadastral (ativa, inapta, suspensa)
- Regime tributário (Simples, Lucro Presumido, Lucro Real)
- Volume e concentração de NF-e emitidas
- Razão crédito/débito de ICMS
- Taxa de cancelamento de NF-e

---

## 📚 Referências Principais

- Receita Federal do Brasil — Manual de Orientação do Contribuinte (NF-e)
- LGPD — Lei nº 13.709/2018
- CTN — Código Tributário Nacional, Art. 198
- Akoglu, L. et al. (2015). *Graph-based Anomaly Detection and Description*
- Chandola, V. et al. (2009). *Anomaly Detection: A Survey*. ACM Computing Surveys
- Liu, F. T. et al. (2008). *Isolation Forest*. IEEE ICDM

---

## 👤 Autor

**Kátia Rios Nóbrega de Mello**  
MBA em Data Science & Analitcs — USP Esalq  
Orientador: Francielly De Fátima Almeida  
Ano: 2026