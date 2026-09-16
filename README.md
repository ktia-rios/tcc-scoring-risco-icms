# 🔍 Machine Learning Aplicado à Detecção de Anomalias Fiscais na Cadeia de Combustíveis do RJ

> **TCC — MBA em Data Science e Analytics | USP ESALQ**
> **Aluna:** Kátia Rios Nóbrega de Mello
> **Ano:** 2026
> **Base de referência:** novembro/2025

---

## 📋 Descrição

Este projeto desenvolve um modelo de scoring de risco fiscal baseado em machine learning não supervisionado para identificar empresas com perfil anômalo na cadeia de combustíveis do Rio de Janeiro, utilizando exclusivamente dados públicos.

A abordagem combina o algoritmo **Isolation Forest** para detecção de anomalias multivariadas, o algoritmo **DBSCAN** para identificação de clusters de comportamento empresarial e um escore de risco estruturado, resultando em um protocolo de priorização de auditorias fiscais com ganho de eficiência estimado em **10 vezes** em relação ao método tradicional de sorteio aleatório.

**Pergunta de pesquisa:**
> *É possível identificar sistematicamente empresas com perfil anômalo na cadeia de combustíveis do Rio de Janeiro a partir de dados cadastrais públicos e técnicas de machine learning não supervisionado?*

---

## 🗂️ Estrutura do Projeto

```
tcc-scoring-risco-icms/
│
├── data/
│   ├── raw/                        # Dados brutos (não versionados — LGPD)
│   └── processed/                  # Datasets processados (não versionados)
│
├── src/
│   ├── 01_coleta.py                # Coleta e extração dos dados públicos
│   ├── 02_preprocessamento.py      # Limpeza, filtros e anonimização
│   ├── 03_feature_engineering.py   # Construção das 51 variáveis analíticas
│   ├── 04_modelagem.py             # Isolation Forest + DBSCAN + score combinado
│   ├── 05_avaliacao.py             # Métricas, visualizações e relatório executivo
│   ├── gerar_graficos_eda.py       # Gráficos da análise exploratória (paleta azul)
│   └── utils.py                    # Funções auxiliares
│
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb    # EDA completa — 7 gráficos
│   ├── 02_analise_anomalias.ipynb       # Análise profunda dos casos críticos
│   └── 03_experimentos_parametros.ipynb # Justificativa empírica dos parâmetros
│
├── outputs/
│   ├── graficos/                   # 26 gráficos (EDA + modelagem + experimentos)
│   └── tabelas/                    # Resultados, top 50, lista de auditoria
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🗃️ Fontes de Dados

| Fonte | Conteúdo | Referência |
|---|---|---|
| Receita Federal do Brasil | Cadastro CNPJ — nov/2025 | dadosabertos.rfb.gov.br/CNPJ |
| Portal da Transparência | CEIS — empresas sancionadas | portaldatransparencia.gov.br |
| ANP | Vendas anuais de combustíveis por município 2024 | gov.br/anp |

> **Conformidade legal:** todos os dados utilizados são públicos e abertos.
> CNPJs substituídos por identificadores anônimos (EMP_RJ_XXXXX) em conformidade
> com a LGPD (Lei nº 13.709/2018) e o Art. 198 do Código Tributário Nacional.
> Arquivos com dados brutos **não** são versionados neste repositório.

---

## ⚙️ Como Executar

### 1. Configurar o ambiente

```bash
# Criar e ativar o ambiente virtual
python3 -m venv venv
source venv/bin/activate       # Linux/Mac

# Instalar dependências
pip install -r requirements.txt
```

### 2. Executar o pipeline completo

```bash
# Etapa 1 — Coleta de dados
python src/01_coleta.py

# Etapa 2 — Pré-processamento e anonimização
python src/02_preprocessamento.py

# Etapa 3 — Engenharia de atributos (51 variáveis)
python src/03_feature_engineering.py

# Etapa 4 — Modelagem (IF + DBSCAN + score combinado)
python src/04_modelagem.py

# Etapa 5 — Avaliação e relatório executivo
python src/05_avaliacao.py
```

### 3. Exploração interativa

```bash
jupyter notebook notebooks/
```

---

## 🤖 Modelos e Parâmetros

| Modelo | Tipo | Parâmetros | Resultado |
|---|---|---|---|
| Isolation Forest | Não supervisionado | n_estimators=200, contamination=0.10 | 152 anomalias (10,0%) |
| DBSCAN | Não supervisionado | eps=1.5, min_samples=5, PCA=10 | 34 clusters, 250 outliers |
| Score combinado | Híbrido | IF(50%) + Manual(30%) + DBSCAN(20%) | 8 críticas, 142 altas |

---

## 📊 Resultados Principais

- **1.520 empresas** analisadas em 6 segmentos da cadeia de combustíveis do RJ
- **51 variáveis** construídas em 5 grupos temáticos
- **152 anomalias** detectadas pelo Isolation Forest — 94,1% em situação ativa
- **5 padrões** de anomalia fiscal identificados:
  - P1: Capital incompatível com a operação (4 empresas)
  - P2: Empresa fantasma — sem sócios + inapta (11 empresas)
  - P3: Laranja societário — sócio em >10 empresas (44 empresas)
  - P4: Abandono fiscal — antiga + inapta (21 empresas)
  - P5: Ativa suspeita — ativa + IF anômalo + sem sócios (19 empresas)
- **Ganho de eficiência:** 10x em relação ao sorteio aleatório
- **Silhouette DBSCAN:** 0,3705

---

## 📁 Outputs Gerados

| Arquivo | Conteúdo |
|---|---|
| `outputs/graficos/EDA_01 a EDA_07` | Gráficos de análise exploratória |
| `outputs/graficos/01 a 10` | Gráficos da modelagem |
| `outputs/graficos/EXP_01 a EXP_05` | Gráficos dos experimentos de parâmetros |
| `outputs/graficos/11 a 14` | Gráficos da análise de anomalias |
| `outputs/tabelas/top50_suspeitas.csv` | Top 50 empresas mais suspeitas |
| `outputs/tabelas/lista_prioridade_auditoria.csv` | 150 empresas prioritárias |
| `outputs/tabelas/relatorio_executivo.txt` | Relatório completo dos achados |

---

## 📚 Referências Principais

- Liu, F.T.; Ting, K.M.; Zhou, Z.H. (2008). Isolation Forest. IEEE ICDM.
- Ester, M. et al. (1996). DBSCAN. KDD-96.
- Chandola, V.; Banerjee, A.; Kumar, V. (2009). Anomaly Detection: A Survey. ACM.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.
- Xavier, A.R. et al. (2022). Identificação de evasão fiscal com IA. RAP/FGV.
- Lederman, L. (2021). The Fraud Triangle and Tax Evasion. Iowa Law Review.

---

## 📜 Conformidade e Privacidade

Este projeto foi desenvolvido em estrita conformidade com:
- **LGPD** (Lei nº 13.709/2018) — anonimização de todos os CNPJs
- **CTN Art. 198** — sigilo fiscal preservado
- **Dados exclusivamente públicos** — nenhum dado sigiloso foi acessado

---

*Repositório público para fins acadêmicos. Os dados brutos não estão incluídos.*
