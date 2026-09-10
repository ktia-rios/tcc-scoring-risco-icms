# 🔍 Scoring de Risco de Fraude Fiscal — ICMS

> **TCC — MBA em Data Science | USP ESALQ**  
> Tema: Identificação de Uso Indevido de Créditos de ICMS via Análise de Dados Cadastrais  
> Aluna: Kátia Rios Nóbrega de Mello  
> Ano: 2026

---

## 📋 Descrição

Este projeto desenvolve um modelo de detecção de padrões suspeitos de
aproveitamento indevido de créditos de ICMS, utilizando dados públicos de
CNPJs, cadastros de empresas sancionadas (CEIS) e técnicas de Machine Learning
supervisionado e não supervisionado.

**Pergunta de pesquisa:**
> *É possível identificar automaticamente contribuintes com comportamento
> suspeito de aproveitamento indevido de crédito de ICMS a partir de
> padrões cadastrais e societários?*

---

## 🏗️ Estrutura do Projeto

```
tcc-scoring-risco-icms/
│
├── data/
│   ├── raw/                  # Dados brutos baixados (não versionados)
│   ├── temp/                 # Checkpoints intermediários (não versionados)
│   └── processed/            # Datasets limpos e anonimizados (não versionados)
│
├── src/
│   ├── 01_coleta.py          # ✅ Coleta e feature engineering — CNPJ nov/2025
│   ├── 02_preprocessamento.py # 🔜 Cruzamento com CEIS e preparação para modelagem
│   ├── 03_feature_engineering.py # 🔜 Features avançadas
│   ├── 04_modelagem.py       # 🔜 Isolation Forest + XGBoost
│   ├── 05_avaliacao.py       # 🔜 Métricas e visualizações
│   └── utils.py              # Funções auxiliares
│
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_graficos_e_visualizacoes.ipynb
│   └── 03_experimentos_modelos.ipynb
│
├── outputs/
│   ├── graficos/
│   ├── tabelas/
│   └── modelos/
│
├── docs/
│   ├── referencias/
│   └── texto_tcc/
│
├── venv/                     # Ambiente virtual (não versionado)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🗂️ Fontes de Dados

| Fonte | Conteúdo | Referência |
|---|---|---|
| Receita Federal | Cadastro CNPJ — nov/2025 | arquivos.receitafederal.gov.br |
| Portal Transparência | Empresas sancionadas (CEIS) | portaldatransparencia.gov.br |
| CONFAZ | Arrecadação ICMS por estado | fazenda.gov.br/confaz |

> **Conformidade legal:** todos os dados utilizados são públicos e abertos.
> Identificadores foram anonimizados em conformidade com a LGPD (Lei 13.709/2018)
> e o Art. 198 do Código Tributário Nacional.

---

## 📊 Status do Dataset (nov/2025)

| Indicador | Valor |
|---|---|
| Empresas analisadas | 5.060.707 |
| Variáveis criadas | 21 |
| Empresas novas (<1 ano) | 522.096 |
| Empresas inaptas | 1.492.856 (29%) |
| Empresas suspensas | 33.890 |
| Sócio com múltiplas empresas | 348.535 |
| Score de risco máximo | 9 |

---

## 🤖 Abordagem Metodológica

### Não Supervisionado
| Modelo | Aplicação |
|---|---|
| Isolation Forest | Detecção de anomalias cadastrais |
| DBSCAN | Clustering de perfis suspeitos |

### Supervisionado
| Modelo | Aplicação |
|---|---|
| XGBoost | Score de risco com rótulos do CEIS |
| Random Forest | Comparativo e interpretabilidade |

---

## ⚙️ Como Executar

```bash
# 1. Ativar ambiente virtual
source venv/bin/activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar pipeline
python3 src/01_coleta.py        # ✅ Concluído
python3 src/02_preprocessamento.py
python3 src/03_feature_engineering.py
python3 src/04_modelagem.py
python3 src/05_avaliacao.py
```

> **Nota:** O script `01_coleta.py` usa checkpoints — se interrompido,
> retoma automaticamente de onde parou ao ser executado novamente.

---

## 📚 Referências Principais

- Receita Federal do Brasil — Dados Abertos CNPJ
- LGPD — Lei nº 13.709/2018
- CTN — Código Tributário Nacional, Art. 198
- Liu, F. T. et al. (2008). *Isolation Forest*. IEEE ICDM
- Chandola, V. et al. (2009). *Anomaly Detection: A Survey*. ACM Computing Surveys
- Akoglu, L. et al. (2015). *Graph-based Anomaly Detection and Description*
