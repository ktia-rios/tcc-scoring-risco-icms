"""
=============================================================
TCC MBA Data Science — Análise de Fraudes Fiscais (ICMS)
=============================================================
Script: 03_feature_engineering.py
Objetivo: Enriquecer o dataset com features avançadas
          para detecção de anomalias na cadeia de
          combustíveis do RJ
=============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# ─── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("feature_engineering.log")
    ]
)
log = logging.getLogger(__name__)

# ─── Caminhos ─────────────────────────────────────────────
PROC_DIR = Path("data/processed")
INPUT    = PROC_DIR / "dataset_combustiveis_rj.parquet"
OUTPUT   = PROC_DIR / "dataset_features_rj.parquet"


# ══════════════════════════════════════════════════════════
# ETAPA 1 — Carregar dataset
# ══════════════════════════════════════════════════════════

def carregar_dataset() -> pd.DataFrame:
    log.info("Etapa 1 — Carregando dataset...")
    df = pd.read_parquet(INPUT)
    log.info(f"  ✓ {len(df):,} empresas, {df.shape[1]} variáveis")
    log.info(f"  Variáveis disponíveis: {list(df.columns)}")
    return df


# ══════════════════════════════════════════════════════════
# ETAPA 2 — Features de estrutura societária
# ══════════════════════════════════════════════════════════

def features_societarias(df: pd.DataFrame) -> pd.DataFrame:
    log.info("\nEtapa 2 — Features societárias...")

    # Proporção de sócios PJ sobre total
    df["proporcao_socios_pj"] = np.where(
        df["qtd_socios"] > 0,
        df["qtd_socios_pj"] / df["qtd_socios"],
        0
    )

    # Proporção de sócios PF sobre total
    df["proporcao_socios_pf"] = np.where(
        df["qtd_socios"] > 0,
        df["qtd_socios_pf"] / df["qtd_socios"],
        0
    )

    # Empresa sem sócios registrados (suspeito)
    df["sem_socios"] = (df["qtd_socios"] == 0).astype(int)

    # Sócio com muitas empresas — refinado por faixas
    df["socio_risco_baixo"]  = ((df["max_empresas_por_socio"] > 2) &
                                 (df["max_empresas_por_socio"] <= 5)).astype(int)
    df["socio_risco_medio"]  = ((df["max_empresas_por_socio"] > 5) &
                                 (df["max_empresas_por_socio"] <= 10)).astype(int)
    df["socio_risco_alto"]   = (df["max_empresas_por_socio"] > 10).astype(int)

    log.info(f"  ✓ Sem sócios: {df['sem_socios'].sum():,}")
    log.info(f"  ✓ Sócio risco alto (>10 empresas): {df['socio_risco_alto'].sum():,}")
    return df


# ══════════════════════════════════════════════════════════
# ETAPA 3 — Features financeiras
# ══════════════════════════════════════════════════════════

def features_financeiras(df: pd.DataFrame) -> pd.DataFrame:
    log.info("\nEtapa 3 — Features financeiras...")

    # Capital social por filial
    # Empresas com muitas filiais e capital baixo são suspeitas
    df["capital_por_filial"] = np.where(
        df["qtd_filiais"] > 0,
        df["capital_social"] / (df["qtd_filiais"] + 1),
        df["capital_social"]
    )

    # Capital social por sócio
    df["capital_por_socio"] = np.where(
        df["qtd_socios"] > 0,
        df["capital_social"] / df["qtd_socios"],
        df["capital_social"]
    )

    # Capital muito baixo para o setor de combustíveis
    # Postos e distribuidoras precisam de capital relevante
    df["capital_muito_baixo"] = (df["capital_social"] < 10_000).astype(int)
    df["capital_baixo"]       = ((df["capital_social"] >= 10_000) &
                                  (df["capital_social"] < 100_000)).astype(int)
    df["capital_adequado"]    = (df["capital_social"] >= 100_000).astype(int)

    # Porte incompatível com capital
    # Micro empresa com capital alto OU empresa grande com capital baixo
    df["porte_incompativel"] = (
        ((df.get("porte_desc", "") == "Micro") & (df["capital_social"] > 500_000)) |
        ((df.get("porte_desc", "") == "Demais") & (df["capital_social"] < 50_000))
    ).astype(int)

    log.info(f"  ✓ Capital muito baixo (<10k): {df['capital_muito_baixo'].sum():,}")
    log.info(f"  ✓ Capital adequado (>=100k):  {df['capital_adequado'].sum():,}")
    return df


# ══════════════════════════════════════════════════════════
# ETAPA 4 — Features operacionais
# ══════════════════════════════════════════════════════════

def features_operacionais(df: pd.DataFrame) -> pd.DataFrame:
    log.info("\nEtapa 4 — Features operacionais...")

    # Razão filiais/sócios — muitas filiais com poucos sócios
    df["razao_filiais_socios"] = np.where(
        df["qtd_socios"] > 0,
        df["qtd_filiais"] / df["qtd_socios"],
        df["qtd_filiais"]
    )

    # Concentração por município
    # Municípios com muitas empresas suspeitas têm risco maior
    municipio_col = "municipio" if "municipio" in df.columns else None
    if municipio_col:
        qtd_por_municipio = df.groupby(municipio_col).size().reset_index(name="empresas_no_municipio")
        inaptas_por_municipio = (
            df[df["situacao_desc"] == "Inapta"]
            .groupby(municipio_col).size()
            .reset_index(name="inaptas_no_municipio")
        )
        df = df.merge(qtd_por_municipio,   on=municipio_col, how="left")
        df = df.merge(inaptas_por_municipio, on=municipio_col, how="left")
        df["inaptas_no_municipio"] = df["inaptas_no_municipio"].fillna(0)
        df["taxa_inapta_municipio"] = np.where(
            df["empresas_no_municipio"] > 0,
            df["inaptas_no_municipio"] / df["empresas_no_municipio"],
            0
        )
        log.info(f"  ✓ Taxa inapta por município calculada")

    # Simples Nacional no setor de combustíveis é incomum
    # Distribuidoras e grandes operadores não deveriam ser Simples
    if "opcao_simples" in df.columns:
        df["simples_suspeito"] = (
            (df["opcao_simples"] == "S") &
            (df["cnae_4dig"].isin(["0600", "1921", "1922", "4681"]))
        ).astype(int)
        log.info(f"  ✓ Simples suspeito (grande porte no Simples): {df['simples_suspeito'].sum():,}")
    else:
        df["simples_suspeito"] = 0

    return df


# ══════════════════════════════════════════════════════════
# ETAPA 5 — Features temporais
# ══════════════════════════════════════════════════════════

def features_temporais(df: pd.DataFrame) -> pd.DataFrame:
    log.info("\nEtapa 5 — Features temporais...")

    # Anos de operação
    df["anos_operacao"] = (df["idade_empresa_dias"] / 365).round(1)

    # Faixas de idade
    df["empresa_muito_nova"]  = (df["idade_empresa_dias"] < 180).astype(int)
    df["empresa_nova"]        = ((df["idade_empresa_dias"] >= 180) &
                                  (df["idade_empresa_dias"] < 365)).astype(int)
    df["empresa_jovem"]       = ((df["idade_empresa_dias"] >= 365) &
                                  (df["idade_empresa_dias"] < 365*3)).astype(int)
    df["empresa_madura"]      = (df["idade_empresa_dias"] >= 365*3).astype(int)

    # Empresa antiga mas inapta — possível abandono ou fraude histórica
    df["antiga_inapta"] = (
        (df["idade_empresa_dias"] > 365*5) &
        (df["situacao_desc"] == "Inapta")
    ).astype(int)

    # Empresa muito nova já inapta — abertura para fraude
    df["nova_inapta"] = (
        (df["idade_empresa_dias"] < 365*2) &
        (df["situacao_desc"] == "Inapta")
    ).astype(int)

    log.info(f"  ✓ Empresas muito novas (<6 meses): {df['empresa_muito_nova'].sum():,}")
    log.info(f"  ✓ Antigas e inaptas (>5 anos):     {df['antiga_inapta'].sum():,}")
    log.info(f"  ✓ Novas e inaptas (<2 anos):        {df['nova_inapta'].sum():,}")
    return df


# ══════════════════════════════════════════════════════════
# ETAPA 6 — Score de risco refinado
# ══════════════════════════════════════════════════════════

def score_risco_refinado(df: pd.DataFrame) -> pd.DataFrame:
    log.info("\nEtapa 6 — Calculando score de risco refinado...")

    df["score_risco_v2"] = (
        # Situação cadastral
        df["inapta"]                   * 4 +
        df["suspensa"]                 * 2 +

        # Estrutura societária
        df["socio_risco_alto"]         * 4 +
        df["socio_risco_medio"]        * 2 +
        df["socio_risco_baixo"]        * 1 +
        df["alta_concentracao_pj"]     * 2 +
        df["sem_socios"]               * 2 +

        # Temporal
        df["empresa_muito_nova"]       * 3 +
        df["nova_inapta"]              * 3 +
        df["antiga_inapta"]            * 2 +

        # Financeiro
        df["capital_muito_baixo"]      * 2 +
        df["porte_incompativel"]       * 2 +

        # Operacional
        df["simples_suspeito"]         * 2
    )

    log.info(f"  ✓ Score v2 médio:   {df['score_risco_v2'].mean():.2f}")
    log.info(f"  ✓ Score v2 máximo:  {df['score_risco_v2'].max()}")

    # Classificação de risco
    df["classe_risco"] = pd.cut(
        df["score_risco_v2"],
        bins=[-1, 2, 5, 9, 100],
        labels=["Baixo", "Médio", "Alto", "Crítico"]
    )

    dist = df["classe_risco"].value_counts().sort_index()
    for classe, qtd in dist.items():
        log.info(f"  {classe}: {qtd:,} empresas ({qtd/len(df)*100:.1f}%)")

    return df


# ══════════════════════════════════════════════════════════
# ETAPA 7 — Preparar para modelagem
# ══════════════════════════════════════════════════════════

def preparar_modelagem(df: pd.DataFrame) -> pd.DataFrame:
    log.info("\nEtapa 7 — Preparando para modelagem...")

    # Remover colunas não numéricas que não serão usadas no modelo
    # (manter para análise exploratória)
    cols_info = ["id_empresa", "cnae_descricao", "cnae_4dig",
                 "situacao_desc", "municipio", "classe_risco",
                 "sancionada", "uf"]
    cols_modelo = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in ["sancionada"]]

    log.info(f"  ✓ Variáveis para o modelo: {len(cols_modelo)}")
    log.info(f"  Variáveis: {cols_modelo}")

    # Salvar lista de features para usar na modelagem
    features_path = PROC_DIR / "features_modelo.txt"
    with open(features_path, "w") as f:
        f.write("\n".join(cols_modelo))
    log.info(f"  ✓ Lista de features salva em: {features_path}")

    return df


# ══════════════════════════════════════════════════════════
# ETAPA ANP — Volume de vendas por município (2024)
# ══════════════════════════════════════════════════════════

def features_anp(df: pd.DataFrame) -> pd.DataFrame:
    log.info("\nEtapa ANP — Cruzando com dados de vendas ANP 2024...")

    RAW_DIR = Path("data/raw")
    gasolina_path = RAW_DIR / "vendas-anuais-de-gasolina-c-por-municipio.csv"
    diesel_path   = RAW_DIR / "vendas-anuais-de-oleo-diesel-por-municipio.csv"

    if not gasolina_path.exists() or not diesel_path.exists():
        log.warning("  ⚠ Arquivos ANP não encontrados — pulando etapa")
        return df

    # Carregar e filtrar RJ 2024
    def carregar_anp(path):
        d = pd.read_csv(
            path, sep=";",
            encoding="utf-8-sig",
            dtype={"CÓDIGO IBGE": str},
            lineterminator="\n",
            low_memory=False,
        )
        d.columns = d.columns.str.strip()
        d["VENDAS"] = pd.to_numeric(d["VENDAS"], errors="coerce").fillna(0)
        d["ANO"] = pd.to_numeric(d["ANO"], errors="coerce")
        return d[(d["ANO"] == 2024) & (d["UF"] == "RJ")].copy()

    gasolina = carregar_anp(gasolina_path)
    diesel   = carregar_anp(diesel_path)

    # Agregar por município
    gas_mun = (gasolina.groupby("MUNICÍPIO")["VENDAS"]
               .sum().reset_index()
               .rename(columns={"MUNICÍPIO": "municipio_anp", "VENDAS": "vol_gasolina_litros_2024"}))

    die_mun = (diesel.groupby("MUNICÍPIO")["VENDAS"]
               .sum().reset_index()
               .rename(columns={"MUNICÍPIO": "municipio_anp", "VENDAS": "vol_diesel_litros_2024"}))

    anp = gas_mun.merge(die_mun, on="municipio_anp", how="outer").fillna(0)
    anp["vol_total_combustivel_2024"] = anp["vol_gasolina_litros_2024"] + anp["vol_diesel_litros_2024"]

    log.info(f"  ✓ {len(anp):,} municípios do RJ com dados ANP 2024")

    # Cruzar com dataset — via código do município
    # O dataset tem 'municipio' como código numérico
    # A ANP tem nome do município — vamos usar empresas_no_municipio como proxy
    # e criar features de volume per capita de empresas

    # Número de postos por município (do nosso dataset)
    if "municipio" in df.columns:
        postos_mun = (df[df["cnae_4dig"] == "4731"]
                      .groupby("municipio").size()
                      .reset_index(name="postos_no_municipio"))
        df = df.merge(postos_mun, on="municipio", how="left")
        df["postos_no_municipio"] = df["postos_no_municipio"].fillna(0).astype(int)

        # Volume médio esperado por posto no município
        # Usando volume total do RJ / total de postos como referência
        vol_total_rj  = anp["vol_total_combustivel_2024"].sum()
        postos_total  = df[df["cnae_4dig"] == "4731"]["municipio"].count()
        vol_medio_rj  = vol_total_rj / postos_total if postos_total > 0 else 0

        log.info(f"  ✓ Volume total RJ 2024: {vol_total_rj/1e9:.2f} bilhões de litros")
        log.info(f"  ✓ Volume médio por posto RJ: {vol_medio_rj/1e6:.2f} milhões de litros")

        # Feature: número de postos per capita de volume
        # Municípios com muitos postos mas baixo volume são suspeitos
        df["vol_medio_esperado_posto"] = vol_medio_rj
        df["postos_acima_do_esperado"] = (
            df["postos_no_municipio"] > (vol_total_rj / vol_medio_rj / len(anp))
        ).astype(int)

    log.info("  ✓ Features ANP adicionadas")
    return df


# ══════════════════════════════════════════════════════════
# RELATÓRIO FINAL
# ══════════════════════════════════════════════════════════

def relatorio(df: pd.DataFrame):
    print("\n" + "═" * 60)
    print("  DATASET ENRIQUECIDO — Cadeia Combustíveis RJ")
    print("═" * 60)
    print(f"  Empresas:                  {len(df):>10,}")
    print(f"  Variáveis totais:          {df.shape[1]:>10}")
    print(f"  Sancionadas (target=1):    {df['sancionada'].sum():>10,}")
    print(f"\n  Classificação de risco:")
    dist = df["classe_risco"].value_counts().sort_index()
    for classe, qtd in dist.items():
        pct = qtd/len(df)*100
        barra = "█" * int(pct/2)
        print(f"    {classe:<10} {qtd:>5,} ({pct:>5.1f}%) {barra}")

    print(f"\n  TOP 10 por score de risco v2:")
    cols = ["id_empresa", "cnae_descricao", "situacao_desc",
            "anos_operacao", "qtd_socios", "max_empresas_por_socio",
            "capital_social", "score_risco_v2", "classe_risco", "sancionada"]
    print(df.nlargest(10, "score_risco_v2")[[c for c in cols if c in df.columns]].to_string(index=False))
    print("═" * 60)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  TCC MBA — Feature Engineering               ║")
    log.info("║  Cadeia de Combustíveis RJ                   ║")
    log.info("╚══════════════════════════════════════════════╝")

    # Carregar
    df = carregar_dataset()

    # Features
    df = features_societarias(df)
    df = features_financeiras(df)
    df = features_operacionais(df)
    df = features_temporais(df)
    df = features_anp(df)
    df = score_risco_refinado(df)
    df = preparar_modelagem(df)

    # Relatório
    relatorio(df)

    # Exportar
    df.to_parquet(OUTPUT, index=False)
    df.to_csv(PROC_DIR / "dataset_features_rj.csv",
              index=False, encoding="utf-8-sig")

    log.info(f"\n✅ Dataset enriquecido salvo: {OUTPUT}")
    log.info(f"   {len(df):,} empresas | {df.shape[1]} variáveis")
    log.info("   Próximo passo: python3 src/04_modelagem.py")


if __name__ == "__main__":
    main()
