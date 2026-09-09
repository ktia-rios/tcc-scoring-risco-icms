"""
=============================================================
TCC MBA Data Science — Análise de Fraudes Fiscais (ICMS)
=============================================================
Script: 01_coleta.py
Leitura e pré-processamento dos arquivos CNPJ já baixados
Referência: novembro/2025
=============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import zipfile
import glob

# ─── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("coleta_cnpj.log")
    ]
)
log = logging.getLogger(__name__)

# ─── Caminhos ─────────────────────────────────────────────
RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ─── Mapeamento dos arquivos reais no disco ────────────────
# Esses são os nomes exatos encontrados em data/raw/
ARQUIVOS = {
    "empresas":         RAW_DIR / "K3241.K03200Y0.D51108.EMPRECSV",
    "estabelecimentos": RAW_DIR / "K3241.K03200Y0.D51108.ESTABELE",
    "socios":           RAW_DIR / "K3241.K03200Y0.D51108.SOCIOCSV",
    "simples":          RAW_DIR / "F.K03200$W.SIMPLES.CSV.D51108",
    "cnaes":            RAW_DIR / "F.K03200$Z.D51108.CNAECSV",
}

# ─── Layouts oficiais da RFB ──────────────────────────────
LAYOUTS = {
    "empresas": {
        "colunas": [
            "cnpj_basico", "razao_social", "natureza_juridica",
            "qualificacao_responsavel", "capital_social",
            "porte_empresa", "ente_federativo"
        ],
        "tipos": {"cnpj_basico": str, "capital_social": str}
    },
    "estabelecimentos": {
        "colunas": [
            "cnpj_basico", "cnpj_ordem", "cnpj_dv",
            "identificador_matriz_filial", "nome_fantasia",
            "situacao_cadastral", "data_situacao_cadastral",
            "motivo_situacao_cadastral", "nome_cidade_exterior",
            "pais", "data_inicio_atividade", "cnae_fiscal_principal",
            "cnae_fiscal_secundaria", "tipo_logradouro", "logradouro",
            "numero", "complemento", "bairro", "cep", "uf",
            "municipio", "ddd1", "telefone1", "ddd2", "telefone2",
            "ddd_fax", "fax", "email", "situacao_especial",
            "data_situacao_especial"
        ],
        "tipos": {
            "cnpj_basico": str, "cnpj_ordem": str, "cnpj_dv": str,
            "situacao_cadastral": str, "cnae_fiscal_principal": str, "uf": str
        }
    },
    "socios": {
        "colunas": [
            "cnpj_basico", "identificador_socio", "nome_socio",
            "cpf_cnpj_socio", "qualificacao_socio",
            "data_entrada_sociedade", "pais", "representante_legal",
            "nome_representante", "qualificacao_representante", "faixa_etaria"
        ],
        "tipos": {"cnpj_basico": str, "cpf_cnpj_socio": str}
    },
    "simples": {
        "colunas": [
            "cnpj_basico", "opcao_simples", "data_opcao_simples",
            "data_exclusao_simples", "opcao_mei",
            "data_opcao_mei", "data_exclusao_mei"
        ],
        "tipos": {"cnpj_basico": str}
    },
    "cnaes": {
        "colunas": ["codigo", "descricao"],
        "tipos": {"codigo": str}
    },
}


# ══════════════════════════════════════════════════════════
# 1. LEITURA DOS ARQUIVOS
# ══════════════════════════════════════════════════════════

def ler_arquivo(tipo: str, nrows: int = None) -> pd.DataFrame:
    """Lê um arquivo CSV da base CNPJ."""
    caminho = ARQUIVOS[tipo]
    layout  = LAYOUTS[tipo]

    if not caminho.exists():
        log.error(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

    tamanho_mb = caminho.stat().st_size / 1e6
    log.info(f"Lendo {tipo}: {caminho.name} ({tamanho_mb:.0f} MB)")

    try:
        df = pd.read_csv(
            caminho,
            sep=";",
            encoding="latin1",
            header=None,
            names=layout["colunas"],
            dtype=layout["tipos"],
            low_memory=False,
            nrows=nrows,  # None = ler tudo
        )
        log.info(f"  ✓ {len(df):,} registros carregados")
        return df
    except Exception as e:
        log.error(f"  ✗ Erro ao ler {tipo}: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════
# 2. LIMPEZA
# ══════════════════════════════════════════════════════════

def limpar_empresas(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Limpando empresas...")
    df["capital_social"] = (
        df["capital_social"]
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    mapa_porte = {"00": "Não informado", "01": "Micro", "03": "Pequeno", "05": "Demais"}
    df["porte_desc"] = df["porte_empresa"].map(mapa_porte).fillna("Não informado")
    return df


def limpar_estabelecimentos(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Limpando estabelecimentos...")
    df["cnpj_completo"] = (
        df["cnpj_basico"].str.zfill(8) +
        df["cnpj_ordem"].str.zfill(4) +
        df["cnpj_dv"].str.zfill(2)
    )
    for col in ["data_situacao_cadastral", "data_inicio_atividade"]:
        df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")

    mapa_situacao = {"01": "Nula", "02": "Ativa", "03": "Suspensa", "04": "Inapta", "08": "Baixada"}
    df["situacao_desc"] = df["situacao_cadastral"].map(mapa_situacao).fillna("Desconhecida")
    df["idade_empresa_dias"] = (pd.Timestamp("2025-11-30") - df["data_inicio_atividade"]).dt.days
    df["e_matriz"] = df["identificador_matriz_filial"].astype(str).str.strip() == "1"
    return df


def limpar_socios(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Limpando sócios...")
    df["data_entrada_sociedade"] = pd.to_datetime(
        df["data_entrada_sociedade"], format="%Y%m%d", errors="coerce"
    )
    mapa_socio = {"1": "PJ", "2": "PF", "3": "Estrangeiro"}
    df["tipo_socio"] = df["identificador_socio"].map(mapa_socio).fillna("Desconhecido")
    return df


# ══════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

def criar_features(empresas, estabelecimentos, socios, simples, cnaes) -> pd.DataFrame:
    log.info("\n── Feature Engineering ──────────────────────────────")

    # Sócios por empresa
    qtd_socios = socios.groupby("cnpj_basico").agg(
        qtd_socios=("nome_socio", "count"),
        qtd_socios_pj=("tipo_socio", lambda x: (x == "PJ").sum()),
        qtd_socios_pf=("tipo_socio", lambda x: (x == "PF").sum()),
    ).reset_index()

    # Quantas empresas cada sócio PF possui (indicador de laranja)
    empresas_por_socio = (
        socios[socios["tipo_socio"] == "PF"]
        .groupby("cpf_cnpj_socio")["cnpj_basico"]
        .nunique()
        .reset_index()
        .rename(columns={"cnpj_basico": "qtd_empresas_do_socio"})
    )
    socios_enr = socios.merge(empresas_por_socio, on="cpf_cnpj_socio", how="left")
    max_empresas_socio = (
        socios_enr.groupby("cnpj_basico")["qtd_empresas_do_socio"]
        .max().reset_index()
        .rename(columns={"qtd_empresas_do_socio": "max_empresas_por_socio"})
    )

    # Filiais e UFs por empresa
    qtd_filiais = (
        estabelecimentos[~estabelecimentos["e_matriz"]]
        .groupby("cnpj_basico")["cnpj_completo"]
        .count().reset_index()
        .rename(columns={"cnpj_completo": "qtd_filiais"})
    )
    qtd_ufs = (
        estabelecimentos.groupby("cnpj_basico")["uf"]
        .nunique().reset_index()
        .rename(columns={"uf": "qtd_ufs"})
    )

    # Estabelecimentos com situação suspeita
    suspeitos = (
        estabelecimentos[estabelecimentos["situacao_desc"].isin(["Inapta", "Suspensa"])]
        .groupby("cnpj_basico").size().reset_index(name="qtd_estab_suspeitos")
    )

    # Base: apenas matrizes
    # Debug temporário — ver quantas matrizes existem
log.info(f"  e_matriz True: {estabelecimentos['e_matriz'].sum():,}")
log.info(f"  id_matriz sample: {estabelecimentos['identificador_matriz_filial'].unique()[:5]}")

matrizes = estabelecimentos[estabelecimentos["e_matriz"]][[
        "cnpj_basico", "cnpj_completo", "situacao_desc",
        "data_inicio_atividade", "idade_empresa_dias",
        "cnae_fiscal_principal", "uf"
    ]].copy()

    # Enriquecer com descrição do CNAE
    cnaes["codigo"] = cnaes["codigo"].str.strip()
    matrizes["cnae_fiscal_principal"] = matrizes["cnae_fiscal_principal"].str.strip()
    matrizes = matrizes.merge(
        cnaes.rename(columns={"codigo": "cnae_fiscal_principal", "descricao": "cnae_descricao"}),
        on="cnae_fiscal_principal", how="left"
    )

    # Merge de todas as features
    df = matrizes.copy()
    df = df.merge(empresas[["cnpj_basico", "capital_social", "porte_desc"]], on="cnpj_basico", how="left")
    df = df.merge(qtd_socios,          on="cnpj_basico", how="left")
    df = df.merge(max_empresas_socio,  on="cnpj_basico", how="left")
    df = df.merge(qtd_filiais,         on="cnpj_basico", how="left")
    df = df.merge(qtd_ufs,             on="cnpj_basico", how="left")
    df = df.merge(suspeitos,           on="cnpj_basico", how="left")
    df = df.merge(simples[["cnpj_basico", "opcao_simples", "opcao_mei"]], on="cnpj_basico", how="left")

    # Preencher nulos
    for col in ["qtd_socios", "qtd_socios_pj", "qtd_socios_pf", "qtd_filiais", "qtd_ufs", "qtd_estab_suspeitos"]:
        df[col] = df[col].fillna(0).astype(int)
    df["max_empresas_por_socio"] = df["max_empresas_por_socio"].fillna(1)

    # Features de risco
    df["empresa_nova"]             = df["idade_empresa_dias"] < 365
    df["alta_concentracao_pj"]     = df["qtd_socios_pj"] > 2
    df["socio_multiplas_empresas"] = df["max_empresas_por_socio"] > 5
    df["tem_estab_suspeito"]       = df["qtd_estab_suspeitos"] > 0
    df["inapta"]                   = df["situacao_desc"] == "Inapta"

    # Score de risco
    df["score_risco"] = (
        df["empresa_nova"].astype(int)             * 2 +
        df["alta_concentracao_pj"].astype(int)     * 2 +
        df["socio_multiplas_empresas"].astype(int) * 3 +
        df["tem_estab_suspeito"].astype(int)       * 3 +
        df["inapta"].astype(int)                   * 4
    )

    log.info(f"  ✓ Dataset: {len(df):,} empresas, {df.shape[1]} variáveis")
    return df


# ══════════════════════════════════════════════════════════
# 4. ANONIMIZAÇÃO
# ══════════════════════════════════════════════════════════

def anonimizar(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Anonimizando...")
    df = df.copy()
    mapa = {v: f"EMP_{i:07d}" for i, v in enumerate(df["cnpj_basico"].unique())}
    df["id_empresa"] = df["cnpj_basico"].map(mapa)
    df = df.drop(columns=["cnpj_basico", "cnpj_completo"], errors="ignore")
    log.info("  ✓ CNPJs substituídos por IDs anônimos")
    return df


# ══════════════════════════════════════════════════════════
# 5. EXPORTAÇÃO
# ══════════════════════════════════════════════════════════

def exportar(df: pd.DataFrame):
    caminho = PROC_DIR / "dataset_tcc_nov2025.parquet"
    df.to_parquet(caminho, index=False)
    log.info(f"  ✓ Dataset salvo: {caminho} ({caminho.stat().st_size/1e6:.1f} MB)")

    # Também salva uma amostra em CSV para inspeção rápida
    amostra = PROC_DIR / "amostra_100.csv"
    df.head(100).to_csv(amostra, index=False, encoding="utf-8-sig")
    log.info(f"  ✓ Amostra CSV: {amostra}")


# ══════════════════════════════════════════════════════════
# RELATÓRIO
# ══════════════════════════════════════════════════════════

def relatorio(df: pd.DataFrame):
    print("\n" + "═" * 55)
    print("  RESUMO DO DATASET — Nov/2025")
    print("═" * 55)
    print(f"  Empresas (matrizes):       {len(df):>12,}")
    print(f"  Variáveis:                 {df.shape[1]:>12}")
    print(f"  Empresas novas (<1 ano):   {df['empresa_nova'].sum():>12,}")
    print(f"  Alta conc. sócios PJ:      {df['alta_concentracao_pj'].sum():>12,}")
    print(f"  Sócio c/ múlt. empresas:   {df['socio_multiplas_empresas'].sum():>12,}")
    print(f"  Com estab. suspeito:       {df['tem_estab_suspeito'].sum():>12,}")
    print(f"  Inaptas:                   {df['inapta'].sum():>12,}")
    print(f"  Score médio:               {df['score_risco'].mean():>12.2f}")
    print(f"  Score máximo:              {df['score_risco'].max():>12}")
    print("═" * 55)
    print("\n  TOP 10 por score de risco:")
    cols = ["id_empresa", "uf", "situacao_desc", "idade_empresa_dias",
            "qtd_socios", "max_empresas_por_socio", "score_risco"]
    print(df.nlargest(10, "score_risco")[[c for c in cols if c in df.columns]].to_string(index=False))


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  TCC MBA — Leitura Base CNPJ Nov/2025       ║")
    log.info("╚══════════════════════════════════════════════╝")

    # Leitura
    # Para testar rapidamente use nrows=500_000
    # Para processar tudo use nrows=None
    NROWS = None  # processar tudo
    
    empresas         = ler_arquivo("empresas",         nrows=NROWS)
    estabelecimentos = ler_arquivo("estabelecimentos", nrows=NROWS)
    socios           = ler_arquivo("socios",           nrows=NROWS)
    simples          = ler_arquivo("simples",          nrows=NROWS)
    cnaes            = ler_arquivo("cnaes")  # pequeno, lê tudo sempre

    if empresas.empty or estabelecimentos.empty:
        log.error("Arquivos essenciais não carregados. Verifique data/raw/")
        return

    # Limpeza
    empresas         = limpar_empresas(empresas)
    estabelecimentos = limpar_estabelecimentos(estabelecimentos)
    socios           = limpar_socios(socios)

    # Features
    dataset = criar_features(empresas, estabelecimentos, socios, simples, cnaes)

    # Anonimização
    dataset = anonimizar(dataset)

    # Relatório
    relatorio(dataset)

    # Exportação
    exportar(dataset)

    log.info("\n✅ Pipeline concluído! Arquivos em data/processed/")
    log.info("   Próximo passo: python src/02_preprocessamento.py")


if __name__ == "__main__":
    main()