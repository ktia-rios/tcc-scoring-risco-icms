"""
=============================================================
TCC MBA Data Science — Análise de Fraudes Fiscais (ICMS)
=============================================================
Script: 02_preprocessamento.py
Objetivo: Processar dados brutos focando em
          cadeia de combustíveis do RJ + cruzamento CEIS
=============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import gc

# ─── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preprocessamento.log")
    ]
)
log = logging.getLogger(__name__)

# ─── Caminhos ─────────────────────────────────────────────
RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ─── Arquivos de entrada ──────────────────────────────────
ARQUIVOS = {
    "estabelecimentos": RAW_DIR / "K3241.K03200Y0.D51108.ESTABELE",
    "empresas":         RAW_DIR / "K3241.K03200Y0.D51108.EMPRECSV",
    "socios":           RAW_DIR / "K3241.K03200Y0.D51108.SOCIOCSV",
    "simples":          RAW_DIR / "F.K03200$W.SIMPLES.CSV.D51108",
    "cnaes":            RAW_DIR / "F.K03200$Z.D51108.CNAECSV",
    "ceis":             RAW_DIR / "20260908_CEIS.csv",
}

OUTPUT = PROC_DIR / "dataset_combustiveis_rj.parquet"

# ─── Configurações ────────────────────────────────────────
CHUNK_SIZE = 100_000
UF_FOCO    = "RJ"

# CNAEs da cadeia de combustíveis (4 primeiros dígitos)
CNAES_COMBUSTIVEIS = {
    "0600": "Extração de petróleo e gás natural",
    "1921": "Fabricação de produtos do refino de petróleo",
    "1922": "Fabricação de combustíveis",
    "4681": "Comércio atacadista de combustíveis",
    "4731": "Comércio varejista de combustíveis (postos)",
    "5221": "Transporte dutoviário de combustíveis",
}

# ─── Layouts ──────────────────────────────────────────────
COL_ESTABELECIMENTOS = [
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
]
COL_EMPRESAS = [
    "cnpj_basico", "razao_social", "natureza_juridica",
    "qualificacao_responsavel", "capital_social",
    "porte_empresa", "ente_federativo"
]
COL_SOCIOS = [
    "cnpj_basico", "identificador_socio", "nome_socio",
    "cpf_cnpj_socio", "qualificacao_socio",
    "data_entrada_sociedade", "pais", "representante_legal",
    "nome_representante", "qualificacao_representante", "faixa_etaria"
]
COL_SIMPLES = [
    "cnpj_basico", "opcao_simples", "data_opcao_simples",
    "data_exclusao_simples", "opcao_mei",
    "data_opcao_mei", "data_exclusao_mei"
]


# ══════════════════════════════════════════════════════════
# ETAPA 1 — Estabelecimentos (RJ + combustíveis)
# ══════════════════════════════════════════════════════════

def filtrar_estabelecimentos() -> pd.DataFrame:
    log.info("Etapa 1 — Filtrando estabelecimentos RJ + combustíveis...")

    chunks = []
    total_lido = 0

    reader = pd.read_csv(
        ARQUIVOS["estabelecimentos"],
        sep=";", encoding="latin1", header=None,
        names=COL_ESTABELECIMENTOS,
        dtype={"cnpj_basico": str, "cnpj_ordem": str, "cnpj_dv": str,
               "situacao_cadastral": str, "cnae_fiscal_principal": str,
               "uf": str},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for i, chunk in enumerate(reader):
        total_lido += len(chunk)

        # Filtro UF
        chunk = chunk[chunk["uf"] == UF_FOCO]
        if len(chunk) == 0:
            continue

        # Filtro CNAE (4 primeiros dígitos)
        chunk["cnae_4dig"] = chunk["cnae_fiscal_principal"].astype(str).str.strip().str[:4]
        chunk = chunk[chunk["cnae_4dig"].isin(CNAES_COMBUSTIVEIS.keys())]
        if len(chunk) == 0:
            continue

        # Manter apenas matrizes e filiais ativas/suspeitas
        chunk = chunk[chunk["situacao_cadastral"].isin(["02", "03", "04"])]

        if len(chunk) > 0:
            chunk = chunk[[
                "cnpj_basico", "cnpj_ordem", "cnpj_dv",
                "identificador_matriz_filial", "situacao_cadastral",
                "data_inicio_atividade", "cnae_fiscal_principal",
                "cnae_4dig", "uf", "municipio"
            ]].copy()
            chunks.append(chunk)

        if (i + 1) % 50 == 0:
            log.info(f"  chunk {i+1}: {total_lido:,} lidos | {sum(len(c) for c in chunks):,} filtrados")

    if not chunks:
        log.error("Nenhum estabelecimento encontrado!")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df["cnpj_basico"] = df["cnpj_basico"].str.zfill(8)
    df["cnpj_completo"] = (
        df["cnpj_basico"] +
        df["cnpj_ordem"].str.zfill(4) +
        df["cnpj_dv"].str.zfill(2)
    )

    # Limpeza
    df["data_inicio_atividade"] = pd.to_datetime(
        df["data_inicio_atividade"], format="%Y%m%d", errors="coerce"
    )
    df["idade_empresa_dias"] = (
        pd.Timestamp("2025-11-30") - df["data_inicio_atividade"]
    ).dt.days

    mapa_sit = {"02": "Ativa", "03": "Suspensa", "04": "Inapta"}
    df["situacao_desc"] = df["situacao_cadastral"].map(mapa_sit)
    df["e_matriz"] = df["identificador_matriz_filial"].astype(str).str.strip() == "1"
    df["cnae_descricao"] = df["cnae_4dig"].map(CNAES_COMBUSTIVEIS)

    log.info(f"  ✓ {len(df):,} estabelecimentos encontrados")
    log.info(f"  ✓ Matrizes: {df['e_matriz'].sum():,} | Filiais: {(~df['e_matriz']).sum():,}")
    log.info(f"\n  Distribuição por CNAE:")
    for cnae, desc in CNAES_COMBUSTIVEIS.items():
        qtd = (df["cnae_4dig"] == cnae).sum()
        if qtd > 0:
            log.info(f"    {cnae} — {desc}: {qtd:,}")

    gc.collect()
    return df


# ══════════════════════════════════════════════════════════
# ETAPA 2 — Empresas, Sócios e Simples
# ══════════════════════════════════════════════════════════

def carregar_complementares(cnpjs_alvo: set) -> tuple:
    log.info("\nEtapa 2 — Carregando dados complementares...")

    # Empresas
    log.info("  Lendo empresas...")
    emp_chunks = []
    reader = pd.read_csv(
        ARQUIVOS["empresas"], sep=";", encoding="latin1",
        header=None, names=COL_EMPRESAS,
        dtype={"cnpj_basico": str, "capital_social": str},
        chunksize=CHUNK_SIZE, low_memory=False,
    )
    for chunk in reader:
        chunk["cnpj_basico"] = chunk["cnpj_basico"].str.zfill(8)
        chunk = chunk[chunk["cnpj_basico"].isin(cnpjs_alvo)]
        if len(chunk) > 0:
            emp_chunks.append(chunk[["cnpj_basico", "capital_social", "porte_empresa"]])

    empresas = pd.concat(emp_chunks, ignore_index=True) if emp_chunks else pd.DataFrame()
    if not empresas.empty:
        empresas["capital_social"] = (
            empresas["capital_social"].str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce").fillna(0)
        )
        mapa_porte = {"00": "Não informado", "01": "Micro", "03": "Pequeno", "05": "Demais"}
        empresas["porte_desc"] = empresas["porte_empresa"].map(mapa_porte).fillna("Não informado")
    log.info(f"  ✓ {len(empresas):,} empresas")
    gc.collect()

    # Sócios
    log.info("  Lendo sócios...")
    soc_chunks = []
    reader = pd.read_csv(
        ARQUIVOS["socios"], sep=";", encoding="latin1",
        header=None, names=COL_SOCIOS,
        dtype={"cnpj_basico": str, "cpf_cnpj_socio": str},
        chunksize=CHUNK_SIZE, low_memory=False,
    )
    for chunk in reader:
        chunk["cnpj_basico"] = chunk["cnpj_basico"].str.zfill(8)
        chunk = chunk[chunk["cnpj_basico"].isin(cnpjs_alvo)]
        if len(chunk) > 0:
            soc_chunks.append(chunk[["cnpj_basico", "identificador_socio", "cpf_cnpj_socio"]])

    socios = pd.concat(soc_chunks, ignore_index=True) if soc_chunks else pd.DataFrame()
    if not socios.empty:
        mapa_socio = {"1": "PJ", "2": "PF", "3": "Estrangeiro"}
        socios["tipo_socio"] = socios["identificador_socio"].astype(str).map(mapa_socio).fillna("Desconhecido")
    log.info(f"  ✓ {len(socios):,} sócios")
    gc.collect()

    # Simples
    log.info("  Lendo Simples Nacional...")
    sim_chunks = []
    reader = pd.read_csv(
        ARQUIVOS["simples"], sep=";", encoding="latin1",
        header=None, names=COL_SIMPLES,
        dtype={"cnpj_basico": str},
        chunksize=CHUNK_SIZE, low_memory=False,
    )
    for chunk in reader:
        chunk["cnpj_basico"] = chunk["cnpj_basico"].str.zfill(8)
        chunk = chunk[chunk["cnpj_basico"].isin(cnpjs_alvo)]
        if len(chunk) > 0:
            sim_chunks.append(chunk[["cnpj_basico", "opcao_simples", "opcao_mei"]])

    simples = pd.concat(sim_chunks, ignore_index=True) if sim_chunks else pd.DataFrame()
    log.info(f"  ✓ {len(simples):,} registros Simples")
    gc.collect()

    return empresas, socios, simples


# ══════════════════════════════════════════════════════════
# ETAPA 3 — CEIS
# ══════════════════════════════════════════════════════════

def carregar_ceis() -> pd.DataFrame:
    log.info("\nEtapa 3 — Carregando CEIS...")

    df = pd.read_csv(
        ARQUIVOS["ceis"],
        sep=";", encoding="latin1",
        dtype=str, low_memory=False,
    )

    # Filtrar apenas PJ
    col_tipo = [c for c in df.columns if "TIPO" in c.upper() and "PESSOA" in c.upper()][0]
    col_cnpj = [c for c in df.columns if "CPF" in c.upper() or "CNPJ" in c.upper()][0]

    df = df[df[col_tipo] == "J"].copy()

    # Limpar CNPJ
    df["cnpj_basico_ceis"] = (
        df[col_cnpj]
        .str.replace(r"[.\-/\s]", "", regex=True)
        .str.strip()
        .str.zfill(14)
        .str[:8]  # apenas CNPJ básico para cruzamento
    )

    log.info(f"  ✓ {df['cnpj_basico_ceis'].nunique():,} CNPJs únicos sancionados no CEIS")
    return df[["cnpj_basico_ceis"]].drop_duplicates()


# ══════════════════════════════════════════════════════════
# ETAPA 4 — Feature Engineering
# ══════════════════════════════════════════════════════════

def criar_features(estab, empresas, socios, simples, ceis) -> pd.DataFrame:
    log.info("\nEtapa 4 — Feature Engineering...")

    # Agregar por CNPJ basico (empresa)
    # Contar filiais por empresa
    qtd_filiais = (
        estab[~estab["e_matriz"]]
        .groupby("cnpj_basico").size()
        .reset_index(name="qtd_filiais")
    )

    # Situação mais grave por empresa
    ordem_sit = {"Inapta": 3, "Suspensa": 2, "Ativa": 1}
    estab["sit_ordem"] = estab["situacao_desc"].map(ordem_sit).fillna(0)
    sit_empresa = (
        estab.groupby("cnpj_basico")
        .agg(
            situacao_principal=("sit_ordem", "max"),
            idade_empresa_dias=("idade_empresa_dias", "min"),
            cnae_4dig=("cnae_4dig", "first"),
            cnae_descricao=("cnae_descricao", "first"),
            municipio=("municipio", "first"),
        )
        .reset_index()
    )
    mapa_sit_inv = {3: "Inapta", 2: "Suspensa", 1: "Ativa"}
    sit_empresa["situacao_desc"] = sit_empresa["situacao_principal"].map(mapa_sit_inv).fillna("Desconhecida")

    # Sócios por empresa
    if not socios.empty:
        qtd_socios = socios.groupby("cnpj_basico").agg(
            qtd_socios=("tipo_socio", "count"),
            qtd_socios_pj=("tipo_socio", lambda x: (x == "PJ").sum()),
            qtd_socios_pf=("tipo_socio", lambda x: (x == "PF").sum()),
        ).reset_index()

        emp_por_socio = (
            socios[socios["tipo_socio"] == "PF"]
            .groupby("cpf_cnpj_socio")["cnpj_basico"]
            .nunique().reset_index()
            .rename(columns={"cnpj_basico": "qtd_emp_do_socio"})
        )
        socios_enr = socios.merge(emp_por_socio, on="cpf_cnpj_socio", how="left")
        max_emp_socio = (
            socios_enr.groupby("cnpj_basico")["qtd_emp_do_socio"]
            .max().reset_index()
            .rename(columns={"qtd_emp_do_socio": "max_empresas_por_socio"})
        )
    else:
        qtd_socios = pd.DataFrame(columns=["cnpj_basico", "qtd_socios", "qtd_socios_pj", "qtd_socios_pf"])
        max_emp_socio = pd.DataFrame(columns=["cnpj_basico", "max_empresas_por_socio"])

    # Montar dataset base (uma linha por empresa)
    df = sit_empresa.copy()
    df = df.merge(qtd_filiais,   on="cnpj_basico", how="left")
    df = df.merge(qtd_socios,    on="cnpj_basico", how="left")
    df = df.merge(max_emp_socio, on="cnpj_basico", how="left")

    if not empresas.empty:
        df = df.merge(
            empresas[["cnpj_basico", "capital_social", "porte_desc"]],
            on="cnpj_basico", how="left"
        )
    if not simples.empty:
        df = df.merge(
            simples[["cnpj_basico", "opcao_simples", "opcao_mei"]],
            on="cnpj_basico", how="left"
        )

    # Preencher nulos
    df["qtd_filiais"]            = df.get("qtd_filiais", pd.Series(0)).fillna(0).astype(int)
    df["qtd_socios"]             = df.get("qtd_socios", pd.Series(0)).fillna(0).astype(int)
    df["qtd_socios_pj"]          = df.get("qtd_socios_pj", pd.Series(0)).fillna(0).astype(int)
    df["qtd_socios_pf"]          = df.get("qtd_socios_pf", pd.Series(0)).fillna(0).astype(int)
    df["max_empresas_por_socio"] = df.get("max_empresas_por_socio", pd.Series(1)).fillna(1)
    df["capital_social"]         = df.get("capital_social", pd.Series(0)).fillna(0)

    # Features de risco
    df["empresa_nova"]             = (df["idade_empresa_dias"] < 365).astype(int)
    df["alta_concentracao_pj"]     = (df["qtd_socios_pj"] > 2).astype(int)
    df["socio_multiplas_empresas"] = (df["max_empresas_por_socio"] > 5).astype(int)
    df["inapta"]                   = (df["situacao_desc"] == "Inapta").astype(int)
    df["suspensa"]                 = (df["situacao_desc"] == "Suspensa").astype(int)
    df["tem_filiais"]              = (df["qtd_filiais"] > 0).astype(int)

    # Score de risco
    df["score_risco"] = (
        df["empresa_nova"]             * 2 +
        df["alta_concentracao_pj"]     * 2 +
        df["socio_multiplas_empresas"] * 3 +
        df["inapta"]                   * 4 +
        df["suspensa"]                 * 2
    )

    # ── Variável TARGET — cruzamento com CEIS ─────────────
    cnpjs_sancionados = set(ceis["cnpj_basico_ceis"].unique())
    df["sancionada"] = df["cnpj_basico"].isin(cnpjs_sancionados).astype(int)

    log.info(f"  ✓ Dataset: {len(df):,} empresas, {df.shape[1]} variáveis")
    log.info(f"  ✓ Sancionadas (target=1): {df['sancionada'].sum():,}")
    log.info(f"  ✓ Não sancionadas (target=0): {(df['sancionada']==0).sum():,}")
    return df


# ══════════════════════════════════════════════════════════
# ETAPA 5 — Anonimização
# ══════════════════════════════════════════════════════════

def anonimizar(df: pd.DataFrame) -> pd.DataFrame:
    log.info("\nEtapa 5 — Anonimizando...")
    df = df.copy()
    mapa = {v: f"EMP_RJ_{i:05d}" for i, v in enumerate(df["cnpj_basico"].unique())}
    df["id_empresa"] = df["cnpj_basico"].map(mapa)
    df = df.drop(columns=["cnpj_basico", "situacao_principal"], errors="ignore")
    log.info("  ✓ CNPJs substituídos por IDs anônimos")
    return df


# ══════════════════════════════════════════════════════════
# RELATÓRIO
# ══════════════════════════════════════════════════════════

def relatorio(df: pd.DataFrame):
    print("\n" + "═" * 60)
    print("  DATASET FINAL — Cadeia de Combustíveis RJ | Nov/2025")
    print("═" * 60)
    print(f"  Empresas:                  {len(df):>10,}")
    print(f"  Variáveis:                 {df.shape[1]:>10}")
    print(f"  Sancionadas (target=1):    {df['sancionada'].sum():>10,}")
    print(f"  Não sancionadas:           {(df['sancionada']==0).sum():>10,}")
    print(f"  Taxa de sanção:            {df['sancionada'].mean()*100:>9.2f}%")
    print(f"\n  Por situação:")
    print(df["situacao_desc"].value_counts().to_string())
    print(f"\n  Por CNAE:")
    print(df["cnae_descricao"].value_counts().to_string())
    print(f"\n  Score de risco médio:      {df['score_risco'].mean():>10.2f}")
    print(f"  Score de risco máximo:     {df['score_risco'].max():>10}")
    print(f"\n  TOP 10 por score de risco:")
    cols = ["id_empresa", "cnae_descricao", "situacao_desc",
            "qtd_socios", "max_empresas_por_socio", "score_risco", "sancionada"]
    print(df.nlargest(10, "score_risco")[[c for c in cols if c in df.columns]].to_string(index=False))
    print("═" * 60)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  TCC MBA — Pré-processamento Combustíveis RJ ║")
    log.info("╚══════════════════════════════════════════════╝")

    # Etapa 1 — Estabelecimentos
    estab = filtrar_estabelecimentos()
    if estab.empty:
        return

    # CNPJs no escopo
    cnpjs_alvo = set(estab["cnpj_basico"].unique())
    log.info(f"\n  CNPJs únicos no escopo: {len(cnpjs_alvo):,}")

    # Etapa 2 — Complementares
    empresas, socios, simples = carregar_complementares(cnpjs_alvo)

    # Etapa 3 — CEIS
    ceis = carregar_ceis()

    # Etapa 4 — Features + Target
    dataset = criar_features(estab, empresas, socios, simples, ceis)

    # Etapa 5 — Anonimização
    dataset = anonimizar(dataset)

    # Relatório
    relatorio(dataset)

    # Exportar
    dataset.to_parquet(OUTPUT, index=False)
    dataset.to_csv(PROC_DIR / "dataset_combustiveis_rj.csv",
                   index=False, encoding="utf-8-sig")
    log.info(f"\n✅ Dataset salvo em: {OUTPUT}")
    log.info(f"   {len(dataset):,} empresas | {dataset.shape[1]} variáveis")
    log.info("   Próximo passo: python3 src/03_feature_engineering.py")


if __name__ == "__main__":
    main()
