"""
=============================================================
TCC MBA Data Science — Análise de Fraudes Fiscais (ICMS)
=============================================================
Script: 01_coleta.py — versão com checkpoints
Estratégia: salva cada etapa em disco, retoma de onde parou
Referência: novembro/2025
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
        logging.FileHandler("coleta_cnpj.log")
    ]
)
log = logging.getLogger(__name__)

# ─── Caminhos ─────────────────────────────────────────────
RAW_DIR   = Path("data/raw")
PROC_DIR  = Path("data/processed")
TEMP_DIR  = Path("data/temp")          # checkpoints temporários
PROC_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ─── Arquivos de entrada ──────────────────────────────────
ARQUIVOS = {
    "empresas":         RAW_DIR / "K3241.K03200Y0.D51108.EMPRECSV",
    "estabelecimentos": RAW_DIR / "K3241.K03200Y0.D51108.ESTABELE",
    "socios":           RAW_DIR / "K3241.K03200Y0.D51108.SOCIOCSV",
    "simples":          RAW_DIR / "F.K03200$W.SIMPLES.CSV.D51108",
    "cnaes":            RAW_DIR / "F.K03200$Z.D51108.CNAECSV",
}

# ─── Checkpoints ──────────────────────────────────────────
CHECKPOINTS = {
    "estab":    TEMP_DIR / "ckpt_estab.parquet",
    "empresas": TEMP_DIR / "ckpt_empresas.parquet",
    "socios":   TEMP_DIR / "ckpt_socios.parquet",
    "simples":  TEMP_DIR / "ckpt_simples.parquet",
    "cnpjs":    TEMP_DIR / "ckpt_cnpjs.txt",
}

# ─── Configurações ────────────────────────────────────────
CHUNK_SIZE = 50_000

CNAES_FOCO = [
    "45", "46", "47",
    "49", "50", "51",
    "10", "11", "12",
    "13", "14", "15",
    "20", "21", "22",
    "23", "24", "25",
    "26", "27", "28",
]

# ─── Layouts ──────────────────────────────────────────────
COL_EMPRESAS = [
    "cnpj_basico", "razao_social", "natureza_juridica",
    "qualificacao_responsavel", "capital_social",
    "porte_empresa", "ente_federativo"
]
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
COL_CNAES = ["codigo", "descricao"]


# ══════════════════════════════════════════════════════════
# UTILITÁRIOS DE CHECKPOINT
# ══════════════════════════════════════════════════════════

def checkpoint_existe(nome: str) -> bool:
    """Verifica se o checkpoint já existe em disco."""
    return CHECKPOINTS[nome].exists()


def salvar_checkpoint(df: pd.DataFrame, nome: str):
    """Salva DataFrame como checkpoint."""
    CHECKPOINTS[nome].parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CHECKPOINTS[nome], index=False)
    log.info(f"  💾 Checkpoint salvo: {CHECKPOINTS[nome].name} ({CHECKPOINTS[nome].stat().st_size/1e6:.1f} MB)")


def carregar_checkpoint(nome: str) -> pd.DataFrame:
    """Carrega DataFrame de checkpoint."""
    log.info(f"  ⚡ Checkpoint encontrado — pulando etapa ({CHECKPOINTS[nome].name})")
    return pd.read_parquet(CHECKPOINTS[nome])


def salvar_cnpjs(cnpjs: set):
    """Salva lista de CNPJs em arquivo texto."""
    with open(CHECKPOINTS["cnpjs"], "w") as f:
        f.write("\n".join(sorted(cnpjs)))
    log.info(f"  💾 {len(cnpjs):,} CNPJs salvos em disco")


def carregar_cnpjs() -> set:
    """Carrega lista de CNPJs do arquivo texto."""
    with open(CHECKPOINTS["cnpjs"], "r") as f:
        cnpjs = set(line.strip() for line in f if line.strip())
    log.info(f"  ⚡ {len(cnpjs):,} CNPJs carregados do disco")
    return cnpjs


# ══════════════════════════════════════════════════════════
# ETAPA 1 — CNAEs
# ══════════════════════════════════════════════════════════

def ler_cnaes() -> pd.DataFrame:
    log.info("Etapa 1 — Lendo CNAEs...")
    df = pd.read_csv(
        ARQUIVOS["cnaes"], sep=";", encoding="latin1",
        header=None, names=COL_CNAES, dtype=str
    )
    df["codigo"] = df["codigo"].str.strip().str.zfill(7)
    df["cnae_key"] = df["codigo"]
    log.info(f"  ✓ {len(df):,} CNAEs carregados")
    return df


# ══════════════════════════════════════════════════════════
# ETAPA 2 — Estabelecimentos
# ══════════════════════════════════════════════════════════

def filtrar_estabelecimentos(cnaes_df: pd.DataFrame) -> pd.DataFrame:
    if checkpoint_existe("estab"):
        return carregar_checkpoint("estab")

    log.info("\nEtapa 2 — Filtrando estabelecimentos em chunks...")
    chunks_filtrados = []
    total_lido = 0
    total_filtrado = 0

    reader = pd.read_csv(
        ARQUIVOS["estabelecimentos"],
        sep=";", encoding="latin1", header=None,
        names=COL_ESTABELECIMENTOS,
        dtype={"cnpj_basico": str, "cnpj_ordem": str, "cnpj_dv": str,
               "situacao_cadastral": str, "cnae_fiscal_principal": str, "uf": str},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for i, chunk in enumerate(reader):
        total_lido += len(chunk)
        chunk = chunk[chunk["identificador_matriz_filial"].astype(str).str.strip() == "1"]
        chunk = chunk[chunk["situacao_cadastral"].isin(["02", "03", "04"])]
        chunk["divisao_cnae"] = chunk["cnae_fiscal_principal"].astype(str).str.strip().str[:2]
        chunk = chunk[chunk["divisao_cnae"].isin(CNAES_FOCO)]

        if len(chunk) > 0:
            chunk = chunk[[
                "cnpj_basico", "cnpj_ordem", "cnpj_dv",
                "situacao_cadastral", "data_inicio_atividade",
                "cnae_fiscal_principal", "uf"
            ]].copy()
            chunks_filtrados.append(chunk)
            total_filtrado += len(chunk)

        if (i + 1) % 20 == 0:
            log.info(f"  chunk {i+1}: {total_lido:,} lidos | {total_filtrado:,} filtrados")

    df = pd.concat(chunks_filtrados, ignore_index=True)
    df["cnpj_basico"] = df["cnpj_basico"].str.zfill(8)
    df["cnpj_completo"] = (df["cnpj_basico"] +
                           df["cnpj_ordem"].str.zfill(4) +
                           df["cnpj_dv"].str.zfill(2))
    df["data_inicio_atividade"] = pd.to_datetime(
        df["data_inicio_atividade"], format="%Y%m%d", errors="coerce"
    )
    df["idade_empresa_dias"] = (
        pd.Timestamp("2025-11-30") - df["data_inicio_atividade"]
    ).dt.days

    mapa_sit = {"02": "Ativa", "03": "Suspensa", "04": "Inapta"}
    df["situacao_desc"] = df["situacao_cadastral"].map(mapa_sit).fillna("Outra")

    # Enriquecer com descrição CNAE
    df["cnae_key"] = df["cnae_fiscal_principal"].astype(str).str.strip().str.zfill(7)
    df = df.merge(
        cnaes_df[["cnae_key", "descricao"]].rename(columns={"descricao": "cnae_descricao"}),
        on="cnae_key", how="left"
    ).drop(columns=["cnae_key"])

    log.info(f"  ✓ {len(df):,} estabelecimentos filtrados")
    salvar_checkpoint(df, "estab")
    gc.collect()
    return df


# ══════════════════════════════════════════════════════════
# ETAPA 3 — Empresas
# ══════════════════════════════════════════════════════════

def filtrar_empresas(cnpjs_alvo: set) -> pd.DataFrame:
    if checkpoint_existe("empresas"):
        return carregar_checkpoint("empresas")

    log.info("\nEtapa 3 — Filtrando empresas em chunks...")
    chunks = []
    total_lido = 0

    reader = pd.read_csv(
        ARQUIVOS["empresas"],
        sep=";", encoding="latin1", header=None,
        names=COL_EMPRESAS,
        dtype={"cnpj_basico": str, "capital_social": str},
        chunksize=CHUNK_SIZE, low_memory=False,
    )

    for i, chunk in enumerate(reader):
        total_lido += len(chunk)
        chunk["cnpj_basico"] = chunk["cnpj_basico"].str.zfill(8)
        cnpjs_series = pd.Series(list(cnpjs_alvo), name="cnpj_basico")
        chunk = chunk.merge(cnpjs_series.to_frame(), on="cnpj_basico", how="inner")
        if len(chunk) > 0:
            chunks.append(chunk[["cnpj_basico", "capital_social", "porte_empresa"]])
        if (i + 1) % 20 == 0:
            log.info(f"  chunk {i+1}: {total_lido:,} lidos")

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if not df.empty:
        df["capital_social"] = (
            df["capital_social"].str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce").fillna(0)
        )
        mapa_porte = {"00": "Não informado", "01": "Micro", "03": "Pequeno", "05": "Demais"}
        df["porte_desc"] = df["porte_empresa"].map(mapa_porte).fillna("Não informado")

    log.info(f"  ✓ {len(df):,} empresas filtradas")
    salvar_checkpoint(df, "empresas")
    gc.collect()
    return df


# ══════════════════════════════════════════════════════════
# ETAPA 4 — Sócios
# ══════════════════════════════════════════════════════════

def filtrar_socios(cnpjs_alvo: set) -> pd.DataFrame:
    if checkpoint_existe("socios"):
        return carregar_checkpoint("socios")

    log.info("\nEtapa 4 — Filtrando sócios em chunks...")
    chunks = []
    total_lido = 0

    reader = pd.read_csv(
        ARQUIVOS["socios"],
        sep=";", encoding="latin1", header=None,
        names=COL_SOCIOS,
        dtype={"cnpj_basico": str, "cpf_cnpj_socio": str},
        chunksize=CHUNK_SIZE, low_memory=False,
    )

    for i, chunk in enumerate(reader):
        total_lido += len(chunk)
        chunk["cnpj_basico"] = chunk["cnpj_basico"].str.zfill(8)
        cnpjs_series = pd.Series(list(cnpjs_alvo), name="cnpj_basico")
        chunk = chunk.merge(cnpjs_series.to_frame(), on="cnpj_basico", how="inner")
        if len(chunk) > 0:
            chunks.append(chunk[["cnpj_basico", "identificador_socio", "cpf_cnpj_socio"]])
        if (i + 1) % 20 == 0:
            log.info(f"  chunk {i+1}: {total_lido:,} lidos")

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if not df.empty:
        df["cnpj_basico"] = df["cnpj_basico"].str.zfill(8)
        mapa_socio = {"1": "PJ", "2": "PF", "3": "Estrangeiro"}
        df["tipo_socio"] = df["identificador_socio"].astype(str).map(mapa_socio).fillna("Desconhecido")

    log.info(f"  ✓ {len(df):,} sócios filtrados")
    salvar_checkpoint(df, "socios")
    gc.collect()
    return df


# ══════════════════════════════════════════════════════════
# ETAPA 5 — Simples Nacional
# ══════════════════════════════════════════════════════════

def filtrar_simples(cnpjs_alvo: set) -> pd.DataFrame:
    if checkpoint_existe("simples"):
        return carregar_checkpoint("simples")

    log.info("\nEtapa 5 — Filtrando Simples Nacional em chunks...")
    chunks = []
    total_lido = 0

    reader = pd.read_csv(
        ARQUIVOS["simples"],
        sep=";", encoding="latin1", header=None,
        names=COL_SIMPLES,
        dtype={"cnpj_basico": str},
        chunksize=CHUNK_SIZE, low_memory=False,
    )

    for i, chunk in enumerate(reader):
        total_lido += len(chunk)
        chunk["cnpj_basico"] = chunk["cnpj_basico"].str.zfill(8)
        cnpjs_series = pd.Series(list(cnpjs_alvo), name="cnpj_basico")
        chunk = chunk.merge(cnpjs_series.to_frame(), on="cnpj_basico", how="inner")
        if len(chunk) > 0:
            chunks.append(chunk[["cnpj_basico", "opcao_simples", "opcao_mei"]])
        if (i + 1) % 20 == 0:
            log.info(f"  chunk {i+1}: {total_lido:,} lidos")

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    log.info(f"  ✓ {len(df):,} registros Simples filtrados")
    salvar_checkpoint(df, "simples")
    gc.collect()
    return df


# ══════════════════════════════════════════════════════════
# ETAPA 6 — Feature Engineering
# ══════════════════════════════════════════════════════════

def criar_features(estab, empresas, socios, simples) -> pd.DataFrame:
    log.info("\nEtapa 6 — Feature Engineering...")

    mapa_socio = {"1": "PJ", "2": "PF", "3": "Estrangeiro"}
    if "tipo_socio" not in socios.columns:
        socios["tipo_socio"] = socios["identificador_socio"].astype(str).map(mapa_socio).fillna("Desconhecido")

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

    df = estab.copy()
    df = df.merge(empresas[["cnpj_basico", "capital_social", "porte_desc"]], on="cnpj_basico", how="left")
    df = df.merge(qtd_socios,    on="cnpj_basico", how="left")
    df = df.merge(max_emp_socio, on="cnpj_basico", how="left")
    df = df.merge(simples[["cnpj_basico", "opcao_simples", "opcao_mei"]], on="cnpj_basico", how="left")

    for col in ["qtd_socios", "qtd_socios_pj", "qtd_socios_pf"]:
        df[col] = df[col].fillna(0).astype(int)
    df["max_empresas_por_socio"] = df["max_empresas_por_socio"].fillna(1)
    df["capital_social"] = df["capital_social"].fillna(0)

    df["empresa_nova"]             = df["idade_empresa_dias"] < 365
    df["alta_concentracao_pj"]     = df["qtd_socios_pj"] > 2
    df["socio_multiplas_empresas"] = df["max_empresas_por_socio"] > 5
    df["inapta"]                   = df["situacao_desc"] == "Inapta"
    df["suspensa"]                 = df["situacao_desc"] == "Suspensa"

    df["score_risco"] = (
        df["empresa_nova"].astype(int)             * 2 +
        df["alta_concentracao_pj"].astype(int)     * 2 +
        df["socio_multiplas_empresas"].astype(int) * 3 +
        df["inapta"].astype(int)                   * 4 +
        df["suspensa"].astype(int)                 * 2
    )

    log.info(f"  ✓ Dataset: {len(df):,} empresas, {df.shape[1]} variáveis")
    return df


# ══════════════════════════════════════════════════════════
# ETAPA 7 — Anonimização
# ══════════════════════════════════════════════════════════

def anonimizar(df: pd.DataFrame) -> pd.DataFrame:
    log.info("\nEtapa 7 — Anonimizando...")
    df = df.copy()
    mapa = {v: f"EMP_{i:07d}" for i, v in enumerate(df["cnpj_basico"].unique())}
    df["id_empresa"] = df["cnpj_basico"].map(mapa)
    df = df.drop(columns=["cnpj_basico", "cnpj_completo", "cnpj_ordem",
                           "cnpj_dv", "situacao_cadastral"], errors="ignore")
    log.info("  ✓ Anonimização concluída")
    return df


# ══════════════════════════════════════════════════════════
# RELATÓRIO
# ══════════════════════════════════════════════════════════

def relatorio(df: pd.DataFrame):
    print("\n" + "═" * 55)
    print("  RESUMO DO DATASET — Nov/2025")
    print("═" * 55)
    print(f"  Empresas:                  {len(df):>12,}")
    print(f"  Variáveis:                 {df.shape[1]:>12}")
    print(f"  Empresas novas (<1 ano):   {df['empresa_nova'].sum():>12,}")
    print(f"  Alta conc. sócios PJ:      {df['alta_concentracao_pj'].sum():>12,}")
    print(f"  Sócio c/ múlt. empresas:   {df['socio_multiplas_empresas'].sum():>12,}")
    print(f"  Inaptas:                   {df['inapta'].sum():>12,}")
    print(f"  Suspensas:                 {df['suspensa'].sum():>12,}")
    print(f"  Score médio:               {df['score_risco'].mean():>12.2f}")
    print(f"  Score máximo:              {df['score_risco'].max():>12}")
    print("═" * 55)
    print("\n  TOP 10 por score de risco:")
    cols = ["id_empresa", "uf", "situacao_desc", "idade_empresa_dias",
            "qtd_socios", "max_empresas_por_socio", "score_risco"]
    print(df.nlargest(10, "score_risco")[[c for c in cols if c in df.columns]].to_string(index=False))


# ══════════════════════════════════════════════════════════
# EXPORTAÇÃO
# ══════════════════════════════════════════════════════════

def exportar(df: pd.DataFrame):
    caminho = PROC_DIR / "dataset_tcc_nov2025.parquet"
    df.to_parquet(caminho, index=False)
    log.info(f"  ✓ Parquet: {caminho} ({caminho.stat().st_size/1e6:.1f} MB)")
    df.head(100).to_csv(PROC_DIR / "amostra_100.csv", index=False, encoding="utf-8-sig")
    log.info("  ✓ Amostra CSV salva")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  TCC MBA — Pipeline CNPJ Nov/2025            ║")
    log.info("║  Versão com checkpoints — retoma do meio     ║")
    log.info("╚══════════════════════════════════════════════╝")

    # Etapa 1 — CNAEs
    cnaes = ler_cnaes()

    # Etapa 2 — Estabelecimentos
    estab = filtrar_estabelecimentos(cnaes)

    # CNPJs do escopo
    if CHECKPOINTS["cnpjs"].exists():
        cnpjs_alvo = carregar_cnpjs()
    else:
        cnpjs_alvo = set(estab["cnpj_basico"].unique())
        salvar_cnpjs(cnpjs_alvo)
        log.info(f"  CNPJs no escopo: {len(cnpjs_alvo):,}")

    # Liberar memória do estab (já salvo em checkpoint)
    del estab
    gc.collect()

    # Etapa 3 — Empresas
    empresas = filtrar_empresas(cnpjs_alvo)

    # Etapa 4 — Sócios
    socios = filtrar_socios(cnpjs_alvo)

    # Etapa 5 — Simples
    simples = filtrar_simples(cnpjs_alvo)

    # Recarregar estab do checkpoint (já liberado da memória)
    log.info("\nRecarregando estabelecimentos do checkpoint...")
    estab = carregar_checkpoint("estab")

    # Etapa 6 — Features
    dataset = criar_features(estab, empresas, socios, simples)

    # Etapa 7 — Anonimização
    dataset = anonimizar(dataset)

    # Relatório e exportação
    relatorio(dataset)
    exportar(dataset)

    log.info("\n✅ Pipeline concluído! Arquivos em data/processed/")
    log.info("   Próximo passo: python3 src/02_preprocessamento.py")


if __name__ == "__main__":
    main()
