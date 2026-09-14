"""
=============================================================
TCC MBA Data Science — Análise de Fraudes Fiscais (ICMS)
=============================================================
Script: 04_modelagem.py
Objetivo: Detecção de anomalias na cadeia de combustíveis RJ
          Camada 1: Isolation Forest (não supervisionado)
          Camada 2: DBSCAN (clustering)
          Camada 3: Validação com score manual e CEIS
Paleta:   Tons de azul + tracejado vermelho
=============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ─── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("modelagem.log")
    ]
)
log = logging.getLogger(__name__)

# ─── Caminhos ─────────────────────────────────────────────
PROC_DIR   = Path("data/processed")
GRAF_DIR   = Path("outputs/graficos")
TAB_DIR    = Path("outputs/tabelas")
MOD_DIR    = Path("outputs/modelos")
for d in [GRAF_DIR, TAB_DIR, MOD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INPUT = PROC_DIR / "dataset_features_rj.parquet"

# ─── Paleta acadêmica ─────────────────────────────────────
AZUL = {
    'escuro':  '#003366',
    'medio':   '#1a5276',
    'claro':   '#2e86c1',
    'palido':  '#85c1e9',
    'cinza':   '#566573',
}
VERMELHO = '#C0392B'  # tracejados de referência

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlecolor'] = AZUL['escuro']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# ══════════════════════════════════════════════════════════
# ETAPA 1 — Carregar e preparar dados
# ══════════════════════════════════════════════════════════

def carregar_preparar() -> tuple:
    log.info("Etapa 1 — Carregando e preparando dados...")
    df = pd.read_parquet(INPUT)
    log.info(f"  ✓ {len(df):,} empresas, {df.shape[1]} variáveis")

    cols_info_candidatas = [
        "id_empresa", "cnae_descricao", "cnae_4dig",
        "situacao_desc", "classe_risco", "sancionada",
        "municipio", "porte_desc", "opcao_simples",
        "opcao_mei", "score_risco", "score_risco_v2"
    ]
    cols_info = [c for c in cols_info_candidatas if c in df.columns]

    cols_excluir = cols_info + ["municipio", "score_risco",
                                "score_risco_v2", "vol_medio_esperado_posto"]
    cols_modelo = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in cols_excluir and c != "sancionada"
    ]

    log.info(f"  ✓ Features selecionadas: {len(cols_modelo)}")

    X = df[cols_modelo].fillna(0)
    meta = df[cols_info].copy()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=cols_modelo)
    log.info("  ✓ Dados normalizados com StandardScaler")
    return df, X, X_scaled, meta, cols_modelo


# ══════════════════════════════════════════════════════════
# ETAPA 2 — Isolation Forest
# ══════════════════════════════════════════════════════════

def rodar_isolation_forest(X_scaled, meta):
    log.info("\nEtapa 2 — Isolation Forest...")

    IF = IsolationForest(
        n_estimators=200,
        contamination=0.10,
        random_state=42,
        n_jobs=-1
    )
    IF.fit(X_scaled)

    scores    = IF.decision_function(X_scaled)
    predicoes = IF.predict(X_scaled)

    resultado = meta.copy()
    resultado["if_score"]      = scores
    resultado["if_anomalia"]   = (predicoes == -1).astype(int)
    resultado["if_score_norm"] = (scores - scores.min()) / (scores.max() - scores.min())
    resultado["if_score_risco"] = 1 - resultado["if_score_norm"]

    anomalias = resultado["if_anomalia"].sum()
    log.info(f"  ✓ Anomalias detectadas: {anomalias:,} ({anomalias/len(resultado)*100:.1f}%)")

    if "sancionada" in resultado.columns:
        for _, row in resultado[resultado["sancionada"]==1].iterrows():
            det = "✅ DETECTADA" if row["if_anomalia"]==1 else "❌ NÃO detectada"
            log.info(f"  Empresa sancionada {row['id_empresa']}: {det} | Score IF: {row['if_score']:.4f}")

    return resultado


# ══════════════════════════════════════════════════════════
# ETAPA 3 — DBSCAN
# ══════════════════════════════════════════════════════════

def rodar_dbscan(X_scaled, resultado):
    log.info("\nEtapa 3 — DBSCAN...")

    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    variancia = pca.explained_variance_ratio_.cumsum()[-1]
    log.info(f"  PCA: 10 componentes explicam {variancia*100:.1f}% da variância")

    db = DBSCAN(eps=1.5, min_samples=5, n_jobs=-1)
    clusters = db.fit_predict(X_pca)

    resultado["dbscan_cluster"] = clusters
    resultado["dbscan_outlier"] = (clusters == -1).astype(int)

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_outliers  = (clusters == -1).sum()
    log.info(f"  ✓ Clusters: {n_clusters} | Outliers: {n_outliers:,} ({n_outliers/len(resultado)*100:.1f}%)")

    resultado["pca_1"] = X_pca[:, 0]
    resultado["pca_2"] = X_pca[:, 1]
    return resultado


# ══════════════════════════════════════════════════════════
# ETAPA 4 — Score combinado
# ══════════════════════════════════════════════════════════

def score_combinado(resultado, df_original):
    log.info("\nEtapa 4 — Score combinado...")

    score_manual = df_original["score_risco_v2"]
    score_manual_norm = (score_manual - score_manual.min()) / \
                        (score_manual.max() - score_manual.min())
    resultado["score_manual_norm"] = score_manual_norm.values

    resultado["score_final"] = (
        resultado["if_score_risco"]    * 0.50 +
        resultado["score_manual_norm"] * 0.30 +
        resultado["dbscan_outlier"]    * 0.20
    )
    resultado["ranking"] = resultado["score_final"].rank(
        ascending=False, method="min").astype(int)
    resultado["classe_final"] = pd.cut(
        resultado["score_final"],
        bins=[-0.01, 0.30, 0.50, 0.70, 1.01],
        labels=["Baixo","Médio","Alto","Crítico"]
    )

    top_if     = set(resultado.nsmallest(152, "if_score")["id_empresa"])
    top_manual = set(resultado.nlargest(152, "score_manual_norm")["id_empresa"])
    concordancia = len(top_if & top_manual) / 152 * 100
    log.info(f"  Concordância IF vs Score Manual (top 10%): {concordancia:.1f}%")

    dist = resultado["classe_final"].value_counts().sort_index()
    for classe, qtd in dist.items():
        log.info(f"    {classe}: {qtd:,} ({qtd/len(resultado)*100:.1f}%)")

    return resultado


# ══════════════════════════════════════════════════════════
# ETAPA 5 — Visualizações
# ══════════════════════════════════════════════════════════

def gerar_visualizacoes(resultado, X, cols_modelo):
    log.info("\nEtapa 5 — Gerando visualizações...")

    CORES_RISCO = {
        'Crítico': AZUL['escuro'],
        'Alto':    AZUL['medio'],
        'Médio':   AZUL['claro'],
        'Baixo':   AZUL['palido'],
    }

    # ── Gráfico 1: Distribuição score IF ──────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(resultado["if_score"], bins=40, color=AZUL['claro'],
            edgecolor='white', linewidth=1.2, alpha=0.85)
    limiar = np.percentile(resultado["if_score"], 10)
    ax.axvline(limiar, color=VERMELHO, linestyle='--', linewidth=2.5,
               label=f'Limiar 10% (anômalo): {limiar:.3f}')
    ax.set_title("Distribuição do Score de Anomalia — Isolation Forest\nCadeia de Combustíveis RJ | Nov/2025")
    ax.set_xlabel("Score de Anomalia (mais negativo = mais suspeito)")
    ax.set_ylabel("Número de empresas")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(GRAF_DIR/"01_distribuicao_score_IF.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    log.info("  ✓ Gráfico 01")

    # ── Gráfico 2: PCA ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    normais  = resultado[resultado["if_anomalia"]==0]
    anomalos = resultado[resultado["if_anomalia"]==1]
    axes[0].scatter(normais["pca_1"], normais["pca_2"],
                    c=AZUL['claro'], alpha=0.4, s=15, label="Normal")
    axes[0].scatter(anomalos["pca_1"], anomalos["pca_2"],
                    c=AZUL['escuro'], alpha=0.8, s=25, label="Anômalo", zorder=5)
    axes[0].set_title("Isolation Forest\n● Anômalo (escuro) | ● Normal (claro)")
    axes[0].set_xlabel("PCA 1"); axes[0].set_ylabel("PCA 2")
    axes[0].legend()

    outliers = resultado[resultado["dbscan_cluster"]==-1]
    clusters = resultado[resultado["dbscan_cluster"]>=0]
    axes[1].scatter(clusters["pca_1"], clusters["pca_2"],
                    c=AZUL['palido'], alpha=0.4, s=15, label="Cluster")
    axes[1].scatter(outliers["pca_1"], outliers["pca_2"],
                    c=VERMELHO, marker="x", s=40, alpha=0.7, label="Outliers", zorder=5)
    axes[1].set_title("DBSCAN Clustering\n✖ Outliers em vermelho")
    axes[1].set_xlabel("PCA 1"); axes[1].set_ylabel("PCA 2")
    axes[1].legend()

    plt.suptitle("Visualização PCA — Detecção de Anomalias\nCadeia de Combustíveis RJ",
                 fontsize=13, fontweight='bold', color=AZUL['escuro'])
    plt.tight_layout()
    plt.savefig(GRAF_DIR/"02_pca_anomalias.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    log.info("  ✓ Gráfico 02")

    # ── Gráfico 3: Classificação risco final ──────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ordem = ["Baixo","Médio","Alto","Crítico"]
    dist = resultado["classe_final"].value_counts().reindex(
        [c for c in ordem if c in resultado["classe_final"].cat.categories])
    cores_c = [AZUL['palido'], AZUL['claro'], AZUL['medio'], AZUL['escuro']]
    bars = ax.bar(dist.index, dist.values, color=cores_c,
                   edgecolor='white', linewidth=1.5, alpha=0.9)
    for bar, val in zip(bars, dist.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f"{val:,}\n({val/len(resultado)*100:.1f}%)",
                ha='center', fontsize=10, fontweight='bold', color=AZUL['escuro'])
    ax.set_title("Classificação de Risco Final — Score Combinado\nCadeia de Combustíveis RJ | Nov/2025")
    ax.set_xlabel("Classe de Risco"); ax.set_ylabel("Número de empresas")
    ax.set_ylim(0, dist.max()*1.2)
    plt.tight_layout()
    plt.savefig(GRAF_DIR/"03_classificacao_risco_final.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    log.info("  ✓ Gráfico 03")

    # ── Gráfico 4: Correlação features ───────────────────
    cols_heatmap = [
        "idade_empresa_dias","qtd_socios","capital_social",
        "max_empresas_por_socio","qtd_filiais","inapta",
        "capital_muito_baixo","socio_risco_alto",
        "taxa_inapta_municipio","postos_no_municipio"
    ]
    cols_heatmap = [c for c in cols_heatmap if c in X.columns]
    corr = X[cols_heatmap].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues",
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 8}, vmin=-0.5, vmax=1)
    ax.set_title("Correlação entre Features — Cadeia de Combustíveis RJ",
                 fontsize=12, fontweight='bold', color=AZUL['escuro'])
    plt.tight_layout()
    plt.savefig(GRAF_DIR/"04_correlacao_features.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    log.info("  ✓ Gráfico 04")

    # ── Gráfico 5: Top 20 suspeitas ───────────────────────
    top20 = resultado.nlargest(20, "score_final")[[
        "id_empresa","situacao_desc","score_final","classe_final"
    ]].reset_index(drop=True)
    top20["label"] = top20["id_empresa"] + " | " + top20["situacao_desc"]
    top20 = top20.sort_values("score_final")

    fig, ax = plt.subplots(figsize=(12, 8))
    cores_top = [AZUL['escuro'] if c=="Crítico" else AZUL['claro']
                 for c in top20["classe_final"]]
    ax.barh(range(len(top20)), top20["score_final"],
             color=cores_top, edgecolor='white', linewidth=1.2, alpha=0.85)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["label"], fontsize=8)
    ax.axvline(x=0.70, color=VERMELHO, linestyle='--', linewidth=2,
               label='Limiar crítico (0.70)')
    ax.set_title("Top 20 Empresas Mais Suspeitas\nCadeia de Combustíveis RJ | Nov/2025")
    ax.set_xlabel("Score de Risco Final")
    ax.legend()
    plt.tight_layout()
    plt.savefig(GRAF_DIR/"05_top20_suspeitas.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    log.info("  ✓ Gráfico 05")


# ══════════════════════════════════════════════════════════
# ETAPA 6 — Exportar resultados
# ══════════════════════════════════════════════════════════

def exportar_resultados(resultado):
    log.info("\nEtapa 6 — Exportando resultados...")
    resultado.to_parquet(PROC_DIR/"resultado_modelagem.parquet", index=False)

    top50 = resultado.nlargest(50, "score_final")[[
        "id_empresa","cnae_descricao","situacao_desc",
        "classe_risco","if_score","dbscan_outlier",
        "score_manual_norm","score_final","classe_final",
        "ranking","sancionada"
    ]]
    top50.to_csv(TAB_DIR/"top50_suspeitas.csv", index=False, encoding="utf-8-sig")
    resultado.to_csv(TAB_DIR/"resultado_completo.csv", index=False, encoding="utf-8-sig")

    resumo = resultado.groupby("classe_final").agg(
        qtd=("id_empresa","count"),
        score_medio=("score_final","mean"),
        if_score_medio=("if_score","mean"),
        pct_inaptas=("situacao_desc", lambda x: (x=="Inapta").mean()*100),
    ).round(3)
    resumo.to_csv(TAB_DIR/"resumo_por_classe.csv", encoding="utf-8-sig")
    log.info("  ✓ Resultados exportados")


# ══════════════════════════════════════════════════════════
# RELATÓRIO
# ══════════════════════════════════════════════════════════

def relatorio_final(resultado):
    print("\n" + "═"*65)
    print("  RESULTADOS DA MODELAGEM — Cadeia Combustíveis RJ")
    print("═"*65)
    print(f"  Empresas analisadas:       {len(resultado):>10,}")
    print(f"\n  ISOLATION FOREST:")
    print(f"  Anomalias detectadas:      {resultado['if_anomalia'].sum():>10,} ({resultado['if_anomalia'].mean()*100:.1f}%)")
    print(f"\n  DBSCAN:")
    print(f"  Outliers detectados:       {resultado['dbscan_outlier'].sum():>10,} ({resultado['dbscan_outlier'].mean()*100:.1f}%)")
    n_cl = resultado[resultado['dbscan_cluster']>=0]['dbscan_cluster'].nunique()
    print(f"  Clusters identificados:    {n_cl:>10}")
    print(f"\n  SCORE COMBINADO:")
    dist = resultado["classe_final"].value_counts().sort_index()
    for classe, qtd in dist.items():
        print(f"  {classe:<12}             {qtd:>6,} ({qtd/len(resultado)*100:.1f}%)")
    print(f"\n  TOP 10 EMPRESAS MAIS SUSPEITAS:")
    cols = ["ranking","id_empresa","cnae_descricao","situacao_desc",
            "score_final","classe_final","sancionada"]
    print(resultado.nlargest(10,"score_final")[
        [c for c in cols if c in resultado.columns]].to_string(index=False))
    print("═"*65)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  TCC MBA — Modelagem                         ║")
    log.info("║  Isolation Forest + DBSCAN                   ║")
    log.info("║  Cadeia de Combustíveis RJ                   ║")
    log.info("╚══════════════════════════════════════════════╝")

    df, X, X_scaled, meta, cols_modelo = carregar_preparar()
    resultado = rodar_isolation_forest(X_scaled, meta)
    resultado = rodar_dbscan(X_scaled, resultado)
    resultado = score_combinado(resultado, df)
    gerar_visualizacoes(resultado, X, cols_modelo)
    exportar_resultados(resultado)
    relatorio_final(resultado)

    log.info("\n✅ Modelagem concluída!")
    log.info(f"   Próximo passo: python3 src/05_avaliacao.py")


if __name__ == "__main__":
    main()
