"""
=============================================================
TCC MBA Data Science — Análise de Fraudes Fiscais (ICMS)
=============================================================
Script: 04_modelagem.py
Objetivo: Detecção de anomalias na cadeia de combustíveis RJ
          Camada 1: Isolation Forest (não supervisionado)
          Camada 2: DBSCAN (clustering)
          Camada 3: Validação com score manual e CEIS
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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # sem interface gráfica
import matplotlib.pyplot as plt
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
PROC_DIR    = Path("data/processed")
OUTPUT_DIR  = Path("outputs")
GRAF_DIR    = OUTPUT_DIR / "graficos"
TAB_DIR     = OUTPUT_DIR / "tabelas"
MOD_DIR     = OUTPUT_DIR / "modelos"

for d in [GRAF_DIR, TAB_DIR, MOD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INPUT = PROC_DIR / "dataset_features_rj.parquet"


# ══════════════════════════════════════════════════════════
# ETAPA 1 — Carregar e preparar dados
# ══════════════════════════════════════════════════════════

def carregar_preparar() -> tuple:
    log.info("Etapa 1 — Carregando e preparando dados...")
    df = pd.read_parquet(INPUT)
    log.info(f"  ✓ {len(df):,} empresas, {df.shape[1]} variáveis")

    # Colunas de informação (não entram no modelo)
    cols_info_candidatas = [
        "id_empresa", "cnae_descricao", "cnae_4dig",
        "situacao_desc", "classe_risco", "sancionada",
        "municipio", "porte_desc", "opcao_simples",
        "opcao_mei", "score_risco", "score_risco_v2"
    ]
    cols_info = [c for c in cols_info_candidatas if c in df.columns]

    # Features para o modelo — apenas numéricas, sem identificadores
    cols_remover = cols_info + ["municipio"]
    cols_modelo = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in cols_remover and c != "sancionada"
    ]

    # Remover features derivadas de score (evitar redundância)
    cols_excluir = ["score_risco", "score_risco_v2",
                    "vol_medio_esperado_posto"]
    cols_modelo = [c for c in cols_modelo if c not in cols_excluir]

    log.info(f"  ✓ Features selecionadas: {len(cols_modelo)}")
    log.info(f"  Features: {cols_modelo}")

    X = df[cols_modelo].fillna(0)
    meta = df[cols_info].copy()

    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=cols_modelo)

    log.info(f"  ✓ Dados normalizados com StandardScaler")
    return df, X, X_scaled, meta, cols_modelo


# ══════════════════════════════════════════════════════════
# ETAPA 2 — Isolation Forest
# ══════════════════════════════════════════════════════════

def rodar_isolation_forest(X_scaled: pd.DataFrame,
                            meta: pd.DataFrame) -> pd.DataFrame:
    log.info("\nEtapa 2 — Isolation Forest...")

    # Parâmetros:
    # contamination = proporção esperada de anomalias
    # Usamos 0.1 (10%) — conservador para o setor
    # random_state para reprodutibilidade
    IF = IsolationForest(
        n_estimators=200,      # número de árvores
        contamination=0.10,    # 10% de anomalias esperadas
        max_samples="auto",
        random_state=42,
        n_jobs=-1              # usar todos os cores
    )

    IF.fit(X_scaled)

    # Score de anomalia: mais negativo = mais anômalo
    scores = IF.decision_function(X_scaled)
    predicoes = IF.predict(X_scaled)  # -1 = anômalo, 1 = normal

    resultado = meta.copy()
    resultado["if_score"]     = scores
    resultado["if_anomalia"]  = (predicoes == -1).astype(int)
    resultado["if_score_norm"] = (scores - scores.min()) / (scores.max() - scores.min())
    resultado["if_score_risco"] = 1 - resultado["if_score_norm"]  # maior = mais suspeito

    anomalias = resultado["if_anomalia"].sum()
    log.info(f"  ✓ Anomalias detectadas: {anomalias:,} ({anomalias/len(resultado)*100:.1f}%)")

    # Verificar se a empresa sancionada foi detectada
    if "sancionada" in resultado.columns:
        sancionadas = resultado[resultado["sancionada"] == 1]
        if len(sancionadas) > 0:
            for _, row in sancionadas.iterrows():
                detectada = "✅ DETECTADA" if row["if_anomalia"] == 1 else "❌ NÃO detectada"
                log.info(f"  Empresa sancionada {row['id_empresa']}: {detectada}")
                log.info(f"  Score IF: {row['if_score']:.4f}")

    return resultado


# ══════════════════════════════════════════════════════════
# ETAPA 3 — DBSCAN
# ══════════════════════════════════════════════════════════

def rodar_dbscan(X_scaled: pd.DataFrame,
                 resultado: pd.DataFrame) -> pd.DataFrame:
    log.info("\nEtapa 3 — DBSCAN...")

    # Reduzir dimensionalidade com PCA para DBSCAN funcionar melhor
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    variancia = pca.explained_variance_ratio_.cumsum()[-1]
    log.info(f"  PCA: 10 componentes explicam {variancia*100:.1f}% da variância")

    # DBSCAN
    # eps: raio de vizinhança — ajustado para o tamanho do dataset
    # min_samples: mínimo de pontos para formar cluster
    db = DBSCAN(eps=1.5, min_samples=5, n_jobs=-1)
    clusters = db.fit_predict(X_pca)

    resultado["dbscan_cluster"] = clusters
    resultado["dbscan_outlier"] = (clusters == -1).astype(int)

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_outliers = (clusters == -1).sum()

    log.info(f"  ✓ Clusters encontrados: {n_clusters}")
    log.info(f"  ✓ Outliers DBSCAN: {n_outliers:,} ({n_outliers/len(resultado)*100:.1f}%)")

    # Distribuição por cluster
    dist = pd.Series(clusters).value_counts().sort_index()
    for cluster, qtd in dist.items():
        label = "Outliers" if cluster == -1 else f"Cluster {cluster}"
        log.info(f"    {label}: {qtd:,} empresas")

    # Salvar componentes PCA para visualização
    resultado["pca_1"] = X_pca[:, 0]
    resultado["pca_2"] = X_pca[:, 1]

    return resultado


# ══════════════════════════════════════════════════════════
# ETAPA 4 — Score combinado e ranking final
# ══════════════════════════════════════════════════════════

def score_combinado(resultado: pd.DataFrame,
                    df_original: pd.DataFrame) -> pd.DataFrame:
    log.info("\nEtapa 4 — Score combinado...")

    # Normalizar score manual para 0-1
    score_manual = df_original["score_risco_v2"]
    score_manual_norm = (score_manual - score_manual.min()) / \
                        (score_manual.max() - score_manual.min())
    resultado["score_manual_norm"] = score_manual_norm.values

    # Score combinado: IF (50%) + manual (30%) + DBSCAN (20%)
    resultado["score_final"] = (
        resultado["if_score_risco"]    * 0.50 +
        resultado["score_manual_norm"] * 0.30 +
        resultado["dbscan_outlier"]    * 0.20
    )

    # Ranking final
    resultado["ranking"] = resultado["score_final"].rank(
        ascending=False, method="min"
    ).astype(int)

    # Classificação final
    resultado["classe_final"] = pd.cut(
        resultado["score_final"],
        bins=[-0.01, 0.30, 0.50, 0.70, 1.01],
        labels=["Baixo", "Médio", "Alto", "Crítico"]
    )

    dist = resultado["classe_final"].value_counts().sort_index()
    log.info("  Distribuição do score combinado:")
    for classe, qtd in dist.items():
        pct = qtd/len(resultado)*100
        log.info(f"    {classe}: {qtd:,} ({pct:.1f}%)")

    # Concordância IF vs score manual
    top_if     = set(resultado.nsmallest(152, "if_score")["id_empresa"])
    top_manual = set(resultado.nlargest(152, "score_manual_norm")["id_empresa"])
    concordancia = len(top_if & top_manual) / 152 * 100
    log.info(f"\n  Concordância IF vs Score Manual (top 10%): {concordancia:.1f}%")

    return resultado


# ══════════════════════════════════════════════════════════
# ETAPA 5 — Visualizações
# ══════════════════════════════════════════════════════════

def gerar_visualizacoes(resultado: pd.DataFrame,
                        X: pd.DataFrame,
                        cols_modelo: list):
    log.info("\nEtapa 5 — Gerando visualizações...")

    plt.style.use("seaborn-v0_8-whitegrid")
    cores = {"Crítico": "#d62728", "Alto": "#ff7f0e",
             "Médio": "#1f77b4", "Baixo": "#2ca02c"}

    # ── Gráfico 1: Distribuição do score IF ───────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(resultado["if_score"], bins=40, color="#1f77b4",
            edgecolor="white", alpha=0.8)
    ax.axvline(resultado["if_score"].quantile(0.10), color="red",
               linestyle="--", linewidth=2, label="Limiar 10% (anômalo)")
    ax.set_title("Distribuição do Score de Anomalia — Isolation Forest\nCadeia de Combustíveis RJ | Nov/2025",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Score de Anomalia (mais negativo = mais suspeito)")
    ax.set_ylabel("Número de empresas")
    ax.legend()
    plt.tight_layout()
    plt.savefig(GRAF_DIR / "01_distribuicao_score_IF.png", dpi=150)
    plt.close()
    log.info("  ✓ Gráfico 1: distribuição score IF")

    # ── Gráfico 2: PCA — IF vs DBSCAN ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # IF
    colors_if = resultado["if_anomalia"].map({1: "#d62728", 0: "#1f77b4"})
    axes[0].scatter(resultado["pca_1"], resultado["pca_2"],
                    c=colors_if, alpha=0.5, s=20)
    axes[0].set_title("Isolation Forest\n🔴 Anômalo | 🔵 Normal", fontsize=12)
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")

    # DBSCAN
    n_clusters = resultado["dbscan_cluster"].nunique()
    palette = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    cluster_colors = resultado["dbscan_cluster"].map(
        {c: i for i, c in enumerate(resultado["dbscan_cluster"].unique())}
    )
    axes[1].scatter(resultado["pca_1"], resultado["pca_2"],
                    c=cluster_colors, alpha=0.5, s=20, cmap="tab20")
    outliers = resultado[resultado["dbscan_cluster"] == -1]
    axes[1].scatter(outliers["pca_1"], outliers["pca_2"],
                    c="red", marker="x", s=50, label="Outliers", zorder=5)
    axes[1].set_title("DBSCAN Clustering\n✖ Outliers em vermelho", fontsize=12)
    axes[1].set_xlabel("PCA 1")
    axes[1].set_ylabel("PCA 2")
    axes[1].legend()

    plt.suptitle("Visualização PCA — Detecção de Anomalias\nCadeia de Combustíveis RJ",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAF_DIR / "02_pca_anomalias.png", dpi=150)
    plt.close()
    log.info("  ✓ Gráfico 2: PCA anomalias")

    # ── Gráfico 3: Score final por classe de risco ────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ordem = ["Baixo", "Médio", "Alto", "Crítico"]
    dist = resultado["classe_final"].value_counts().reindex(ordem)
    bars = ax.bar(dist.index, dist.values,
                  color=[cores.get(c, "gray") for c in dist.index],
                  edgecolor="white")
    for bar, val in zip(bars, dist.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val:,}\n({val/len(resultado)*100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Classificação de Risco Final — Score Combinado\nCadeia de Combustíveis RJ | Nov/2025",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Classe de Risco")
    ax.set_ylabel("Número de empresas")
    plt.tight_layout()
    plt.savefig(GRAF_DIR / "03_classificacao_risco_final.png", dpi=150)
    plt.close()
    log.info("  ✓ Gráfico 3: classificação risco final")

    # ── Gráfico 4: Heatmap de correlação das features ─────
    cols_heatmap = [
        "idade_empresa_dias", "qtd_socios", "capital_social",
        "max_empresas_por_socio", "qtd_filiais", "inapta",
        "capital_muito_baixo", "socio_risco_alto",
        "taxa_inapta_municipio", "postos_no_municipio"
    ]
    cols_heatmap = [c for c in cols_heatmap if c in X.columns]
    corr = X[cols_heatmap].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Correlação entre Features — Cadeia de Combustíveis RJ",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAF_DIR / "04_correlacao_features.png", dpi=150)
    plt.close()
    log.info("  ✓ Gráfico 4: correlação features")

    # ── Gráfico 5: Top 20 empresas mais suspeitas ─────────
    top20 = resultado.nlargest(20, "score_final")[
        ["id_empresa", "cnae_descricao", "situacao_desc",
         "score_final", "classe_final"]
    ].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors_top = [cores.get(str(c), "gray") for c in top20["classe_final"]]
    bars = ax.barh(range(len(top20)), top20["score_final"],
                   color=colors_top, edgecolor="white")
    ax.set_yticks(range(len(top20)))
    labels = [f"{row['id_empresa']} | {row['situacao_desc']}"
              for _, row in top20.iterrows()]
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Score de Risco Final")
    ax.set_title("Top 20 Empresas Mais Suspeitas\nCadeia de Combustíveis RJ | Nov/2025",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAF_DIR / "05_top20_suspeitas.png", dpi=150)
    plt.close()
    log.info("  ✓ Gráfico 5: top 20 suspeitas")


# ══════════════════════════════════════════════════════════
# ETAPA 6 — Exportar resultados
# ══════════════════════════════════════════════════════════

def exportar_resultados(resultado: pd.DataFrame):
    log.info("\nEtapa 6 — Exportando resultados...")

    # Dataset completo com scores
    resultado.to_parquet(PROC_DIR / "resultado_modelagem.parquet", index=False)

    # Top 50 mais suspeitas (para análise)
    top50 = resultado.nlargest(50, "score_final")[[
        "id_empresa", "cnae_descricao", "situacao_desc",
        "classe_risco", "if_score", "dbscan_outlier",
        "score_manual_norm", "score_final", "classe_final",
        "ranking", "sancionada"
    ]]
    top50.to_csv(TAB_DIR / "top50_suspeitas.csv",
                 index=False, encoding="utf-8-sig")
    log.info(f"  ✓ Top 50 suspeitas: {TAB_DIR}/top50_suspeitas.csv")

    # Resumo por classe
    resumo = resultado.groupby("classe_final").agg(
        qtd=("id_empresa", "count"),
        score_medio=("score_final", "mean"),
        if_score_medio=("if_score", "mean"),
        pct_inaptas=("situacao_desc", lambda x: (x == "Inapta").mean() * 100),
        pct_dbscan_outlier=("dbscan_outlier", "mean"),
    ).round(3)
    resumo.to_csv(TAB_DIR / "resumo_por_classe.csv", encoding="utf-8-sig")
    log.info(f"  ✓ Resumo por classe: {TAB_DIR}/resumo_por_classe.csv")

    # Resultado completo em CSV
    resultado.to_csv(TAB_DIR / "resultado_completo.csv",
                     index=False, encoding="utf-8-sig")
    log.info(f"  ✓ Resultado completo: {TAB_DIR}/resultado_completo.csv")


# ══════════════════════════════════════════════════════════
# RELATÓRIO FINAL
# ══════════════════════════════════════════════════════════

def relatorio_final(resultado: pd.DataFrame):
    print("\n" + "═" * 65)
    print("  RESULTADOS DA MODELAGEM — Cadeia Combustíveis RJ")
    print("═" * 65)
    print(f"  Empresas analisadas:       {len(resultado):>10,}")
    print(f"\n  ISOLATION FOREST:")
    print(f"  Anomalias detectadas:      {resultado['if_anomalia'].sum():>10,} ({resultado['if_anomalia'].mean()*100:.1f}%)")
    print(f"\n  DBSCAN:")
    print(f"  Outliers detectados:       {resultado['dbscan_outlier'].sum():>10,} ({resultado['dbscan_outlier'].mean()*100:.1f}%)")
    n_clusters = resultado[resultado["dbscan_cluster"] >= 0]["dbscan_cluster"].nunique()
    print(f"  Clusters identificados:    {n_clusters:>10}")
    print(f"\n  SCORE COMBINADO:")
    dist = resultado["classe_final"].value_counts().sort_index()
    for classe, qtd in dist.items():
        print(f"  {classe:<12}             {qtd:>6,} ({qtd/len(resultado)*100:.1f}%)")

    print(f"\n  TOP 10 EMPRESAS MAIS SUSPEITAS:")
    cols = ["ranking", "id_empresa", "cnae_descricao", "situacao_desc",
            "score_final", "classe_final", "sancionada"]
    top10 = resultado.nlargest(10, "score_final")[
        [c for c in cols if c in resultado.columns]
    ]
    print(top10.to_string(index=False))
    print("═" * 65)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  TCC MBA — Modelagem                         ║")
    log.info("║  Isolation Forest + DBSCAN                   ║")
    log.info("║  Cadeia de Combustíveis RJ                   ║")
    log.info("╚══════════════════════════════════════════════╝")

    # Etapa 1 — Preparar dados
    df, X, X_scaled, meta, cols_modelo = carregar_preparar()

    # Etapa 2 — Isolation Forest
    resultado = rodar_isolation_forest(X_scaled, meta)

    # Etapa 3 — DBSCAN
    resultado = rodar_dbscan(X_scaled, resultado)

    # Etapa 4 — Score combinado
    resultado = score_combinado(resultado, df)

    # Etapa 5 — Visualizações
    gerar_visualizacoes(resultado, X, cols_modelo)

    # Etapa 6 — Exportar
    exportar_resultados(resultado)

    # Relatório final
    relatorio_final(resultado)

    log.info("\n✅ Modelagem concluída!")
    log.info(f"   Gráficos: {GRAF_DIR}/")
    log.info(f"   Tabelas:  {TAB_DIR}/")
    log.info("   Próximo passo: python3 src/05_avaliacao.py")


if __name__ == "__main__":
    main()
