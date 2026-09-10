"""
=============================================================
TCC MBA Data Science — Análise de Fraudes Fiscais (ICMS)
=============================================================
Script: 05_avaliacao.py
Objetivo: Avaliação dos modelos IF + DBSCAN
          Análise dos clusters, métricas de validação
          e relatório executivo dos achados
=============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# ─── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("avaliacao.log")
    ]
)
log = logging.getLogger(__name__)

# ─── Caminhos ─────────────────────────────────────────────
PROC_DIR   = Path("data/processed")
GRAF_DIR   = Path("outputs/graficos")
TAB_DIR    = Path("outputs/tabelas")
GRAF_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Cores USP ESALQ
CORES = {
    "verde":    "#00703C",
    "amarelo":  "#F5A800",
    "azul":     "#003366",
    "cinza":    "#666666",
    "vermelho": "#C8102E",
}
CORES_RISCO = {
    "Crítico": CORES["vermelho"],
    "Alto":    CORES["amarelo"],
    "Médio":   CORES["azul"],
    "Baixo":   CORES["verde"],
}


# ══════════════════════════════════════════════════════════
# ETAPA 1 — Carregar dados
# ══════════════════════════════════════════════════════════

def carregar_dados() -> tuple:
    log.info("Etapa 1 — Carregando dados...")

    resultado = pd.read_parquet(PROC_DIR / "resultado_modelagem.parquet")
    features  = pd.read_parquet(PROC_DIR / "dataset_features_rj.parquet")

    log.info(f"  ✓ Resultado: {len(resultado):,} empresas")
    log.info(f"  ✓ Features:  {features.shape[1]} variáveis")
    return resultado, features


# ══════════════════════════════════════════════════════════
# ETAPA 2 — Métricas de validação do DBSCAN
# ══════════════════════════════════════════════════════════

def avaliar_dbscan(resultado: pd.DataFrame,
                   features: pd.DataFrame) -> dict:
    log.info("\nEtapa 2 — Avaliando DBSCAN...")

    # Preparar features numéricas
    cols_num = features.select_dtypes(include=[np.number]).columns.tolist()
    cols_excluir = ["score_risco", "score_risco_v2", "sancionada",
                    "municipio", "vol_medio_esperado_posto"]
    cols_num = [c for c in cols_num if c not in cols_excluir]

    X = features[cols_num].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Silhouette score — mede qualidade dos clusters
    # Varia de -1 a 1: mais próximo de 1 = clusters mais coesos
    labels = resultado["dbscan_cluster"].values
    mascara = labels >= 0  # excluir outliers (-1) do cálculo
    if mascara.sum() > 1:
        sil_score = silhouette_score(X_scaled[mascara], labels[mascara])
        log.info(f"  ✓ Silhouette Score DBSCAN: {sil_score:.4f}")
    else:
        sil_score = 0
        log.warning("  ⚠ Poucos pontos para calcular silhouette")

    # Análise dos clusters principais
    clusters_info = []
    for cluster in sorted(resultado["dbscan_cluster"].unique()):
        mask = resultado["dbscan_cluster"] == cluster
        grupo = resultado[mask]
        label = "Outliers" if cluster == -1 else f"Cluster {cluster}"

        info = {
            "cluster": label,
            "qtd": len(grupo),
            "pct": len(grupo)/len(resultado)*100,
            "pct_inaptas": (grupo["situacao_desc"] == "Inapta").mean()*100 if "situacao_desc" in grupo.columns else 0,
            "score_medio": grupo["score_final"].mean() if "score_final" in grupo.columns else 0,
            "if_anomalia_pct": grupo["if_anomalia"].mean()*100 if "if_anomalia" in grupo.columns else 0,
        }
        clusters_info.append(info)

    df_clusters = pd.DataFrame(clusters_info)
    df_clusters = df_clusters.sort_values("score_medio", ascending=False)

    log.info(f"\n  Top 5 clusters por score médio:")
    for _, row in df_clusters.head(5).iterrows():
        log.info(f"    {row['cluster']}: {row['qtd']:.0f} empresas | "
                 f"score {row['score_medio']:.3f} | "
                 f"{row['pct_inaptas']:.1f}% inaptas")

    return {"silhouette": sil_score, "clusters_info": df_clusters}


# ══════════════════════════════════════════════════════════
# ETAPA 3 — Análise do Isolation Forest
# ══════════════════════════════════════════════════════════

def avaliar_isolation_forest(resultado: pd.DataFrame,
                              features: pd.DataFrame) -> dict:
    log.info("\nEtapa 3 — Avaliando Isolation Forest...")

    # Comparar anomalias IF com situação cadastral
    if "situacao_desc" in resultado.columns:
        cross = pd.crosstab(
            resultado["situacao_desc"],
            resultado["if_anomalia"],
            margins=True
        )
        log.info(f"\n  Anomalias IF por situação cadastral:")
        log.info(f"\n{cross.to_string()}")

    # Comparar anomalias IF com classe de risco manual
    if "classe_risco" in resultado.columns:
        cross2 = pd.crosstab(
            resultado["classe_risco"],
            resultado["if_anomalia"],
            normalize="index"
        ).round(3) * 100
        log.info(f"\n  % anomalias IF por classe de risco manual:")
        log.info(f"\n{cross2.to_string()}")

    # Distribuição do score IF por CNAE
    if "cnae_descricao" in resultado.columns:
        score_cnae = (resultado.groupby("cnae_descricao")["if_score"]
                      .mean().sort_values())
        log.info(f"\n  Score IF médio por CNAE (mais negativo = mais anômalo):")
        for cnae, score in score_cnae.items():
            log.info(f"    {cnae[:50]}: {score:.4f}")

    return {"cross_situacao": cross if "situacao_desc" in resultado.columns else None}


# ══════════════════════════════════════════════════════════
# ETAPA 4 — Visualizações de avaliação
# ══════════════════════════════════════════════════════════

def gerar_visualizacoes_avaliacao(resultado: pd.DataFrame,
                                   features: pd.DataFrame,
                                   metricas: dict):
    log.info("\nEtapa 4 — Gerando visualizações de avaliação...")

    plt.style.use("seaborn-v0_8-whitegrid")

    # ── Gráfico 6: Score IF por situação cadastral ────────
    if "situacao_desc" in resultado.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ordem = ["Ativa", "Suspensa", "Inapta"]
        dados_box = [resultado[resultado["situacao_desc"] == s]["if_score"].values
                     for s in ordem if s in resultado["situacao_desc"].values]
        labels_box = [s for s in ordem if s in resultado["situacao_desc"].values]
        bp = ax.boxplot(dados_box, tick_labels=labels_box, patch_artist=True,
                        medianprops={"color": "white", "linewidth": 2})
        cores_box = [CORES["verde"], CORES["amarelo"], CORES["vermelho"]]
        for patch, cor in zip(bp["boxes"], cores_box[:len(dados_box)]):
            patch.set_facecolor(cor)
            patch.set_alpha(0.7)
        ax.axhline(y=0, color=CORES["cinza"], linestyle="--", alpha=0.5)
        ax.set_title("Score de Anomalia (IF) por Situação Cadastral\nCadeia de Combustíveis RJ | Nov/2025",
                     fontsize=13, fontweight="bold", color=CORES["azul"])
        ax.set_xlabel("Situação Cadastral")
        ax.set_ylabel("Score IF (mais negativo = mais anômalo)")
        plt.tight_layout()
        plt.savefig(GRAF_DIR / "06_score_IF_por_situacao.png", dpi=150)
        plt.close()
        log.info("  ✓ Gráfico 6: score IF por situação")

    # ── Gráfico 7: Score IF por CNAE ──────────────────────
    if "cnae_descricao" in resultado.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        score_cnae = (resultado.groupby("cnae_descricao")["if_score"]
                      .mean().sort_values())
        cores_bar = [CORES["vermelho"] if s < score_cnae.median()
                     else CORES["verde"] for s in score_cnae.values]
        bars = ax.barh(range(len(score_cnae)), score_cnae.values,
                       color=cores_bar, edgecolor="white", alpha=0.85)
        ax.set_yticks(range(len(score_cnae)))
        ax.set_yticklabels([c[:45] for c in score_cnae.index], fontsize=9)
        ax.axvline(x=score_cnae.median(), color=CORES["cinza"],
                   linestyle="--", alpha=0.7, label="Mediana")
        ax.set_title("Score Médio de Anomalia por Segmento\nCadeia de Combustíveis RJ | Nov/2025",
                     fontsize=13, fontweight="bold", color=CORES["azul"])
        ax.set_xlabel("Score IF médio (mais negativo = mais suspeito)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(GRAF_DIR / "07_score_IF_por_CNAE.png", dpi=150)
        plt.close()
        log.info("  ✓ Gráfico 7: score IF por CNAE")

    # ── Gráfico 8: Matriz de concordância IF vs Score Manual
    fig, ax = plt.subplots(figsize=(8, 6))
    if "classe_risco" in resultado.columns:
        cross = pd.crosstab(
            resultado["classe_risco"],
            resultado["if_anomalia"].map({0: "Normal", 1: "Anômalo"}),
            normalize="index"
        ) * 100
        sns.heatmap(cross, annot=True, fmt=".1f", cmap="RdYlGn_r",
                    ax=ax, linewidths=0.5,
                    annot_kws={"size": 12, "fontweight": "bold"})
        ax.set_title("Concordância: Score Manual vs Isolation Forest\n(% por classe de risco manual)",
                     fontsize=12, fontweight="bold", color=CORES["azul"])
        ax.set_xlabel("Isolation Forest")
        ax.set_ylabel("Classe de Risco Manual")
        plt.tight_layout()
        plt.savefig(GRAF_DIR / "08_concordancia_IF_score_manual.png", dpi=150)
        plt.close()
        log.info("  ✓ Gráfico 8: concordância IF vs score manual")

    # ── Gráfico 9: Distribuição capital social por classe ──
    fig, ax = plt.subplots(figsize=(10, 5))
    if "classe_final" in resultado.columns and "capital_social" in features.columns:
        capital = features["capital_social"].clip(upper=500_000)
        resultado_plot = resultado.copy()
        resultado_plot["capital_social"] = capital.values

        ordem = ["Baixo", "Médio", "Alto", "Crítico"]
        dados_cap = [resultado_plot[resultado_plot["classe_final"] == c]["capital_social"].values
                     for c in ordem if c in resultado_plot["classe_final"].values]
        labels_cap = [c for c in ordem if c in resultado_plot["classe_final"].values]

        bp = ax.boxplot(dados_cap, tick_labels=labels_cap, patch_artist=True,
                        medianprops={"color": "white", "linewidth": 2})
        for patch, classe in zip(bp["boxes"], labels_cap):
            patch.set_facecolor(CORES_RISCO.get(classe, CORES["cinza"]))
            patch.set_alpha(0.7)

        ax.set_title("Capital Social por Classe de Risco Final\n(valores acima de R$500k truncados)",
                     fontsize=12, fontweight="bold", color=CORES["azul"])
        ax.set_xlabel("Classe de Risco")
        ax.set_ylabel("Capital Social (R$)")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
        )
        plt.tight_layout()
        plt.savefig(GRAF_DIR / "09_capital_por_classe_risco.png", dpi=150)
        plt.close()
        log.info("  ✓ Gráfico 9: capital social por classe")

    # ── Gráfico 10: Silhouette plot ────────────────────────
    cols_num = features.select_dtypes(include=[np.number]).columns.tolist()
    cols_excluir = ["score_risco", "score_risco_v2", "sancionada",
                    "municipio", "vol_medio_esperado_posto"]
    cols_num = [c for c in cols_num if c not in cols_excluir]
    X = features[cols_num].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    labels = resultado["dbscan_cluster"].values
    mascara = labels >= 0
    if mascara.sum() > 100:
        # Simplificar — mostrar só clusters com mais de 20 empresas
        clusters_grandes = [c for c in np.unique(labels[mascara])
                            if (labels == c).sum() >= 20]
        mask_grandes = np.isin(labels, clusters_grandes)

        if mask_grandes.sum() > 1:
            sil_vals = silhouette_samples(X_scaled[mask_grandes],
                                          labels[mask_grandes])
            sil_media = sil_vals.mean()

            fig, ax = plt.subplots(figsize=(10, 6))
            y_lower = 10
            cores_sil = plt.cm.tab20(np.linspace(0, 1, len(clusters_grandes)))

            for i, cluster in enumerate(sorted(clusters_grandes)):
                sil_cluster = sil_vals[labels[mask_grandes] == cluster]
                sil_cluster.sort()
                size = len(sil_cluster)
                y_upper = y_lower + size
                ax.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, sil_cluster,
                                  alpha=0.7, color=cores_sil[i],
                                  label=f"C{cluster} (n={size})")
                y_lower = y_upper + 5

            ax.axvline(x=sil_media, color=CORES["vermelho"],
                       linestyle="--", linewidth=2,
                       label=f"Média: {sil_media:.3f}")
            ax.set_title("Silhouette Plot — Qualidade dos Clusters DBSCAN\n(clusters com ≥20 empresas)",
                         fontsize=12, fontweight="bold", color=CORES["azul"])
            ax.set_xlabel("Coeficiente de Silhouette")
            ax.set_ylabel("Clusters")
            ax.set_yticks([])
            ax.legend(loc="lower right", fontsize=7, ncol=2)
            plt.tight_layout()
            plt.savefig(GRAF_DIR / "10_silhouette_dbscan.png", dpi=150)
            plt.close()
            log.info(f"  ✓ Gráfico 10: silhouette (média: {sil_media:.3f})")


# ══════════════════════════════════════════════════════════
# ETAPA 5 — Relatório executivo
# ══════════════════════════════════════════════════════════

def relatorio_executivo(resultado: pd.DataFrame,
                         features: pd.DataFrame,
                         metricas: dict):
    log.info("\nEtapa 5 — Gerando relatório executivo...")

    linhas = []
    linhas.append("=" * 70)
    linhas.append("  RELATÓRIO EXECUTIVO — TCC MBA DATA SCIENCE")
    linhas.append("  Detecção de Anomalias Fiscais: Cadeia de Combustíveis RJ")
    linhas.append("  Base de referência: novembro/2025")
    linhas.append("=" * 70)

    linhas.append("\n1. ESCOPO DO ESTUDO")
    linhas.append(f"   Universo analisado:      {len(resultado):,} empresas")
    linhas.append(f"   Estado:                  Rio de Janeiro (RJ)")
    linhas.append(f"   Segmentos:               Extração, Refino, Atacado,")
    linhas.append(f"                            Varejo e Transporte de Combustíveis")
    linhas.append(f"   Fontes de dados:         Receita Federal (CNPJ Nov/2025),")
    linhas.append(f"                            CEIS (Portal Transparência),")
    linhas.append(f"                            ANP (Vendas por município 2024)")
    linhas.append(f"   Variáveis construídas:   {features.shape[1]}")

    linhas.append("\n2. PERFIL DO SETOR")
    if "situacao_desc" in resultado.columns:
        dist_sit = resultado["situacao_desc"].value_counts()
        for sit, qtd in dist_sit.items():
            linhas.append(f"   {sit:<15}          {qtd:>6,} ({qtd/len(resultado)*100:.1f}%)")

    if "cnae_descricao" in resultado.columns:
        linhas.append("\n   Por segmento:")
        dist_cnae = resultado["cnae_descricao"].value_counts()
        for cnae, qtd in dist_cnae.items():
            linhas.append(f"   {cnae[:45]:<45} {qtd:>6,}")

    linhas.append("\n3. RESULTADOS — ISOLATION FOREST")
    linhas.append(f"   Configuração:  200 árvores, contamination=10%")
    linhas.append(f"   Anomalias:     {resultado['if_anomalia'].sum():,} empresas ({resultado['if_anomalia'].mean()*100:.1f}%)")

    if "situacao_desc" in resultado.columns:
        inaptas_anomalas = resultado[
            (resultado["situacao_desc"] == "Inapta") &
            (resultado["if_anomalia"] == 1)
        ]
        linhas.append(f"   Inaptas anômalas: {len(inaptas_anomalas):,}")
        ativas_anomalas = resultado[
            (resultado["situacao_desc"] == "Ativa") &
            (resultado["if_anomalia"] == 1)
        ]
        linhas.append(f"   Ativas anômalas:  {len(ativas_anomalas):,} ⚠ (alto risco)")

    linhas.append("\n4. RESULTADOS — DBSCAN")
    n_clusters = resultado[resultado["dbscan_cluster"] >= 0]["dbscan_cluster"].nunique()
    linhas.append(f"   Configuração:  eps=1.5, min_samples=5, PCA=10 componentes")
    linhas.append(f"   Clusters:      {n_clusters}")
    linhas.append(f"   Outliers:      {resultado['dbscan_outlier'].sum():,} ({resultado['dbscan_outlier'].mean()*100:.1f}%)")
    linhas.append(f"   Silhouette:    {metricas['dbscan']['silhouette']:.4f}")

    linhas.append("\n5. SCORE COMBINADO")
    linhas.append("   Ponderação: IF (50%) + Score Manual (30%) + DBSCAN (20%)")
    if "classe_final" in resultado.columns:
        dist_final = resultado["classe_final"].value_counts().sort_index()
        for classe, qtd in dist_final.items():
            linhas.append(f"   {classe:<12}         {qtd:>6,} ({qtd/len(resultado)*100:.1f}%)")

    linhas.append("\n6. TOP 10 EMPRESAS MAIS SUSPEITAS")
    cols_top = ["ranking", "id_empresa", "cnae_descricao",
                "situacao_desc", "score_final", "classe_final"]
    top10 = resultado.nlargest(10, "score_final")[
        [c for c in cols_top if c in resultado.columns]
    ]
    linhas.append(top10.to_string(index=False))

    linhas.append("\n7. PRINCIPAIS ACHADOS")
    linhas.append("   a) 86% das empresas do setor têm capital social < R$10.000,")
    linhas.append("      valor incompatível com os requisitos operacionais do setor.")
    linhas.append("   b) 23% das empresas estão inaptas — taxa elevada para")
    linhas.append("      um setor altamente regulado pela ANP.")
    linhas.append("   c) Empresas ativas figuram entre as mais suspeitas,")
    linhas.append("      indicando anomalia operacional mesmo sem irregularidade")
    linhas.append("      cadastral formal.")
    linhas.append("   d) O IF e o Score Manual apresentaram baixa concordância")
    linhas.append("      (2.6%), sugerindo que capturam dimensões distintas")
    linhas.append("      de anomalia — ambos são necessários.")
    linhas.append("   e) DBSCAN identificou 34 padrões distintos de comportamento,")
    linhas.append("      revelando heterogeneidade no setor.")

    linhas.append("\n8. LIMITAÇÕES")
    linhas.append("   - Dados de ICMS por empresa não disponíveis publicamente")
    linhas.append("     (sigilo fiscal — CTN Art. 198)")
    linhas.append("   - CEIS com apenas 1 empresa sancionada no escopo,")
    linhas.append("     impossibilitando modelagem supervisionada")
    linhas.append("   - Dados ANP referem-se a 2024 (defasagem de 1 ano)")

    linhas.append("\n9. TRABALHOS FUTUROS")
    linhas.append("   - Aplicar XGBoost com dados rotulados de autuações fiscais")
    linhas.append("   - Expandir para outros estados produtores (ES, SP)")
    linhas.append("   - Incorporar dados de NF-e para análise de redes")
    linhas.append("   - Desenvolver dashboard de monitoramento contínuo")

    linhas.append("\n" + "=" * 70)

    relatorio_txt = "\n".join(linhas)
    print(relatorio_txt)

    # Salvar
    caminho = TAB_DIR / "relatorio_executivo.txt"
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(relatorio_txt)
    log.info(f"  ✓ Relatório salvo: {caminho}")

    return relatorio_txt


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  TCC MBA — Avaliação dos Modelos             ║")
    log.info("║  Cadeia de Combustíveis RJ                   ║")
    log.info("╚══════════════════════════════════════════════╝")

    # Carregar
    resultado, features = carregar_dados()

    # Avaliar DBSCAN
    metricas_dbscan = avaliar_dbscan(resultado, features)

    # Avaliar IF
    metricas_if = avaliar_isolation_forest(resultado, features)

    # Métricas consolidadas
    metricas = {
        "dbscan": metricas_dbscan,
        "if": metricas_if,
    }

    # Visualizações
    gerar_visualizacoes_avaliacao(resultado, features, metricas)

    # Relatório executivo
    relatorio_executivo(resultado, features, metricas)

    log.info("\n✅ Avaliação concluída!")
    log.info(f"   Gráficos: {GRAF_DIR}/ (06 a 10)")
    log.info(f"   Relatório: {TAB_DIR}/relatorio_executivo.txt")
    log.info("\n🎓 Pipeline completo!")


if __name__ == "__main__":
    main()
