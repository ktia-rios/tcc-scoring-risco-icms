"""
=============================================================
TCC MBA Data Science — Análise de Fraudes Fiscais (ICMS)
=============================================================
Script: 05_avaliacao.py
Objetivo: Avaliação dos modelos IF + DBSCAN
Paleta:   Tons de azul + tracejado vermelho
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
import matplotlib.ticker as mticker
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("avaliacao.log")]
)
log = logging.getLogger(__name__)

PROC_DIR = Path("data/processed")
GRAF_DIR = Path("outputs/graficos")
TAB_DIR  = Path("outputs/tabelas")
GRAF_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ─── Paleta ───────────────────────────────────────────────
AZUL = {
    'escuro':  '#003366',
    'medio':   '#1a5276',
    'claro':   '#2e86c1',
    'palido':  '#85c1e9',
    'cinza':   '#566573',
}
VERMELHO = '#C0392B'
CORES_RISCO = {
    'Crítico': AZUL['escuro'],
    'Alto':    AZUL['medio'],
    'Médio':   AZUL['claro'],
    'Baixo':   AZUL['palido'],
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlecolor'] = AZUL['escuro']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def carregar_dados():
    log.info("Etapa 1 — Carregando dados...")
    resultado = pd.read_parquet(PROC_DIR/"resultado_modelagem.parquet")
    features  = pd.read_parquet(PROC_DIR/"dataset_features_rj.parquet")
    log.info(f"  ✓ Resultado: {len(resultado):,} empresas")
    return resultado, features


def avaliar_dbscan(resultado, features):
    log.info("\nEtapa 2 — Avaliando DBSCAN...")

    cols_excluir = ["score_risco","score_risco_v2","sancionada",
                    "municipio","vol_medio_esperado_posto"]
    cols_num = [c for c in features.select_dtypes(include=[np.number]).columns
                if c not in cols_excluir]
    X = features[cols_num].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    labels = resultado["dbscan_cluster"].values
    mascara = labels >= 0
    if mascara.sum() > 1:
        sil = silhouette_score(X_scaled[mascara], labels[mascara])
        log.info(f"  ✓ Silhouette Score: {sil:.4f}")
    else:
        sil = 0

    clusters_info = []
    for cluster in sorted(resultado["dbscan_cluster"].unique()):
        mask = resultado["dbscan_cluster"] == cluster
        grupo = resultado[mask]
        label = "Outliers" if cluster == -1 else f"Cluster {cluster}"
        clusters_info.append({
            "cluster": label, "qtd": len(grupo),
            "pct": len(grupo)/len(resultado)*100,
            "pct_inaptas": (grupo["situacao_desc"]=="Inapta").mean()*100
                           if "situacao_desc" in grupo.columns else 0,
            "score_medio": grupo["score_final"].mean()
                           if "score_final" in grupo.columns else 0,
        })

    df_cl = pd.DataFrame(clusters_info).sort_values("score_medio", ascending=False)
    log.info("  Top 5 clusters por score médio:")
    for _, r in df_cl.head(5).iterrows():
        log.info(f"    {r['cluster']}: {r['qtd']:.0f} emp | "
                 f"score {r['score_medio']:.3f} | {r['pct_inaptas']:.1f}% inaptas")

    return {"silhouette": sil, "clusters_info": df_cl, "X_scaled": X_scaled}


def avaliar_isolation_forest(resultado, features):
    log.info("\nEtapa 3 — Avaliando Isolation Forest...")

    if "situacao_desc" in resultado.columns:
        cross = pd.crosstab(resultado["situacao_desc"],
                            resultado["if_anomalia"], margins=True)
        log.info(f"\n  Anomalias IF por situação:\n{cross.to_string()}")

    if "classe_risco" in resultado.columns:
        cross2 = pd.crosstab(resultado["classe_risco"],
                             resultado["if_anomalia"],
                             normalize="index").round(3)*100
        log.info(f"\n  % anomalias IF por classe de risco:\n{cross2.to_string()}")

    if "cnae_descricao" in resultado.columns:
        score_cnae = resultado.groupby("cnae_descricao")["if_score"].mean().sort_values()
        log.info("\n  Score IF médio por CNAE:")
        for cnae, score in score_cnae.items():
            log.info(f"    {cnae[:50]}: {score:.4f}")

    return {}


def gerar_visualizacoes(resultado, features, metricas):
    log.info("\nEtapa 4 — Gerando visualizações...")

    # ── Gráfico 06: Score IF por situação ─────────────────
    if "situacao_desc" in resultado.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ordem = ["Ativa","Suspensa","Inapta"]
        dados = [resultado[resultado["situacao_desc"]==s]["if_score"].values
                 for s in ordem if s in resultado["situacao_desc"].values]
        labels = [s for s in ordem if s in resultado["situacao_desc"].values]
        cores_box = [AZUL['claro'], AZUL['medio'], AZUL['escuro']]
        bp = ax.boxplot(dados, tick_labels=labels, patch_artist=True,
                        medianprops={"color": VERMELHO, "linewidth": 2.5})
        for patch, cor in zip(bp["boxes"], cores_box):
            patch.set_facecolor(cor); patch.set_alpha(0.85)
        for el in ["whiskers","caps"]:
            for item in bp[el]:
                item.set_color(AZUL['escuro']); item.set_linewidth(1.5)
        for flier in bp["fliers"]:
            flier.set(marker='o', color=AZUL['cinza'], alpha=0.4, markersize=4)
        ax.axhline(y=0, color=VERMELHO, linestyle='--', linewidth=1.5,
                   alpha=0.7, label="Score = 0")
        ax.set_title("Score de Anomalia (IF) por Situação Cadastral\nCadeia de Combustíveis RJ | Nov/2025")
        ax.set_xlabel("Situação Cadastral")
        ax.set_ylabel("Score IF (mais negativo = mais anômalo)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(GRAF_DIR/"06_score_IF_por_situacao.png",
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        log.info("  ✓ Gráfico 06")

    # ── Gráfico 07: Score IF por CNAE ─────────────────────
    if "cnae_descricao" in resultado.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        score_cnae = resultado.groupby("cnae_descricao")["if_score"].mean().sort_values()
        media_if = score_cnae.median()
        cores_cn = [AZUL['escuro'] if v < media_if else AZUL['claro']
                    for v in score_cnae.values]
        ax.barh(range(len(score_cnae)), score_cnae.values,
                 color=cores_cn, alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.set_yticks(range(len(score_cnae)))
        ax.set_yticklabels([c[:45] for c in score_cnae.index], fontsize=9)
        ax.axvline(x=media_if, color=VERMELHO, linestyle='--', linewidth=2,
                   label=f"Mediana: {media_if:.4f}")
        ax.set_title("Score Médio de Anomalia por Segmento\nCadeia de Combustíveis RJ | Nov/2025")
        ax.set_xlabel("Score IF médio (mais negativo = mais suspeito)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(GRAF_DIR/"07_score_IF_por_CNAE.png",
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        log.info("  ✓ Gráfico 07")

    # ── Gráfico 08: Concordância IF vs Score Manual ────────
    if "classe_risco" in resultado.columns:
        cross = pd.crosstab(
            resultado["classe_risco"],
            resultado["if_anomalia"].map({0:"Normal",1:"Anômalo"}),
            normalize="index"
        ) * 100
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cross, annot=True, fmt=".1f", cmap="Blues",
                    ax=ax, linewidths=0.5,
                    annot_kws={"size": 13, "fontweight": "bold"},
                    vmin=0, vmax=100)
        ax.set_title("Concordância: Score Manual vs Isolation Forest\n"
                     "(% por classe de risco manual)",
                     fontsize=12, fontweight='bold', color=AZUL['escuro'])
        ax.set_xlabel("Isolation Forest")
        ax.set_ylabel("Classe de Risco Manual")
        plt.tight_layout()
        plt.savefig(GRAF_DIR/"08_concordancia_IF_score_manual.png",
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        log.info("  ✓ Gráfico 08")

    # ── Gráfico 09: Capital por classe de risco ───────────
    if "classe_final" in resultado.columns and "capital_social" in features.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ordem_c = ["Baixo","Médio","Alto","Crítico"]
        capital  = features["capital_social"].clip(upper=500_000)
        res_plot = resultado.copy()
        res_plot["capital_social"] = capital.values
        dados_cap = [res_plot[res_plot["classe_final"]==c]["capital_social"].values
                     for c in ordem_c if c in res_plot["classe_final"].cat.categories]
        labels_cap = [c for c in ordem_c if c in res_plot["classe_final"].cat.categories]
        cores_cap = [AZUL['palido'], AZUL['claro'], AZUL['medio'], AZUL['escuro']]
        bp = ax.boxplot(dados_cap, tick_labels=labels_cap, patch_artist=True,
                        medianprops={"color": VERMELHO, "linewidth": 2.5})
        for patch, cor in zip(bp["boxes"], cores_cap[:len(dados_cap)]):
            patch.set_facecolor(cor); patch.set_alpha(0.85)
        for el in ["whiskers","caps"]:
            for item in bp[el]:
                item.set_color(AZUL['escuro']); item.set_linewidth(1.5)
        for flier in bp["fliers"]:
            flier.set(marker='o', color=AZUL['cinza'], alpha=0.4, markersize=4)
        ax.set_title("Capital Social por Classe de Risco Final\n"
                     "(valores acima de R$500k truncados)")
        ax.set_xlabel("Classe de Risco")
        ax.set_ylabel("Capital Social (R$)")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x,p: f"R${x:,.0f}"))
        plt.tight_layout()
        plt.savefig(GRAF_DIR/"09_capital_por_classe_risco.png",
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        log.info("  ✓ Gráfico 09")

    # ── Gráfico 10: Silhouette DBSCAN ─────────────────────
    X_scaled = metricas["dbscan"]["X_scaled"]
    labels   = resultado["dbscan_cluster"].values
    mascara  = labels >= 0
    clusters_grandes = [c for c in np.unique(labels[mascara])
                        if (labels==c).sum() >= 20]
    mask_grandes = np.isin(labels, clusters_grandes)

    if mask_grandes.sum() > 1:
        sil_vals = silhouette_samples(X_scaled[mask_grandes],
                                      labels[mask_grandes])
        sil_media = sil_vals.mean()

        fig, ax = plt.subplots(figsize=(10, 7))
        y_lower = 10
        paleta  = plt.cm.Blues(np.linspace(0.3, 0.9, len(clusters_grandes)))

        for i, cluster in enumerate(sorted(clusters_grandes)):
            mask_c = labels[mask_grandes] == cluster
            sil_c  = np.sort(sil_vals[mask_c])
            size   = len(sil_c)
            y_upper = y_lower + size
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_c,
                              alpha=0.85, color=paleta[i],
                              label=f"C{cluster} (n={size})")
            y_lower = y_upper + 5

        ax.axvline(x=sil_media, color=VERMELHO, linestyle='--', linewidth=2.5,
                   label=f"Média: {sil_media:.3f}")
        ax.set_title("Silhouette Plot — Qualidade dos Clusters DBSCAN\n"
                     "(clusters com ≥20 empresas)",
                     fontsize=12, fontweight='bold', color=AZUL['escuro'])
        ax.set_xlabel("Coeficiente de Silhouette")
        ax.set_ylabel("Clusters")
        ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(GRAF_DIR/"10_silhouette_dbscan.png",
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        log.info(f"  ✓ Gráfico 10 (silhouette médio: {sil_media:.3f})")


def relatorio_executivo(resultado, features, metricas):
    log.info("\nEtapa 5 — Gerando relatório executivo...")
    linhas = []
    linhas.append("="*70)
    linhas.append("  RELATÓRIO EXECUTIVO — TCC MBA DATA SCIENCE")
    linhas.append("  Detecção de Anomalias Fiscais: Cadeia de Combustíveis RJ")
    linhas.append("  Base de referência: novembro/2025")
    linhas.append("="*70)
    linhas.append(f"\n  Empresas analisadas: {len(resultado):,}")
    linhas.append(f"  Variáveis: {features.shape[1]}")
    linhas.append(f"\n  ISOLATION FOREST:")
    linhas.append(f"  Anomalias: {resultado['if_anomalia'].sum():,} ({resultado['if_anomalia'].mean()*100:.1f}%)")
    linhas.append(f"\n  DBSCAN:")
    linhas.append(f"  Outliers: {resultado['dbscan_outlier'].sum():,} ({resultado['dbscan_outlier'].mean()*100:.1f}%)")
    linhas.append(f"  Silhouette: {metricas['dbscan']['silhouette']:.4f}")
    linhas.append(f"\n  SCORE COMBINADO:")
    if "classe_final" in resultado.columns:
        for classe, qtd in resultado["classe_final"].value_counts().sort_index().items():
            linhas.append(f"  {classe}: {qtd:,} ({qtd/len(resultado)*100:.1f}%)")
    linhas.append("="*70)

    txt = "\n".join(linhas)
    print(txt)
    caminho = TAB_DIR/"relatorio_executivo.txt"
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(txt)
    log.info(f"  ✓ Relatório salvo: {caminho}")


def main():
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  TCC MBA — Avaliação dos Modelos             ║")
    log.info("╚══════════════════════════════════════════════╝")

    resultado, features = carregar_dados()
    metricas_dbscan = avaliar_dbscan(resultado, features)
    metricas_if     = avaliar_isolation_forest(resultado, features)
    metricas = {"dbscan": metricas_dbscan, "if": metricas_if}

    gerar_visualizacoes(resultado, features, metricas)
    relatorio_executivo(resultado, features, metricas)

    log.info("\n✅ Avaliação concluída!")
    log.info(f"   Gráficos: {GRAF_DIR}/ (06 a 10)")


if __name__ == "__main__":
    main()
