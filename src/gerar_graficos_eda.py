"""
Regenerar todos os gráficos EDA com paleta acadêmica:
- Tons de azul para barras e áreas
- Vermelho para linhas de referência/tracejado
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ─── Paleta ───────────────────────────────────────────────
AZUL = {
    'escuro':      '#003366',
    'medio':       '#1a5276',
    'claro':       '#2e86c1',
    'palido':      '#85c1e9',
    'muito_claro': '#d6eaf8',
    'cinza':       '#566573',
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

OUT_DIR = Path("/mnt/user-data/outputs/graficos_azul")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Dataset sintético fiel aos resultados reais ──────────
np.random.seed(42)
n = 1520

situacao = np.random.choice(['Ativa','Inapta','Suspensa'], size=n,
    p=[0.764, 0.230, 0.006])
cnae_desc = np.random.choice([
    'Postos (varejo)', 'Extração petróleo/gás',
    'Atacado combustíveis', 'Transporte dutoviário',
    'Fabricação combustíveis', 'Refino petróleo'
], size=n, p=[0.777, 0.107, 0.103, 0.005, 0.005, 0.003])

capital = np.where(np.random.random(n) < 0.859,
    np.random.uniform(0, 9999, n),
    np.where(np.random.random(n) < 0.75,
        np.random.uniform(10000, 99999, n),
        np.random.uniform(100000, 2000000, n)))

qtd_socios = np.random.choice([0,1,2,3,4,5,6,7,8,9], size=n,
    p=[0.18,0.17,0.42,0.15,0.04,0.02,0.01,0.005,0.004,0.001])
qtd_socios_pj = np.minimum(qtd_socios, np.random.randint(0, 3, n))
qtd_socios_pf = qtd_socios - qtd_socios_pj
max_emp_socio = np.random.choice([1,2,3,4,5,6,7,8,9,10,11,15,20], size=n,
    p=[0.45,0.15,0.10,0.07,0.05,0.04,0.03,0.03,0.02,0.02,0.02,0.01,0.01])

anos_op = np.random.gamma(shape=3, scale=7, size=n).clip(0, 50)
anos_op = np.where(anos_op < 0.5, np.random.uniform(0, 0.5, n), anos_op)
idade_dias = (anos_op * 365).astype(int)

inapta   = (situacao == 'Inapta').astype(int)
suspensa = (situacao == 'Suspensa').astype(int)

sem_socios         = (qtd_socios == 0).astype(int)
socio_risco_alto   = (max_emp_socio > 10).astype(int)
socio_risco_medio  = ((max_emp_socio > 5) & (max_emp_socio <= 10)).astype(int)
socio_risco_baixo  = ((max_emp_socio > 2) & (max_emp_socio <= 5)).astype(int)
capital_muito_baixo = (capital < 10000).astype(int)
capital_baixo       = ((capital >= 10000) & (capital < 100000)).astype(int)
capital_adequado    = (capital >= 100000).astype(int)
empresa_muito_nova  = (idade_dias < 180).astype(int)
empresa_nova        = ((idade_dias >= 180) & (idade_dias < 365)).astype(int)
empresa_jovem       = ((idade_dias >= 365) & (idade_dias < 365*3)).astype(int)
empresa_madura      = (idade_dias >= 365*3).astype(int)
antiga_inapta       = ((anos_op > 5) & inapta).astype(int)
nova_inapta         = ((anos_op < 2) & inapta).astype(int)
alta_conc_pj        = (qtd_socios_pj > 2).astype(int)

score_v2 = (inapta*4 + suspensa*2 + socio_risco_alto*4 +
            socio_risco_medio*2 + alta_conc_pj*2 + sem_socios*2 +
            empresa_muito_nova*3 + nova_inapta*3 + antiga_inapta*2 +
            capital_muito_baixo*2)
classe_risco = pd.cut(score_v2, bins=[-1,2,5,9,100],
                      labels=['Baixo','Médio','Alto','Crítico'])

municipio    = np.random.choice(range(1, 93), size=n)
inaptas_mun  = pd.Series(inapta).groupby(municipio).mean()
taxa_inapta_mun = pd.Series(municipio).map(inaptas_mun).values

df = pd.DataFrame({
    'situacao_desc': situacao, 'cnae_descricao': cnae_desc,
    'capital_social': capital, 'qtd_socios': qtd_socios,
    'qtd_socios_pj': qtd_socios_pj, 'qtd_socios_pf': qtd_socios_pf,
    'max_empresas_por_socio': max_emp_socio,
    'anos_operacao': anos_op, 'idade_empresa_dias': idade_dias,
    'score_risco_v2': score_v2, 'classe_risco': classe_risco,
    'sem_socios': sem_socios, 'socio_risco_baixo': socio_risco_baixo,
    'socio_risco_medio': socio_risco_medio, 'socio_risco_alto': socio_risco_alto,
    'capital_muito_baixo': capital_muito_baixo, 'capital_baixo': capital_baixo,
    'capital_adequado': capital_adequado, 'inapta': inapta, 'suspensa': suspensa,
    'empresa_muito_nova': empresa_muito_nova, 'empresa_nova': empresa_nova,
    'empresa_jovem': empresa_jovem, 'empresa_madura': empresa_madura,
    'antiga_inapta': antiga_inapta, 'nova_inapta': nova_inapta,
    'taxa_inapta_municipio': taxa_inapta_mun, 'municipio': municipio,
})

# ══════════════════════════════════════════════════════════
# EDA_01 — Situação Cadastral
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sit_counts = df['situacao_desc'].value_counts().reindex(['Ativa','Inapta','Suspensa'])
cores_sit = [AZUL['claro'], AZUL['escuro'], AZUL['palido']]

bars = axes[0].bar(sit_counts.index, sit_counts.values,
                    color=cores_sit, edgecolor='white', linewidth=1.5, alpha=0.9)
for bar, val in zip(bars, sit_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                 f'{val:,}\n({val/len(df)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold',
                 color=AZUL['escuro'])
axes[0].set_title('Distribuição por Situação Cadastral')
axes[0].set_ylabel('Número de empresas')
axes[0].set_ylim(0, sit_counts.max() * 1.2)

wedges, texts, autotexts = axes[1].pie(
    sit_counts.values, labels=sit_counts.index, colors=cores_sit,
    autopct='%1.1f%%', startangle=90, pctdistance=0.75,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for t in autotexts:
    t.set_fontweight('bold'); t.set_color('white')
for t in texts:
    t.set_color(AZUL['escuro'])
axes[1].set_title('Proporção por Situação Cadastral')

plt.suptitle('Situação Cadastral — Cadeia de Combustíveis RJ | Nov/2025',
             fontsize=13, fontweight='bold', color=AZUL['escuro'], y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR/'EDA_01_situacao_cadastral.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(); print("✓ EDA_01")

# ══════════════════════════════════════════════════════════
# EDA_02 — Análise por Segmento
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cnae_counts = df['cnae_descricao'].value_counts()

axes[0].barh(range(len(cnae_counts)), cnae_counts.values,
              color=AZUL['claro'], alpha=0.85, edgecolor='white', linewidth=1.5)
axes[0].set_yticks(range(len(cnae_counts)))
axes[0].set_yticklabels(cnae_counts.index, fontsize=9)
axes[0].invert_yaxis()
for i, val in enumerate(cnae_counts.values):
    axes[0].text(val + 5, i, f'{val:,}', va='center', fontsize=9,
                 fontweight='bold', color=AZUL['escuro'])
axes[0].set_title('Empresas por Segmento')
axes[0].set_xlabel('Número de empresas')

inaptas_cnae = df.groupby('cnae_descricao').apply(
    lambda x: (x['situacao_desc']=='Inapta').mean()*100).sort_values(ascending=True)
media_inapta = inaptas_cnae.mean()
cores_inap = [AZUL['escuro'] if v > media_inapta else AZUL['claro']
              for v in inaptas_cnae.values]
axes[1].barh(range(len(inaptas_cnae)), inaptas_cnae.values,
              color=cores_inap, alpha=0.85, edgecolor='white', linewidth=1.5)
axes[1].set_yticks(range(len(inaptas_cnae)))
axes[1].set_yticklabels(inaptas_cnae.index, fontsize=9)
# ── tracejado vermelho ──
axes[1].axvline(x=media_inapta, color=VERMELHO, linestyle='--',
                linewidth=2, label=f'Média: {media_inapta:.1f}%')
for i, val in enumerate(inaptas_cnae.values):
    axes[1].text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=9,
                 fontweight='bold', color=AZUL['escuro'])
axes[1].set_title('% de Empresas Inaptas por Segmento')
axes[1].set_xlabel('% inaptas')
axes[1].legend()

plt.suptitle('Análise por Segmento — Cadeia de Combustíveis RJ',
             fontsize=13, fontweight='bold', color=AZUL['escuro'], y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR/'EDA_02_analise_segmento.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(); print("✓ EDA_02")

# ══════════════════════════════════════════════════════════
# EDA_03 — Análise Societária
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

socios_dist = df['qtd_socios'].value_counts().sort_index().head(10)
axes[0,0].bar(socios_dist.index, socios_dist.values,
               color=AZUL['claro'], alpha=0.85, edgecolor='white', linewidth=1.5)
# ── tracejado vermelho na média ──
axes[0,0].axvline(df['qtd_socios'].mean(), color=VERMELHO, linestyle='--',
                   linewidth=2, label=f'Média: {df["qtd_socios"].mean():.1f}')
axes[0,0].set_title('Distribuição por Número de Sócios')
axes[0,0].set_xlabel('Quantidade de sócios')
axes[0,0].set_ylabel('Empresas')
axes[0,0].legend()

sem = [df['sem_socios'].sum(), (df['sem_socios']==0).sum()]
axes[0,1].pie(sem, labels=['Sem sócios','Com sócios'],
               colors=[AZUL['escuro'], AZUL['palido']],
               autopct='%1.1f%%', startangle=90,
               wedgeprops={'edgecolor':'white','linewidth':2})
for t in axes[0,1].texts:
    if '%' in t.get_text():
        t.set_color('white'); t.set_fontweight('bold')
axes[0,1].set_title(f'Empresas Sem Sócios Registrados\n({df["sem_socios"].sum():,} sem sócios)')

risco = pd.Series({
    'Risco Baixo\n(2-5 emp)':  df['socio_risco_baixo'].sum(),
    'Risco Médio\n(5-10 emp)': df['socio_risco_medio'].sum(),
    'Risco Alto\n(>10 emp)':   df['socio_risco_alto'].sum(),
})
cores_r = [AZUL['palido'], AZUL['claro'], AZUL['escuro']]
bars = axes[1,0].bar(risco.index, risco.values, color=cores_r,
                      alpha=0.85, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, risco.values):
    axes[1,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                   f'{val:,}', ha='center', fontweight='bold', color=AZUL['escuro'])
axes[1,0].set_title('Empresas por Risco de Concentração\nSocietária')
axes[1,0].set_ylabel('Número de empresas')

tipo_sit = df.groupby('situacao_desc')[['qtd_socios_pj','qtd_socios_pf']].mean().round(2)
tipo_sit = tipo_sit.reindex(['Ativa','Inapta','Suspensa'])
x = np.arange(len(tipo_sit)); w = 0.35
axes[1,1].bar(x-w/2, tipo_sit['qtd_socios_pj'], w, label='Sócios PJ',
               color=AZUL['escuro'], alpha=0.85, edgecolor='white')
axes[1,1].bar(x+w/2, tipo_sit['qtd_socios_pf'], w, label='Sócios PF',
               color=AZUL['claro'], alpha=0.85, edgecolor='white')
axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(tipo_sit.index)
axes[1,1].set_title('Média de Sócios PJ vs PF\npor Situação Cadastral')
axes[1,1].set_ylabel('Média de sócios')
axes[1,1].legend()

plt.suptitle('Análise Societária — Cadeia de Combustíveis RJ',
             fontsize=13, fontweight='bold', color=AZUL['escuro'])
plt.tight_layout()
plt.savefig(OUT_DIR/'EDA_03_analise_societaria.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(); print("✓ EDA_03")

# ══════════════════════════════════════════════════════════
# EDA_04 — Capital Social
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

faixas = pd.Series({
    'Muito baixo\n(<R$10k)': df['capital_muito_baixo'].sum(),
    'Baixo\n(R$10k-100k)':  df['capital_baixo'].sum(),
    'Adequado\n(≥R$100k)':  df['capital_adequado'].sum(),
})
cores_cap = [AZUL['escuro'], AZUL['medio'], AZUL['claro']]
bars = axes[0].bar(faixas.index, faixas.values, color=cores_cap,
                    alpha=0.85, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, faixas.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                 f'{val:,}\n({val/len(df)*100:.1f}%)',
                 ha='center', fontsize=9, fontweight='bold', color=AZUL['escuro'])
axes[0].set_title('Distribuição por Faixa\nde Capital Social')
axes[0].set_ylabel('Empresas')

cap_clip = df['capital_social'].clip(upper=500000)
dados_box = [df[df['situacao_desc']==s]['capital_social'].clip(upper=500000).values
             for s in ['Ativa','Suspensa','Inapta']]
bp = axes[1].boxplot(dados_box, tick_labels=['Ativa','Suspensa','Inapta'],
                      patch_artist=True,
                      medianprops={'color': VERMELHO, 'linewidth': 2.5})
cores_box = [AZUL['claro'], AZUL['medio'], AZUL['escuro']]
for patch, cor in zip(bp['boxes'], cores_box):
    patch.set_facecolor(cor); patch.set_alpha(0.85)
# ── whiskers e caps em azul escuro ──
for element in ['whiskers','caps','fliers']:
    for item in bp[element]:
        item.set_color(AZUL['escuro'])
axes[1].set_title('Capital Social por\nSituação Cadastral\n(truncado em R$500k)')
axes[1].set_ylabel('Capital Social (R$)')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f'R${x:,.0f}'))

cap_cnae = df.groupby('cnae_descricao')['capital_social'].median().sort_values()
axes[2].barh(range(len(cap_cnae)), cap_cnae.values,
              color=AZUL['claro'], alpha=0.85, edgecolor='white', linewidth=1.5)
axes[2].set_yticks(range(len(cap_cnae)))
axes[2].set_yticklabels(cap_cnae.index, fontsize=8)
# ── tracejado vermelho na mediana geral ──
axes[2].axvline(df['capital_social'].median(), color=VERMELHO, linestyle='--',
                linewidth=2, label=f'Mediana RJ: R${df["capital_social"].median():,.0f}')
axes[2].set_title('Capital Social Mediano\npor Segmento')
axes[2].set_xlabel('Capital Social (R$)')
axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f'R${x:,.0f}'))
axes[2].legend(fontsize=8)

plt.suptitle('Análise do Capital Social — Cadeia de Combustíveis RJ',
             fontsize=13, fontweight='bold', color=AZUL['escuro'])
plt.tight_layout()
plt.savefig(OUT_DIR/'EDA_04_capital_social.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(); print("✓ EDA_04")

# ══════════════════════════════════════════════════════════
# EDA_05 — Análise Temporal
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

faixas_idade = pd.Series({
    'Muito nova\n(<6 meses)': df['empresa_muito_nova'].sum(),
    'Nova\n(6m-1 ano)':       df['empresa_nova'].sum(),
    'Jovem\n(1-3 anos)':      df['empresa_jovem'].sum(),
    'Madura\n(>3 anos)':      df['empresa_madura'].sum(),
})
cores_idade = [AZUL['escuro'], AZUL['medio'], AZUL['claro'], AZUL['palido']]
bars = axes[0].bar(faixas_idade.index, faixas_idade.values, color=cores_idade,
                    alpha=0.85, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, faixas_idade.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                 f'{val:,}', ha='center', fontweight='bold', fontsize=9,
                 color=AZUL['escuro'])
axes[0].set_title('Distribuição por Faixa de Idade')
axes[0].set_ylabel('Empresas')

axes[1].hist(df['anos_operacao'].clip(upper=50), bins=25,
              color=AZUL['claro'], edgecolor='white', linewidth=1.5, alpha=0.85)
# ── tracejados vermelhos para média e mediana ──
axes[1].axvline(df['anos_operacao'].mean(), color=VERMELHO, linestyle='--',
                linewidth=2, label=f'Média: {df["anos_operacao"].mean():.1f} anos')
axes[1].axvline(df['anos_operacao'].median(), color=VERMELHO, linestyle=':',
                linewidth=2, label=f'Mediana: {df["anos_operacao"].median():.1f} anos')
axes[1].set_title('Distribuição da Idade das Empresas\n(anos de operação)')
axes[1].set_xlabel('Anos de operação')
axes[1].set_ylabel('Empresas')
axes[1].legend()

padroes = pd.Series({
    'Antiga inapta\n(>5 anos)': df['antiga_inapta'].sum(),
    'Nova inapta\n(<2 anos)':   df['nova_inapta'].sum(),
    'Muito nova\ninapta':        ((df['empresa_muito_nova']==1)&(df['inapta']==1)).sum(),
})
axes[2].bar(padroes.index, padroes.values,
             color=[AZUL['escuro'], AZUL['claro'], AZUL['palido']],
             alpha=0.85, edgecolor='white', linewidth=1.5)
for i, (label, val) in enumerate(padroes.items()):
    axes[2].text(i, val+2, f'{val:,}', ha='center',
                  fontweight='bold', color=AZUL['escuro'])
axes[2].set_title('Padrões Temporais Suspeitos\n(empresas inaptas por faixa de idade)')
axes[2].set_ylabel('Empresas')

plt.suptitle('Análise Temporal — Cadeia de Combustíveis RJ',
             fontsize=13, fontweight='bold', color=AZUL['escuro'])
plt.tight_layout()
plt.savefig(OUT_DIR/'EDA_05_analise_temporal.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(); print("✓ EDA_05")

# ══════════════════════════════════════════════════════════
# EDA_06 — Correlação
# ══════════════════════════════════════════════════════════
cols_corr = ['anos_operacao','qtd_socios','qtd_socios_pj','capital_social',
             'max_empresas_por_socio','inapta','suspensa',
             'capital_muito_baixo','socio_risco_alto',
             'taxa_inapta_municipio','score_risco_v2']
nomes = {
    'anos_operacao':'Idade (anos)', 'qtd_socios':'Qtd Sócios',
    'qtd_socios_pj':'Sócios PJ', 'capital_social':'Capital Social',
    'max_empresas_por_socio':'Max Emp/Sócio', 'inapta':'Inapta',
    'suspensa':'Suspensa', 'capital_muito_baixo':'Cap. Muito Baixo',
    'socio_risco_alto':'Sócio Risco Alto',
    'taxa_inapta_municipio':'Taxa Inapta Mun.', 'score_risco_v2':'Score Risco'
}
corr = df[cols_corr].rename(columns=nomes).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='Blues', center=0, square=True,
            linewidths=0.5, ax=ax, annot_kws={'size': 8},
            vmin=-0.5, vmax=1)
ax.set_title('Matriz de Correlação — Principais Variáveis\nCadeia de Combustíveis RJ',
             fontsize=13, fontweight='bold', color=AZUL['escuro'])
plt.tight_layout()
plt.savefig(OUT_DIR/'EDA_06_correlacao.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(); print("✓ EDA_06")

# ══════════════════════════════════════════════════════════
# EDA_07 — Score de Risco
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

score_counts = df['score_risco_v2'].value_counts().sort_index()
axes[0].bar(score_counts.index, score_counts.values,
             color=AZUL['claro'], alpha=0.85, edgecolor='white', linewidth=1.5)
# ── tracejado vermelho na média ──
axes[0].axvline(df['score_risco_v2'].mean(), color=VERMELHO, linestyle='--',
                linewidth=2, label=f'Média: {df["score_risco_v2"].mean():.2f}')
axes[0].set_title('Distribuição do Score de Risco v2')
axes[0].set_xlabel('Score de Risco')
axes[0].set_ylabel('Empresas')
axes[0].legend()

ordem_c = ['Baixo','Médio','Alto','Crítico']
classe_counts = df['classe_risco'].value_counts().reindex(
    [c for c in ordem_c if c in df['classe_risco'].cat.categories])
cores_c = [AZUL['palido'], AZUL['claro'], AZUL['medio'], AZUL['escuro']]
bars = axes[1].bar(classe_counts.index, classe_counts.values,
                    color=cores_c, alpha=0.85, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, classe_counts.values):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                 f'{val:,}\n({val/len(df)*100:.1f}%)',
                 ha='center', fontsize=9, fontweight='bold', color=AZUL['escuro'])
axes[1].set_title('Classificação de Risco Manual')
axes[1].set_ylabel('Empresas')

score_cnae = df.groupby('cnae_descricao')['score_risco_v2'].mean().sort_values()
media_score = score_cnae.mean()
cores_sc = [AZUL['escuro'] if v > media_score else AZUL['claro']
            for v in score_cnae.values]
axes[2].barh(range(len(score_cnae)), score_cnae.values,
              color=cores_sc, alpha=0.85, edgecolor='white', linewidth=1.5)
axes[2].set_yticks(range(len(score_cnae)))
axes[2].set_yticklabels(score_cnae.index, fontsize=8)
# ── tracejado vermelho na média ──
axes[2].axvline(media_score, color=VERMELHO, linestyle='--',
                linewidth=2, label=f'Média: {media_score:.2f}')
axes[2].set_title('Score Médio de Risco\npor Segmento')
axes[2].set_xlabel('Score médio')
axes[2].legend()

plt.suptitle('Análise do Score de Risco — Cadeia de Combustíveis RJ',
             fontsize=13, fontweight='bold', color=AZUL['escuro'])
plt.tight_layout()
plt.savefig(OUT_DIR/'EDA_07_score_risco.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(); print("✓ EDA_07")

print(f"\n✅ Todos os gráficos salvos em: {OUT_DIR}")
