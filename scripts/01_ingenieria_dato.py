"""
Script 01 — Ingeniería del Dato
TFG Business Analytics | UFV 2025-2026
Autor: Leopoldo Fernández de Villavicencio Alberola

Pipeline ETL completo sobre UCI Adult Census Income (Becker & Kohavi, 1996).
Produce: data/processed/adult_clean.csv + 9 figuras + reports/reporte_ingenieria.txt

Cifras esperadas tras ejecución:
  - Registros originales : 48.842
  - Duplicados eliminados: 52
  - Registros finales    : 48.790
  - Variables finales    : 23 (15 originales + 6 derivadas + 2 log)
  - Target >50K          : 23,94 %
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ────────────────────────────────────────────────────────────────────
RAW_PATH  = os.path.join('data', 'raw',       'adult_census_income.csv')
PROC_PATH = os.path.join('data', 'processed', 'adult_clean.csv')
FIG_DIR   = 'figures'
REP_DIR   = 'reports'

for d in [os.path.join('data', 'processed'), FIG_DIR, REP_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Paleta corporativa ────────────────────────────────────────────────────────
BLUE   = '#1F4E79'
LBLUE  = '#2E75B6'
ORANGE = '#C55A11'
GREY   = '#404040'
GREEN  = '#375623'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

reporte = []

def log(msg):
    print(msg)
    reporte.append(msg)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA
# ═══════════════════════════════════════════════════════════════════════════════
COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

log("=" * 60)
log("SCRIPT 01 — INGENIERÍA DEL DATO")
log("=" * 60)

try:
    df = pd.read_csv(RAW_PATH, header=None, names=COLUMNS,
                     na_values=['?', ' ?'], skipinitialspace=True)
    log(f"\n[CARGA] {len(df):,} registros × {df.shape[1]} variables")
except FileNotFoundError:
    # Segundo intento: el CSV puede tener cabecera
    df = pd.read_csv(RAW_PATH, na_values=['?', ' ?'], skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    log(f"\n[CARGA] {len(df):,} registros × {df.shape[1]} variables (con cabecera)")

# Limpiar espacios en strings
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Normalizar income: '>50K.' → '>50K'
df['income'] = df['income'].str.replace('.', '', regex=False)

n_original = len(df)
log(f"  Registros originales: {n_original:,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ANÁLISIS EXPLORATORIO INICIAL — nulos
# ═══════════════════════════════════════════════════════════════════════════════
null_counts_antes = df.isnull().sum()
null_pct_antes    = (null_counts_antes / len(df) * 100).round(2)

log("\n[NULOS ANTES DE IMPUTACIÓN]")
for col in null_counts_antes[null_counts_antes > 0].index:
    log(f"  {col}: {null_counts_antes[col]:,} ({null_pct_antes[col]:.2f} %)")

# ── Figura 1: Nulos antes ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
cols_con_nulos = null_pct_antes[null_pct_antes > 0]
bars = ax.bar(cols_con_nulos.index, cols_con_nulos.values, color=LBLUE, edgecolor='white')
ax.axhline(5, color=ORANGE, linestyle='--', linewidth=1.5, label='Umbral 5 %')
ax.set_title('Perfil de valores nulos antes de la imputación', fontweight='bold', color=BLUE)
ax.set_ylabel('% de nulos')
ax.set_ylim(0, max(cols_con_nulos.values) * 1.3)
for bar, val in zip(bars, cols_con_nulos.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.2f} %', ha='center', va='bottom', fontsize=10)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_nulls_antes.png'), dpi=150)
plt.close()
log("  → fig_nulls_antes.png guardada")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ELIMINACIÓN DE DUPLICADOS
# ═══════════════════════════════════════════════════════════════════════════════
n_dup = df.duplicated().sum()
df    = df.drop_duplicates().reset_index(drop=True)
log(f"\n[DUPLICADOS] {n_dup} registros duplicados eliminados → {len(df):,} registros")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. IMPUTACIÓN POR MODA
# ═══════════════════════════════════════════════════════════════════════════════
COLS_NULOS = ['workclass', 'occupation', 'native-country']
log("\n[IMPUTACIÓN] Moda en variables categóricas con nulos:")
for col in COLS_NULOS:
    moda = df[col].mode()[0]
    n_imp = df[col].isnull().sum()
    df[col].fillna(moda, inplace=True)
    log(f"  {col}: {n_imp:,} valores → '{moda}'")

# ── Figura 2: Nulos después ──────────────────────────────────────────────────
null_counts_despues = df.isnull().sum()
null_pct_despues    = (null_counts_despues / len(df) * 100).round(4)

fig, ax = plt.subplots(figsize=(10, 4))
cols_show = null_pct_antes[null_pct_antes > 0].index
vals_despues = [null_pct_despues[c] for c in cols_show]
bars = ax.bar(cols_show, vals_despues, color=GREEN, edgecolor='white')
ax.axhline(5, color=ORANGE, linestyle='--', linewidth=1.5, label='Umbral 5 %')
ax.set_title('Perfil de valores nulos después de la imputación', fontweight='bold', color=BLUE)
ax.set_ylabel('% de nulos')
ax.set_ylim(0, 6)
for bar, val in zip(bars, vals_despues):
    ax.text(bar.get_x() + bar.get_width()/2, 0.05,
            f'{val:.2f} %', ha='center', va='bottom', fontsize=10, color='white', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_nulls_despues.png'), dpi=150)
plt.close()
log("  → fig_nulls_despues.png guardada")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRATAMIENTO DE OUTLIERS
# ═══════════════════════════════════════════════════════════════════════════════
log("\n[OUTLIERS]")

# ── Figura 3: Boxplots antes ─────────────────────────────────────────────────
VARS_NUM = ['fnlwgt', 'hours-per-week', 'capital-gain', 'capital-loss']
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, col in zip(axes, VARS_NUM):
    ax.boxplot(df[col].dropna(), patch_artist=True,
               boxprops=dict(facecolor=LBLUE, alpha=0.7),
               medianprops=dict(color=ORANGE, linewidth=2))
    ax.set_title(col, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
fig.suptitle('Boxplots de variables numéricas — ANTES del tratamiento',
             fontweight='bold', color=BLUE, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_outliers_antes.png'), dpi=150, bbox_inches='tight')
plt.close()
log("  → fig_outliers_antes.png guardada")

# IQR capping: fnlwgt y hours-per-week
for col in ['fnlwgt', 'hours-per-week']:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR    = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out  = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col] = df[col].clip(lo, hi)
    log(f"  IQR capping {col}: {n_out:,} outliers cappados "
        f"[{lo:,.1f} – {hi:,.1f}]")

# log1p: capital-gain y capital-loss -> se crean DOS columnas nuevas
# (>91 % de valores son 0 → Q1=Q3=0 → IQR inviable; se conservan las originales)
for col in ['capital-gain', 'capital-loss']:
    pct_cero = (df[col] == 0).mean() * 100
    df[col + '_log'] = np.log1p(df[col])
    log(f"  log1p {col} -> {col}_log: {pct_cero:.1f} % de ceros — IQR capping inviable")

# ── Figura 4: Boxplots después ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
titles = ['fnlwgt (IQR capping)', 'hours-per-week (IQR capping)',
          'capital-gain (log1p)',  'capital-loss (log1p)']
cols_despues = ['fnlwgt', 'hours-per-week', 'capital-gain_log', 'capital-loss_log']
for ax, col, title in zip(axes, cols_despues, titles):
    ax.boxplot(df[col].dropna(), patch_artist=True,
               boxprops=dict(facecolor=GREEN, alpha=0.7),
               medianprops=dict(color=ORANGE, linewidth=2))
    ax.set_title(title, fontsize=9)
fig.suptitle('Boxplots de variables numéricas — DESPUÉS del tratamiento',
             fontweight='bold', color=BLUE, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_outliers_despues.png'), dpi=150, bbox_inches='tight')
plt.close()
log("  → fig_outliers_despues.png guardada")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
log("\n[FEATURE ENGINEERING]")

# income_binary
df['income_binary'] = (df['income'] == '>50K').astype(int)

# has_capital — sobre los valores originales (intactos)
df['has_capital'] = ((df['capital-gain'] > 0) | (df['capital-loss'] > 0)).astype(int)

# full_time — hours-per-week fue cappado con IQR (no log1p), umbral 40 directo
df['full_time'] = (df['hours-per-week'] >= 40).astype(int)

# age_group
df['age_group'] = pd.cut(df['age'],
                          bins=[0, 25, 35, 50, 65, 100],
                          labels=['<25', '25-35', '35-50', '50-65', '>65'])

# edu_level
df['edu_level'] = pd.cut(df['education-num'],
                          bins=[0, 8, 12, 14, 16],
                          labels=['Básico', 'Secundaria', 'Superior', 'Postgrado'])

# capital_ratio — sobre los valores originales de capital
df['capital_ratio'] = (df['capital-gain'] - df['capital-loss']) / (df['fnlwgt'] + 1)

log(f"  income_binary : {df['income_binary'].sum():,} positivos ({df['income_binary'].mean()*100:.2f} %)")
log(f"  has_capital   : {df['has_capital'].sum():,} con capital activo ({df['has_capital'].mean()*100:.1f} %)")
log(f"  full_time     : {df['full_time'].sum():,} jornada completa ({df['full_time'].mean()*100:.1f} %)")
log(f"  capital_ratio : r con income_binary = {df['capital_ratio'].corr(df['income_binary']):.3f}")

# ── Figura 5: Feature engineering ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# % >50K por age_group
ag = df.groupby('age_group', observed=True)['income_binary'].mean() * 100
axes[0].bar(ag.index.astype(str), ag.values, color=LBLUE, edgecolor='white')
axes[0].set_title('% >50K por grupo de edad', fontweight='bold')
axes[0].set_ylabel('% >50K')
for i, v in enumerate(ag.values):
    axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

# % >50K por edu_level
el = df.groupby('edu_level', observed=True)['income_binary'].mean() * 100
axes[1].bar(el.index.astype(str), el.values, color=ORANGE, edgecolor='white')
axes[1].set_title('% >50K por nivel educativo', fontweight='bold')
axes[1].set_ylabel('% >50K')
for i, v in enumerate(el.values):
    axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

# % >50K por has_capital
hc = df.groupby('has_capital')['income_binary'].mean() * 100
axes[2].bar(['Sin capital\nactivo', 'Con capital\nactivo'], hc.values,
            color=[GREY, GREEN], edgecolor='white')
axes[2].set_title('% >50K por presencia de capital', fontweight='bold')
axes[2].set_ylabel('% >50K')
for i, v in enumerate(hc.values):
    axes[2].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

fig.suptitle('Variables derivadas en feature engineering', fontweight='bold',
             color=BLUE, fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_feature_engineering.png'), dpi=150)
plt.close()
log("  → fig_feature_engineering.png guardada")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. EDA POST-LIMPIEZA
# ═══════════════════════════════════════════════════════════════════════════════
log("\n[EDA POST-LIMPIEZA]")

# ── Figura 6: Distribución variable objetivo ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
counts = df['income'].value_counts()
bars   = ax.bar(counts.index, counts.values,
                color=[GREY, LBLUE], edgecolor='white', width=0.5)
ax.set_title('Distribución de la variable objetivo (income)',
             fontweight='bold', color=BLUE)
ax.set_ylabel('Número de registros')
for bar, val in zip(bars, counts.values):
    pct = val / len(df) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{val:,}\n({pct:.2f} %)', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(counts.values) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_income_dist.png'), dpi=150)
plt.close()
log(f"  Distribución: {counts.to_dict()}")
log("  → fig_income_dist.png guardada")

# ── Figura 7: Sesgos por atributos sensibles ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Sexo
sex_rate = df.groupby('sex')['income_binary'].mean() * 100
axes[0].bar(sex_rate.index, sex_rate.values, color=[LBLUE, ORANGE], edgecolor='white')
axes[0].set_title('Tasa real >50K por sexo', fontweight='bold')
axes[0].set_ylabel('% con ingresos >50K')
axes[0].axhline(10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Umbral DPD ref.')
for i, (idx, val) in enumerate(sex_rate.items()):
    axes[0].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold')

# Raza
race_rate = df.groupby('race')['income_binary'].mean() * 100
race_rate = race_rate.sort_values(ascending=False)
axes[1].bar(range(len(race_rate)), race_rate.values, color=LBLUE, edgecolor='white')
axes[1].set_xticks(range(len(race_rate)))
axes[1].set_xticklabels(race_rate.index, rotation=25, ha='right', fontsize=9)
axes[1].set_title('Tasa real >50K por raza', fontweight='bold')
axes[1].set_ylabel('% con ingresos >50K')
for i, val in enumerate(race_rate.values):
    axes[1].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

# Educación
edu_rate = df.groupby('education-num')['income_binary'].mean() * 100
axes[2].plot(edu_rate.index, edu_rate.values, color=LBLUE, marker='o', linewidth=2)
axes[2].set_title('Tasa real >50K por nivel educativo (num)', fontweight='bold')
axes[2].set_xlabel('education-num')
axes[2].set_ylabel('% con ingresos >50K')
axes[2].fill_between(edu_rate.index, edu_rate.values, alpha=0.15, color=LBLUE)

fig.suptitle('Disparidades reales por atributos sensibles (antes de cualquier modelo)',
             fontweight='bold', color=BLUE, fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_sesgos.png'), dpi=150)
plt.close()

brecha_sexo = sex_rate.max() - sex_rate.min()
brecha_raza = race_rate.max() - race_rate.min()
log(f"  Brecha sexo (Male-Female): {brecha_sexo:.1f} pp")
log(f"  Brecha raza (máx-mín)    : {brecha_raza:.1f} pp")
log("  → fig_sesgos.png guardada")

# ── Figura 8: Correlaciones ──────────────────────────────────────────────────
num_cols = ['age', 'education-num', 'hours-per-week', 'fnlwgt',
            'capital-gain', 'capital-loss', 'capital_ratio', 'income_binary']
corr_matrix = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, square=True)
ax.set_title('Matriz de correlaciones con income_binary',
             fontweight='bold', color=BLUE, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_correlacion.png'), dpi=150)
plt.close()

corr_target = corr_matrix['income_binary'].drop('income_binary').sort_values(ascending=False)
log("\n  Correlación con income_binary:")
for col, val in corr_target.items():
    log(f"    {col:20s}: {val:+.3f}")
log("  → fig_correlacion.png guardada")

# ── Figura 9: Edad e horas ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribución de edad por target
for label, color in [('<=50K', GREY), ('>50K', LBLUE)]:
    subset = df[df['income'] == label]['age']
    axes[0].hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
axes[0].set_title('Distribución de edad por nivel de ingresos', fontweight='bold')
axes[0].set_xlabel('Edad')
axes[0].set_ylabel('Frecuencia')
axes[0].legend()

# Histograma de horas por target
for label, color in [('<=50K', GREY), ('>50K', ORANGE)]:
    subset = df[df['income'] == label]['hours-per-week']
    axes[1].hist(subset, bins=25, alpha=0.6, color=color, label=label, edgecolor='white')
axes[1].set_title('Horas semanales por nivel de ingresos', fontweight='bold')
axes[1].set_xlabel('Horas por semana')
axes[1].set_ylabel('Frecuencia')
axes[1].legend()

fig.suptitle('Distribución de edad y horas trabajadas por nivel de ingresos',
             fontweight='bold', color=BLUE, fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_edad_horas.png'), dpi=150)
plt.close()
log("  → fig_edad_horas.png guardada")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. GUARDAR DATASET LIMPIO
# ═══════════════════════════════════════════════════════════════════════════════
df.to_csv(PROC_PATH, index=False)

log("\n" + "=" * 60)
log("RESUMEN FINAL")
log("=" * 60)
log(f"  Registros originales : {n_original:,}")
log(f"  Duplicados eliminados: {n_dup}")
log(f"  Registros finales    : {len(df):,}")
log(f"  Variables finales    : {df.shape[1]}")
log(f"  Target <=50K         : {(df['income_binary']==0).sum():,} ({(df['income_binary']==0).mean()*100:.2f} %)")
log(f"  Target >50K          : {(df['income_binary']==1).sum():,} ({(df['income_binary']==1).mean()*100:.2f} %)")
log(f"\n  Dataset limpio guardado en: {PROC_PATH}")
log(f"  Figuras guardadas en      : {FIG_DIR}/")

# ── Guardar reporte ──────────────────────────────────────────────────────────
rep_path = os.path.join(REP_DIR, 'reporte_ingenieria.txt')
with open(rep_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(reporte))
log(f"  Reporte guardado en       : {rep_path}")
log("\n[OK] Script 01 completado.")
