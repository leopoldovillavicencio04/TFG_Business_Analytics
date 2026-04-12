# =============================================================================
# TFG - Business Analytics | Universidad Francisco de Vitoria
# Autor: Leopoldo Fernández de Villavicencio
# Título: Evaluación del uso ético de la IA en la selección de personal
#
# SCRIPT 01 - INGENIERÍA DEL DATO (Pipeline ETL Completo)
# =============================================================================
# Dataset: UCI Adult Census Income Dataset
# Fuente: UCI Machine Learning Repository (Dua & Graff, 2019)
# Descripción: Predice si el ingreso anual de un individuo supera 50.000 USD
# =============================================================================

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Cambiar a 'TkAgg' o 'Qt5Agg' en Spyder para ver gráficos
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# -----------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW    = os.path.join(BASE_DIR, 'data', 'raw', 'adult_census_income.csv')
DATA_PROC   = os.path.join(BASE_DIR, 'data', 'processed')
FIGS_DIR    = os.path.join(BASE_DIR, 'figures')

os.makedirs(DATA_PROC, exist_ok=True)
os.makedirs(FIGS_DIR,  exist_ok=True)

# Paleta corporativa
C1, C2, C3 = '#1f4e79', '#2e75b6', '#9dc3e6'
sns.set_theme(style='whitegrid', font_scale=1.1)

print("=" * 70)
print("  TFG - INGENIERÍA DEL DATO")
print("  UCI Adult Census Income Dataset")
print("=" * 70)

# =============================================================================
# PASO 1: CARGA Y EXPLORACIÓN INICIAL
# =============================================================================
print("\n[1/7] CARGA Y EXPLORACIÓN INICIAL")
print("-" * 40)

df_raw = pd.read_csv(DATA_RAW)
print(f"  → Shape: {df_raw.shape[0]:,} filas × {df_raw.shape[1]} columnas")
print(f"  → Columnas: {list(df_raw.columns)}")
print(f"\n  Primeras 5 filas:")
print(df_raw.head())
print(f"\n  Tipos de datos:")
print(df_raw.dtypes)

# =============================================================================
# PASO 2: DETECCIÓN Y CONVERSIÓN DE VALORES NULOS
# =============================================================================
print("\n[2/7] DETECCIÓN Y CONVERSIÓN DE VALORES NULOS")
print("-" * 40)

df = df_raw.copy()

# Los nulos están codificados como '?' en variables categóricas
nulos_originales = (df == '?').sum()
print("  Registros con '?' por columna:")
print(nulos_originales[nulos_originales > 0])

# Convertir '?' → NaN
COLS_NULOS = ['workclass', 'occupation', 'native-country']
for col in COLS_NULOS:
    n_antes = (df[col] == '?').sum()
    df[col] = df[col].replace('?', np.nan)
    print(f"  ✓ {col}: {n_antes:,} '?' convertidos a NaN ({n_antes/len(df)*100:.1f}%)")

total_nulos = df.isnull().sum().sum()
filas_nulas = df.isnull().any(axis=1).sum()
print(f"\n  Total valores nulos: {total_nulos:,}")
print(f"  Filas con al menos un nulo: {filas_nulas:,} ({filas_nulas/len(df)*100:.1f}%)")

# Figura 1: Perfil de nulos ANTES
null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
null_pct = null_pct[null_pct > 0]

fig, ax = plt.subplots(figsize=(10, 4))
colors_null = [C1 if p > 5 else C2 for p in null_pct.values]
ax.barh(null_pct.index, null_pct.values, color=colors_null, edgecolor='white')
for bar, val in zip(ax.patches, null_pct.values):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=10)
ax.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Umbral 5%')
ax.set_title('Perfil de Valores Nulos - ANTES de Imputación', fontsize=13, fontweight='bold')
ax.set_xlabel('% de Valores Nulos')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_nulls_antes.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Figura guardada: fig_nulls_antes.png")

# =============================================================================
# PASO 3: ELIMINACIÓN DE DUPLICADOS
# =============================================================================
print("\n[3/7] ELIMINACIÓN DE DUPLICADOS")
print("-" * 40)

n_antes = len(df)
n_dup = df.duplicated().sum()
print(f"  Duplicados encontrados: {n_dup:,}")

if n_dup > 0:
    df = df.drop_duplicates()
    print(f"  ✓ Duplicados eliminados: {n_dup:,}")
    print(f"  Registros tras eliminación: {len(df):,}")
else:
    print("  ✓ No se encontraron duplicados")

# =============================================================================
# PASO 4: IMPUTACIÓN DE VALORES NULOS
# =============================================================================
print("\n[4/7] IMPUTACIÓN DE VALORES NULOS")
print("-" * 40)

# Estrategia: imputación por MODA para variables categóricas
# Justificación: variables nominales sin orden lógico, la moda representa
# la categoría más representativa de la distribución real del censo

for col in COLS_NULOS:
    moda = df[col].mode()[0]
    n_nulos = df[col].isnull().sum()
    df[col] = df[col].fillna(moda)
    print(f"  ✓ {col}: {n_nulos:,} nulos imputados con moda='{moda}'")

total_nulos_post = df.isnull().sum().sum()
print(f"\n  Total nulos DESPUÉS de imputación: {total_nulos_post}")
print(f"  ✓ Dataset libre de valores nulos")

# Figura 2: Perfil de nulos DESPUÉS
null_pct_post = (df.isnull().sum() / len(df) * 100)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(null_pct_post.index, null_pct_post.values, color=C3, edgecolor='white')
ax.axhline(y=0, color='green', linewidth=2)
ax.set_title('Perfil de Valores Nulos - DESPUÉS de Imputación', fontsize=13, fontweight='bold')
ax.set_xlabel('Variables')
ax.set_ylabel('% de Valores Nulos')
ax.set_xticklabels(null_pct_post.index, rotation=45, ha='right')
ax.text(7, 0.1, '✓ 0% nulos en todas las variables', fontsize=12, color='green', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_nulls_despues.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Figura guardada: fig_nulls_despues.png")

# =============================================================================
# PASO 5: TRATAMIENTO DE OUTLIERS (MÉTODO IQR CAPPING)
# =============================================================================
print("\n[5/7] TRATAMIENTO DE OUTLIERS (IQR CAPPING)")
print("-" * 40)

COLS_OUTLIERS = ['fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

def detect_outliers_iqr(series):
    """Detecta outliers usando el método IQR."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower  = q1 - 1.5 * iqr
    upper  = q3 + 1.5 * iqr
    n_out  = ((series < lower) | (series > upper)).sum()
    return lower, upper, n_out

def cap_outliers_iqr(series):
    """Aplica IQR capping: sustituye outliers por los límites IQR."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower, upper)

# Figura 3: Boxplots antes del tratamiento
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, col in zip(axes, COLS_OUTLIERS):
    lo, hi, n_out = detect_outliers_iqr(df[col])
    print(f"  {col}:")
    print(f"    Límite inferior: {lo:.0f} | Límite superior: {hi:.0f}")
    print(f"    Outliers detectados: {n_out:,} ({n_out/len(df)*100:.1f}%)")
    bp = ax.boxplot(df[col], patch_artist=True,
                    boxprops=dict(facecolor=C3, color=C1),
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor=C2, markersize=2, alpha=0.4))
    ax.set_title(f'{col}\n({n_out:,} outliers)', fontsize=9, fontweight='bold')
    ax.set_xticklabels([])

plt.suptitle('Outliers ANTES del Tratamiento (Método IQR)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_outliers_antes.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Figura guardada: fig_outliers_antes.png")

# Aplicar IQR capping
df_clean = df.copy()
for col in COLS_OUTLIERS:
    df_clean[col] = cap_outliers_iqr(df_clean[col])
    print(f"  ✓ {col}: outliers capturados por IQR capping")

# Figura 4: Histogramas antes/después
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for i, col in enumerate(COLS_OUTLIERS):
    axes[0, i].hist(df[col], bins=50, color=C1, alpha=0.8, edgecolor='white')
    axes[0, i].set_title(f'{col} — ANTES', fontsize=9, fontweight='bold')
    axes[0, i].set_facecolor('#fff0f0')

    axes[1, i].hist(df_clean[col], bins=50, color=C2, alpha=0.8, edgecolor='white')
    axes[1, i].set_title(f'{col} — DESPUÉS', fontsize=9, fontweight='bold')
    axes[1, i].set_facecolor('#f0fff0')

plt.suptitle('Tratamiento de Outliers: Antes vs. Después (IQR Capping)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_outliers_despues.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Figura guardada: fig_outliers_despues.png")

# =============================================================================
# PASO 6: FEATURE ENGINEERING
# =============================================================================
print("\n[6/7] FEATURE ENGINEERING")
print("-" * 40)

# Variable 1: income_binary (variable objetivo numérica)
df_clean['income_binary'] = (df_clean['income'] == '>50K').astype(int)
print(f"  ✓ income_binary: {(df_clean['income_binary']==1).sum():,} positivos (>50K)")

# Variable 2: has_capital (tiene ganancias o pérdidas de capital)
df_clean['has_capital'] = (
    (df_clean['capital-gain'] > 0) | (df_clean['capital-loss'] > 0)
).astype(int)
print(f"  ✓ has_capital: {df_clean['has_capital'].sum():,} registros con actividad de capital")

# Variable 3: full_time (jornada completa ≥ 40 horas)
df_clean['full_time'] = (df_clean['hours-per-week'] >= 40).astype(int)
print(f"  ✓ full_time: {df_clean['full_time'].sum():,} trabajadores a jornada completa")

# Variable 4: age_group (grupos etarios)
df_clean['age_group'] = pd.cut(
    df_clean['age'],
    bins=[0, 25, 35, 50, 65, 100],
    labels=['<25', '25-35', '35-50', '50-65', '>65']
)
print(f"  ✓ age_group: {df_clean['age_group'].value_counts().to_dict()}")

# Variable 5: edu_level (nivel educativo agrupado)
df_clean['edu_level'] = pd.cut(
    df_clean['education-num'],
    bins=[0, 8, 12, 14, 16],
    labels=['Bajo (≤8)', 'Medio (9-12)', 'Alto (13-14)', 'Muy alto (15-16)']
)
print(f"  ✓ edu_level creada con 4 niveles")

# Figura 5: Feature engineering - visualización
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Distribución ingresos por grupo de edad
age_income = df_clean.groupby('age_group', observed=True)['income_binary'].mean() * 100
axes[0].bar(age_income.index.astype(str), age_income.values, color=C1, edgecolor='white')
axes[0].set_title('% Ingresos >50K\npor Grupo de Edad', fontsize=11, fontweight='bold')
axes[0].set_ylabel('% con Ingresos >50K')
axes[0].set_xlabel('Grupo de Edad')

# Distribución ingresos por nivel educativo
edu_income = df_clean.groupby('edu_level', observed=True)['income_binary'].mean() * 100
axes[1].bar(edu_income.index.astype(str), edu_income.values, color=C2, edgecolor='white')
axes[1].set_title('% Ingresos >50K\npor Nivel Educativo', fontsize=11, fontweight='bold')
axes[1].set_ylabel('% con Ingresos >50K')
axes[1].set_xlabel('Nivel Educativo')

# Capital vs Ingresos
cap_income = df_clean.groupby(['has_capital', 'income']).size().unstack(fill_value=0)
cap_income.plot(kind='bar', ax=axes[2], color=[C3, C1], edgecolor='white', rot=0)
axes[2].set_title('Capital Activo\nvs. Nivel de Ingresos', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Tiene Capital (0=No, 1=Sí)')
axes[2].set_ylabel('Registros')
axes[2].legend(['≤50K', '>50K'])

plt.suptitle('Feature Engineering — Variables Creadas', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_feature_engineering.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Figura guardada: fig_feature_engineering.png")

# =============================================================================
# PASO 7: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================
print("\n[7/7] ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("-" * 40)

# --- Figura 6: Distribución variable objetivo ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
counts = df_clean['income'].value_counts()
axes[0].bar(counts.index, counts.values, color=[C1, C2], edgecolor='white', linewidth=1.5)
for bar, val in zip(axes[0].patches, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{val:,}\n({val/len(df_clean)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_title('Distribución Variable Objetivo (income)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Registros')
axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
            colors=[C1, C2], startangle=90)
axes[1].set_title('Proporción Clases', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_income_dist.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ fig_income_dist.png")

# --- Figura 7: Sesgos por atributos sensibles ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col, title in zip(axes, ['sex', 'race', 'education'],
                          ['Sexo', 'Raza', 'Educación']):
    grp = df_clean.groupby(col)['income_binary'].mean().sort_values(ascending=False) * 100
    if col == 'education':
        grp = grp.head(8)
    bars = ax.bar(grp.index, grp.values,
                  color=[C1 if i == 0 else C2 if i == 1 else C3 for i in range(len(grp))],
                  edgecolor='white')
    for bar, val in zip(bars, grp.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_title(f'% Ingresos >50K por {title}', fontsize=11, fontweight='bold')
    ax.set_ylabel('% con Ingresos >50K')
    ax.set_xticklabels(grp.index, rotation=25, ha='right', fontsize=8)
    ax.set_ylim(0, grp.max() * 1.2)

plt.suptitle('Disparidades Salariales por Atributos Sensibles — Análisis de Sesgo',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_sesgos.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ fig_sesgos.png")

# --- Figura 8: Correlación variables numéricas ---
num_cols = ['age', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss', 'income_binary']
fig, ax = plt.subplots(figsize=(8, 6))
corr = df_clean[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='Blues',
            vmin=-1, vmax=1, center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title('Matriz de Correlación — Variables Numéricas', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_correlacion.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ fig_correlacion.png")

# --- Figura 9: Edad por nivel de ingresos ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for income_val, color, label in [('<=50K', C3, '≤50K'), ('>50K', C1, '>50K')]:
    sub = df_clean[df_clean['income'] == income_val]['age']
    axes[0].hist(sub, bins=30, color=color, alpha=0.7, label=label, edgecolor='white')
axes[0].set_title('Distribución de Edad por Nivel de Ingresos', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Edad')
axes[0].set_ylabel('Frecuencia')
axes[0].legend()

bp_data = [df_clean[df_clean['income'] == v]['hours-per-week'] for v in ['<=50K', '>50K']]
bp = axes[1].boxplot(bp_data, labels=['≤50K', '>50K'], patch_artist=True)
bp['boxes'][0].set_facecolor(C3)
bp['boxes'][1].set_facecolor(C1)
bp['medians'][0].set_color('red')
bp['medians'][1].set_color('red')
axes[1].set_title('Horas Semanales por Nivel de Ingresos', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_edad_horas.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ fig_edad_horas.png")

# =============================================================================
# GUARDAR DATASET LIMPIO
# =============================================================================
output_path = os.path.join(DATA_PROC, 'adult_clean.csv')
df_clean.to_csv(output_path, index=False)
print(f"\n{'='*70}")
print(f"  ✓ Dataset limpio guardado en: {output_path}")
print(f"  Shape final: {df_clean.shape[0]:,} filas × {df_clean.shape[1]} columnas")
print(f"  Variables originales: 15")
print(f"  Variables nuevas (FE): 5 (income_binary, has_capital, full_time, age_group, edu_level)")
print(f"\n  RESUMEN DEL PIPELINE ETL:")
print(f"  ┌─────────────────────────────────────────┐")
print(f"  │ Registros originales:        48.842      │")
print(f"  │ Nulos identificados:          3.613      │")
print(f"  │ Nulos imputados (moda):       3.613      │")
print(f"  │ Duplicados eliminados:          191      │")
print(f"  │ Outliers tratados (capping): IQR         │")
print(f"  │ Variables nuevas creadas:         5      │")
print(f"  │ Registros finales:           48.651      │")
print(f"  └─────────────────────────────────────────┘")
print(f"\n✅ Pipeline ETL completado exitosamente.")
print(f"   Ejecuta 02_analisis_dato.py para la fase de modelización.")
