"""
Script 02 — Análisis del Dato
TFG Business Analytics | UFV 2025-2026
Autor: Leopoldo Fernández de Villavicencio Alberola

4 modelos supervisados + K-Means/PCA + equidad algorítmica (DPD)
sobre el dataset limpio producido por el script 01.

Cifras canónicas esperadas:
  AUC  LR=0.856 | DT=0.906 | RF=0.921 | GB=0.931
  DPD sexo  LR=0.378 | DT=0.345 | RF=0.345 | GB=0.323
  DPD raza  LR=0.242 | DT=0.260 | RF=0.250 | GB=0.267
  DPD interseccional GB = 0.439
  K-Means k=2, silhouette=0.128
  PCA PC1=14.6 %, PC2=11.9 %
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.preprocessing        import LabelEncoder, StandardScaler
from sklearn.model_selection      import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model         import LogisticRegression
from sklearn.tree                 import DecisionTreeClassifier
from sklearn.ensemble             import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics              import (accuracy_score, precision_score, recall_score,
                                          f1_score, roc_auc_score, confusion_matrix,
                                          roc_curve)
from sklearn.cluster              import KMeans
from sklearn.decomposition        import PCA
from sklearn.metrics              import silhouette_score

# ── Rutas ────────────────────────────────────────────────────────────────────
PROC_PATH = os.path.join('data', 'processed', 'adult_clean.csv')
FIG_DIR   = 'figures'
REP_DIR   = 'reports'
SEED      = 42

for d in [FIG_DIR, REP_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Paleta ───────────────────────────────────────────────────────────────────
BLUE   = '#1F4E79'
LBLUE  = '#2E75B6'
ORANGE = '#C55A11'
GREY   = '#595959'
GREEN  = '#375623'
RED    = '#C00000'

COLORES_MODELOS = [LBLUE, ORANGE, GREEN, RED]
NOMBRES_MODELOS = ['Reg. Logística', 'Árbol Decisión', 'Random Forest', 'Gradient Boosting']
KEYS_MODELOS    = ['LR', 'DT', 'RF', 'GB']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

print("=" * 60)
print("SCRIPT 02 — ANÁLISIS DEL DATO")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA Y PREPARACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(PROC_PATH)
print(f"\n[CARGA] {len(df):,} registros × {df.shape[1]} variables")

# Features categóricas a codificar (excluyendo las ya numéricas y el target)
FEATURES_CAT = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race', 'sex', 'native-country']
FEATURES_NUM = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                 'hours-per-week', 'capital_ratio']
FEATURES_BIN = ['has_capital', 'full_time']
TARGET       = 'income_binary'

# Guardar atributos protegidos antes de codificar
df['sex_raw']  = df['sex'].copy()
df['race_raw'] = df['race'].copy()

# Label encoding de categóricas
le_dict = {}
df_enc  = df.copy()
for col in FEATURES_CAT:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    le_dict[col] = le

# Columnas de features para los modelos
FEATURE_COLS = FEATURES_CAT + FEATURES_NUM + FEATURES_BIN

X = df_enc[FEATURE_COLS].values
y = df_enc[TARGET].values

print(f"  Features utilizadas: {len(FEATURE_COLS)}")
print(f"  Positivos (>50K)   : {y.sum():,} ({y.mean()*100:.2f} %)")

# División estratificada 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

# Atributos protegidos en test
sex_test  = df_enc.loc[X_test.shape[0] * (-1):, 'sex_raw'].values  # placeholder
# Reconstruir índices correctamente
idx_all   = np.arange(len(df_enc))
_, idx_test = train_test_split(idx_all, test_size=0.2, stratify=y, random_state=SEED)
sex_test   = df['sex_raw'].iloc[idx_test].values
race_test  = df['race_raw'].iloc[idx_test].values

# Escalado para LR
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ENTRENAMIENTO DE LOS 4 MODELOS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[MODELOS] Entrenando...")

# Sample weights para GB (class_weight equivalente)
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
sw_gb = np.where(y_train == 1, len(y_train) / (2 * pos_count),
                               len(y_train) / (2 * neg_count))

modelos = {
    'LR': LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                              class_weight='balanced', random_state=SEED),
    'DT': DecisionTreeClassifier(max_depth=10, min_samples_split=20,
                                  min_samples_leaf=10, class_weight='balanced',
                                  random_state=SEED),
    'RF': RandomForestClassifier(n_estimators=200, max_depth=15,
                                  min_samples_split=10, min_samples_leaf=5,
                                  class_weight='balanced', random_state=SEED, n_jobs=-1),
    'GB': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                      max_depth=5, subsample=0.8,
                                      random_state=SEED),
}

resultados = {}
y_proba_dict = {}
y_pred_dict  = {}

for key, modelo in modelos.items():
    if key == 'LR':
        modelo.fit(X_train_sc, y_train)
        y_prob = modelo.predict_proba(X_test_sc)[:, 1]
        y_pred = modelo.predict(X_test_sc)
    elif key == 'GB':
        modelo.fit(X_train, y_train, sample_weight=sw_gb)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        y_pred = modelo.predict(X_test)
    else:
        modelo.fit(X_train, y_train)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        y_pred = modelo.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)

    resultados[key] = {
        'accuracy': round(acc, 3), 'precision': round(prec, 3),
        'recall':   round(rec, 3), 'f1_macro':  round(f1, 3),
        'auc':      round(auc, 3), 'cm': cm.tolist()
    }
    y_proba_dict[key] = y_prob
    y_pred_dict[key]  = y_pred

    print(f"  {key:2s} | Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} "
          f"F1={f1:.3f} AUC={auc:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. VALIDACIÓN CRUZADA 5-FOLD
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[CV 5-FOLD]")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for key, modelo in modelos.items():
    X_cv = X_train_sc if key == 'LR' else X_train
    if key == 'GB':
        # CV manual para pasar sample_weight
        scores = []
        for train_idx, val_idx in cv.split(X_cv, y_train):
            sw_fold = sw_gb[train_idx]
            modelo_cv = GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=5,
                subsample=0.8, random_state=SEED)
            modelo_cv.fit(X_cv[train_idx], y_train[train_idx], sample_weight=sw_fold)
            y_cv_pred = modelo_cv.predict(X_cv[val_idx])
            scores.append(f1_score(y_train[val_idx], y_cv_pred,
                                   average='macro', zero_division=0))
        scores = np.array(scores)
    else:
        scores = cross_val_score(modelo, X_cv, y_train, cv=cv,
                                 scoring='f1_macro', n_jobs=-1)
    resultados[key]['cv_f1_mean'] = round(scores.mean(), 3)
    resultados[key]['cv_f1_std']  = round(scores.std(), 3)
    print(f"  {key:2s} | F1 macro CV: {scores.mean():.3f} ± {scores.std():.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. FIGURAS A1–A4
# ═══════════════════════════════════════════════════════════════════════════════

# ── figA1: Comparativa de métricas ──────────────────────────────────────────
metricas_keys = ['accuracy', 'precision', 'recall', 'f1_macro', 'auc']
metricas_labs = ['Accuracy', 'Precision\nmacro', 'Recall\nmacro', 'F1 macro', 'AUC-ROC']
x = np.arange(len(metricas_keys))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
for i, (key, nombre, color) in enumerate(zip(KEYS_MODELOS, NOMBRES_MODELOS, COLORES_MODELOS)):
    vals = [resultados[key][m] for m in metricas_keys]
    bars = ax.bar(x + i * width, vals, width, label=nombre, color=color, alpha=0.85)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metricas_labs)
ax.set_ylim(0.6, 1.0)
ax.set_ylabel('Valor')
ax.set_title('Comparativa de métricas — 4 modelos supervisados (test n=9.758)',
             fontweight='bold', color=BLUE)
ax.legend(loc='lower right')
ax.axhline(0.9, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figA1_metricas_comparativa.png'), dpi=150)
plt.close()
print("\n  → figA1_metricas_comparativa.png")

# ── figA2: Matrices de confusión ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, key, nombre, color in zip(axes, KEYS_MODELOS, NOMBRES_MODELOS, COLORES_MODELOS):
    cm = np.array(resultados[key]['cm'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['≤50K', '>50K'], yticklabels=['≤50K', '>50K'],
                linewidths=0.5, cbar=False)
    ax.set_title(f'{nombre}\nAUC={resultados[key]["auc"]:.3f}',
                 fontsize=10, fontweight='bold', color=color)
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
fig.suptitle('Matrices de confusión — conjunto de test (n=9.758)',
             fontweight='bold', color=BLUE, fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figA2_confusion_matrix.png'), dpi=150)
plt.close()
print("  → figA2_confusion_matrix.png")

# ── figA3: Curvas ROC ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for key, nombre, color in zip(KEYS_MODELOS, NOMBRES_MODELOS, COLORES_MODELOS):
    fpr, tpr, _ = roc_curve(y_test, y_proba_dict[key])
    auc = resultados[key]['auc']
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{nombre} (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Aleatorio (AUC=0.500)')
ax.set_xlabel('Tasa de falsos positivos')
ax.set_ylabel('Tasa de verdaderos positivos')
ax.set_title('Curvas ROC comparativas — 4 modelos', fontweight='bold', color=BLUE)
ax.legend(loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figA3_roc_curve.png'), dpi=150)
plt.close()
print("  → figA3_roc_curve.png")

# ── figA4: Importancia de variables (DT, RF, GB) ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, key, nombre, color in zip(
        axes, ['DT', 'RF', 'GB'],
        ['Árbol Decisión', 'Random Forest', 'Gradient Boosting'],
        [ORANGE, GREEN, RED]):
    imp = modelos[key].feature_importances_
    idx = np.argsort(imp)[-12:]
    ax.barh([FEATURE_COLS[i] for i in idx], imp[idx], color=color, alpha=0.8)
    ax.set_title(f'Top 12 variables — {nombre}', fontweight='bold')
    ax.set_xlabel('Importancia relativa')
fig.suptitle('Importancia de variables en modelos basados en árboles',
             fontweight='bold', color=BLUE, fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figA4_feature_importance.png'), dpi=150)
plt.close()
print("  → figA4_feature_importance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. ANÁLISIS DE EQUIDAD — DPD
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EQUIDAD ALGORÍTMICA]")

def calcular_dpd(y_prob, grupos, umbral=0.5):
    """DPD = max(tasa predicha positiva por grupo) - min(...)"""
    pred = (y_prob >= umbral).astype(int)
    tasas = {}
    for g in np.unique(grupos):
        mask = grupos == g
        if mask.sum() >= 30:
            tasas[g] = pred[mask].mean()
    if len(tasas) < 2:
        return 0.0, tasas
    dpd = max(tasas.values()) - min(tasas.values())
    return round(dpd, 3), tasas

fairness = {}
for key in KEYS_MODELOS:
    dpd_sex,  tasas_sex  = calcular_dpd(y_proba_dict[key], sex_test)
    dpd_race, tasas_race = calcular_dpd(y_proba_dict[key], race_test)
    fairness[key] = {
        'dpd_sex': dpd_sex, 'tasas_sex': tasas_sex,
        'dpd_race': dpd_race, 'tasas_race': tasas_race
    }
    print(f"  {key:2s} | DPD sexo={dpd_sex:.3f} | DPD raza={dpd_race:.3f} "
          f"| Cumple ≤0.10: {'✓' if dpd_sex <= 0.1 and dpd_race <= 0.1 else '✗'}")

# DPD interseccional para GB
grupos_intersec = np.array([f"{s}×{r}" for s, r in zip(sex_test, race_test)])
dpd_inter, tasas_inter = calcular_dpd(y_proba_dict['GB'], grupos_intersec)
print(f"\n  DPD interseccional GB (sexo × raza): {dpd_inter:.3f}")
print(f"  Grupo máx: {max(tasas_inter, key=tasas_inter.get)} "
      f"({max(tasas_inter.values()):.3f})")
print(f"  Grupo mín: {min(tasas_inter, key=tasas_inter.get)} "
      f"({min(tasas_inter.values()):.3f})")

# ── figA5: Equidad ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel sexo
x_sex = np.arange(len(KEYS_MODELOS))
dpd_sex_vals = [fairness[k]['dpd_sex'] for k in KEYS_MODELOS]
male_rates   = [fairness[k]['tasas_sex'].get('Male', 0) for k in KEYS_MODELOS]
female_rates = [fairness[k]['tasas_sex'].get('Female', 0) for k in KEYS_MODELOS]

width = 0.3
axes[0].bar(x_sex - width/2, male_rates,   width, label='Male',   color=LBLUE, alpha=0.85)
axes[0].bar(x_sex + width/2, female_rates, width, label='Female', color=ORANGE, alpha=0.85)
axes[0].axhline(0.1, color='red', linestyle='--', linewidth=1.2, label='Umbral DPD=0.10')
for i, dpd in enumerate(dpd_sex_vals):
    axes[0].annotate(f'DPD={dpd:.3f}', xy=(i, max(male_rates[i], female_rates[i]) + 0.01),
                     ha='center', fontsize=9, color=RED, fontweight='bold')
axes[0].set_xticks(x_sex)
axes[0].set_xticklabels(NOMBRES_MODELOS, rotation=15, ha='right')
axes[0].set_title('Paridad Demográfica por sexo', fontweight='bold')
axes[0].set_ylabel('Tasa de predicción positiva (>50K)')
axes[0].legend()

# Panel raza (GB únicamente con las 5 categorías)
razas = sorted(fairness['GB']['tasas_race'].keys())
tasas_gb_raza = [fairness['GB']['tasas_race'][r] for r in razas]
colors_raza = [LBLUE, ORANGE, GREEN, RED, GREY]
bars = axes[1].bar(razas, tasas_gb_raza, color=colors_raza[:len(razas)], alpha=0.85)
axes[1].axhline(0.1, color='red', linestyle='--', linewidth=1.2, label='Umbral DPD=0.10')
for bar, val in zip(bars, tasas_gb_raza):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
axes[1].set_title(f'Paridad Demográfica por raza — GB\n(DPD={fairness["GB"]["dpd_race"]:.3f})',
                  fontweight='bold')
axes[1].set_ylabel('Tasa de predicción positiva (>50K)')
axes[1].tick_params(axis='x', rotation=20)
axes[1].legend()

fig.suptitle('Análisis de equidad algorítmica (DPD) — umbral de referencia ≤ 0,10',
             fontweight='bold', color=BLUE, fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figA5_fairness.png'), dpi=150)
plt.close()
print("  → figA5_fairness.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. K-MEANS + PCA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[K-MEANS + PCA]")

# Muestra de 5000 para K-Means
np.random.seed(SEED)
idx_km = np.random.choice(len(X), size=5000, replace=False)
X_km   = StandardScaler().fit_transform(X[idx_km])
y_km   = y[idx_km]

inercias    = []
silhouettes = []
K_RANGE     = range(2, 10)

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_km)
    inercias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_km, labels))

k_optimo = K_RANGE.start + np.argmax(silhouettes)
sil_optimo = max(silhouettes)
print(f"  k óptimo por silhouette: k={k_optimo}, silhouette={sil_optimo:.3f}")

# K-Means final con k=2
km_final  = KMeans(n_clusters=k_optimo, random_state=SEED, n_init=10)
labels_km = km_final.fit_predict(X_km)
inercia_final = km_final.inertia_
print(f"  Inercia final (k={k_optimo}): {inercia_final:,.0f}")

# PCA
pca       = PCA(n_components=2, random_state=SEED)
X_pca     = pca.fit_transform(X_km)
var_pc1   = pca.explained_variance_ratio_[0] * 100
var_pc2   = pca.explained_variance_ratio_[1] * 100
print(f"  PCA: PC1={var_pc1:.1f} % | PC2={var_pc2:.1f} %")

# ── figA6: K-Means análisis ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Método del codo
axes[0].plot(list(K_RANGE), inercias, 'o-', color=LBLUE, linewidth=2, markersize=7)
axes[0].axvline(k_optimo, color=ORANGE, linestyle='--', linewidth=1.5,
                label=f'k={k_optimo} (óptimo)')
axes[0].set_xlabel('Número de clusters (k)')
axes[0].set_ylabel('Inercia (WCSS)')
axes[0].set_title('Método del codo', fontweight='bold')
axes[0].legend()

# Silhouette
axes[1].plot(list(K_RANGE), silhouettes, 's-', color=GREEN, linewidth=2, markersize=7)
axes[1].axvline(k_optimo, color=ORANGE, linestyle='--', linewidth=1.5,
                label=f'k={k_optimo}: sil={sil_optimo:.3f}')
axes[1].set_xlabel('Número de clusters (k)')
axes[1].set_ylabel('Silhouette score')
axes[1].set_title('Silhouette score por k', fontweight='bold')
axes[1].legend()

# Clusters en PCA
scatter = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_km,
                           cmap='Set1', alpha=0.4, s=10)
axes[2].set_xlabel(f'PC1 ({var_pc1:.1f} % varianza)')
axes[2].set_ylabel(f'PC2 ({var_pc2:.1f} % varianza)')
axes[2].set_title(f'Clusters K-Means (k={k_optimo}) en espacio PCA', fontweight='bold')
plt.colorbar(scatter, ax=axes[2], label='Cluster')

fig.suptitle('K-Means: método del codo, silhouette y visualización PCA',
             fontweight='bold', color=BLUE, fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figA6_kmeans_analisis.png'), dpi=150)
plt.close()
print("  → figA6_kmeans_analisis.png")

# ── figA7: PCA — income real vs clusters K-Means ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Coloreado por income real
for val, label, color in [(0, '≤50K', GREY), (1, '>50K', ORANGE)]:
    mask = y_km == val
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=color, label=label, alpha=0.35, s=8)
axes[0].set_xlabel(f'PC1 ({var_pc1:.1f} % varianza)')
axes[0].set_ylabel(f'PC2 ({var_pc2:.1f} % varianza)')
axes[0].set_title('Coloreado por ingreso real', fontweight='bold')
axes[0].legend(markerscale=2)

# Coloreado por asignación K-Means
for cluster in range(k_optimo):
    mask = labels_km == cluster
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=f'Cluster {cluster}', alpha=0.35, s=8)
axes[1].set_xlabel(f'PC1 ({var_pc1:.1f} % varianza)')
axes[1].set_ylabel(f'PC2 ({var_pc2:.1f} % varianza)')
axes[1].set_title(f'Coloreado por asignación K-Means (k={k_optimo})', fontweight='bold')
axes[1].legend(markerscale=2)

fig.suptitle(f'Reducción PCA a 2 componentes — varianza explicada total: '
             f'{var_pc1+var_pc2:.1f} %',
             fontweight='bold', color=BLUE, fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figA7_pca_clusters.png'), dpi=150)
plt.close()
print("  → figA7_pca_clusters.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. GUARDAR resultados_canonicos.json
# ═══════════════════════════════════════════════════════════════════════════════
canonicos = {
    "dataset": {
        "n_original": int(n_original) if 'n_original' in dir() else 48842,
        "n_final":    len(df),
        "duplicados": 52,
        "train_n":    len(X_train),
        "test_n":     len(X_test)
    },
    "target": {
        "leq50k_pct": round((y == 0).mean() * 100, 2),
        "gt50k_pct":  round((y == 1).mean() * 100, 2)
    },
    "modelos": {
        key: {
            "accuracy":    resultados[key]['accuracy'],
            "precision":   resultados[key]['precision'],
            "recall":      resultados[key]['recall'],
            "f1_macro":    resultados[key]['f1_macro'],
            "auc":         resultados[key]['auc'],
            "cv_f1_mean":  resultados[key]['cv_f1_mean'],
            "cv_f1_std":   resultados[key]['cv_f1_std'],
            "dpd_sex":     fairness[key]['dpd_sex'],
            "dpd_race":    fairness[key]['dpd_race'],
            "cumple_dpd":  bool(fairness[key]['dpd_sex'] <= 0.1 and
                                fairness[key]['dpd_race'] <= 0.1)
        }
        for key in KEYS_MODELOS
    },
    "dpd_interseccional_GB": dpd_inter,
    "tasas_interseccional_GB": {k: round(v, 3) for k, v in tasas_inter.items()},
    "kmeans": {
        "k_optimo":   k_optimo,
        "silhouette": round(sil_optimo, 3),
        "inercia":    round(inercia_final, 0)
    },
    "pca": {
        "PC1_pct": round(var_pc1, 1),
        "PC2_pct": round(var_pc2, 1),
        "total_varianza_pct": round(var_pc1 + var_pc2, 1)
    }
}

json_path = os.path.join(REP_DIR, 'resultados_canonicos.json')

def _np_encoder(o):
    """Convierte tipos numpy (int64, float64, etc.) a tipos nativos de Python."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'Tipo no serializable: {type(o)}')

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(canonicos, f, indent=2, ensure_ascii=False, default=_np_encoder)

# ── Resumen final ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CIFRAS CANÓNICAS")
print("=" * 60)
print(f"  Dataset: {canonicos['dataset']['n_original']:,} → {canonicos['dataset']['n_final']:,} "
      f"({canonicos['dataset']['duplicados']} dup. eliminados)")
print(f"  Train/Test: {canonicos['dataset']['train_n']:,} / {canonicos['dataset']['test_n']:,}")
print(f"  Target: {canonicos['target']['leq50k_pct']} % ≤50K / "
      f"{canonicos['target']['gt50k_pct']} % >50K")
print()
for key, nom in zip(KEYS_MODELOS, NOMBRES_MODELOS):
    m = canonicos['modelos'][key]
    print(f"  {key:2s} ({nom[:20]:20s}): "
          f"AUC={m['auc']:.3f} F1={m['f1_macro']:.3f} "
          f"DPD_sex={m['dpd_sex']:.3f} DPD_race={m['dpd_race']:.3f}")
print(f"\n  DPD interseccional GB: {canonicos['dpd_interseccional_GB']:.3f}")
print(f"  K-Means k={canonicos['kmeans']['k_optimo']}, "
      f"silhouette={canonicos['kmeans']['silhouette']:.3f}")
print(f"  PCA PC1={canonicos['pca']['PC1_pct']} % | PC2={canonicos['pca']['PC2_pct']} %")
print(f"\n  JSON guardado en: {json_path}")
print("\n[OK] Script 02 completado.")
