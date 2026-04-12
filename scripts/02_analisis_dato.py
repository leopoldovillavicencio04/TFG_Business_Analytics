# =============================================================================
# TFG - Business Analytics | Universidad Francisco de Vitoria
# Autor: Leopoldo Fernández de Villavicencio
# Título: Evaluación del uso ético de la IA en la selección de personal
#
# SCRIPT 02 - ANÁLISIS DEL DATO (Modelos Predictivos + Métricas de Equidad)
# =============================================================================
# IMPORTANTE: Ejecutar DESPUÉS de 01_ingenieria_dato.py
# Input:  data/processed/adult_clean.csv
# Output: figures/ + reports/resultados_modelos.txt
# =============================================================================

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Cambiar a 'TkAgg' en Spyder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# -----------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC  = os.path.join(BASE_DIR, 'data', 'processed', 'adult_clean.csv')
FIGS_DIR   = os.path.join(BASE_DIR, 'figures')
REP_DIR    = os.path.join(BASE_DIR, 'reports')

os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(REP_DIR,  exist_ok=True)

C1, C2, C3 = '#1f4e79', '#2e75b6', '#9dc3e6'
sns.set_theme(style='whitegrid', font_scale=1.1)

print("=" * 70)
print("  TFG - ANÁLISIS DEL DATO")
print("  Modelos Predictivos + Métricas de Equidad Algorítmica")
print("=" * 70)

# =============================================================================
# PASO 1: CARGA Y PREPARACIÓN
# =============================================================================
print("\n[1/5] CARGA Y PREPARACIÓN DEL DATASET")
print("-" * 40)

df = pd.read_csv(DATA_PROC)
print(f"  Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

# Seleccionar features para el modelo
FEATURES_CAT = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race', 'sex', 'native-country']
FEATURES_NUM = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                'capital-loss', 'hours-per-week']
TARGET = 'income_binary'

# Codificación Label Encoding para variables categóricas
df_model = df[FEATURES_CAT + FEATURES_NUM + [TARGET, 'sex', 'race']].copy()
le = LabelEncoder()
for col in FEATURES_CAT:
    df_model[col + '_enc'] = le.fit_transform(df_model[col].astype(str))

# Features finales para el modelo
FEATURE_COLS = [col + '_enc' for col in FEATURES_CAT] + FEATURES_NUM

X = df_model[FEATURE_COLS]
y = df_model[TARGET]

print(f"  Features: {len(FEATURE_COLS)} variables")
print(f"  Variable objetivo: income_binary (0=≤50K, 1=>50K)")
print(f"  Distribución target: {y.value_counts().to_dict()}")

# Train/Test Split (80/20 estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# Escalado para Regresión Logística
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# =============================================================================
# PASO 2: MODELOS PREDICTIVOS
# =============================================================================
print("\n[2/5] ENTRENAMIENTO DE MODELOS")
print("-" * 40)

# --- MODELO 1: Regresión Logística (Baseline) ---
print("  Entrenando Regresión Logística...")
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    C=1.0
)
lr_model.fit(X_train_sc, y_train)
y_pred_lr  = lr_model.predict(X_test_sc)
y_prob_lr  = lr_model.predict_proba(X_test_sc)[:, 1]
print("  ✓ Regresión Logística entrenada")

# --- MODELO 2: Random Forest (Avanzado) ---
print("  Entrenando Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf  = rf_model.predict(X_test)
y_prob_rf  = rf_model.predict_proba(X_test)[:, 1]
print("  ✓ Random Forest entrenado (200 árboles)")

# =============================================================================
# PASO 3: MÉTRICAS DE EVALUACIÓN
# =============================================================================
print("\n[3/5] MÉTRICAS DE EVALUACIÓN")
print("-" * 40)

def compute_metrics(y_true, y_pred, y_prob, model_name):
    """Calcula y muestra todas las métricas de evaluación."""
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, average='macro')
    rec   = recall_score(y_true, y_pred, average='macro')
    f1    = f1_score(y_true, y_pred, average='macro')
    auc   = roc_auc_score(y_true, y_prob)

    print(f"\n  [{model_name}]")
    print(f"  {'Accuracy':<20}: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  {'Precision (macro)':<20}: {prec:.4f}")
    print(f"  {'Recall (macro)':<20}: {rec:.4f}")
    print(f"  {'F1-Score (macro)':<20}: {f1:.4f}")
    print(f"  {'AUC-ROC':<20}: {auc:.4f}")
    print(f"\n  Informe de clasificación:")
    print(classification_report(y_true, y_pred, target_names=['≤50K', '>50K']))

    return {'model': model_name, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1_macro': f1, 'auc_roc': auc}

metrics_lr = compute_metrics(y_test, y_pred_lr, y_prob_lr, "Regresión Logística")
metrics_rf = compute_metrics(y_test, y_pred_rf, y_prob_rf, "Random Forest")

# Cross-validation (5-fold)
print("\n  Validación Cruzada (5-fold StratifiedKFold):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_lr = cross_val_score(
    LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    X_train_sc, y_train, cv=cv, scoring='f1_macro'
)
cv_rf = cross_val_score(
    RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    X_train, y_train, cv=cv, scoring='f1_macro'
)
print(f"  LR  F1 CV: {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")
print(f"  RF  F1 CV: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

# =============================================================================
# PASO 4: VISUALIZACIONES
# =============================================================================
print("\n[4/5] GENERANDO VISUALIZACIONES")
print("-" * 40)

# --- Figura 1: Matrices de Confusión ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, y_pred, title, color in zip(
    axes,
    [y_pred_lr, y_pred_rf],
    ['Regresión Logística', 'Random Forest'],
    [C1, C2]
):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['≤50K', '>50K'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Matriz de Confusión\n{title}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicción', fontsize=10)
    ax.set_ylabel('Valor Real', fontsize=10)

plt.suptitle('Matrices de Confusión — Comparativa de Modelos',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ fig_confusion_matrix.png")

# --- Figura 2: Curvas ROC ---
fig, ax = plt.subplots(figsize=(8, 6))
for y_prob, label, color in [
    (y_prob_lr, f"Regresión Logística (AUC={metrics_lr['auc_roc']:.3f})", C1),
    (y_prob_rf, f"Random Forest      (AUC={metrics_rf['auc_roc']:.3f})", C2),
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, linewidth=2.5, color=color, label=label)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6, label='Clasificador Aleatorio')
ax.fill_between(fpr, tpr, alpha=0.05, color=C2)
ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
ax.set_title('Curvas ROC — Comparativa de Modelos\n(UCI Adult Census Income)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_roc_curve.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ fig_roc_curve.png")

# --- Figura 3: Feature Importance (Random Forest) ---
feat_imp = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS)
feat_imp_sorted = feat_imp.sort_values(ascending=False).head(15)

# Nombres legibles
name_map = {col + '_enc': col.replace('-', ' ').title() for col in FEATURES_CAT}
for col in FEATURES_NUM:
    name_map[col] = col.replace('-', ' ').title()
labels = [name_map.get(c, c) for c in feat_imp_sorted.index]

fig, ax = plt.subplots(figsize=(10, 7))
colors_fi = [C1 if i < 5 else C2 if i < 10 else C3 for i in range(len(feat_imp_sorted))]
bars = ax.barh(labels[::-1], feat_imp_sorted.values[::-1],
               color=colors_fi[::-1], edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, feat_imp_sorted.values[::-1]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
ax.set_xlabel('Importancia (Gini)', fontsize=11)
ax.set_title('Top 15 Variables Más Importantes — Random Forest\n(Impacto en la predicción de ingresos >50K)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ fig_feature_importance.png")

# --- Figura 4: Comparativa de métricas ---
metrics_df = pd.DataFrame([metrics_lr, metrics_rf]).set_index('model')
metrics_plot = metrics_df[['accuracy', 'precision', 'recall', 'f1_macro', 'auc_roc']]

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(metrics_plot.columns))
width = 0.35
bars1 = ax.bar(x - width/2, metrics_plot.iloc[0].values, width, label='Regresión Logística',
                color=C1, edgecolor='white', linewidth=1.2)
bars2 = ax.bar(x + width/2, metrics_plot.iloc[1].values, width, label='Random Forest',
                color=C2, edgecolor='white', linewidth=1.2)

for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(['Accuracy', 'Precision\n(macro)', 'Recall\n(macro)', 'F1-Score\n(macro)', 'AUC-ROC'],
                    fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Valor de la Métrica', fontsize=11)
ax.set_title('Comparativa de Métricas: Regresión Logística vs. Random Forest',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, linewidth=1.2, label='Umbral 0.80')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_metricas_comparativa.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ fig_metricas_comparativa.png")

# =============================================================================
# PASO 5: ANÁLISIS DE EQUIDAD ALGORÍTMICA (FAIRNESS)
# =============================================================================
print("\n[5/5] ANÁLISIS DE EQUIDAD ALGORÍTMICA (FAIRNESS)")
print("-" * 40)

# Añadir predicciones al test set con atributos sensibles
df_test = df_model.iloc[y_test.index].copy()
df_test['y_true'] = y_test.values
df_test['y_pred_lr'] = y_pred_lr
df_test['y_pred_rf'] = y_pred_rf

def fairness_by_group(df_eval, group_col, pred_col):
    """Calcula métricas de equidad por grupo."""
    results = []
    for group in df_eval[group_col].unique():
        mask = df_eval[group_col] == group
        sub = df_eval[mask]
        if len(sub) < 30:
            continue
        acc  = accuracy_score(sub['y_true'], sub[pred_col])
        f1   = f1_score(sub['y_true'], sub[pred_col], average='macro', zero_division=0)
        dr   = sub[pred_col].mean()  # Demographic Rate (% predicted >50K)
        n    = len(sub)
        results.append({'Grupo': group, 'N': n, 'Accuracy': acc, 'F1-macro': f1,
                        'Tasa Pred. >50K': dr})
    return pd.DataFrame(results).sort_values('Tasa Pred. >50K', ascending=False)

print("\n  Equidad por SEXO:")
sex_fair_lr = fairness_by_group(df_test, 'sex', 'y_pred_lr')
print(sex_fair_lr.to_string(index=False))

print("\n  Equidad por RAZA:")
race_fair_rf = fairness_by_group(df_test, 'race', 'y_pred_rf')
print(race_fair_rf.to_string(index=False))

# Figura 5: Fairness Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, fair_df, model_name, group_col in zip(
    axes,
    [fairness_by_group(df_test, 'sex', 'y_pred_rf'),
     fairness_by_group(df_test, 'race', 'y_pred_rf')],
    ['Random Forest — Sexo', 'Random Forest — Raza'],
    ['sex', 'race']
):
    colors_f = [C1 if v > fair_df['Tasa Pred. >50K'].mean() else C3
                for v in fair_df['Tasa Pred. >50K']]
    bars = ax.bar(fair_df['Grupo'], fair_df['Tasa Pred. >50K'] * 100,
                  color=colors_f, edgecolor='white')
    for bar, val in zip(bars, fair_df['Tasa Pred. >50K'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)
    mean_rate = fair_df['Tasa Pred. >50K'].mean() * 100
    ax.axhline(y=mean_rate, color='red', linestyle='--', linewidth=1.5,
               label=f'Media: {mean_rate:.1f}%')
    ax.set_title(f'Tasa de Predicción >50K\n{model_name}', fontsize=11, fontweight='bold')
    ax.set_ylabel('% Predichos como >50K')
    ax.set_xticklabels(fair_df['Grupo'], rotation=20, ha='right', fontsize=9)
    ax.legend(fontsize=9)

plt.suptitle('Análisis de Equidad Algorítmica (Demographic Parity)\nRandom Forest — UCI Adult Census Income',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'fig_fairness.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ fig_fairness.png")

# =============================================================================
# GUARDAR REPORTE
# =============================================================================
report_path = os.path.join(REP_DIR, 'resultados_modelos.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("  RESULTADOS — ANÁLISIS DEL DATO\n")
    f.write("  TFG Business Analytics — UFV\n")
    f.write("=" * 70 + "\n\n")
    f.write("MÉTRICAS FINALES:\n")
    f.write(f"{'Métrica':<25} {'Reg. Logística':>18} {'Random Forest':>18}\n")
    f.write("-" * 62 + "\n")
    for key, label in [('accuracy','Accuracy'), ('precision','Precision (macro)'),
                        ('recall','Recall (macro)'), ('f1_macro','F1-Score (macro)'),
                        ('auc_roc','AUC-ROC')]:
        f.write(f"{label:<25} {metrics_lr[key]:>18.4f} {metrics_rf[key]:>18.4f}\n")
    f.write("\nCross-Validation F1-macro (5-fold):\n")
    f.write(f"  Reg. Logística: {cv_lr.mean():.4f} ± {cv_lr.std():.4f}\n")
    f.write(f"  Random Forest:  {cv_rf.mean():.4f} ± {cv_rf.std():.4f}\n")

print(f"\n{'='*70}")
print(f"  ✓ Reporte guardado: {report_path}")
print(f"\n  MÉTRICAS FINALES:")
print(f"  {'Métrica':<25} {'Reg. Logística':>18} {'Random Forest':>18}")
print(f"  {'-'*62}")
for key, label in [('accuracy','Accuracy'), ('precision','Precision (macro)'),
                    ('recall','Recall (macro)'), ('f1_macro','F1-Score (macro)'),
                    ('auc_roc','AUC-ROC')]:
    print(f"  {label:<25} {metrics_lr[key]:>18.4f} {metrics_rf[key]:>18.4f}")
print(f"\n✅ Análisis del dato completado.")
