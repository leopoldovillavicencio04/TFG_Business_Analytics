# =============================================================================
# TFG - Business Analytics | Universidad Francisco de Vitoria
# Autor: Leopoldo Fernández de Villavicencio
# Título: Evaluación del uso ético de la IA en la selección de personal:
#         un análisis del equilibrio entre equidad, objetividad y eficiencia
#
# SCRIPT 02 — ANÁLISIS DEL DATO
# Modelos Supervisados: Reg. Logística + Árbol de Decisión + Random Forest + Gradient Boosting
# Modelos No Supervisados: K-Means + PCA
# Métricas de Equidad Algorítmica (EU AI Act, 2024)
# =============================================================================
# PREREQUISITO: Ejecutar ANTES 01_ingenieria_dato.py
# Input:  data/processed/adult_clean.csv
# Output: figures/fig_*.png  +  reports/resultados_modelos.txt
# =============================================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster           import KMeans
from sklearn.decomposition     import PCA
from sklearn.metrics           import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, silhouette_score
)

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN GLOBAL
# -----------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(BASE_DIR, 'data', 'processed', 'adult_clean.csv')
FIGS_DIR  = os.path.join(BASE_DIR, 'figures')
REP_DIR   = os.path.join(BASE_DIR, 'reports')

os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(REP_DIR,  exist_ok=True)

# Paleta de colores corporativa UFV
C1, C2, C3, C4 = '#1f4e79', '#2e75b6', '#9dc3e6', '#e67e22'
C5 = '#27ae60'   # verde para Gradient Boosting
C6 = '#8e44ad'   # morado para Árbol de Decisión
sns.set_theme(style='whitegrid', font_scale=1.1)

# Variables del modelo
FEATURES_CAT = ['workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'native-country']
FEATURES_NUM = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                'capital-loss', 'hours-per-week']
TARGET = 'income_binary'

LABEL_MAP = {col + '_enc': col.replace('-', ' ').title() for col in FEATURES_CAT}
LABEL_MAP.update({col: col.replace('-', ' ').title() for col in FEATURES_NUM})

print("=" * 70)
print("  TFG — ANÁLISIS DEL DATO")
print("  Modelos Supervisados + No Supervisados + Equidad Algorítmica")
print("  Universidad Francisco de Vitoria · Business Analytics 2024-2025")
print("=" * 70)


# =============================================================================
# PASO 1: CARGA Y PREPARACIÓN DEL DATASET
# =============================================================================
def cargar_y_preparar(path: str):
    """
    Carga el dataset limpio generado por el pipeline ETL y prepara las
    features para el modelado. Se aplica Label Encoding a las variables
    categóricas y estandarización para los modelos que lo requieren.
    """
    print("\n[1/7] CARGA Y PREPARACIÓN")
    print("-" * 40)

    df = pd.read_csv(path)
    print(f"  Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    df_model = df[FEATURES_CAT + FEATURES_NUM + [TARGET, 'sex', 'race']].copy()
    le = LabelEncoder()
    for col in FEATURES_CAT:
        df_model[col + '_enc'] = le.fit_transform(df_model[col].astype(str))

    feature_cols = [col + '_enc' for col in FEATURES_CAT] + FEATURES_NUM

    X = df_model[feature_cols]
    y = df_model[TARGET]

    dist = y.value_counts(normalize=True)
    print(f"  Distribución target: ≤50K={dist[0]*100:.1f}% | >50K={dist[1]*100:.1f}%")
    print(f"  Dataset desbalanceado — se aplicará class_weight='balanced'")

    # Split estratificado 80/20 con semilla fija para reproducibilidad
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {X_train.shape[0]:,} registros | Test: {X_test.shape[0]:,} registros")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return (X, y, df_model, X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc, scaler, feature_cols)


# =============================================================================
# PASO 2: MODELOS SUPERVISADOS
# =============================================================================
def entrenar_supervisados(X_train, X_test, y_train, y_test,
                           X_train_sc, X_test_sc):
    """
    Entrena cuatro modelos de clasificación supervisada con complejidad creciente:
      1. Regresión Logística: modelo lineal interpretable, usado como baseline.
      2. Árbol de Decisión: modelo no lineal simple; precursor lógico del RF.
      3. Random Forest: ensemble de árboles (bagging), robusto a overfitting.
      4. Gradient Boosting: boosting secuencial, generalmente el más preciso.

    Todos usan class_weight='balanced' o sample_weight equivalente para
    compensar el desbalanceo del dataset (≈75.7% ≤50K vs. 24.3% >50K).
    """
    print("\n[2/7] MODELOS SUPERVISADOS (4 modelos)")
    print("-" * 40)

    # Modelo 1: Regresión Logística — baseline lineal interpretable
    print("  [1/4] Entrenando Regresión Logística (baseline)...")
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=1.0,
        solver='lbfgs',
        random_state=42
    )
    lr_model.fit(X_train_sc, y_train)
    y_pred_lr = lr_model.predict(X_test_sc)
    y_prob_lr = lr_model.predict_proba(X_test_sc)[:, 1]
    print(f"       Positivos predichos: {y_pred_lr.sum():,} ({y_pred_lr.mean()*100:.1f}%)")

    # Modelo 2: Árbol de Decisión — modelo no lineal simple
    # max_depth=10 para evitar sobreajuste total; min_samples_leaf=10 para
    # que las hojas tengan representatividad estadística suficiente.
    print("  [2/4] Entrenando Árbol de Decisión (max_depth=10)...")
    dt_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
    print(f"       Positivos predichos: {y_pred_dt.sum():,} ({y_pred_dt.mean()*100:.1f}%)")

    # Modelo 3: Random Forest — ensemble de árboles (bagging)
    print("  [3/4] Entrenando Random Forest (200 árboles)...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    print(f"       Positivos predichos: {y_pred_rf.sum():,} ({y_pred_rf.mean()*100:.1f}%)")

    # Modelo 4: Gradient Boosting — boosting secuencial estocástico
    # GradientBoostingClassifier no acepta class_weight directamente:
    # se calculan sample_weight inversamente proporcionales a la frecuencia de clase.
    print("  [4/4] Entrenando Gradient Boosting (150 estimadores, subsample=0.8)...")
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=8,
        subsample=0.8,
        random_state=42
    )
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    ratio = n_neg / n_pos
    sample_weights = np.where(y_train == 1, ratio, 1.0)
    gb_model.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred_gb = gb_model.predict(X_test)
    y_prob_gb = gb_model.predict_proba(X_test)[:, 1]
    print(f"       Positivos predichos: {y_pred_gb.sum():,} ({y_pred_gb.mean()*100:.1f}%)")

    return (lr_model, dt_model, rf_model, gb_model,
            y_pred_lr, y_pred_dt, y_pred_rf, y_pred_gb,
            y_prob_lr, y_prob_dt, y_prob_rf, y_prob_gb)


# =============================================================================
# PASO 3: MODELOS NO SUPERVISADOS (K-Means + PCA)
# =============================================================================
def analisis_no_supervisado(X_sc: np.ndarray, y: pd.Series, feature_cols: list):
    """
    Análisis no supervisado sobre el dataset completo:
      - K-Means: agrupa registros sin usar la etiqueta de ingresos.
        Se evalúa el número óptimo de clusters por Elbow Method y Silhouette.
      - PCA: reduce la dimensionalidad a 2 componentes para visualizar
        la estructura del espacio de features.
    """
    print("\n[3/7] ANÁLISIS NO SUPERVISADO (K-Means + PCA)")
    print("-" * 40)

    idx_sample = np.random.RandomState(42).choice(len(X_sc), 5000, replace=False)
    X_sample   = X_sc[idx_sample]
    y_sample   = y.values[idx_sample]

    inertias    = []
    silhouettes = []
    K_range     = range(2, 10)

    print("  Evaluando número óptimo de clusters (k=2 hasta k=9)...")
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_sample)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_sample, labels))

    best_k = list(K_range)[np.argmax(silhouettes)]
    print(f"  Mejor k por Silhouette Score: k={best_k} (score={max(silhouettes):.4f})")

    print(f"  Entrenando K-Means final (k={best_k}) sobre dataset completo...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20, max_iter=300)
    kmeans.fit(X_sc)
    cluster_labels = kmeans.labels_

    df_cluster = pd.DataFrame({'cluster': cluster_labels, 'income': y.values})
    cluster_income = df_cluster.groupby('cluster')['income'].agg(['mean', 'count'])
    cluster_income.columns = ['Tasa >50K', 'N']
    print("\n  Alineación Clusters ↔ Ingresos >50K:")
    print(cluster_income.to_string())

    sil_full = silhouette_score(X_sc[:5000], cluster_labels[:5000])
    inertia  = kmeans.inertia_

    metrics_kmeans = {
        'best_k': best_k,
        'silhouette': sil_full,
        'inertia': inertia,
        'cluster_income': cluster_income
    }

    print("\n  Aplicando PCA (2 componentes principales)...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    var_exp = pca.explained_variance_ratio_
    print(f"  Varianza explicada: PC1={var_exp[0]*100:.1f}% | PC2={var_exp[1]*100:.1f}% | Total={sum(var_exp)*100:.1f}%")

    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'income': y.values,
        'cluster': cluster_labels
    })

    return kmeans, pca, df_pca, metrics_kmeans, K_range, inertias, silhouettes


# =============================================================================
# PASO 4: MÉTRICAS DE EVALUACIÓN
# =============================================================================
def calcular_metricas(y_true, y_pred, y_prob, nombre: str) -> dict:
    """
    Calcula las 5 métricas estándar para clasificación binaria.
    Se usa average='macro' para dar el mismo peso a ambas clases
    independientemente de su frecuencia en el dataset.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)

    print(f"\n  [{nombre}]")
    print(f"  {'Accuracy':<22}: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  {'Precision (macro)':<22}: {prec:.4f}")
    print(f"  {'Recall (macro)':<22}: {rec:.4f}")
    print(f"  {'F1-Score (macro)':<22}: {f1:.4f}")
    print(f"  {'AUC-ROC':<22}: {auc:.4f}")
    print("\n  Informe de clasificación completo:")
    print(classification_report(y_true, y_pred,
                                 target_names=['≤50K', '>50K'],
                                 zero_division=0))

    return {'model': nombre, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1_macro': f1, 'auc_roc': auc}


def validacion_cruzada(X_train, y_train, X_train_sc):
    """
    Validación cruzada 5-fold estratificada para estimar la capacidad de
    generalización de cada modelo. Se usa F1-macro como métrica principal
    para evaluar el rendimiento en ambas clases por igual.
    """
    print("\n  Validación Cruzada (5-fold StratifiedKFold — F1 macro):")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_lr = cross_val_score(
        LogisticRegression(max_iter=1000, class_weight='balanced',
                           random_state=42, C=1.0),
        X_train_sc, y_train, cv=cv, scoring='f1_macro', n_jobs=-1
    )
    cv_dt = cross_val_score(
        DecisionTreeClassifier(max_depth=10, class_weight='balanced',
                               random_state=42),
        X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1
    )
    cv_rf = cross_val_score(
        RandomForestClassifier(n_estimators=100, class_weight='balanced',
                               random_state=42, n_jobs=-1),
        X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1
    )
    cv_gb = cross_val_score(
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                   max_depth=5, random_state=42),
        X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1
    )
    print(f"  Reg. Logística  F1 CV: {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")
    print(f"  Árbol Decisión  F1 CV: {cv_dt.mean():.4f} ± {cv_dt.std():.4f}")
    print(f"  Random Forest   F1 CV: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
    print(f"  Grad. Boosting  F1 CV: {cv_gb.mean():.4f} ± {cv_gb.std():.4f}")
    return cv_lr, cv_dt, cv_rf, cv_gb


# =============================================================================
# PASO 5: VISUALIZACIONES
# =============================================================================
def _save_fig(filename: str):
    path = os.path.join(FIGS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {filename}")


def generar_visualizaciones(y_test,
                             y_pred_lr, y_pred_dt, y_pred_rf, y_pred_gb,
                             y_prob_lr,  y_prob_dt,  y_prob_rf,  y_prob_gb,
                             metrics_lr, metrics_dt, metrics_rf, metrics_gb,
                             dt_model, rf_model, gb_model, feature_cols,
                             df_pca, K_range, inertias, silhouettes,
                             metrics_kmeans):
    """
    Genera y guarda todas las figuras del análisis (4 modelos).
    """
    print("\n[5/7] GENERANDO VISUALIZACIONES")
    print("-" * 40)

    palette = [C1, C2, C3, C4, C5, C6, '#e74c3c']

    # ---- figA1: Matrices de Confusión (4 modelos, layout 2×2) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, y_pred, title in zip(
        axes.flat,
        [y_pred_lr, y_pred_dt, y_pred_rf, y_pred_gb],
        ['Regresión Logística\n(Baseline lineal)',
         'Árbol de Decisión\n(max_depth=10)',
         'Random Forest\n(200 árboles)',
         'Gradient Boosting\n(150 estimadores)']
    ):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=['≤50K', '>50K'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f'Matriz de Confusión\n{title}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicción', fontsize=9)
        ax.set_ylabel('Valor Real', fontsize=9)

    plt.suptitle('Matrices de Confusión — Comparativa de Cuatro Modelos Supervisados\n'
                 'UCI Adult Census Income',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_fig('figA1_confusion_matrix.png')

    # ---- figA2: Curvas ROC (4 modelos) ----
    fig, ax = plt.subplots(figsize=(9, 7))
    for y_prob, label, color in [
        (y_prob_lr, f"Regresión Logística  (AUC = {metrics_lr['auc_roc']:.3f})", C1),
        (y_prob_dt, f"Árbol de Decisión    (AUC = {metrics_dt['auc_roc']:.3f})", C6),
        (y_prob_rf, f"Random Forest        (AUC = {metrics_rf['auc_roc']:.3f})", C2),
        (y_prob_gb, f"Gradient Boosting    (AUC = {metrics_gb['auc_roc']:.3f})", C5),
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, linewidth=2.5, color=color, label=label)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Azar (AUC = 0.500)')
    fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)
    ax.fill_between(fpr_gb, tpr_gb, alpha=0.07, color=C5)
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    ax.set_title('Curvas ROC — Comparativa de Cuatro Modelos Supervisados\n(UCI Adult Census Income)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    _save_fig('figA2_roc_curve.png')

    # ---- figA3: Feature Importance (DT, RF, GB) — layout 1×3 ----
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    for ax, model, title, color in [
        (axes[0], dt_model, 'Árbol de Decisión', C6),
        (axes[1], rf_model, 'Random Forest',     C2),
        (axes[2], gb_model, 'Gradient Boosting', C5),
    ]:
        feat_imp = pd.Series(model.feature_importances_, index=feature_cols)
        top12    = feat_imp.sort_values(ascending=False).head(12)
        labels   = [LABEL_MAP.get(c, c) for c in top12.index]
        colors_fi = [C1 if i < 4 else color if i < 8 else C3 for i in range(len(top12))]
        bars = ax.barh(labels[::-1], top12.values[::-1],
                       color=colors_fi[::-1], edgecolor='white', linewidth=1.2)
        for bar, val in zip(bars, top12.values[::-1]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9)
        ax.set_xlabel('Importancia relativa', fontsize=11)
        ax.set_title(f'Top 12 Variables — {title}', fontsize=11, fontweight='bold')

    plt.suptitle('Importancia de Variables: Árbol de Decisión, Random Forest y Gradient Boosting\n'
                 '(Predicción de ingresos >50K — UCI Adult Census Income)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_fig('figA3_feature_importance.png')

    # ---- figA4: Comparativa de Métricas (4 modelos) ----
    metrics_names = ['Accuracy', 'Precision\n(macro)', 'Recall\n(macro)',
                     'F1-Score\n(macro)', 'AUC-ROC']
    vals_lr = [metrics_lr[k] for k in ['accuracy', 'precision', 'recall', 'f1_macro', 'auc_roc']]
    vals_dt = [metrics_dt[k] for k in ['accuracy', 'precision', 'recall', 'f1_macro', 'auc_roc']]
    vals_rf = [metrics_rf[k] for k in ['accuracy', 'precision', 'recall', 'f1_macro', 'auc_roc']]
    vals_gb = [metrics_gb[k] for k in ['accuracy', 'precision', 'recall', 'f1_macro', 'auc_roc']]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(metrics_names))
    w = 0.19
    b1 = ax.bar(x - 1.5*w, vals_lr, w, label='Regresión Logística',  color=C1, edgecolor='white')
    b2 = ax.bar(x - 0.5*w, vals_dt, w, label='Árbol de Decisión',    color=C6, edgecolor='white')
    b3 = ax.bar(x + 0.5*w, vals_rf, w, label='Random Forest',        color=C2, edgecolor='white')
    b4 = ax.bar(x + 1.5*w, vals_gb, w, label='Gradient Boosting',    color=C5, edgecolor='white')
    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f'{bar.get_height():.3f}', ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel('Valor de la Métrica', fontsize=11)
    ax.set_title('Comparativa de Métricas — Cuatro Modelos Supervisados\n'
                 '(UCI Adult Census Income — Conjunto de Test)',
                 fontsize=13, fontweight='bold')
    ax.axhline(0.8, color='red', ls='--', alpha=0.5, lw=1.5, label='Umbral referencia 0.80')
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_fig('figA4_metricas_comparativa.png')

    # ---- figA6: K-Means — Elbow + Silhouette + Clusters en PCA ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(list(K_range), inertias, 'o-', color=C1, linewidth=2, markersize=7)
    ax.set_xlabel('Número de Clusters (k)', fontsize=11)
    ax.set_ylabel('Inercia (Within-Cluster SSE)', fontsize=11)
    ax.set_title('Método del Codo (Elbow Method)\nSelección óptima de k',
                 fontsize=11, fontweight='bold')
    ax.axvline(x=metrics_kmeans['best_k'], color='red', ls='--', alpha=0.7,
               label=f"k óptimo = {metrics_kmeans['best_k']}")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(list(K_range), silhouettes, 's-', color=C2, linewidth=2, markersize=7)
    ax.set_xlabel('Número de Clusters (k)', fontsize=11)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_title('Silhouette Score por k\n(Mayor = Clusters más cohesionados)',
                 fontsize=11, fontweight='bold')
    ax.axvline(x=metrics_kmeans['best_k'], color='red', ls='--', alpha=0.7,
               label=f"k óptimo = {metrics_kmeans['best_k']}")
    ax.legend(fontsize=9)

    ax = axes[2]
    sample_idx = np.random.RandomState(42).choice(len(df_pca), 3000, replace=False)
    df_s = df_pca.iloc[sample_idx]
    for c_id in sorted(df_s['cluster'].unique()):
        mask = df_s['cluster'] == c_id
        ax.scatter(df_s.loc[mask, 'PC1'], df_s.loc[mask, 'PC2'],
                   alpha=0.4, s=15, color=palette[c_id % len(palette)],
                   label=f'Cluster {c_id}')
    ax.set_xlabel('PC1', fontsize=11); ax.set_ylabel('PC2', fontsize=11)
    ax.set_title(f'K-Means (k={metrics_kmeans["best_k"]}) — Espacio PCA\n'
                 f'Silhouette = {metrics_kmeans["silhouette"]:.4f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, markerscale=2)

    plt.suptitle('Análisis No Supervisado: K-Means Clustering\nUCI Adult Census Income',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_fig('figA6_kmeans_analisis.png')

    # ---- figA7: PCA — Separabilidad por nivel de ingresos ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sample_idx = np.random.RandomState(42).choice(len(df_pca), 5000, replace=False)
    df_s = df_pca.iloc[sample_idx]

    ax = axes[0]
    for label, color, name in [(0, C3, '≤50K'), (1, C1, '>50K')]:
        m = df_s['income'] == label
        ax.scatter(df_s.loc[m, 'PC1'], df_s.loc[m, 'PC2'],
                   alpha=0.35, s=12, color=color, label=name)
    ax.set_xlabel('PC1', fontsize=11); ax.set_ylabel('PC2', fontsize=11)
    ax.set_title('PCA — Separabilidad Real\nColoreado por Ingresos (≤50K / >50K)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, markerscale=2)

    ax = axes[1]
    for c_id in sorted(df_s['cluster'].unique()):
        m = df_s['cluster'] == c_id
        ax.scatter(df_s.loc[m, 'PC1'], df_s.loc[m, 'PC2'],
                   alpha=0.35, s=12, color=palette[c_id % len(palette)],
                   label=f'Cluster {c_id}')
    ax.set_xlabel('PC1', fontsize=11); ax.set_ylabel('PC2', fontsize=11)
    ax.set_title(f'PCA — Asignación K-Means\nColoreado por Cluster (k={metrics_kmeans["best_k"]})',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, markerscale=2)

    plt.suptitle('Reducción de Dimensionalidad PCA (2 Componentes Principales)\n'
                 'Comparación: Etiqueta Real vs. Clusters K-Means',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save_fig('figA7_pca_clusters.png')

    print("  ✓ Todas las figuras generadas correctamente")


# =============================================================================
# PASO 6: ANÁLISIS DE EQUIDAD ALGORÍTMICA (EU AI Act, 2024)
# =============================================================================
def analisis_equidad(df_model, y_test, y_pred_lr, y_pred_dt, y_pred_rf, y_pred_gb):
    """
    Análisis de Equidad Algorítmica conforme al EU AI Act (2024).

    Los sistemas de IA para selección de personal se clasifican como
    'alto riesgo' (Anexo III, punto 1a). Se evalúa la Paridad Demográfica
    (Demographic Parity Difference, DPD) por sexo y raza para los tres modelos.

    Un DPD < 0.1 se considera aceptable en la literatura de fairness
    (Barocas, Hardt & Narayanan, 2023). Valores superiores indican
    sesgo sistemático hacia grupos protegidos.
    """
    print("\n[6/7] EQUIDAD ALGORÍTMICA (EU AI Act)")
    print("-" * 40)

    df_test = df_model.iloc[y_test.index].copy()
    df_test['y_true']    = y_test.values
    df_test['y_pred_lr'] = y_pred_lr
    df_test['y_pred_dt'] = y_pred_dt
    df_test['y_pred_rf'] = y_pred_rf
    df_test['y_pred_gb'] = y_pred_gb

    def fairness_by_group(df_eval, group_col, pred_col):
        rows = []
        for group in sorted(df_eval[group_col].unique()):
            mask = df_eval[group_col] == group
            sub  = df_eval[mask]
            if len(sub) < 30:
                continue
            rows.append({
                'Grupo': group,
                'N': len(sub),
                'Accuracy': accuracy_score(sub['y_true'], sub[pred_col]),
                'F1-macro': f1_score(sub['y_true'], sub[pred_col], average='macro', zero_division=0),
                'Tasa Pred >50K': sub[pred_col].mean()
            })
        return pd.DataFrame(rows).sort_values('Tasa Pred >50K', ascending=False)

    models_info = [
        ('y_pred_lr', 'Regresión Logística'),
        ('y_pred_dt', 'Árbol de Decisión'),
        ('y_pred_rf', 'Random Forest'),
        ('y_pred_gb', 'Gradient Boosting'),
    ]

    results_fairness = {}
    for pred_col, model_name in models_info:
        print(f"\n  --- {model_name} ---")
        sex_df  = fairness_by_group(df_test, 'sex', pred_col)
        race_df = fairness_by_group(df_test, 'race', pred_col)
        dpd_sex  = sex_df['Tasa Pred >50K'].max() - sex_df['Tasa Pred >50K'].min()
        dpd_race = race_df['Tasa Pred >50K'].max() - race_df['Tasa Pred >50K'].min()
        print(f"  DPD Sexo: {dpd_sex:.4f} | DPD Raza: {dpd_race:.4f}")
        status_s = "✓ PASA (<0.10)" if dpd_sex < 0.10 else "✗ FALLA (≥0.10)"
        status_r = "✓ PASA (<0.10)" if dpd_race < 0.10 else "✗ FALLA (≥0.10)"
        print(f"  Sexo: {status_s} | Raza: {status_r}")
        results_fairness[model_name] = {
            'sex': sex_df, 'race': race_df,
            'dpd_sex': dpd_sex, 'dpd_race': dpd_race
        }

    # Figura de Equidad (figA5) — 4 modelos × 2 atributos
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))

    for row_idx, (pred_col, model_name) in enumerate(models_info):
        fair_sex  = results_fairness[model_name]['sex']
        fair_race = results_fairness[model_name]['race']
        for col_idx, (fair_df, attr) in enumerate([(fair_sex, 'Sexo'), (fair_race, 'Raza')]):
            ax = axes[row_idx][col_idx]
            mean_rate = fair_df['Tasa Pred >50K'].mean()
            colors_f  = [C1 if v >= mean_rate else C3 for v in fair_df['Tasa Pred >50K']]
            bars = ax.bar(fair_df['Grupo'], fair_df['Tasa Pred >50K'] * 100,
                          color=colors_f, edgecolor='white', linewidth=1.2)
            for bar, val in zip(bars, fair_df['Tasa Pred >50K']):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                        f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)
            ax.axhline(mean_rate * 100, color='red', ls='--', lw=1.5,
                       label=f'Media: {mean_rate*100:.1f}%')
            dpd = results_fairness[model_name][f'dpd_{attr.lower()[:3].replace("sex","sex").replace("raz","race")[:4]}']
            dpd_key = 'dpd_sex' if attr == 'Sexo' else 'dpd_race'
            dpd_val = results_fairness[model_name][dpd_key]
            ax.set_title(f'{model_name} — por {attr}\n(DPD = {dpd_val:.4f})',
                         fontsize=10, fontweight='bold')
            ax.set_ylabel('% Predichos como >50K')
            ax.set_xticklabels(fair_df['Grupo'], rotation=20, ha='right', fontsize=8)
            ax.legend(fontsize=8)

    plt.suptitle('Análisis de Equidad Algorítmica — Paridad Demográfica (DPD)\n'
                 'EU AI Act (2024): Sistemas IA Alto Riesgo en Selección de Personal',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_fig('figA5_fairness.png')

    return df_test, results_fairness


# =============================================================================
# PASO 7: REPORTE FINAL
# =============================================================================
def guardar_reporte(metrics_lr, metrics_dt, metrics_rf, metrics_gb,
                    cv_lr, cv_dt, cv_rf, cv_gb, metrics_kmeans, results_fairness):
    """
    Guarda un reporte completo en texto con todas las métricas,
    resultados de cross-validation, clustering y análisis de equidad.
    """
    path = os.path.join(REP_DIR, 'resultados_modelos.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 75 + "\n")
        f.write("  RESULTADOS — ANÁLISIS DEL DATO\n")
        f.write("  TFG Business Analytics — UFV 2024-2025\n")
        f.write("  Autor: Leopoldo Fernández de Villavicencio\n")
        f.write("=" * 75 + "\n\n")

        f.write("MODELOS SUPERVISADOS (4 modelos)\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Métrica':<25} {'Reg. Logística':>16} {'Árbol Dec.':>14} {'Random Forest':>14} {'Grad. Boosting':>16}\n")
        f.write("-" * 90 + "\n")
        for k, label in [('accuracy','Accuracy'), ('precision','Precision (macro)'),
                          ('recall','Recall (macro)'), ('f1_macro','F1-Score (macro)'),
                          ('auc_roc','AUC-ROC')]:
            f.write(f"{label:<25} {metrics_lr[k]:>16.4f} {metrics_dt[k]:>14.4f} {metrics_rf[k]:>14.4f} {metrics_gb[k]:>16.4f}\n")

        f.write(f"\nValidación Cruzada 5-fold (F1-macro):\n")
        f.write(f"  Reg. Logística:  {cv_lr.mean():.4f} ± {cv_lr.std():.4f}\n")
        f.write(f"  Árbol Decisión:  {cv_dt.mean():.4f} ± {cv_dt.std():.4f}\n")
        f.write(f"  Random Forest:   {cv_rf.mean():.4f} ± {cv_rf.std():.4f}\n")
        f.write(f"  Grad. Boosting:  {cv_gb.mean():.4f} ± {cv_gb.std():.4f}\n")

        f.write("\nMODELO NO SUPERVISADO — K-Means\n")
        f.write("-" * 75 + "\n")
        f.write(f"  k óptimo:              {metrics_kmeans['best_k']}\n")
        f.write(f"  Silhouette Score:      {metrics_kmeans['silhouette']:.4f}\n")
        f.write(f"  Inercia (WSS):         {metrics_kmeans['inertia']:.2f}\n")
        f.write(f"\n  Alineación clusters ↔ ingresos >50K:\n")
        f.write(metrics_kmeans['cluster_income'].to_string() + "\n")

        f.write("\nANÁLISIS DE EQUIDAD ALGORÍTMICA (EU AI Act)\n")
        f.write("-" * 75 + "\n")
        f.write(f"  {'Modelo':<25} {'DPD Sexo':>15} {'DPD Raza':>15}\n")
        f.write("-" * 55 + "\n")
        for model_name, res in results_fairness.items():
            f.write(f"  {model_name:<25} {res['dpd_sex']:>15.4f} {res['dpd_race']:>15.4f}\n")
        f.write("\n  Umbral aceptable (Barocas et al., 2023): DPD < 0.10\n")

    print(f"\n  ✓ Reporte guardado en: {path}")
    return path


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == '__main__':

    # 1. Carga y preparación
    (X, y, df_model, X_train, X_test, y_train, y_test,
     X_train_sc, X_test_sc, scaler, feature_cols) = cargar_y_preparar(DATA_PROC)

    # 2. Modelos supervisados (LR + DT + RF + GB)
    (lr_model, dt_model, rf_model, gb_model,
     y_pred_lr, y_pred_dt, y_pred_rf, y_pred_gb,
     y_prob_lr, y_prob_dt, y_prob_rf, y_prob_gb) = entrenar_supervisados(
         X_train, X_test, y_train, y_test, X_train_sc, X_test_sc)

    # 3. Modelos no supervisados (K-Means + PCA)
    X_all_sc = scaler.transform(X)
    (kmeans, pca, df_pca, metrics_kmeans,
     K_range, inertias, silhouettes) = analisis_no_supervisado(
         X_all_sc, y, feature_cols)

    # 4. Métricas de evaluación
    print("\n[4/7] MÉTRICAS DE EVALUACIÓN")
    print("-" * 40)
    metrics_lr = calcular_metricas(y_test, y_pred_lr, y_prob_lr, "Regresión Logística")
    metrics_dt = calcular_metricas(y_test, y_pred_dt, y_prob_dt, "Árbol de Decisión")
    metrics_rf = calcular_metricas(y_test, y_pred_rf, y_prob_rf, "Random Forest")
    metrics_gb = calcular_metricas(y_test, y_pred_gb, y_prob_gb, "Gradient Boosting")
    cv_lr, cv_dt, cv_rf, cv_gb = validacion_cruzada(X_train, y_train, X_train_sc)

    # 5. Visualizaciones
    generar_visualizaciones(
        y_test,
        y_pred_lr, y_pred_dt, y_pred_rf, y_pred_gb,
        y_prob_lr,  y_prob_dt,  y_prob_rf,  y_prob_gb,
        metrics_lr, metrics_dt, metrics_rf, metrics_gb,
        dt_model, rf_model, gb_model, feature_cols,
        df_pca, K_range, inertias, silhouettes, metrics_kmeans
    )

    # 6. Equidad algorítmica
    df_test, results_fairness = analisis_equidad(
        df_model, y_test, y_pred_lr, y_pred_dt, y_pred_rf, y_pred_gb)

    # 7. Reporte final
    guardar_reporte(metrics_lr, metrics_dt, metrics_rf, metrics_gb,
                    cv_lr, cv_dt, cv_rf, cv_gb, metrics_kmeans, results_fairness)

    # Resumen en consola
    print(f"\n{'='*90}")
    print("  RESUMEN FINAL — COMPARATIVA DE 4 MODELOS")
    print(f"{'='*90}")
    print(f"  {'Métrica':<25} {'Reg. Logística':>16} {'Árbol Dec.':>14} {'Random Forest':>14} {'Grad. Boosting':>16}")
    print(f"  {'-'*88}")
    for k, label in [('accuracy','Accuracy'), ('f1_macro','F1-Score (macro)'), ('auc_roc','AUC-ROC')]:
        print(f"  {label:<25} {metrics_lr[k]:>16.4f} {metrics_dt[k]:>14.4f} {metrics_rf[k]:>14.4f} {metrics_gb[k]:>16.4f}")
    print(f"\n  K-Means: k={metrics_kmeans['best_k']}, Silhouette={metrics_kmeans['silhouette']:.4f}")
    print(f"\n✅ Análisis completado. Figuras en /figures/, reporte en /reports/")
