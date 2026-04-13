# TFG Business Analytics — UFV 2024-2025
## Evaluación del uso ético de la IA en la selección de personal: un análisis del equilibrio entre equidad, objetividad y eficiencia

**Autor:** Leopoldo Fernández de Villavicencio  
**Grado:** Business Analytics | Universidad Francisco de Vitoria  
**Curso:** 2024-2025  

---

## Descripción del proyecto

Este TFG analiza el dataset **UCI Adult Census Income** mediante técnicas de ingeniería del dato y análisis predictivo. El objetivo central es predecir el nivel de ingresos anuales (≤50K / >50K) a partir de variables sociodemográficas, comparando cuatro modelos de clasificación supervisada y evaluando la equidad algorítmica de cada modelo bajo el marco del Reglamento de IA de la UE (EU AI Act, 2024).

---

## Estructura del repositorio

```
TFG_Business_Analytics/
├── data/
│   ├── raw/               → Dataset original (adult_census_income.csv) — no versionado
│   └── processed/         → Dataset limpio generado por el script 01 — no versionado
├── figures/               → Figuras generadas automáticamente por los scripts
│   ├── fig_*.png          → Figuras de Ingeniería del Dato (ETL + EDA)
│   └── figA*.png          → Figuras de Análisis del Dato (modelos + equidad)
├── reports/               → Informe de resultados en texto plano
├── scripts/
│   ├── 01_ingenieria_dato.py   → Pipeline ETL completo
│   └── 02_analisis_dato.py     → 4 modelos + no supervisado + equidad
├── .github/workflows/
│   └── run_analysis.yml   → GitHub Actions: ejecución automática del pipeline
└── requirements.txt       → Dependencias Python
```

---

## Dataset

**UCI Adult Census Income** — Kohavi & Becker (1996)  
- 48.651 registros válidos × 20 variables (tras limpieza + feature engineering)
- Variable objetivo: `income` (≤50K / >50K)
- Fuente oficial: https://archive.ics.uci.edu/dataset/2/adult

> ⚠️ El archivo `adult_census_income.csv` debe colocarse en `data/raw/` antes de ejecutar los scripts (no está versionado por privacidad de datos).

---

## Instalación

```bash
pip install -r requirements.txt
```

Dependencias: `pandas>=2.0`, `numpy>=1.24`, `matplotlib>=3.7`, `seaborn>=0.12`, `scikit-learn>=1.3`

---

## Ejecución local

Los scripts deben ejecutarse **en orden**:

```bash
# Paso 1: ETL + EDA
python scripts/01_ingenieria_dato.py

# Paso 2: Modelado + Equidad
python scripts/02_analisis_dato.py
```

---

## Ejecución en GitHub Actions (sin Python local)

Ir a **Actions → "Ejecutar Análisis TFG" → Run workflow**. Se puede elegir ejecutar ambos scripts, solo el 01 o solo el 02. Las figuras generadas quedan disponibles como artefactos descargables del workflow.

---

## Script 01 — Ingeniería del Dato

**Input:** `data/raw/adult_census_income.csv`  
**Output:** `data/processed/adult_clean.csv` + figuras ETL/EDA

| Paso | Descripción |
|------|-------------|
| Carga | Lectura del CSV con codificación correcta |
| Nulos | Detección y conversión de `?` → NaN |
| Duplicados | Eliminación de 24 registros duplicados |
| Imputación | Moda para variables categóricas (interpolación para numéricas si hubiera nulos) |
| Outliers | IQR capping en variables numéricas continuas |
| Feature engineering | 5 nuevas variables: `capital_ratio`, `high_hours`, `age_group`, `university_edu`, `hours_ratio` |
| EDA | Distribución de clases, correlaciones, sesgos, scatter edad×horas |

**Figuras generadas (ETL):**
- `fig_nulls_antes/despues.png` — Nulos antes y después de la imputación
- `fig_outliers_antes/despues.png` — Outliers antes y después del IQR capping
- `fig_feature_engineering.png` — Variables derivadas
- `fig_income_dist.png` — Distribución de la variable objetivo
- `fig_sesgos.png` — Distribución de variables sensibles (sexo, raza)
- `fig_correlacion.png` — Mapa de correlaciones
- `fig_edad_horas.png` — Scatter edad × horas trabajadas por nivel de ingresos

---

## Script 02 — Análisis del Dato

**Prerequisito:** ejecutar antes el script 01.  
**Input:** `data/processed/adult_clean.csv`  
**Output:** figuras + `reports/resultados_modelos.txt`

### Modelos supervisados (4 modelos con complejidad creciente)

| Modelo | Parámetros clave | Rol |
|--------|-----------------|-----|
| Regresión Logística | `C=1.0`, `max_iter=1000`, `class_weight=balanced` | Baseline lineal interpretable |
| Árbol de Decisión | `max_depth=10`, `min_samples_leaf=10`, `class_weight=balanced` | Modelo no lineal interpretable |
| Random Forest | `n_estimators=200`, `max_depth=15`, `class_weight=balanced` | Ensamble por bagging |
| Gradient Boosting | `n_estimators=150`, `lr=0.1`, `max_depth=5`, `subsample=0.8` | Ensamble por boosting estocástico |

### Análisis no supervisado
- **K-Means**: número óptimo de clusters por método del codo + silhouette score
- **PCA**: reducción a 2 componentes para visualización de separabilidad de clases

### Equidad algorítmica (EU AI Act, 2024)
- Métrica: Diferencia de Paridad Demográfica (DPD)
- Ejes evaluados: sexo (Male vs. Female) y raza (White vs. Non-White)
- Umbral de aceptabilidad: DPD ≤ 0.10

### Figuras generadas (análisis):
- `figA1_confusion_matrix.png` — Matrices de confusión (2×2, 4 modelos)
- `figA2_roc_curve.png` — Curvas ROC comparativas (4 modelos)
- `figA3_feature_importance.png` — Importancia de variables (DT, RF, GB)
- `figA4_metricas_comparativa.png` — Comparativa de métricas (4 modelos)
- `figA5_fairness.png` — Equidad algorítmica: DPD por sexo y raza (4 modelos)
- `figA6_kmeans_analisis.png` — K-Means: Elbow + Silhouette + clusters en PCA
- `figA7_pca_clusters.png` — PCA: separabilidad real vs. clusters K-Means

---

## Referencias principales

- Kohavi, R. & Becker, B. (1996). Adult Dataset. UCI ML Repository. https://doi.org/10.24432/C5XW20
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Ann. Stat.*, 29(5).
- Barocas, S., Hardt, M. & Narayanan, A. (2023). *Fairness and Machine Learning*. MIT Press. https://fairmlbook.org/
- Parlamento Europeo. (2024). Reglamento (UE) 2024/1689 — EU AI Act. https://eur-lex.europa.eu/legal-content/ES/TXT/?uri=OJ:L_202401689
