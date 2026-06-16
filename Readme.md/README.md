# TFG Business Analytics — UFV 2025-2026
## Evaluación del uso ético de la IA en la selección de personal: un análisis del equilibrio entre equidad, objetividad y eficiencia

**Autor:** Leopoldo Fernández de Villavicencio Alberola  
**Grado:** Business Analytics | Universidad Francisco de Vitoria  
**Curso:** 2025-2026  

---

## Descripción del proyecto

Este TFG analiza el dataset **UCI Adult Census Income** (Becker & Kohavi, 1996) mediante técnicas de ingeniería del dato y análisis predictivo. El objetivo central es predecir el nivel de ingresos anuales (≤50K / >50K) a partir de variables sociodemográficas, comparando cuatro modelos de clasificación supervisada y evaluando la equidad algorítmica de cada modelo bajo el marco del Reglamento (UE) 2024/1689 (EU AI Act).

La variable objetivo actúa como **proxy de la decisión de selección de personal** ("candidato aceptado / rechazado"), lo que permite cuantificar el sesgo algorítmico que el EU AI Act prohíbe en sistemas de IA de alto riesgo (Anexo III, punto 1a).

---

## Estructura del TFG

| Entrega | Documento |
|---------|-----------|
| 1 | Anteproyecto |
| 2 | Análisis de Negocio (FairHire Audit) |
| 3 | Ingeniería del Dato ← este repositorio |
| 4 | Análisis del Dato ← este repositorio |
| 5 | Memoria Final (en elaboración) |

---

## Estructura del repositorio

```
TFG_Business_Analytics/
├── data/
│   ├── raw/               → Dataset original (adult_census_income.csv)
│   └── processed/         → Dataset limpio generado por el script 01
├── figures/               → Figuras generadas automáticamente por los scripts
│   ├── fig_*.png          → Figuras de Ingeniería del Dato (ETL + EDA)
│   └── figA*.png          → Figuras de Análisis del Dato (modelos + equidad)
├── reports/
│   ├── reporte_ingenieria.txt      → Reporte del pipeline ETL
│   └── resultados_canonicos.json   → Cifras canónicas del análisis
├── scripts/
│   ├── 01_ingenieria_dato.py   → Pipeline ETL completo
│   └── 02_analisis_dato.py     → 4 modelos + no supervisado + equidad
├── .github/workflows/
│   └── run_analysis.yml   → GitHub Actions: ejecución automática del pipeline
└── requirements.txt       → Dependencias Python
```

---

## Dataset

**UCI Adult Census Income** — Becker & Kohavi (1996)  
- **48.842 registros originales** → **48.790 tras eliminar 52 duplicados exactos**
- **23 variables** (15 originales + 6 derivadas en feature engineering + 2 transformaciones log)
- Variable objetivo: `income` (≤50K / >50K) — desbalance 76,06 % / 23,94 %
- Fuente oficial: https://doi.org/10.24432/C5XW20

**Cómo obtener el dataset:**

```bash
# El CSV está disponible públicamente — descarga directa:
curl -L https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv \
     -o data/raw/adult_census_income.csv
```

> Verificación de integridad: el archivo correcto tiene **48.842 filas**, **52 duplicados** y una tasa de positivos (`>50K`) del **23,93 %**. Si estos números no coinciden, el dataset no es el oficial y los scripts producirán cifras distintas a las de los documentos.

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
**Output:** `data/processed/adult_clean.csv` + figuras ETL/EDA + `reports/reporte_ingenieria.txt`

| Paso | Descripción |
|------|-------------|
| Carga | Lectura del CSV con codificación correcta |
| Nulos | Conversión de `?` → NaN en workclass (5,73 %), occupation (5,75 %) y native-country (1,75 %) |
| Duplicados | Eliminación de **52 registros duplicados exactos** |
| Imputación | Moda para las tres variables categóricas con nulos |
| Outliers | **IQR capping** en fnlwgt y hours-per-week · **log1p** en capital-gain y capital-loss (el IQR es inviable: >91 % de valores son cero, Q1=Q3=0) |
| Feature engineering | **6 nuevas variables** (ver tabla inferior) |
| EDA | Distribución de clases, correlaciones, sesgos por sexo/raza, scatter edad×horas |

### Variables generadas en feature engineering

| Variable | Fórmula | Tipo |
|----------|---------|------|
| `income_binary` | 1 si income == '>50K', 0 si no | Binaria |
| `has_capital` | 1 si capital-gain > 0 o capital-loss > 0 | Binaria |
| `full_time` | 1 si hours-per-week ≥ 40 | Binaria |
| `age_group` | pd.cut en bins [0,25,35,50,65,100] | Ordinal |
| `edu_level` | pd.cut en bins [0,8,12,14,16] | Ordinal |
| `capital_ratio` | (capital-gain − capital-loss) / (fnlwgt + 1) | Continua |

### Figuras generadas (Ingeniería del Dato)

- `fig_nulls_antes.png` / `fig_nulls_despues.png` — Perfil de nulos antes y después de la imputación
- `fig_outliers_antes.png` / `fig_outliers_despues.png` — Outliers antes y después del tratamiento
- `fig_feature_engineering.png` — Variables derivadas
- `fig_income_dist.png` — Distribución de la variable objetivo (76,06 % / 23,94 %)
- `fig_sesgos.png` — Tasa real de >50K por sexo, raza y nivel educativo
- `fig_correlacion.png` — Matriz de correlaciones con income_binary
- `fig_edad_horas.png` — Distribución de edad e histograma de horas por nivel de ingresos

---

## Script 02 — Análisis del Dato

**Prerequisito:** ejecutar antes el script 01.  
**Input:** `data/processed/adult_clean.csv`  
**Output:** figuras + `reports/resultados_canonicos.json`

### Modelos supervisados (4 modelos con complejidad creciente)

| Modelo | Parámetros clave | Rol |
|--------|-----------------|-----|
| Regresión Logística | `C=1.0`, `solver=lbfgs`, `class_weight=balanced` | Baseline lineal interpretable (art. 11 EU AI Act) |
| Árbol de Decisión | `max_depth=10`, `min_samples_leaf=10`, `class_weight=balanced` | Modelo no lineal auditable |
| Random Forest | `n_estimators=200`, `max_depth=15`, `class_weight=balanced` | Ensamble por bagging |
| Gradient Boosting | `n_estimators=150`, `lr=0.1`, `max_depth=5`, `subsample=0.8` | Estado del arte en datos tabulares |

### Análisis no supervisado
- **K-Means**: búsqueda de k óptimo entre 2 y 9 (método del codo + silhouette score)
- **PCA**: reducción a 2 componentes para visualización de separabilidad de clases

### Equidad algorítmica (EU AI Act, art. 10)
- **Métrica:** Diferencia de Paridad Demográfica (DPD = max − min de tasas de predicción positiva entre grupos)
- **Ejes evaluados:** sexo (Male / Female), raza (5 categorías: White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other) y análisis **interseccional** sexo × raza (10 combinaciones)
- **Umbral de referencia:** DPD ≤ 0,10 (Barocas, Hardt & Narayanan, 2023)

### Figuras generadas (Análisis del Dato)

- `figA1_metricas_comparativa.png` — Comparativa de 5 métricas para los 4 modelos
- `figA2_confusion_matrix.png` — Matrices de confusión (4 modelos)
- `figA3_roc_curve.png` — Curvas ROC comparativas (4 modelos)
- `figA4_feature_importance.png` — Importancia de variables (DT, RF, GB) incluyendo capital_ratio
- `figA5_fairness.png` — DPD por sexo y raza (4 modelos)
- `figA6_kmeans_analisis.png` — K-Means: método del codo + silhouette + clusters en PCA
- `figA7_pca_clusters.png` — PCA: separabilidad real vs. asignación K-Means

---

## Cifras canónicas

Todos los documentos del TFG usan estas cifras como referencia. Son reproducibles ejecutando los dos scripts en orden. Ante cualquier discrepancia entre documentos, el árbitro es `reports/resultados_canonicos.json`.

| Métrica | Valor |
|---------|-------|
| Dataset inicial / final | 48.842 / 48.790 (52 duplicados eliminados) |
| Train / Test | 39.032 / 9.758 (80/20 estratificado) |
| Distribución target | 76,06 % ≤50K / 23,94 % >50K |
| Tasa real >50K — Male / Female | 30,4 % / 10,9 % |
| Tasa real >50K — máx/mín por raza | 27,0 % (Asian-Pac) / 11,7 % (Amer-Indian) |
| AUC — Regresión Logística | 0,856 |
| AUC — Árbol de Decisión | 0,906 |
| AUC — Random Forest | 0,921 |
| AUC — Gradient Boosting | **0,931** |
| F1 macro — Gradient Boosting | 0,802 |
| CV 5-fold F1 macro — GB | 0,811 ± 0,007 |
| DPD sexo (rango 4 modelos) | 0,323 – 0,378 |
| DPD raza (rango 4 modelos) | 0,242 – 0,267 |
| DPD interseccional GB (sexo × raza) | **0,439** |
| K-Means k óptimo / silhouette | 2 / 0,128 |
| PCA varianza PC1 / PC2 | 14,6 % / 11,9 % |

> Ninguno de los cuatro modelos cumple el umbral DPD ≤ 0,10 en ningún eje protegido.

---

## Referencias

- Becker, B. & Kohavi, R. (1996). Adult [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
- Barocas, S., Hardt, M. & Narayanan, A. (2023). *Fairness and Machine Learning: Limitations and Opportunities*. MIT Press. https://fairmlbook.org/
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Chouldechova, A. (2017). Fair prediction with disparate impact. *Big Data*, 5(2), 153–163.
- Fabris, A. et al. (2024). Fairness and bias in algorithmic hiring: a multidisciplinary survey. *ACM TIST*, 15(4), 1–54.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Ann. Stat.*, 29(5), 1189–1232.
- Parlamento Europeo. (2024). Reglamento (UE) 2024/1689 — EU AI Act. https://eur-lex.europa.eu/legal-content/ES/TXT/?uri=OJ:L_202401689
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
