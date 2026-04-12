# TFG Business Analytics — Evaluación del Uso Ético de la IA en la Selección de Personal

**Autor:** Leopoldo Fernández de Villavicencio  
**Universidad:** Universidad Francisco de Vitoria (UFV)  
**Grado:** Business Analytics  
**Curso académico:** 2024-2025  

---

## 📋 Descripción del Proyecto

Este Trabajo de Fin de Grado analiza el uso ético de la inteligencia artificial en los procesos de selección de personal, evaluando el equilibrio entre **equidad, objetividad y eficiencia**. Se utiliza el dataset UCI Adult Census Income para construir modelos predictivos de clasificación de ingresos y estudiar posibles sesgos algorítmicos por género y raza, en el contexto del **Reglamento Europeo de IA (EU AI Act, 2024)**.

---

## 🎯 Objetivos

- Aplicar un pipeline ETL completo sobre datos reales del Census Income dataset
- Desarrollar modelos de clasificación predictiva (Regresión Logística y Random Forest) implementados desde cero con NumPy
- Evaluar la equidad algorítmica (Demographic Parity) por sexo y raza
- Contextualizar el análisis en el marco normativo del EU AI Act y la ética en RRHH
- Demostrar el riesgo de la "paradoja de la accuracy" con datos desbalanceados

---

## 📁 Estructura del Repositorio

```
TFG_Business_Analytics/
│
├── scripts/                          # Scripts Python del análisis
│   ├── 01_ingenieria_dato.py         # Pipeline ETL completo (Entrega 2)
│   └── 02_analisis_dato.py           # Modelos ML + Análisis de Equidad (Entrega 3)
│
├── figures/                          # Visualizaciones generadas
│   ├── fig_correlacion.png           # Mapa de correlaciones
│   ├── fig_edad_horas.png            # Distribución edad y horas semanales
│   ├── fig_feature_engineering.png   # Variables engineeradas
│   ├── fig_income_dist.png           # Distribución de la variable objetivo
│   ├── fig_nulls_antes.png           # Valores nulos antes de limpieza
│   ├── fig_nulls_despues.png         # Valores nulos después de limpieza
│   ├── fig_outliers_antes.png        # Outliers antes de tratamiento
│   ├── fig_outliers_despues.png      # Outliers después de tratamiento (IQR capping)
│   ├── fig_sesgos.png                # Análisis de sesgos por grupo demográfico
│   ├── figA1_confusion_matrix.png    # Matrices de confusión (LR y RF)
│   ├── figA2_roc_curve.png           # Curvas ROC comparativas
│   ├── figA3_feature_importance.png  # Importancia de variables (Random Forest)
│   ├── figA4_metricas_comparativa.png # Comparativa métricas LR vs RF
│   ├── figA5_fairness.png            # Paridad demográfica por sexo y raza
│   ├── figA6_prob_dist.png           # Distribución de probabilidades predichas
│   └── figA7_bootstrap.png          # Intervalos de confianza Bootstrap
│
├── .gitignore                        # Archivos excluidos del control de versiones
├── README.md                         # Este archivo
└── requirements.txt                  # Dependencias Python
```

---

## 🗄️ Dataset

**UCI Adult Census Income Dataset**
- **Fuente:** UCI Machine Learning Repository (Dua, D. & Graff, C., 2019)
- **Instancias:** 48,842 registros
- **Variables:** 15 (6 numéricas, 9 categóricas)
- **Variable objetivo:** income (≤50K / >50K)
- **Valores perdidos:** ~7.4% codificados como "?" en workclass, occupation y native-country
- **Distribución:** 82.7% ≤50K — 17.3% >50K (desbalanceado)

> **Nota:** El fichero CSV no se incluye en el repositorio por su tamaño. Puede generarse ejecutando `scripts/01_ingenieria_dato.py`.

---

## ⚙️ Entrega 2 — Ingeniería del Dato

El script `01_ingenieria_dato.py` implementa el pipeline ETL completo:

| Paso | Descripción |
|------|-------------|
| 1. Carga | Lectura del CSV con detección de encoding |
| 2. Perfilado | Estadísticos descriptivos, tipos, cardinalidad |
| 3. Nulos | Conversión "?" → NaN, imputación por moda |
| 4. Duplicados | Eliminación de 24 registros duplicados |
| 5. Outliers | Capping con método IQR (±1.5×IQR) |
| 6. Feature Engineering | 5 variables: income_binary, has_capital, full_time, age_group, edu_level |
| 7. EDA | 9 visualizaciones exploratorias guardadas en figures/ |

**Output:** `adult_clean.csv` con 48,818 registros limpios.

---

## 🤖 Entrega 3 — Análisis del Dato

El script `02_analisis_dato.py` implementa modelos predictivos **desde cero con NumPy puro** (sin scikit-learn):

### Resultados de Modelos

| Modelo | Accuracy | Precision Macro | Recall Macro | F1-Macro | AUC-ROC |
|--------|----------|----------------|--------------|----------|---------|
| Regresión Logística | 82.72% | 41.36% | 50.00% | 45.27% | 0.637 |
| Random Forest | 63.06% | 56.59% | 61.04% | 54.67% | 0.650 |

### Análisis de Equidad Algorítmica

Evaluación de **Paridad Demográfica** (Demographic Parity Difference) por:
- **Sexo:** Male vs. Female
- **Raza:** White vs. non-White

### Hallazgos Clave

- La Regresión Logística ilustra la **"paradoja de la accuracy"**: con datos desbalanceados (82.7% clase mayoritaria), un modelo que predice siempre ≤50K obtiene alta accuracy pero AUC cercano a 0.5
- El Random Forest equilibra mejor entre ambas clases y obtiene mayor F1-Macro
- Se detectan disparidades en tasas de predicción positiva por género, planteando riesgos éticos en RRHH
- Se aplica **Bootstrap** (1000 iteraciones) para estimar intervalos de confianza de las métricas

---

## 🛡️ Marco Normativo — EU AI Act

Este análisis se enmarca en el **Reglamento Europeo de IA (EU AI Act, Reglamento UE 2024/1689)**:

- Los sistemas de IA para **selección de personal** se clasifican como **"alto riesgo"** (Anexo III, punto 4)
- Obligaciones clave: transparencia, supervisión humana, evaluación de conformidad, no discriminación
- La equidad algorítmica no es solo una cuestión técnica, sino un **requisito legal en la UE**

---

## 🔧 Instalación y Ejecución

### Requisitos previos
```bash
pip install -r requirements.txt
```

### Ejecutar Pipeline ETL (Entrega 2)
```bash
python scripts/01_ingenieria_dato.py
```

### Ejecutar Análisis Predictivo (Entrega 3)
```bash
python scripts/02_analisis_dato.py
```

Las figuras se guardarán automáticamente en la carpeta `figures/`.

---

## 📚 Referencias Principales

- Dua, D. & Graff, C. (2019). *UCI Machine Learning Repository*. University of California, Irvine. https://archive.ics.uci.edu/ml/datasets/adult
- European Parliament (2024). *Artificial Intelligence Act*. Official Journal of the European Union, Regulation (EU) 2024/1689.
- Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning*. MIT Press.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Mehrabi, N. et al. (2021). A Survey on Bias and Fairness in Machine Learning. *ACM Computing Surveys*, 54(6), 1–35.

---

## 📄 Licencia

Repositorio privado de uso académico exclusivo para el TFG — Business Analytics, UFV (2024-2025).

---

*Desarrollado con Python 3.11 · NumPy · Pandas · Matplotlib · Seaborn*
