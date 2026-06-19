"""
Script 03 — Mitigación del Sesgo Algorítmico
TFG Business Analytics | UFV 2025-2026
Autor: Leopoldo Fernández de Villavicencio Alberola

Toma el dataset limpio (data/processed/adult_clean.csv, salida del Script 01),
reproduce el Gradient Boosting baseline del Script 02 y aplica tres técnicas de
mitigación —una por familia del pipeline— midiendo la Paridad Demográfica (DPD)
por sexo antes y después de cada una.

Usa EXACTAMENTE las mismas FEATURE_COLS, codificación, partición e hiperparámetros
que el Script 02, de modo que el baseline es directamente comparable con el Cap. 7.

Produce: figures/fig_mitig_dpd.png, figures/fig_mitig_tradeoff.png
         reports/mitigacion_resultados.json

Requiere: pandas, numpy, scikit-learn, fairlearn, matplotlib
Uso:      python 03_mitigacion.py   (ejecutar DESPUÉS del Script 01)
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.model_selection           import train_test_split
from sklearn.preprocessing             import LabelEncoder
from sklearn.ensemble                  import GradientBoostingClassifier
from sklearn.metrics                   import accuracy_score, f1_score, roc_auc_score
from fairlearn.metrics                 import demographic_parity_difference
from fairlearn.postprocessing          import ThresholdOptimizer
from fairlearn.reductions              import ExponentiatedGradient, DemographicParity

# ── Rutas (idénticas a los Scripts 01 y 02) ──────────────────────────────────
PROC_PATH = os.path.join('data', 'processed', 'adult_clean.csv')
FIG_DIR   = 'figures'
REP_DIR   = 'reports'
for d in [FIG_DIR, REP_DIR]:
    os.makedirs(d, exist_ok=True)

SEED  = 42
BLUE  = '#1F4E79'; LBLUE = '#2E75B6'; RED = '#C0392B'; GREEN = '#27AE60'

# ── Carga ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(PROC_PATH)
print(f"[CARGA] {len(df):,} registros x {df.shape[1]} variables")

# ── Features EXACTAS del Script 02 ───────────────────────────────────────────
FEATURES_CAT = ['workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'native-country']
FEATURES_NUM = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                'hours-per-week', 'capital_ratio']
FEATURES_BIN = ['has_capital', 'full_time']
TARGET       = 'income_binary'
FEATURE_COLS = FEATURES_CAT + FEATURES_NUM + FEATURES_BIN

# Atributo protegido en claro antes de codificar
sex_raw = df['sex'].copy()

# Label encoding de categóricas (igual que el Script 02)
df_enc = df.copy()
for col in FEATURES_CAT:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

X = df_enc[FEATURE_COLS].values
y = df_enc[TARGET].values

# Partición idéntica al Script 02 (80/20 estratificada, semilla 42)
X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
    X, y, sex_raw, test_size=0.20, stratify=y, random_state=SEED)

# Sample weights del GB (misma fórmula que el Script 02)
def gb_sample_weight(yv):
    neg, pos = (yv == 0).sum(), (yv == 1).sum()
    return np.where(yv == 1, len(yv) / (2 * pos), len(yv) / (2 * neg))

def dpd(y_pred):
    return demographic_parity_difference(y_te, y_pred, sensitive_features=s_te)

def gb_model():
    return GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                      max_depth=5, subsample=0.8, random_state=SEED)

res = {}

# ── BASELINE: Gradient Boosting (config canónica del Script 02) ──────────────
gb = gb_model(); gb.fit(X_tr, y_tr, sample_weight=gb_sample_weight(y_tr))
p = gb.predict(X_te)
res['baseline'] = dict(acc=accuracy_score(y_te, p), f1=f1_score(y_te, p, average='macro'),
                       auc=roc_auc_score(y_te, gb.predict_proba(X_te)[:, 1]), dpd=dpd(p))

# ── 1. REWEIGHING (pre-procesamiento, Kamiran & Calders 2012) ────────────────
dft = pd.DataFrame({'s': s_tr.values, 'y': y_tr}); w = np.ones(len(dft))
for sv in dft['s'].unique():
    for yv in (0, 1):
        m = (dft['s'] == sv) & (dft['y'] == yv)
        if m.mean() > 0:
            w[m.values] = ((dft['s'] == sv).mean() * (dft['y'] == yv).mean()) / m.mean()
gb_rw = gb_model(); gb_rw.fit(X_tr, y_tr, sample_weight=w)
p = gb_rw.predict(X_te)
res['reweighing'] = dict(acc=accuracy_score(y_te, p), f1=f1_score(y_te, p, average='macro'),
                         auc=roc_auc_score(y_te, gb_rw.predict_proba(X_te)[:, 1]), dpd=dpd(p))

# ── 2. EXPONENTIATED GRADIENT (in-processing, DemographicParity) ─────────────
eg = ExponentiatedGradient(gb_model(), constraints=DemographicParity(), eps=0.02)
eg.fit(X_tr, y_tr, sensitive_features=s_tr)
p = eg.predict(X_te)
res['expgrad'] = dict(acc=accuracy_score(y_te, p), f1=f1_score(y_te, p, average='macro'),
                      auc=np.nan, dpd=dpd(p))

# ── 3. THRESHOLD OPTIMIZER (post-procesamiento, Hardt et al. 2016) ───────────
to = ThresholdOptimizer(estimator=gb, constraints='demographic_parity',
                        objective='accuracy_score', prefit=True,
                        predict_method='predict_proba')
to.fit(X_tr, y_tr, sensitive_features=s_tr)
p = to.predict(X_te, sensitive_features=s_te)
res['threshold'] = dict(acc=accuracy_score(y_te, p), f1=f1_score(y_te, p, average='macro'),
                        auc=np.nan, dpd=dpd(p))

# ── Guardar resultados ────────────────────────────────────────────────────────
with open(os.path.join(REP_DIR, 'mitigacion_resultados.json'), 'w') as f:
    json.dump(res, f, indent=2, default=float)

print("\n[RESULTADOS] DPD por sexo (baseline -> mitigado):")
for k in ('baseline', 'reweighing', 'expgrad', 'threshold'):
    print(f"  {k:12s}  DPD={res[k]['dpd']:.3f}  acc={res[k]['acc']:.3f}  f1={res[k]['f1']:.3f}")

# ── Figuras ───────────────────────────────────────────────────────────────────
labels = ["Baseline\nGB", "Reweighing\n(pre)", "Exp. Gradient\n(in)", "Threshold Opt.\n(post)"]
keys   = ["baseline", "reweighing", "expgrad", "threshold"]
dpd_v  = [res[k]["dpd"] for k in keys]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, dpd_v, color=[RED, LBLUE, LBLUE, LBLUE], edgecolor="white", width=.62)
ax.axhline(0.10, color=GREEN, ls="--", lw=1.8, label="Umbral de referencia (DPD <= 0,10)")
for b, v in zip(bars, dpd_v):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("DPD por sexo"); ax.set_ylim(0, 0.40)
ax.set_title("Paridad Demografica (sexo): efecto de las tecnicas de mitigacion",
             fontsize=12, fontweight="bold", color=BLUE)
ax.legend(); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.25)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_mitig_dpd.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(9, 5.4))
mk = ["o", "s", "^", "D"]; cols = [RED, LBLUE, BLUE, GREEN]
for k, l, m, c in zip(keys, labels, mk, cols):
    ax.scatter(res[k]["dpd"], res[k]["acc"], s=230, marker=m, color=c,
               edgecolor="white", linewidth=1.5, zorder=3, label=l.replace("\n", " "))
ax.axvline(0.10, color=GREEN, ls="--", lw=1.6, alpha=.8)
ax.set_xlabel("DPD por sexo  (menor = mas equitativo)")
ax.set_ylabel("Accuracy")
ax.set_title("Frente precision-equidad", fontsize=11.5, fontweight="bold", color=BLUE)
ax.legend(fontsize=9.5, loc="lower center", ncol=2)
ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.22); ax.set_xlim(-0.02, 0.37)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_mitig_tradeoff.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\n[FIGURAS] fig_mitig_dpd.png y fig_mitig_tradeoff.png guardadas en figures/")
print("[OK] Script 03 completado.")
