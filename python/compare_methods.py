import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from tensorflow import keras

# ------------------------------- configuracao --------------------------------
CSV_PATH = Path("../csv/merged_ML_ch2-600to800.csv")   # amostra de calibracao
BATCH = 8192

STAGE2_DIR = Path("models_stage2_qqbar_binary")
FOUR_DIR   = Path("models_four_class")
THREE_DIR  = Path("models_stage2_matched_3class")
GMM_DIR    = Path("models_unsupervised_gmm_matched")
GMM_KS     = [3, 4, 5, 6]
GMM_K_PLOT = None        # None: desenha o K cujo melhor cluster tem maior pureza

# ------------------------------- dados ---------------------------------------
df = pd.read_csv(CSV_PATH)
df = df[df["channel2"].isin([1, 2, 3])].copy()
if "dpsi12" in df.columns:
    df = df[np.isfinite(df["dpsi12"])]
df = df.reset_index(drop=True)
y = (df["channel2"].to_numpy() == 1).astype(int)
baseline = y.mean()
print(f"Splittings matched: {len(df):,}   fracao g->qqbar = {baseline:.3f}\n")

# ------------------------------- features ------------------------------------
def engineer_stage2(d):
    """Igual ao apply_ML.py e ao notebook do stage 2."""
    d = d.copy()
    for col in ["pt_dispersion3", "pt_dispersion4"]:
        if col in d.columns:
            d[col] = np.log1p(d[col].clip(lower=0))
    if "kt2" in d.columns:
        d["kt2"] = np.log1p(d["kt2"].clip(lower=0))
    d["z_sum"] = d["z1"] + d["z2"]
    d["z_absdiff"] = np.abs(d["z1"] - d["z2"])
    d["z_min"] = np.minimum(d["z1"], d["z2"])
    d["z_max"] = np.maximum(d["z1"], d["z2"])
    d["z_minmax_ratio"] = d["z_min"] / (d["z_max"] + 1e-8)
    for c3, c4, name in [("N_charged3", "N_charged4", "N_charged"),
                         ("N_all3", "N_all4", "N_all"),
                         ("pt_weight3", "pt_weight4", "pt_weight"),
                         ("pt_dispersion3", "pt_dispersion4", "pt_dispersion")]:
        v3, v4 = d[c3], d[c4]
        vmin, vmax = np.minimum(v3, v4), np.maximum(v3, v4)
        d[f"{name}_sum"] = v3 + v4
        d[f"{name}_absdiff"] = np.abs(v3 - v4)
        d[f"{name}_min"] = vmin
        d[f"{name}_max"] = vmax
        d[f"{name}_minmax_ratio"] = vmin / (vmax + 1e-8)
    return d

def add_clustering_features(d):
    """Igual ao notebook do GMM."""
    d = d.copy()
    for col in ["pt_dispersion3", "pt_dispersion4"]:
        d[col] = np.log1p(d[col].clip(lower=0))
    d["kt2"] = np.log1p(d["kt2"].clip(lower=0))
    d["deltaR34"] = np.log1p(1.0 / d["deltaR34"].clip(lower=1e-4))
    for c3, c4, name in [("N_charged3", "N_charged4", "N_charged"),
                         ("N_all3", "N_all4", "N_all"),
                         ("pt_weight3", "pt_weight4", "pt_weight"),
                         ("pt_dispersion3", "pt_dispersion4", "pt_dispersion")]:
        x3, x4 = d[c3], d[c4]
        d[f"{name}_sum"] = x3 + x4
        d[f"{name}_absdiff"] = np.abs(x3 - x4)
        d[f"{name}_min"] = np.minimum(x3, x4)
        d[f"{name}_max"] = np.maximum(x3, x4)
    return d

def clean(X):
    return X.replace([np.inf, -np.inf], np.nan)

def feature_names(prep):
    """Nomes das features vistos pelo primeiro passo de um Pipeline do sklearn."""
    step = prep.steps[0][1] if hasattr(prep, "steps") else prep
    return list(step.feature_names_in_)

def predict(model, X):
    return model.predict(X.astype("float32"), batch_size=BATCH, verbose=0)

# ------------------------------- scores --------------------------------------
scores = {}

# Stage 2 da selecao em dois stages
pp2 = load(STAGE2_DIR / "qqbar_binary_preprocessor.joblib")
m2 = keras.models.load_model(STAGE2_DIR / "qqbar_binary_dnn_best.keras", compile=False)
d2 = engineer_stage2(df)
X2 = pp2["scaler"].transform(pp2["imputer"].transform(clean(d2[pp2["features"]])))
scores["Two-stage selection, second stage"] = predict(m2, X2).ravel()

# Rede de 4 classes (classes 0..3, g->qqbar e o indice 1)
prep4 = load(FOUR_DIR / "four_class_preprocessor.joblib")
m4 = keras.models.load_model(FOUR_DIR / "four_class_dnn_best.keras", compile=False)
f4 = feature_names(prep4)
scores["Four-class network"] = predict(m4, prep4.transform(clean(df[f4])))[:, 1]

# Rede de 3 classes (mapa 1->0, g->qqbar e o indice 0)
prep3 = load(THREE_DIR / "stage2_matched_3class_preprocessor.joblib")
m3 = keras.models.load_model(THREE_DIR / "stage2_matched_3class_dnn_best.keras", compile=False)
f3 = feature_names(prep3)
scores["Three-class network"] = predict(m3, prep3.transform(clean(df[f3])))[:, 0]

for name, feats in [("Four-class network", f4), ("Three-class network", f3),
                    ("Two-stage selection, second stage", pp2["features"])]:
    if "Jet_pt" in feats:
        print(f"ATENCAO: '{name}' usa Jet_pt como input.")
print()

# ------------------------------- tabela --------------------------------------
print(f"{'Method':38s} {'ROC AUC':>8s} {'PR AUC':>8s}")
for name, s in scores.items():
    print(f"{name:38s} {roc_auc_score(y, s):8.3f} {average_precision_score(y, s):8.3f}")
print(f"{'Random classifier':38s} {0.5:8.3f} {baseline:8.3f}\n")

print("Linhas LaTeX para tab:methods:")
for name, s in scores.items():
    print(f"    {name:34s} & ${roc_auc_score(y, s):.3f}$ & ${average_precision_score(y, s):.3f}$ \\\\")
print()

# ------------------------------- GMM -----------------------------------------
gmm_points = {}
prep_g = load(GMM_DIR / "gmm_ggcandidate_pca_preprocessor.joblib")
dg = add_clustering_features(df)
Z = prep_g.transform(clean(dg[feature_names(prep_g)]))
n_qq = y.sum()
for k in GMM_KS:
    path = GMM_DIR / f"gmm_ggcandidate_pca_K{k}.joblib"
    if not path.exists():
        continue
    labels = load(path).predict(Z)
    pts = []
    print(f"GMM K={k}")
    for c in range(k):
        mask = labels == c
        if mask.sum() == 0:
            continue
        pur = y[mask].mean()
        eff = y[mask].sum() / n_qq
        pts.append((eff, pur))
        print(f"   cluster {c}: jatos = {mask.sum():9d}   pureza = {pur:.3f}   eficiencia = {eff:.3f}")
    gmm_points[k] = pts
if gmm_points:
    best_k = max(gmm_points, key=lambda k: max(p for _, p in gmm_points[k]))
    k_plot = GMM_K_PLOT if GMM_K_PLOT in gmm_points else best_k
    print(f"\nPureza maxima de um cluster: {max(p for _, p in gmm_points[best_k]):.3f} (K={best_k})")

# ------------------------------- figura --------------------------------------
fig, ax = plt.subplots(figsize=(7, 5.5))
styles = {"Two-stage selection, second stage": ("C0", "-"),
          "Three-class network": ("C1", "--"),
          "Four-class network": ("C2", "-.")}
for name, s in scores.items():
    prec, rec, _ = precision_recall_curve(y, s)
    color, ls = styles[name]
    ax.plot(rec, prec, color=color, ls=ls, lw=2,
            label=f"{name} (AP = {average_precision_score(y, s):.3f})")
if gmm_points:
    e, p = zip(*gmm_points[k_plot])
    ax.scatter(e, p, color="C3", marker="s", s=40, zorder=3,
               label=f"Gaussian mixture, {k_plot} clusters")
ax.axhline(baseline, color="grey", ls=":", lw=1.5, label=f"No selection ({baseline:.3f})")
ax.set_xlabel("Efficiency (recall)")
ax.set_ylabel("Purity (precision)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("pr_methods.png", dpi=160)
