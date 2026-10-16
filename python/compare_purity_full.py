from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import precision_recall_curve
from tensorflow import keras

# ------------------------------- configuracao --------------------------------
CSV_PATH = Path("../csv/merged_ML_ch2-600to800.csv")   # amostra de calibracao
BATCH = 8192
STAGE1_CUT = 0.8
INCLUDE_REST = True        # True: a classe "rest" conta como fundo
LOG_X = True               # eixo da eficiencia em escala logaritmica
EFFS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]   # eficiencias para a tabela

STAGE1_DIR = Path("models_stage1_unmatched_vs_matched")
STAGE2_DIR = Path("models_stage2_qqbar_binary")
FOUR_DIR   = Path("models_four_class")
THREE_DIR  = Path("models_stage2_matched_3class")

# ------------------------------- dados ---------------------------------------
keep = [0, 1, 2, 3, 4] if INCLUDE_REST else [0, 1, 2, 3]
df = pd.read_csv(CSV_PATH)
df = df[df["channel2"].isin(keep)].copy()
if "dpsi12" in df.columns:
    df = df[np.isfinite(df["dpsi12"])]
df = df.reset_index(drop=True)
y = (df["channel2"].to_numpy() == 1).astype(int)
n_qq = y.sum()
print(f"Splittings com etiqueta: {len(df):,}   g->qqbar: {n_qq:,}   "
      f"pureza sem selecao = {y.mean():.4f}\n")

# ------------------------------- features ------------------------------------
def engineer_stage2(d):
    """Igual ao apply_ML.py e ao notebook do stage 2."""
    d = d.copy()
    for col in ["pt_dispersion3", "pt_dispersion4"]:
        d[col] = np.log1p(d[col].clip(lower=0))
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

def clean(X):
    return X.replace([np.inf, -np.inf], np.nan)

def feature_names(prep):
    step = prep.steps[0][1] if hasattr(prep, "steps") else prep
    return list(step.feature_names_in_)

def predict(model, X):
    return model.predict(X.astype("float32"), batch_size=BATCH, verbose=0)

# ------------------------------- scores --------------------------------------
prep1 = load(STAGE1_DIR / "stage1_preprocess.joblib")
m1 = keras.models.load_model(STAGE1_DIR / "stage1_unmatched_vs_matched_dnn_best.keras", compile=False)
p1 = predict(m1, prep1.transform(clean(df[feature_names(prep1)])))
s_matched = p1[:, 1] if (p1.ndim == 2 and p1.shape[1] == 2) else p1.ravel()

pp2 = load(STAGE2_DIR / "qqbar_binary_preprocessor.joblib")
m2 = keras.models.load_model(STAGE2_DIR / "qqbar_binary_dnn_best.keras", compile=False)
d2 = engineer_stage2(df)
X2 = pp2["scaler"].transform(pp2["imputer"].transform(clean(d2[pp2["features"]])))
s_stage2 = predict(m2, X2).ravel()

prep3 = load(THREE_DIR / "stage2_matched_3class_preprocessor.joblib")
m3 = keras.models.load_model(THREE_DIR / "stage2_matched_3class_dnn_best.keras", compile=False)
s_three = predict(m3, prep3.transform(clean(df[feature_names(prep3)])))[:, 0]   # canal 1 -> classe 0

prep4 = load(FOUR_DIR / "four_class_preprocessor.joblib")
m4 = keras.models.load_model(FOUR_DIR / "four_class_dnn_best.keras", compile=False)
s_four = predict(m4, prep4.transform(clean(df[feature_names(prep4)])))[:, 1]     # classe 1 = qqbar

# ------------------------------- curvas --------------------------------------
def curve(score, premask=None):
    """Pureza e eficiencia ao variar o threshold no score, depois de um corte
    previo opcional. A eficiencia e relativa a TODOS os jatos qqbar."""
    if premask is None:
        premask = np.ones(len(score), dtype=bool)
    prec, rec, _ = precision_recall_curve(y[premask], score[premask])
    eff = rec * y[premask].sum() / n_qq
    return eff, prec

stage1_mask = s_matched > STAGE1_CUT
curves = {
    "Two-stage selection": curve(s_stage2, stage1_mask),
    "Stage 1 + three-class network": curve(s_three, stage1_mask),
    "Four-class network": curve(s_four),
}
eff_stage1 = y[stage1_mask].sum() / n_qq
print(f"Eficiencia maxima depois do stage 1 (matched > {STAGE1_CUT}): {eff_stage1:.3f}\n")

# ------------------------------- tabela --------------------------------------
def purity_at(eff, pur, target):
    """Maior pureza alcancada com eficiencia >= target."""
    ok = eff >= target
    return pur[ok].max() if ok.any() else float("nan")

header = "Eficiencia " + "".join(f"{name:>32s}" for name in curves)
print(header)
for e in EFFS:
    row = f"{e:10.3f} " + "".join(f"{purity_at(*curves[n], e):32.3f}" for n in curves)
    print(row)

# ------------------------------- figura --------------------------------------
fig, ax = plt.subplots(figsize=(7, 5.5))
styles = {"Two-stage selection": ("C0", "-"),
          "Stage 1 + three-class network": ("C1", "--"),
          "Four-class network": ("C2", "-.")}
for name, (eff, pur) in curves.items():
    color, ls = styles[name]
    ok = eff > 0
    ax.plot(eff[ok], pur[ok], color=color, ls=ls, lw=2, label=name)
ax.axhline(y.mean(), color="grey", ls=":", lw=1.5, label=f"No selection ({y.mean():.3f})")
ax.axvline(eff_stage1, color="grey", ls="--", lw=1, alpha=0.6)
ax.text(eff_stage1, 0.95, " stage 1 limit", color="grey", fontsize=8, va="top")
if LOG_X:
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
else:
    ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel(r"Efficiency for $g\to q\bar{q}$")
ax.set_ylabel(r"Purity in $g\to q\bar{q}$")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig("purity_full_methods.png", dpi=160)
