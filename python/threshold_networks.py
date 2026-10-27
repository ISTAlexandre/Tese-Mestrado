# threshold_networks.py
# <cos 2 dpsi12> em funcao do threshold no score g->qqbar para as tres redes:
#   1) dois stages:          matched score > STAGE1_CUT  e  score qq (stage 2) > t
#   2) stage 1 + 3 classes:  matched score > STAGE1_CUT  e  prob. qq (3 classes) > t
#   3) stage 1 + 4 classes:  matched score > STAGE1_CUT  e  prob. qq (4 classes) > t
#      (com FOUR_CLASS_WITH_STAGE1 = False a rede de 4 classes e usada sozinha)
#
# Os scores das quatro redes sao calculados aqui, a partir das variaveis lidas
# diretamente dos ficheiros ROOT, para todos os jatos com dpsi12 valido
# (incluindo os jatos sem etiqueta, como nas Figuras 4.13 e 4.15).
#
# O ROOT e o TensorFlow nao podem estar no mesmo processo (no macOS os dois
# trazem o seu proprio LLVM e o cling da segmentation violation). Por isso:
#   - a leitura dos ficheiros ROOT corre num subprocesso, que so importa o ROOT
#     e grava as variaveis em threshold_networks_sim.npz / _data.npz,
#   - o processo principal so importa o TensorFlow e le esses ficheiros.
# Se os .npz ja existirem e corresponderem aos mesmos ficheiros ROOT, a leitura
# nao e repetida (REREAD = True obriga a reler).
#
# Produz threshold_networks.png / .pdf
#   painel esquerdo: <cos 2 dpsi12> em funcao do threshold
#   painel direito:  <cos 2 dpsi12> em funcao da pureza na simulacao
# e grava a tabela em threshold_networks.csv.
#
# Uso (na pasta python/, onde estao os modelos):  python threshold_networks.py

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------- configuracao --------------------------------
SIM_FILE = "../root_ML/merged_ML-600to800.root"   # amostra de calibracao
DATA_FILE = "../root/merged.root"                 # None para mostrar so a simulacao
TREE = "jetTree"
BATCH = 8192
REREAD = False

STAGE1_CUT = 0.8
FOUR_CLASS_WITH_STAGE1 = True
THRESHOLDS = np.round(np.arange(0.10, 0.951, 0.05), 2)
MIN_JETS_SIM = 200          # pontos com menos jatos selecionados nao sao desenhados
MIN_JETS_DATA = 30

STAGE1_DIR = Path("models_stage1_unmatched_vs_matched")
STAGE2_DIR = Path("models_stage2_qqbar_binary")
THREE_DIR  = Path("models_stage2_matched_3class")
FOUR_DIR   = Path("models_four_class")

CACHE = {"sim": ("threshold_networks_sim.npz", SIM_FILE),
         "data": ("threshold_networks_data.npz", DATA_FILE)}

NETWORKS = ["two", "three", "four"]
NAMES = {"two":   "Two-stage selection",
         "three": "Stage 1 + three-class network",
         "four":  "Stage 1 + four-class network" if FOUR_CLASS_WITH_STAGE1
                  else "Four-class network"}
STYLE = {"two": ("C0", "o"), "three": ("C1", "s"), "four": ("C2", "^")}

# ------------------------------- leitura (so ROOT) ---------------------------
# variavel do CSV -> expressao C++ para o jato i (k1, k2 = splittings de maior kT)
BRANCHES = {
    "Jet_pt":         "jet_pt[i]",
    "z1":             "lund_z_sd[i][k1]",
    "z2":             "lund_z_secondary_sd[i][k2]",
    "deltaR34":       "lund_delta_secondary_sd[i][k2]",
    "kt2":            "lund_kt_secondary_sd[i][k2]",
    "N_charged3":     "lund_p3_n_charged[i]",
    "N_charged4":     "lund_p4_n_charged[i]",
    "N_all3":         "lund_p3_n_all[i]",
    "N_all4":         "lund_p4_n_all[i]",
    "pt_weight3":     "lund_p3_sigma[i]",
    "pt_weight4":     "lund_p4_sigma[i]",
    "pt_dispersion3": "lund_p3_ptD[i]",
    "pt_dispersion4": "lund_p4_ptD[i]",
    "dpsi12":         "lund_psi12_sd[i]",
}
# lidas so se existirem no ficheiro: (ramo, expressao)
OPTIONAL = {
    "label":          ("lund_secondary_idx_sd", "lund_secondary_idx_sd[i]"),
    "stored_matched": ("lund_ML_matched",       "lund_ML_matched[i]"),
    "stored_qq":      ("lund_ML_qq",            "lund_ML_qq[i]"),
}

COLLECTOR = r"""
#include <vector>
#include "ROOT/RVec.hxx"
namespace thrnet {
std::vector<std::vector<double>> store;
void reset(int ncol) { store.assign(ncol, std::vector<double>()); }
int push(const ROOT::RVecD &v, int icol) {
    auto &s = store[icol];
    s.insert(s.end(), v.begin(), v.end());
    return (int) v.size();
}
const std::vector<double> &column(int icol) { return store[icol]; }
}
"""


def per_jet(expr, sizes):
    """Valor de expr para cada jato com dpsi12 e os dois splittings validos.
    A condicao e a mesma para todas as variaveis, por isso as colunas ficam
    alinhadas jato a jato."""
    n_expr = ", ".join(f"(size_t) {b}.size()" for b in sizes)
    return f"""
    ROOT::RVecD out;
    const size_t n = std::min<size_t>({{{n_expr}}});
    for (size_t i = 0; i < n; ++i) {{
        const double p = lund_psi12_sd[i];
        const int k1 = lund_max_kt_sd[i];
        const int k2 = lund_max_kt_secondary_sd[i];
        if (!(p == p) || k1 < 0 || k2 < 0) continue;
        if ((size_t) k1 >= lund_z_sd[i].size()) continue;
        if ((size_t) k2 >= lund_z_secondary_sd[i].size() ||
            (size_t) k2 >= lund_delta_secondary_sd[i].size() ||
            (size_t) k2 >= lund_kt_secondary_sd[i].size()) continue;
        out.push_back({expr});
    }}
    return out;
    """


def read_sample(ROOT, path):
    """Uma linha por jato valido, com as variaveis de entrada das redes,
    dpsi12 e, se existirem, a etiqueta e os scores guardados no ficheiro."""
    df = ROOT.RDataFrame(TREE, path)
    have = {str(c) for c in df.GetColumnNames()}
    cols = dict(BRANCHES)
    for name, (branch, expr) in OPTIONAL.items():
        if branch in have:
            cols[name] = expr
    sizes = sorted({e.split("[")[0] for e in cols.values()}
                   | {"lund_max_kt_sd", "lund_max_kt_secondary_sd"})
    missing = [b for b in sizes if b not in have]
    if missing:
        raise RuntimeError(f"{path}: faltam os ramos {missing}")

    ROOT.thrnet.reset(len(cols))
    trig = []
    for j, expr in enumerate(cols.values()):
        df = (df.Define(f"v_{j}", per_jet(expr, sizes))
                .Define(f"n_{j}", f"thrnet::push(v_{j}, {j})"))
        trig.append(f"n_{j}")
    total = df.Define("trigger", " + ".join(trig)).Sum("trigger").GetValue()

    out = {name: np.asarray(ROOT.thrnet.column(j), dtype=np.float64).copy()
           for j, name in enumerate(cols)}
    ROOT.thrnet.reset(0)
    n = {len(v) for v in out.values()}
    if len(n) != 1 or total != len(cols) * n.pop():
        raise RuntimeError("as colunas lidas nao estao alinhadas")
    print(f"{path}: {len(out['dpsi12']):,} jatos com dpsi12 valido")
    return out


def read_root_files():
    """Corre no subprocesso. Importa so o ROOT e grava os .npz."""
    import ROOT
    ROOT.gROOT.SetBatch(True)
    if ROOT.IsImplicitMTEnabled():     # a leitura junta os jatos por ordem, tem de ser sequencial
        ROOT.DisableImplicitMT()
    ROOT.gInterpreter.Declare(COLLECTOR)
    for key, (npz, src) in CACHE.items():
        if src is None:
            continue
        out = read_sample(ROOT, src)
        np.savez(npz, _source=np.array(src), **out)
        print(f"   gravado em {npz}")


def cache_ok(npz, src):
    if not Path(npz).exists():
        return False
    with np.load(npz) as f:
        return "_source" in f.files and str(f["_source"]) == src


def load_cache(npz):
    with np.load(npz) as f:
        return pd.DataFrame({k: f[k] for k in f.files if k != "_source"})


if len(sys.argv) > 1 and sys.argv[1] == "read":
    read_root_files()
    sys.exit(0)

# ------------------------------- leitura -------------------------------------
need = [k for k, (npz, src) in CACHE.items() if src is not None and (REREAD or not cache_ok(npz, src))]
if need:
    print("A ler os ficheiros ROOT num processo separado ...")
    subprocess.run([sys.executable, os.path.abspath(__file__), "read"], check=True)
else:
    print("A usar as variaveis ja lidas em threshold_networks_*.npz (REREAD = True para reler)")
sim = load_cache(CACHE["sim"][0])
data = load_cache(CACHE["data"][0]) if DATA_FILE else None

# ------------------------------- redes (so TensorFlow) -----------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from joblib import load
import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel("ERROR")


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
    return model.predict(np.asarray(X, dtype="float32"), batch_size=BATCH, verbose=0)


prep1 = load(STAGE1_DIR / "stage1_preprocess.joblib")
m1 = keras.models.load_model(STAGE1_DIR / "stage1_unmatched_vs_matched_dnn_best.keras", compile=False)
pp2 = load(STAGE2_DIR / "qqbar_binary_preprocessor.joblib")
m2 = keras.models.load_model(STAGE2_DIR / "qqbar_binary_dnn_best.keras", compile=False)
prep3 = load(THREE_DIR / "stage2_matched_3class_preprocessor.joblib")
m3 = keras.models.load_model(THREE_DIR / "stage2_matched_3class_dnn_best.keras", compile=False)
prep4 = load(FOUR_DIR / "four_class_preprocessor.joblib")
m4 = keras.models.load_model(FOUR_DIR / "four_class_dnn_best.keras", compile=False)
if "Jet_pt" in feature_names(prep4):
    print("ATENCAO: a rede de 4 classes usa Jet_pt como input "
          "(treinada em 800-1000 GeV, aplicada aqui a outra gama).")


def score_stage1(d):
    p = predict(m1, prep1.transform(clean(d[feature_names(prep1)])))
    return p[:, 1] if (p.ndim == 2 and p.shape[1] == 2) else p.ravel()


def score_two(d):
    d2 = engineer_stage2(d)
    X = pp2["scaler"].transform(pp2["imputer"].transform(clean(d2[pp2["features"]])))
    return predict(m2, X).ravel()


def score_three(d):
    return predict(m3, prep3.transform(clean(d[feature_names(prep3)])))[:, 0]   # canal 1 -> classe 0


def score_four(d):
    return predict(m4, prep4.transform(clean(d[feature_names(prep4)])))[:, 1]   # classe 1 = qqbar


def add_scores(d, tag):
    """Calcula os scores. As redes do segundo passo so correm nos jatos que
    passam o stage 1 (e a de 4 classes em todos, se for usada sozinha)."""
    d["matched"] = score_stage1(d)
    s1 = d["matched"].to_numpy() > STAGE1_CUT
    for name, fn in [("two", score_two), ("three", score_three), ("four", score_four)]:
        rows = s1 if (name != "four" or FOUR_CLASS_WITH_STAGE1) else np.ones(len(d), bool)
        s = np.full(len(d), np.nan)
        if rows.any():
            s[rows] = fn(d.loc[rows])
        d[name] = s
    print(f"{tag}: {len(d):,} jatos, {s1.sum():,} com matched score > {STAGE1_CUT}")
    # verificacao contra os scores guardados pelo apply_ML.py
    for mine, stored in [("matched", "stored_matched"), ("two", "stored_qq")]:
        if stored in d:
            ok = np.isfinite(d[mine]) & (d[stored] >= 0)
            if ok.any():
                diff = np.abs(d.loc[ok, mine] - d.loc[ok, stored]).max()
                flag = "" if diff < 1e-3 else "   <-- ATENCAO: diferentes do ficheiro"
                print(f"   {mine:8s} max |calculado - guardado| = {diff:.2e}{flag}")
    return d

# ------------------------------- scan ----------------------------------------
def mean_err(c):
    n = len(c)
    if n < 2:
        return n, float("nan"), float("nan")
    return n, c.mean(), c.std(ddof=1) / math.sqrt(n)


def selection(d, net, t):
    passed = d[net].to_numpy() > t                       # NaN da sempre False
    if net == "four" and not FOUR_CLASS_WITH_STAGE1:
        return passed
    return passed & (d["matched"].to_numpy() > STAGE1_CUT)


def scan(sim, data):
    c2_sim = np.cos(2.0 * sim["dpsi12"].to_numpy())
    c2_dat = np.cos(2.0 * data["dpsi12"].to_numpy()) if data is not None else None
    lab = sim["label"].to_numpy() if "label" in sim else None
    n_qq_all = (lab == 1).sum() if lab is not None else 0
    rows = []
    for net in NETWORKS:
        for t in THRESHOLDS:
            m = selection(sim, net, t)
            n, mu, er = mean_err(c2_sim[m])
            pur = (lab[m] == 1).mean() if (lab is not None and n > 0) else float("nan")
            eff = (lab[m] == 1).sum() / n_qq_all if n_qq_all > 0 else float("nan")
            row = dict(network=net, threshold=t, n_sim=n, purity=pur, efficiency=eff,
                       cos_sim=mu, err_sim=er)
            if data is not None:
                nd, mud, erd = mean_err(c2_dat[selection(data, net, t)])
                row.update(n_data=nd, cos_data=mud, err_data=erd)
            rows.append(row)
    return pd.DataFrame(rows)


def references(sim):
    """Valores dos jatos com etiqueta depois do stage 1 apenas."""
    if "label" not in sim:
        return {}
    s1 = sim["matched"].to_numpy() > STAGE1_CUT
    lab = sim["label"].to_numpy()
    c2 = np.cos(2.0 * sim["dpsi12"].to_numpy())
    return {"all": mean_err(c2[s1]),
            "qq":  mean_err(c2[s1 & (lab == 1)]),
            "bkg": mean_err(c2[s1 & ((lab == 2) | (lab == 3))])}

# ------------------------------- main ----------------------------------------
sim = add_scores(sim, "simulacao")
data = add_scores(data, "dados") if data is not None else None
res = scan(sim, data)
ref = references(sim)
res.to_csv("threshold_networks.csv", index=False)

print()
for net in NETWORKS:
    print(NAMES[net])
    head = f"{'thr':>5s} {'N sim':>10s} {'pureza':>7s} {'efic.':>7s} {'<cos> sim':>18s}"
    if data is not None:
        head += f" {'N dados':>8s} {'<cos> dados':>18s}"
    print(head)
    for _, r in res[(res.network == net) & (res.n_sim > 0)].iterrows():
        line = (f"{r.threshold:5.2f} {r.n_sim:10.0f} {r.purity:7.3f} {r.efficiency:7.4f}"
                f"   {r.cos_sim:+.4f} +- {r.err_sim:.4f}")
        if data is not None:
            line += f" {r.n_data:8.0f}   {r.cos_data:+.4f} +- {r.err_data:.4f}"
        print(line)
    print()
for key, txt in [("all", "todos os jatos"), ("qq", "etiqueta g->qqbar"),
                 ("bkg", "etiquetas q->qg + g->gg")]:
    if key in ref:
        n, mu, er = ref[key]
        print(f"Depois do stage 1, {txt:24s}: N = {n:9d}   <cos> = {mu:+.4f} +- {er:.4f}")

# ------------------------------- figura --------------------------------------
import matplotlib.pyplot as plt

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
offsets = dict(zip(NETWORKS, [-0.008, 0.0, 0.008]))
for net in NETWORKS:
    color, mk = STYLE[net]
    r = res[res.network == net]
    ok = (r.n_sim >= MIN_JETS_SIM).to_numpy()
    for ax, x in [(a1, r.threshold.to_numpy() + offsets[net]), (a2, r.purity.to_numpy())]:
        ax.errorbar(x[ok], r.cos_sim[ok], yerr=r.err_sim[ok], color=color, marker=mk,
                    ls="-", ms=5, capsize=3, label=f"{NAMES[net]}, simulation")
    if data is not None:
        okd = (r.n_data >= MIN_JETS_DATA).to_numpy() & ok
        for ax, x in [(a1, r.threshold.to_numpy() + offsets[net] + 0.004),
                      (a2, r.purity.to_numpy())]:
            ax.errorbar(x[okd], r.cos_data[okd], yerr=r.err_data[okd], color=color,
                        marker=mk, mfc="white", ls="none", ms=5, capsize=3,
                        label=f"{NAMES[net]}, data")

for key, color, ls, txt in [("qq", "k", ":", r"$g\to q\bar{q}$ label, after stage 1"),
                            ("bkg", "grey", ":", r"$q\to qg$ + $g\to gg$ labels, after stage 1"),
                            ("all", "grey", "--", "All jets after stage 1")]:
    if key in ref:
        for ax in (a1, a2):
            ax.axhline(ref[key][1], color=color, ls=ls, lw=1.2,
                       label=txt if ax is a1 else None)

a1.set_xlabel(r"Threshold on the $g\to q\bar{q}$ score")
a2.set_xlabel(r"Purity in $g\to q\bar{q}$ (simulation)")
for ax in (a1, a2):
    ax.set_ylabel(r"$\langle\cos 2\Delta\psi_{12}\rangle$")
    ax.grid(alpha=0.3)
a1.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig("threshold_networks.png", dpi=160)
fig.savefig("threshold_networks.pdf")
print("\nFigura gravada em threshold_networks.png e threshold_networks.pdf")
print("Tabela gravada em threshold_networks.csv")
