import ROOT
import matplotlib.pyplot as plt
import numpy as np

ROOT.gROOT.SetBatch(True)
ROOT.EnableImplicitMT()

SIM_FILE = "../root_ML/merged_ML-600to800.root"
STAGE1_CUT = 0.8
THR_PLOT = [0.45, 0.60]          # thresholds mostrados no painel esquerdo
NB_Z, Z_MIN, Z_MAX = 20, 0.1, 0.5
NB_PROF = 8

def per_jet(what, thr=None, label_cond=None):
    """Grandeza por jato (z1, z2 ou cos2) para os jatos selecionados."""
    expr = {
        "z1":  "lund_z_sd[i][k1]",
        "z2":  "lund_z_secondary_sd[i][k2]",
        "c2":  "std::cos(2.0 * lund_psi12_sd[i])",
    }[what]
    qq_cut = f"&& lund_ML_qq[i] > {thr}" if thr is not None else ""
    lab = f"&& ({label_cond})" if label_cond else ""
    return f"""
    ROOT::RVecD out;
    const size_t n = std::min({{lund_psi12_sd.size(), lund_secondary_idx_sd.size(),
                               lund_ML_qq.size(), lund_ML_matched.size(),
                               lund_max_kt_sd.size(), lund_max_kt_secondary_sd.size()}});
    for (size_t i = 0; i < n; ++i) {{
        const double p = lund_psi12_sd[i];
        const int l = lund_secondary_idx_sd[i];
        const int k1 = lund_max_kt_sd[i];
        const int k2 = lund_max_kt_secondary_sd[i];
        (void) l;
        if (!(p == p) || k1 < 0 || k2 < 0) continue;
        if ((size_t) k1 >= lund_z_sd[i].size() || (size_t) k2 >= lund_z_secondary_sd[i].size()) continue;
        if (lund_ML_matched[i] > {STAGE1_CUT} {qq_cut} {lab})
            out.push_back({expr});
    }}
    return out;
    """

df = ROOT.RDataFrame("jetTree", SIM_FILE)

# populacoes: (nome, threshold, condicao na etiqueta)
pops = [("qq_s1",  None, "l == 1"),
        ("bkg_s1", None, "l == 2 || l == 3"),
        ("all_s1", None, None)]
def tag(t):
    return f"all_{int(round(t * 100)):03d}"

pops += [(tag(t), t, None) for t in [0.30, 0.45, 0.60]]

hz, means = {}, {}
for name, thr, cond in pops:
    df = (df.Define(f"z1_{name}", per_jet("z1", thr, cond))
            .Define(f"z2_{name}", per_jet("z2", thr, cond))
            .Define(f"c2_{name}", per_jet("c2", thr, cond)))
    hz[name] = df.Histo1D((f"hz_{name}", "", NB_Z, Z_MIN, Z_MAX), f"z2_{name}")
    means[name] = (df.Mean(f"z1_{name}"), df.Mean(f"z2_{name}"))

prof = {}
for name in ["qq_s1", "bkg_s1"]:
    prof[name] = df.Profile1D((f"p_{name}", "", NB_PROF, Z_MIN, Z_MAX), f"z2_{name}", f"c2_{name}")

print(f"{'Selecao':12s} {'<z1>':>7s} {'<z2>':>7s}")
for name, _, _ in pops:
    print(f"{name:12s} {means[name][0].GetValue():7.3f} {means[name][1].GetValue():7.3f}")

def hist_arrays(h):
    h = h.GetValue()
    x = np.array([h.GetBinCenter(b) for b in range(1, h.GetNbinsX() + 1)])
    y = np.array([h.GetBinContent(b) for b in range(1, h.GetNbinsX() + 1)])
    return x, y / y.sum() if y.sum() > 0 else y

print("\n<cos 2 dpsi12> em funcao de z2 (depois do stage 1):")
print(f"{'z2':>6s} {'g->qqbar':>18s} {'q->qg + g->gg':>18s}")
pq, pb = prof["qq_s1"].GetValue(), prof["bkg_s1"].GetValue()
for b in range(1, NB_PROF + 1):
    print(f"{pq.GetBinCenter(b):6.3f} {pq.GetBinContent(b):+9.4f} +- {pq.GetBinError(b):.4f}"
          f" {pb.GetBinContent(b):+9.4f} +- {pb.GetBinError(b):.4f}")

# ---------------- figura ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
styles = {"qq_s1":  ("C0", "-",  r"$g\to q\bar{q}$ label, after stage 1"),
          "bkg_s1": ("grey", ":", r"$q\to qg$ + $g\to gg$ labels, after stage 1")}
for t, col in zip(THR_PLOT, ["C1", "C3"]):
    styles[tag(t)] = (col, "--", f"All selected jets, threshold {t:.2f}")
for name, (color, ls, label) in styles.items():
    x, y = hist_arrays(hz[name])
    a1.step(x, y, where="mid", color=color, ls=ls, lw=2, label=label)
a1.set_xlabel(r"$z_2$")
a1.set_ylabel("Fraction of jets")
a1.grid(alpha=0.3)
a1.legend(fontsize=8.5)

for name, color, marker, label in [("qq_s1", "C0", "o", r"$g\to q\bar{q}$ label"),
                                   ("bkg_s1", "grey", "^", r"$q\to qg$ + $g\to gg$ labels")]:
    p = prof[name].GetValue()
    x = [p.GetBinCenter(b) for b in range(1, NB_PROF + 1)]
    y = [p.GetBinContent(b) for b in range(1, NB_PROF + 1)]
    e = [p.GetBinError(b) for b in range(1, NB_PROF + 1)]
    a2.errorbar(x, y, yerr=e, fmt=marker + "-", color=color, ms=5, capsize=3, label=label)
a2.set_xlabel(r"$z_2$")
a2.set_ylabel(r"$\langle\cos 2\Delta\psi_{12}\rangle$")
a2.grid(alpha=0.3)
a2.legend(fontsize=9)
fig.tight_layout()
fig.savefig("z_kinematics.png", dpi=160)
fig.savefig("z_kinematics.pdf")
print("\nFigura gravada em z_kinematics.png e z_kinematics.pdf")
