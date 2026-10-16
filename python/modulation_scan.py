import math
import ROOT
import matplotlib.pyplot as plt

ROOT.gROOT.SetBatch(True)
ROOT.EnableImplicitMT()

SIM_FILE  = "../root_ML/merged_ML-600to800.root"
DATA_FILE = "../root/merged.root"
STAGE1_CUT = 0.8
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

def selected(thr, label_cond=None, has_labels=True):
    """cos(2 dpsi12) dos jatos selecionados (vetor por evento)."""
    lab = "lund_secondary_idx_sd.size()" if has_labels else "lund_psi12_sd.size()"
    cond_lab = f"&& ({label_cond})" if label_cond else ""
    get_l = "const int l = lund_secondary_idx_sd[i];" if has_labels else "const int l = -1;"
    qq_cut = f"&& lund_ML_qq[i] > {thr}" if thr is not None else ""
    return f"""
    ROOT::RVecD out;
    const size_t n = std::min({{lund_psi12_sd.size(), {lab},
                               lund_ML_qq.size(), lund_ML_matched.size()}});
    for (size_t i = 0; i < n; ++i) {{
        {get_l}
        (void) l;
        const double p = lund_psi12_sd[i];
        if (p == p && lund_ML_matched[i] > {STAGE1_CUT} {qq_cut} {cond_lab})
            out.push_back(std::cos(2.0 * p));
    }}
    return out;
    """

def book(df, col):
    df = df.Define(f"n_{col}", f"(double) {col}.size()")
    return df, dict(mean=df.Mean(col), std=df.StdDev(col), n=df.Sum(f"n_{col}"))

def value(b):
    n = float(b["n"].GetValue())
    if n == 0:
        return 0.0, float("nan"), float("nan")
    m = float(b["mean"].GetValue())
    e = float(b["std"].GetValue()) / math.sqrt(n) if n > 1 else float("nan")
    return n, m, e

# ---------------- simulacao ----------------
sim = ROOT.RDataFrame("jetTree", SIM_FILE)
sb = {}
cats = {"all": None, "lab": "l >= 0", "qq": "l == 1", "oth": "l == 2 || l == 3"}
for i, thr in enumerate([None] + THRESHOLDS):
    for c, cond in cats.items():
        col = f"s_{c}_{i}"
        sim = sim.Define(col, selected(thr, cond))
        sim, sb[(i, c)] = book(sim, col)
sim = sim.Define("s_bkg_ref", selected(None, "l == 2 || l == 3"))
sim, sb["bkg_ref"] = book(sim, "s_bkg_ref")

# ---------------- dados ----------------
dat = ROOT.RDataFrame("jetTree", DATA_FILE)
db = {}
for i, thr in enumerate([None] + THRESHOLDS):
    col = f"d_all_{i}"
    dat = dat.Define(col, selected(thr, None, has_labels=False))
    dat, db[i] = book(dat, col)

# ---------------- resultados ----------------
_, qq_ref, qq_ref_e = value(sb[(0, "qq")])
_, bkg_ref, bkg_ref_e = value(sb["bkg_ref"])
print(f"Referencias (matched > {STAGE1_CUT}):  qq = {qq_ref:+.4f} +- {qq_ref_e:.4f}"
      f"   q->qg + g->gg = {bkg_ref:+.4f} +- {bkg_ref_e:.4f}\n")

rows = []
comp = []
print(f"{'thr':>5s} {'N sim':>9s} {'P_low':>6s} {'P_up':>6s} {'<c2> all':>16s}"
      f" {'<c2> qq':>16s} {'<c2> outros':>16s} {'N dados':>8s} {'<c2> dados':>16s}")
for i, thr in enumerate(THRESHOLDS, start=1):
    n_all, m_all, e_all = value(sb[(i, "all")])
    n_lab, _, _ = value(sb[(i, "lab")])
    n_qq, m_qq, e_qq = value(sb[(i, "qq")])
    n_oth, m_oth, e_oth = value(sb[(i, "oth")])
    n_d, m_d, e_d = value(db[i])
    p_low = n_qq / n_all if n_all else float("nan")
    p_up = n_qq / n_lab if n_lab else float("nan")
    rows.append((thr, m_all, e_all, m_d, e_d))
    comp.append((thr, m_all, e_all, m_qq, e_qq, m_oth, e_oth, p_low, p_up))
    print(f"{thr:5.2f} {n_all:9.0f} {p_low:6.3f} {p_up:6.3f} {m_all:+8.4f} +- {e_all:.4f}"
          f" {m_qq:+8.4f} +- {e_qq:.4f} {m_oth:+8.4f} +- {e_oth:.4f}"
          f" {n_d:8.0f} {m_d:+8.4f} +- {e_d:.4f}")

print("\nLinhas LaTeX para tab:modulation-sim:")
for i, thr in enumerate(THRESHOLDS, start=1):
    n_all, m_all, e_all = value(sb[(i, "all")])
    n_lab, _, _ = value(sb[(i, "lab")])
    n_qq, m_qq, e_qq = value(sb[(i, "qq")])
    _, m_oth, e_oth = value(sb[(i, "oth")])
    print(f"    ${thr:.2f}$ & ${int(n_all):,}$ & ${n_qq/n_all:.2f}$--${n_qq/n_lab:.2f}$ & "
          f"${m_all:+.3f} \\pm {e_all:.3f}$ & ${m_qq:+.3f} \\pm {e_qq:.3f}$ & "
          f"${m_oth:+.3f} \\pm {e_oth:.3f}$ \\\\".replace(",", "\\,"))

# ---------------- figura ----------------
thr = [r[0] for r in rows]
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.errorbar(thr, [r[1] for r in rows], yerr=[r[2] for r in rows], fmt="s-",
            color="C3", ms=5, capsize=3, label="Simulation, 600 < $\\hat{p}_T$ < 800 GeV")
ax.errorbar([t + 0.004 for t in thr], [r[3] for r in rows], yerr=[r[4] for r in rows],
            fmt="o", color="black", ms=5, capsize=3, label="Data")
ax.axhline(qq_ref, color="C0", ls="--", lw=1.5,
           label=f"$g\\to q\\bar{{q}}$ label, after stage 1 ({qq_ref:+.3f})")
ax.axhline(bkg_ref, color="grey", ls=":", lw=1.5,
           label=f"$q\\to qg$ + $g\\to gg$ labels, after stage 1 ({bkg_ref:+.3f})")
ax.set_xlabel(r"Threshold on the $g\to q\bar{q}$ score")
ax.set_ylabel(r"$\langle\cos 2\Delta\psi_{12}\rangle$")
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="lower left")
fig.tight_layout()
fig.savefig("modulation_threshold.png", dpi=160)
print("\nFigura gravada em modulation_threshold.png")

# ---------------- figura 2: componentes na simulacao ----------------
t = [c[0] for c in comp]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.06))
a1.errorbar(t, [c[1] for c in comp], yerr=[c[2] for c in comp], fmt="s-", color="C3",
            ms=5, capsize=3, label="All selected jets")
a1.errorbar(t, [c[3] for c in comp], yerr=[c[4] for c in comp], fmt="o-", color="C0",
            ms=5, capsize=3, label=r"Selected jets labelled $g\to q\bar{q}$")
a1.errorbar(t, [c[5] for c in comp], yerr=[c[6] for c in comp], fmt="^-", color="C2",
            ms=5, capsize=3, label=r"Selected jets labelled $q\to qg$ or $g\to gg$")
a1.axhline(qq_ref, color="C0", ls="--", lw=1.2,
           label=rf"$g\to q\bar{{q}}$ label, after stage 1 ({qq_ref:+.3f})")
a1.axhline(bkg_ref, color="grey", ls=":", lw=1.5,
           label=rf"$q\to qg$ + $g\to gg$ labels, after stage 1 ({bkg_ref:+.3f})")
a1.set_ylabel(r"$\langle\cos 2\Delta\psi_{12}\rangle$")
a1.grid(alpha=0.3)
a1.legend(fontsize=8.5, loc="lower left")

a2.fill_between(t, [c[7] for c in comp], [c[8] for c in comp], color="C0", alpha=0.25,
                label="Purity range")
a2.plot(t, [c[7] for c in comp], "o-", color="C0", ms=4, label="Unlabelled jets as background")
a2.plot(t, [c[8] for c in comp], "s--", color="C0", ms=4, label="Unlabelled jets excluded")
a2.set_xlabel(r"Threshold on the $g\to q\bar{q}$ score")
a2.set_ylabel(r"$g\to q\bar{q}$ purity")
a2.set_ylim(0, 1)
a2.grid(alpha=0.3)
a2.legend(fontsize=8.5, loc="upper left")
fig.savefig("modulation_components.png", dpi=160, bbox_inches="tight")
print("Figura gravada em modulation_components.png")
