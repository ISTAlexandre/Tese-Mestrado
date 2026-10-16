import math

import numpy as np

DATA_FILE = "../root/merged.root"
SIM_FILE = "../root_ML/merged_ML-600to800.root"
NB, ZMIN, ZMAX = 16, 0.1, 0.5          # bins de 0.025 em z1 e em z2
MIN_ENTRIES = 50                       # bins com menos jatos ficam em branco nas razoes
RATIO_MAX = 1.6                        # escala das razoes de 1/RATIO_MAX a RATIO_MAX

# (nome, condicao na etiqueta l, legenda)
SIM_CATS = [("all",   None,     "Simulation, all jets"),
            ("qq",    "l == 1", r"$g\to q\bar{q}$ label"),
            ("gg",    "l == 3", r"$g\to gg$ label"),
            ("qg",    "l == 2", r"$q\to qg$ label"),
            ("unm",   "l == 0", "Unmatched"),
            ("nolab", "l < 0",  "No label"),
            ("rest",  "l == 4", "Rest")]
DATA_CATS = [("all", None, "Data, all jets")]


def per_jet(what, cond):
    """z1 ou z2 de cada jato com dpsi12 valido que satisfaz cond (em l)."""
    expr = {"z1": "lund_z_sd[i][k1]", "z2": "lund_z_secondary_sd[i][k2]"}[what]
    lab = f"&& ({cond})" if cond else ""
    return f"""
    ROOT::RVecD out;
    const size_t n = std::min({{lund_psi12_sd.size(), lund_secondary_idx_sd.size(),
                               lund_max_kt_sd.size(), lund_max_kt_secondary_sd.size(),
                               lund_z_sd.size(), lund_z_secondary_sd.size()}});
    for (size_t i = 0; i < n; ++i) {{
        const double p = lund_psi12_sd[i];
        const int l = lund_secondary_idx_sd[i];
        const int k1 = lund_max_kt_sd[i];
        const int k2 = lund_max_kt_secondary_sd[i];
        (void) l;
        if (!(p == p) || k1 < 0 || k2 < 0) continue;
        if ((size_t) k1 >= lund_z_sd[i].size() || (size_t) k2 >= lund_z_secondary_sd[i].size()) continue;
        if (true {lab}) out.push_back({expr});
    }}
    return out;
    """


def compute():
    """Le os dois ficheiros e devolve, por amostra e categoria, o histograma 2D
    (contagens) e as estatisticas."""
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.EnableImplicitMT()

    booked = {}
    for sample, path, cats in [("data", DATA_FILE, DATA_CATS), ("sim", SIM_FILE, SIM_CATS)]:
        df = ROOT.RDataFrame("jetTree", path)
        for c, cond, _ in cats:
            df = (df.Define(f"z1_{c}", per_jet("z1", cond))
                    .Define(f"z2_{c}", per_jet("z2", cond))
                    .Define(f"z12_{c}", f"z1_{c} * z2_{c}")
                    .Define(f"n_{c}", f"(double) z1_{c}.size()"))
        for c, _, _ in cats:
            booked[(sample, c)] = dict(
                h=df.Histo2D((f"h_{sample}_{c}", "", NB, ZMIN, ZMAX, NB, ZMIN, ZMAX),
                             f"z1_{c}", f"z2_{c}"),
                n=df.Sum(f"n_{c}"),
                m1=df.Mean(f"z1_{c}"), m2=df.Mean(f"z2_{c}"), m12=df.Mean(f"z12_{c}"),
                s1=df.StdDev(f"z1_{c}"), s2=df.StdDev(f"z2_{c}"))

    res = {}
    for (sample, c), b in booked.items():
        h = b["h"].GetValue()
        counts = np.array([[h.GetBinContent(ix, iy) for iy in range(1, NB + 1)]
                           for ix in range(1, NB + 1)])          # [z1, z2]
        n = float(b["n"].GetValue())
        if n > 1:
            m1, m2, m12 = b["m1"].GetValue(), b["m2"].GetValue(), b["m12"].GetValue()
            s1, s2 = b["s1"].GetValue(), b["s2"].GetValue()
            corr = (m12 - m1 * m2) / (s1 * s2) if s1 > 0 and s2 > 0 else float("nan")
        else:
            m1 = m2 = corr = float("nan")
        res.setdefault(sample, {})[c] = dict(counts=counts, n=n, m1=m1, m2=m2, corr=corr)
    return res


def print_table(res):
    n_sim = res["sim"]["all"]["n"]
    print(f"\n{'Amostra':7s} {'Categoria':12s} {'Jatos':>11s} {'Fracao':>8s}"
          f" {'<z1>':>7s} {'<z2>':>7s} {'corr(z1,z2)':>12s}")
    for sample, cats in [("data", DATA_CATS), ("sim", SIM_CATS)]:
        for c, _, _ in cats:
            r = res[sample][c]
            frac = f"{100 * r['n'] / n_sim:7.1f}%" if sample == "sim" else f"{'':>8s}"
            print(f"{sample:7s} {c:12s} {r['n']:11.0f} {frac} {r['m1']:7.3f} {r['m2']:7.3f}"
                  f" {r['corr']:+12.3f}")


def density(counts):
    bw = (ZMAX - ZMIN) / NB
    tot = counts.sum()
    return counts / (tot * bw * bw) if tot > 0 else counts * 0.0


def fig_channels(res):
    import matplotlib.pyplot as plt
    panels = [("data", "all", "Data, all jets"),
              ("sim", "all", "Simulation, all jets"),
              ("sim", "qq", r"$g\to q\bar{q}$ label"),
              ("sim", "gg", r"$g\to gg$ label"),
              ("sim", "qg", r"$q\to qg$ label"),
              ("sim", "unm", "Unmatched")]
    dens = [density(res[s][c]["counts"]) for s, c, _ in panels]
    vmax = max(d.max() for d in dens)
    edges = np.linspace(ZMIN, ZMAX, NB + 1)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, (s, c, title), d in zip(axes.flat, panels, dens):
        dm = np.ma.masked_where(d <= 0, d)
        mesh = ax.pcolormesh(edges, edges, dm.T, cmap="viridis", vmin=0, vmax=vmax)
        r = res[s][c]
        ax.set_title(title, fontsize=11)
        ax.text(0.03, 0.97, f"{r['n']:,.0f} jets\n" + rf"$\langle z_1\rangle = {r['m1']:.3f}$"
                + "\n" + rf"$\langle z_2\rangle = {r['m2']:.3f}$",
                transform=ax.transAxes, va="top", fontsize=8.5, color="w")
        ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlabel(r"$z_1$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$z_2$")
    cb = fig.colorbar(mesh, ax=axes, shrink=0.9)
    cb.set_label(r"$\frac{1}{N}\,\frac{d^2N}{dz_1\,dz_2}$")
    fig.savefig("zplane_channels.png", dpi=160)
    plt.close(fig)


def fig_ratios(res):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    ref = density(res["sim"]["all"]["counts"])
    panels = [("data", "all", "Data / simulation"),
              ("sim", "qq", r"$g\to q\bar{q}$ / all"),
              ("sim", "gg", r"$g\to gg$ / all"),
              ("sim", "qg", r"$q\to qg$ / all")]
    edges = np.linspace(ZMIN, ZMAX, NB + 1)
    norm = LogNorm(vmin=1 / RATIO_MAX, vmax=RATIO_MAX)    # simetrica em torno de 1

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, (s, c, title) in zip(axes, panels):
        counts = res[s][c]["counts"]
        d = density(counts)
        ratio = np.divide(d, ref, out=np.full_like(d, np.nan), where=ref > 0)
        ratio = np.where(counts >= MIN_ENTRIES, ratio, np.nan)
        ratio = np.clip(ratio, 1 / RATIO_MAX, RATIO_MAX)
        mesh = ax.pcolormesh(edges, edges, np.ma.masked_invalid(ratio).T, cmap="RdBu_r", norm=norm)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"$z_1$")
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"$z_2$")
    cb = fig.colorbar(mesh, ax=axes, shrink=0.9)
    cb.set_label("Ratio of normalised distributions")
    ticks = [1 / RATIO_MAX, 1 / 1.25, 1.0, 1.25, RATIO_MAX]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.2f}" for t in ticks])
    cb.minorticks_off()
    fig.savefig("zplane_ratios.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    results = compute()
    print_table(results)
    fig_channels(results)
    fig_ratios(results)
    print("\nFiguras gravadas: zplane_channels e zplane_ratios")