import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from numba import njit

# ------------------------------- configuracao --------------------------------
SAMPLES = {
    "data": ("../root/merged.root", False),
    "sim":  ("../root_ML/merged_ML-600to800.root", True),
}
TREE = "jetTree"

MAX_EVENTS = {"data": None, "sim": None}            # None = todos (teste: p.ex. 2000)
MAX_EVENTS_INCL = {"data": None, "sim": None}    # eventos usados em incl, lab_*, s1_*

STAGE1_CUT = 0.8
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

NBINS = 100                      # bins em psi, de -pi a pi (como no eeec_numba.py)
RL_MIN, RL_MAX = math.sqrt(0.1), 1.0
RS_MIN, RS_MAX = 0.01, 0.1

N_WORKERS = min(8, os.cpu_count() or 1)
CHUNKS_PER_WORKER = 4
RESULTS_FILE = "eeec_results.npz"
RECOMPUTE = True

PLOT_REBIN = 1                   # 2 junta pares de bins nas figuras
SHOW_COS_CURVE = True            # curva 1 + 2<cos 2psi> cos 2psi nas figuras
SEL_FIELDS = ["H", "H2", "NJ", "NT", "SW", "SS", "SSS", "SWW", "SSW"]


def tag(t, base="sel"):
    return f"{base}_{int(round(t * 100)):03d}"


# ------------------------------- EEEC de um jato -----------------------------
@njit(cache=True)
def eeec_jet(pts, etas, phis, jet_pt, hist, rl_min, rl_max, rs_min, rs_max):
    """Enche hist (zerado aqui) com o EEEC do jato.
    Devolve (soma dos pesos, soma de peso * cos 2psi, numero de tripletos)."""
    nb = hist.shape[0]
    for b in range(nb):
        hist[b] = 0.0
    n = pts.shape[0]
    if n < 3 or jet_pt <= 0.0:
        return 0.0, 0.0, 0

    width = 2.0 * np.pi / nb
    inv_pt3 = 1.0 / (jet_pt * jet_pt * jet_pt)

    px = np.empty(n)
    py = np.empty(n)
    pz = np.empty(n)
    for i in range(n):
        px[i] = pts[i] * np.cos(phis[i])
        py[i] = pts[i] * np.sin(phis[i])
        pz[i] = pts[i] * np.sinh(etas[i])

    # distancia no plano (eta, phi)
    dr = np.empty((n, n))
    for i in range(n):
        dr[i, i] = 0.0
        for j in range(i + 1, n):
            dphi = phis[i] - phis[j]
            while dphi > np.pi:
                dphi -= 2.0 * np.pi
            while dphi <= -np.pi:
                dphi += 2.0 * np.pi
            deta = etas[i] - etas[j]
            d = np.sqrt(deta * deta + dphi * dphi)
            dr[i, j] = d
            dr[j, i] = d

    w_sum = 0.0
    s_sum = 0.0
    ntrip = 0
    for i in range(n):
        for j in range(i + 1, n):
            dij = dr[i, j]
            if not (dij < rl_max):          # o maior lado tem de ser < rl_max
                continue
            for k in range(j + 1, n):
                dik = dr[i, k]
                djk = dr[j, k]
                # par mais proximo (mesma ordem de desempate do eeec_numba.py)
                if dij <= dik and dij <= djk:
                    rs = dij
                    a, b, c = i, j, k
                elif dik <= djk:
                    rs = dik
                    a, b, c = i, k, j
                else:
                    rs = djk
                    a, b, c = j, k, i
                rl = max(dij, max(dik, djk))
                if not (rl_min < rl < rl_max):
                    continue
                if not (rs_min < rs < rs_max):
                    continue

                # n1 = p_a x p_b
                n1x = py[a] * pz[b] - pz[a] * py[b]
                n1y = pz[a] * px[b] - px[a] * pz[b]
                n1z = px[a] * py[b] - py[a] * px[b]
                # p_ab = p_a + p_b
                abx = px[a] + px[b]
                aby = py[a] + py[b]
                abz = pz[a] + pz[b]
                # n2 = p_ab x p_c
                n2x = aby * pz[c] - abz * py[c]
                n2y = abz * px[c] - abx * pz[c]
                n2z = abx * py[c] - aby * px[c]

                n1n = np.sqrt(n1x * n1x + n1y * n1y + n1z * n1z)
                n2n = np.sqrt(n2x * n2x + n2y * n2y + n2z * n2z)
                abn = np.sqrt(abx * abx + aby * aby + abz * abz)
                if n1n <= 0.0 or n2n <= 0.0 or abn <= 0.0:
                    continue
                n1x /= n1n
                n1y /= n1n
                n1z /= n1n
                n2x /= n2n
                n2y /= n2n
                n2z /= n2n

                cx = n1y * n2z - n1z * n2y
                cy = n1z * n2x - n1x * n2z
                cz = n1x * n2y - n1y * n2x
                num = (abx * cx + aby * cy + abz * cz) / abn
                den = n1x * n2x + n1y * n2y + n1z * n2z
                psi = np.arctan2(num, den)
                if not np.isfinite(psi):
                    continue

                ib = int((psi + np.pi) / width)
                if ib < 0 or ib >= nb:
                    continue
                w = pts[i] * pts[j] * pts[k] * inv_pt3
                hist[ib] += w
                w_sum += w
                s_sum += w * np.cos(2.0 * psi)
                ntrip += 1
    return w_sum, s_sum, ntrip


# ------------------------------- acumuladores --------------------------------
def new_acc():
    return {"H": np.zeros(NBINS), "H2": np.zeros(NBINS), "NJ": 0.0, "NT": 0.0,
            "SW": 0.0, "SS": 0.0, "SSS": 0.0, "SWW": 0.0, "SSW": 0.0}


def add_jet(acc, h, w, s, nt):
    acc["H"] += h
    acc["H2"] += h * h
    acc["NJ"] += 1.0
    acc["NT"] += nt
    acc["SW"] += w
    acc["SS"] += s
    acc["SSS"] += s * s
    acc["SWW"] += w * w
    acc["SSW"] += s * w


def merge(dst, src):
    for cat, a in src.items():
        if cat not in dst:
            dst[cat] = new_acc()
        for k in SEL_FIELDS:
            dst[cat][k] = dst[cat][k] + a[k]


# ------------------------------- um bloco de eventos -------------------------
def process_chunk(path, is_sim, start, stop, incl_stop):
    import ROOT
    f = ROOT.TFile.Open(path, "READ")
    t = f.Get(TREE)
    needed = ["jet_pt", "const_pt", "const_eta", "const_phi",
              "lund_secondary_idx_sd", "lund_ML_qq", "lund_ML_matched"]
    t.SetBranchStatus("*", 0)
    for b in needed:
        if t.GetBranch(b):
            t.SetBranchStatus(b, 1)
    has_ml = bool(t.GetBranch("lund_ML_qq")) and bool(t.GetBranch("lund_ML_matched"))
    has_lab = bool(t.GetBranch("lund_secondary_idx_sd"))

    acc = {}
    hist = np.zeros(NBINS)
    for iev in range(start, stop):
        t.GetEntry(iev)
        in_incl = iev < incl_stop
        nj = len(t.jet_pt)
        labs = t.lund_secondary_idx_sd if has_lab else None
        mats = t.lund_ML_matched if has_ml else None
        qqs = t.lund_ML_qq if has_ml else None
        for ij in range(nj):
            lab = int(labs[ij]) if (has_lab and ij < len(labs)) else -2
            m = float(mats[ij]) if (has_ml and ij < len(mats)) else float("nan")
            q = float(qqs[ij]) if (has_ml and ij < len(qqs)) else float("nan")
            s1 = (m == m) and m > STAGE1_CUT       # NaN falha

            cats = []
            if in_incl:
                cats.append("incl")
                if is_sim:
                    cats.append(f"lab_{lab}")
                    if s1 and lab == 1:
                        cats.append("s1_qq")
                    elif s1 and lab in (2, 3):
                        cats.append("s1_bkg")
            if s1 and q == q:
                for thr in THRESHOLDS:
                    if q > thr:
                        cats.append(tag(thr))
                        if is_sim and lab == 1:
                            cats.append(tag(thr, "selqq"))
            if not cats:
                continue

            pts = np.asarray(t.const_pt[ij], dtype=np.float64)
            etas = np.asarray(t.const_eta[ij], dtype=np.float64)
            phis = np.asarray(t.const_phi[ij], dtype=np.float64)
            ok = pts > 0.0
            w, s, nt = eeec_jet(np.ascontiguousarray(pts[ok]), np.ascontiguousarray(etas[ok]),
                                np.ascontiguousarray(phis[ok]), float(t.jet_pt[ij]), hist,
                                RL_MIN, RL_MAX, RS_MIN, RS_MAX)
            for c in cats:
                if c not in acc:
                    acc[c] = new_acc()
                add_jet(acc[c], hist, w, s, nt)
    f.Close()
    return acc


def compute_sample(name, path, is_sim):
    import ROOT
    f = ROOT.TFile.Open(path, "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Nao consegui abrir {path}")
    n = int(f.Get(TREE).GetEntries())
    f.Close()
    if MAX_EVENTS[name] is not None:
        n = min(n, MAX_EVENTS[name])
    incl_stop = n if MAX_EVENTS_INCL[name] is None else min(n, MAX_EVENTS_INCL[name])

    n_chunks = max(1, N_WORKERS * CHUNKS_PER_WORKER)
    edges = np.linspace(0, n, n_chunks + 1).astype(int)
    total = {}
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
        futs = [ex.submit(process_chunk, path, is_sim, int(a), int(b), incl_stop)
                for a, b in zip(edges[:-1], edges[1:]) if b > a]
        for i, fu in enumerate(as_completed(futs), 1):
            merge(total, fu.result())
            print(f"  {name}: bloco {i}/{len(futs)} ({(time.time() - t0) / 60:.1f} min)")
    print(f"{name}: {n} eventos ({incl_stop} nas categorias inclusivas)")
    return total


def save(results):
    flat = {}
    for s, cats in results.items():
        for c, a in cats.items():
            for k in SEL_FIELDS:
                flat[f"{s}__{c}__{k}"] = np.asarray(a[k])
    np.savez(RESULTS_FILE, **flat)


def load():
    out = {}
    with np.load(RESULTS_FILE) as z:
        for key in z.files:
            s, c, k = key.split("__")
            out.setdefault(s, {}).setdefault(c, {})[k] = z[key]
    return out


# ------------------------------- grandezas finais ----------------------------
def cos2(a):
    """<cos 2 psi> ponderado pelo EEEC e a sua incerteza (jatos independentes)."""
    sw = float(a["SW"])
    if sw <= 0:
        return float("nan"), float("nan")
    r = float(a["SS"]) / sw
    var = float(a["SSS"]) - 2 * r * float(a["SSW"]) + r * r * float(a["SWW"])
    return r, math.sqrt(max(var, 0.0)) / sw


def shape(a, rebin=PLOT_REBIN):
    """Distribuicao normalizada (1/Sigma) dSigma/dpsi e incerteza por bin."""
    h = np.asarray(a["H"]).reshape(-1, rebin).sum(axis=1)
    h2 = np.asarray(a["H2"]).reshape(-1, rebin).sum(axis=1)
    nb = h.size
    width = 2 * np.pi / nb
    x = -np.pi + (np.arange(nb) + 0.5) * width
    tot = h.sum()
    if tot <= 0:
        return x, np.zeros(nb), np.zeros(nb), width
    return x, h / (tot * width), np.sqrt(h2) / (tot * width), width


NAMES = {"incl": "all jets", "lab_-1": "no label", "lab_0": "unmatched",
         "lab_1": "g->qqbar", "lab_2": "q->qg", "lab_3": "g->gg", "lab_4": "rest",
         "s1_qq": "stage 1, g->qqbar", "s1_bkg": "stage 1, q->qg + g->gg"}


def print_table(res):
    print(f"\n{'Amostra':6s} {'Categoria':26s} {'Jatos':>10s} {'Tripletos':>12s}"
          f" {'<cos 2psi>_EEEC':>20s} {'a2/a0':>8s}")
    for s, cats in res.items():
        order = ["incl"] + sorted(c for c in cats if c.startswith("lab_")) + \
                [c for c in ("s1_qq", "s1_bkg") if c in cats] + \
                sorted(c for c in cats if c.startswith("sel_")) + \
                sorted(c for c in cats if c.startswith("selqq_"))
        for c in order:
            if c not in cats:
                continue
            a = cats[c]
            m, e = cos2(a)
            if c in NAMES:
                name = NAMES[c]
            else:
                base, num = c.split("_")
                name = f"{'sel g->qqbar' if base == 'selqq' else 'sel'}, score > {int(num) / 100:.2f}"
            print(f"{s:6s} {name:26s} {float(a['NJ']):10.0f} {float(a['NT']):12.0f}"
                  f"   {m:+.4f} +- {e:.4f} {2 * m:+8.3f}")


# ------------------------------- figuras -------------------------------------
def pi_ticks(ax):
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.set_xlim(-np.pi, np.pi)


def cos_curve(ax, m, color):
    xx = np.linspace(-np.pi, np.pi, 400)
    ax.plot(xx, (1 + 2 * m * np.cos(2 * xx)) / (2 * np.pi), color=color, ls="--", lw=1, alpha=0.7)


def fig_inclusive(res):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    xs, ys, es, w = shape(res["sim"]["incl"])
    xd, yd, ed, _ = shape(res["data"]["incl"])
    ms, _ = cos2(res["sim"]["incl"])
    md, _ = cos2(res["data"]["incl"])
    a1.step(np.append(xs - w / 2, np.pi), np.append(ys, ys[-1]), where="post", color="C3", lw=1.8,
            label=rf"Simulation, $\langle\cos 2\psi\rangle = {ms:+.3f}$")
    a1.errorbar(xd, yd, yerr=ed, fmt="o", color="k", ms=3.5, capsize=0,
                label=rf"Data, $\langle\cos 2\psi\rangle = {md:+.3f}$")
    if SHOW_COS_CURVE:
        cos_curve(a1, ms, "C3")
    a1.set_ylabel(r"$\frac{1}{\Sigma}\,\frac{d\Sigma}{d\psi}$")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=9)
    r = np.divide(yd, ys, out=np.full_like(yd, np.nan), where=ys > 0)
    re = np.divide(ed, ys, out=np.full_like(ed, np.nan), where=ys > 0)
    a2.errorbar(xd, r, yerr=re, fmt="o", color="k", ms=3, capsize=0)
    a2.axhline(1, color="C3", lw=1)
    a2.set_ylabel("Data / sim.")
    a2.set_ylim(0.8, 1.2)
    a2.set_xlabel(r"$\psi$")
    a2.grid(alpha=0.3)
    pi_ticks(a2)
    fig.savefig("eeec_inclusive.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_labels(res):
    import matplotlib.pyplot as plt
    sim = res["sim"]
    curves = [("incl", "k", "-", "All jets"),
              ("lab_1", "C0", "-", r"$g\to q\bar{q}$ label"),
              ("lab_3", "C2", "--", r"$g\to gg$ label"),
              ("lab_2", "C1", "-.", r"$q\to qg$ label")]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    xi, yi, ei, w = shape(sim["incl"])
    for c, col, ls, lab in curves:
        if c not in sim:
            continue
        x, y, e, _ = shape(sim[c])
        m, _ = cos2(sim[c])
        a1.step(np.append(x - w / 2, np.pi), np.append(y, y[-1]), where="post", color=col, ls=ls,
                lw=1.8, label=rf"{lab}, $\langle\cos 2\psi\rangle = {m:+.3f}$")
        if c != "incl":
            r = np.divide(y, yi, out=np.full_like(y, np.nan), where=yi > 0)
            re = np.divide(e, yi, out=np.full_like(e, np.nan), where=yi > 0)
            a2.fill_between(x, r - re, r + re, step="mid", color=col, alpha=0.2, lw=0)
            a2.step(np.append(x - w / 2, np.pi), np.append(r, r[-1]), where="post", color=col, ls=ls, lw=1.5)
    a1.set_ylabel(r"$\frac{1}{\Sigma}\,\frac{d\Sigma}{d\psi}$")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=9)
    a2.axhline(1, color="k", lw=1)
    a2.set_ylabel("Label / all")
    a2.set_ylim(0.7, 1.3)
    a2.set_xlabel(r"$\psi$")
    a2.grid(alpha=0.3)
    pi_ticks(a2)
    fig.savefig("eeec_labels.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_threshold(res):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for c, col, ls, lab in [("s1_qq", "C0", "--", r"$g\to q\bar{q}$ label, after stage 1"),
                            ("s1_bkg", "grey", ":", r"$q\to qg$ + $g\to gg$ labels, after stage 1")]:
        if c in res["sim"]:
            m, _ = cos2(res["sim"][c])
            ax.axhline(m, color=col, ls=ls, lw=1.5, label=f"{lab} ({m:+.3f})")
    for s, col, fmt, ls, dx, lab in [("sim", "C3", "s", "-", 0.0, r"Simulation, $600 < \hat{p}_T < 800$ GeV"),
                                     ("data", "k", "o", "none", 0.004, "Data")]:
        xs, ys, es = [], [], []
        for thr in THRESHOLDS:
            c = tag(thr)
            if c in res.get(s, {}):
                m, e = cos2(res[s][c])
                xs.append(thr + dx)
                ys.append(m)
                es.append(e)
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt=fmt, ls=ls, color=col, ms=5, capsize=3, label=lab)
    ax.set_xlabel(r"Threshold on the $g\to q\bar{q}$ score")
    ax.set_ylabel(r"$\langle\cos 2\psi\rangle_{\mathrm{EEEC}}$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig("eeec_threshold.png", dpi=160)
    plt.close(fig)


# ------------------------------- main ----------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    if RECOMPUTE or not os.path.exists(RESULTS_FILE):
        results = {}
        for name, (path, is_sim) in SAMPLES.items():
            results[name] = compute_sample(name, path, is_sim)
        save(results)
        print(f"Resultados gravados em {RESULTS_FILE} ({(time.time() - t0) / 60:.1f} min)")
    else:
        results = load()
        print(f"Resultados lidos de {RESULTS_FILE}")

    print_table(results)
    if "incl" in results.get("data", {}) and "incl" in results.get("sim", {}):
        fig_inclusive(results)
    if "incl" in results.get("sim", {}):
        fig_labels(results)
    fig_threshold(results)
    print("\nFiguras gravadas: eeec_inclusive, eeec_labels, eeec_threshold")
