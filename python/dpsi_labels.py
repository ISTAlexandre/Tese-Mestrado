import math
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.EnableImplicitMT()

SIM_FILE = "../root_ML/merged_ML-600to800.root"
NBINS = 40
SHOW_BKG = False      # True: acrescenta a curva das classes q->qg + g->gg

# Etiquetas definidas no lund_gen.cpp
CLASSES = [
    (1, "g->qqbar"),
    (3, "g->gg"),
    (2, "q->qg"),
    (0, "unmatched"),
    (4, "rest"),
]

def select(cond):
    """dpsi12 dos jatos com valor valido e etiqueta que satisfaz cond (em l)."""
    return f"""
    ROOT::RVecD out;
    const size_t n = std::min(lund_psi12_sd.size(), lund_secondary_idx_sd.size());
    for (size_t i = 0; i < n; ++i) {{
        const double p = lund_psi12_sd[i];
        const int l = lund_secondary_idx_sd[i];
        if (p == p && ({cond})) out.push_back(p);
    }}
    return out;
    """

df = ROOT.RDataFrame("jetTree", SIM_FILE)

groups = [(f"c{k}", name, f"l == {k}") for k, name in CLASSES]
groups.append(("all", "all labelled jets", "l >= 0"))
groups.append(("bkg", "q->qg + g->gg", "l == 2 || l == 3"))

booked = {}
for key, name, cond in groups:
    df = (df.Define(f"psi_{key}", select(cond))
            .Define(f"c2_{key}", f"cos(2.0 * psi_{key})")
            .Define(f"n_{key}", f"(double) psi_{key}.size()"))
for key, name, cond in groups:
    booked[key] = dict(
        mean=df.Mean(f"c2_{key}"),
        std=df.StdDev(f"c2_{key}"),
        n=df.Sum(f"n_{key}"),
    )
h_all = df.Histo1D(("h_all", "", NBINS, 0.0, math.pi), "psi_all")
h_qq  = df.Histo1D(("h_qq",  "", NBINS, 0.0, math.pi), "psi_c1")
h_bkg = df.Histo1D(("h_bkg", "", NBINS, 0.0, math.pi), "psi_bkg")

# ---- tabela ----------------------------------------------------------------
n_all = float(booked["all"]["n"].GetValue())
stats = {}
print(f"\n{'Classe':22s} {'Jatos':>12s} {'Fracao':>8s}   <cos 2 dpsi12>")
for key, name, cond in groups:
    n = float(booked[key]["n"].GetValue())
    m = float(booked[key]["mean"].GetValue()) if n > 0 else float("nan")
    e = float(booked[key]["std"].GetValue()) / math.sqrt(n) if n > 0 else float("nan")
    stats[key] = (n, m, e)
    frac = 100.0 * n / n_all if (n_all > 0 and key != "bkg") else float("nan")
    print(f"{name:22s} {n:12.0f} {frac:7.1f}%   {m:+.4f} +- {e:.4f}")

# ---- figura ----------------------------------------------------------------
def prep(hr, name, color, marker, width=2):
    h = hr.GetValue().Clone(name)
    h.SetDirectory(0)
    h.Sumw2()
    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())
    h.SetLineColor(color)
    h.SetLineWidth(width)
    h.SetMarkerColor(color)
    h.SetMarkerStyle(marker if marker else 1)
    h.SetMarkerSize(0.8)
    return h

ha = prep(h_all, "ha", ROOT.kRed + 1, 0)
hq = prep(h_qq,  "hq", ROOT.kAzure + 2, 20)
curves = [(ha, f"All labelled jets, #LTcos 2#Delta#psi_{{12}}#GT = {stats['all'][1]:+.3f}", 0),
          (hq, f"g#rightarrowq#bar{{q}} label, #LTcos 2#Delta#psi_{{12}}#GT = {stats['c1'][1]:+.3f}", 20)]
if SHOW_BKG:
    hb = prep(h_bkg, "hb", ROOT.kGray + 2, 0)
    hb.SetLineStyle(2)
    curves.append((hb, f"q#rightarrowqg + g#rightarrowgg, #LTcos 2#Delta#psi_{{12}}#GT = {stats['bkg'][1]:+.3f}", 0))

c = ROOT.TCanvas("c", "", 800, 700)
p1 = ROOT.TPad("p1", "", 0, 0.30, 1, 1)
p2 = ROOT.TPad("p2", "", 0, 0.00, 1, 0.30)
for p in (p1, p2):
    p.SetLeftMargin(0.13)
p1.SetBottomMargin(0.02)
p2.SetTopMargin(0.02)
p2.SetBottomMargin(0.32)
p1.Draw()
p2.Draw()

p1.cd()
ymin = min(h.GetMinimum() for h, _, _ in curves)
ymax = max(h.GetMaximum() for h, _, _ in curves)
span = ymax - ymin
first = True
for h, label, marker in curves:
    h.SetMinimum(ymin - 0.6 * span)
    h.SetMaximum(ymax + 0.9 * span)
    h.GetYaxis().SetTitle("Fraction of jets")
    h.GetXaxis().SetLabelSize(0)
    opt = "E1" if marker else "HIST"
    h.Draw(opt if first else opt + " SAME")
    first = False
leg = ROOT.TLegend(0.30, 0.70, 0.88, 0.88)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
for h, label, marker in curves:
    leg.AddEntry(h, label, "lep" if marker else "l")
leg.Draw()

p2.cd()
r = hq.Clone("ratio")
r.Divide(ha)
r.SetTitle("")
r.SetMinimum(0.7)
r.SetMaximum(1.3)
r.GetXaxis().SetTitle("#Delta#psi_{12}")
r.GetYaxis().SetTitle("q#bar{q} / all")
r.GetXaxis().SetLabelSize(0.10)
r.GetXaxis().SetTitleSize(0.12)
r.GetYaxis().SetLabelSize(0.09)
r.GetYaxis().SetTitleSize(0.10)
r.GetYaxis().SetTitleOffset(0.5)
r.GetYaxis().SetNdivisions(505)
r.Draw("E1")
line = ROOT.TLine(0.0, 1.0, math.pi, 1.0)
line.SetLineColor(ROOT.kRed + 1)
line.Draw()

c.SaveAs("dpsi_labels.png")
c.SaveAs("dpsi_labels.pdf")
