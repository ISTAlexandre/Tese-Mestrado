import math
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.EnableImplicitMT()

samples = [
    ("../root/merged.root",                "Data",                                    ROOT.kBlack,   20),
    ("../root_ML/merged_ML-600to800.root", "Simulation, 600 < #hat{p}_{T} < 800 GeV", ROOT.kRed + 1,  0),
]

NBINS = 40
RATIO_MIN, RATIO_MAX = 0.90, 1.10   # ajusta se a razao sair do intervalo

results = []
for i, (path, label, color, marker) in enumerate(samples):
    df = (ROOT.RDataFrame("jetTree", path)
          .Define("psi", "lund_psi12_sd[lund_psi12_sd == lund_psi12_sd]")  # exclui NaN
          .Define("c2", "cos(2.0 * psi)")
          .Define("n", "(int) psi.size()"))
    # reservar as acoes todas antes de correr
    h_r = df.Histo1D((f"h{i}", "", NBINS, 0.0, math.pi), "psi")
    mean_r = df.Mean("c2")
    std_r = df.StdDev("c2")
    n_r = df.Sum("n")

    n = float(n_r.GetValue())
    mean = float(mean_r.GetValue())
    err = float(std_r.GetValue()) / math.sqrt(n) if n > 0 else 0.0
    print(f"{label:42s} N = {n:12.0f}   <cos2dpsi> = {mean:+.4f} +- {err:.4f}"
          f"   a2/a0 = {2*mean:+.4f} +- {2*err:.4f}")

    h = h_r.GetValue().Clone(f"hc{i}")
    h.SetDirectory(0)
    h.Sumw2()
    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())
    h.SetLineColor(color)
    h.SetLineWidth(2)
    h.SetMarkerColor(color)
    h.SetMarkerStyle(marker if marker else 1)
    h.SetMarkerSize(0.8)
    results.append((h, label, marker, mean, err))

c = ROOT.TCanvas("c", "", 800, 700)
p1 = ROOT.TPad("p1", "", 0, 0.30, 1, 1)
p2 = ROOT.TPad("p2", "", 0, 0.00, 1, 0.30)
p1.SetBottomMargin(0.02)
p1.SetLeftMargin(0.13)
p2.SetTopMargin(0.02)
p2.SetBottomMargin(0.32)
p2.SetLeftMargin(0.13)
p1.Draw()
p2.Draw()

# Eixo vertical sem comecar em zero, para a modulacao ser visivel
p1.cd()
ymin = min(h.GetMinimum() for h, *_ in results)
ymax = max(h.GetMaximum() for h, *_ in results)
span = ymax - ymin
first = True
for h, label, marker, mean, err in results:
    h.SetMinimum(ymin - 0.6 * span)
    h.SetMaximum(ymax + 0.9 * span)
    h.GetYaxis().SetTitle("Fraction of jets")
    h.GetXaxis().SetLabelSize(0)
    opt = "E1" if marker else "HIST"
    h.Draw(opt if first else opt + " SAME")
    first = False

leg = ROOT.TLegend(0.40, 0.70, 0.88, 0.88)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
for h, label, marker, mean, err in results:
    leg.AddEntry(h, f"{label}, #LTcos 2#Delta#psi_{{12}}#GT = {mean:+.3f}",
                 "lep" if marker else "l")
leg.Draw()

# Razao dados / simulacao
p2.cd()
r = results[0][0].Clone("ratio")
r.Divide(results[1][0])
r.SetTitle("")
r.SetMinimum(RATIO_MIN)
r.SetMaximum(RATIO_MAX)
r.GetXaxis().SetTitle("#Delta#psi_{12}")
r.GetYaxis().SetTitle("data / sim")
r.GetXaxis().SetLabelSize(0.10)
r.GetXaxis().SetTitleSize(0.12)
r.GetYaxis().SetLabelSize(0.09)
r.GetYaxis().SetTitleSize(0.10)
r.GetYaxis().SetTitleOffset(0.5)
r.GetYaxis().SetNdivisions(505)
r.SetLineColor(ROOT.kBlack)
r.SetMarkerColor(ROOT.kBlack)
r.Draw("E1")
line = ROOT.TLine(0.0, 1.0, math.pi, 1.0)
line.SetLineColor(ROOT.kRed + 1)
line.Draw()

c.SaveAs("dpsi_inclusive.png")
