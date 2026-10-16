import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.EnableImplicitMT()

samples = [
    ("../root/merged.root",                 "Data",                                     ROOT.kBlack,     20),
    ("../root_ML/merged_ML-600to800.root",  "Simulation, 600 < #hat{p}_{T} < 800 GeV",  ROOT.kRed + 1,    0),
    ("../root_ML/merged_ML-800to1000.root", "Simulation, 800 < #hat{p}_{T} < 1000 GeV", ROOT.kAzure + 2,  0),
]

ONLY_VALID_DPSI = False             # True: so jatos com dpsi12 valido
NBINS, XMIN, XMAX = 52, 700, 2000   # bins de 25 GeV
LOGY = False

pt_valid = """
ROOT::RVecF v;
for (size_t i = 0; i < jet_pt.size(); ++i)
    if (i < lund_psi12_sd.size() && lund_psi12_sd[i] == lund_psi12_sd[i])
        v.push_back(jet_pt[i]);
return v;
"""

hists = []
for i, (path, label, color, marker) in enumerate(samples):
    df = ROOT.RDataFrame("jetTree", path)
    if ONLY_VALID_DPSI:
        df = df.Define("pt_sel", pt_valid)
        col = "pt_sel"
    else:
        col = "jet_pt"
    h = df.Histo1D((f"h{i}", "", NBINS, XMIN, XMAX), col).GetValue()
    h = h.Clone(f"hc{i}")
    h.SetDirectory(0)
    print(f"{label:45s} jatos = {h.GetEntries():12.0f}   media = {h.GetMean():7.1f} GeV")
    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())
    h.SetLineColor(color)
    h.SetLineWidth(2)
    h.SetMarkerColor(color)
    h.SetMarkerStyle(marker if marker else 1)
    hists.append((h, label, marker))

c = ROOT.TCanvas("c", "", 800, 600)
c.SetLeftMargin(0.13)
if LOGY:
    c.SetLogy()
    positive = [h.GetBinContent(b) for h, _, _ in hists
                for b in range(1, h.GetNbinsX() + 1) if h.GetBinContent(b) > 0]
    ymin = 0.5 * min(positive) if positive else 1e-6
    ymax = 5.0 * max(h.GetMaximum() for h, _, _ in hists)
else:
    ymin, ymax = 0.0, 1.25 * max(h.GetMaximum() for h, _, _ in hists)

first = True
for h, label, marker in hists:
    h.SetMinimum(ymin)
    h.SetMaximum(ymax)
    h.GetXaxis().SetTitle("#it{p}_{T} [GeV]")
    h.GetYaxis().SetTitle("Fraction of jets")
    opt = "E1" if marker else "HIST"
    h.Draw(opt if first else opt + " SAME")
    first = False

leg = ROOT.TLegend(0.48, 0.72, 0.88, 0.88)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
for h, label, marker in hists:
    leg.AddEntry(h, label, "lep" if marker else "l")
leg.Draw()

c.SaveAs("jetpt_jets.png")