import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.EnableImplicitMT()          # usa varios cores, importante para as simulacoes

samples = [
    ("../root/merged.root",                 "Data",                                         ROOT.kBlack,   20),
    ("../root_ML/merged_ML-600to800.root",  "Simulation, 600 < #hat{p}_{T} < 800 GeV",      ROOT.kRed + 1,  0),
    ("../root_ML/merged_ML-800to1000.root", "Simulation, 800 < #hat{p}_{T} < 1000 GeV",     ROOT.kAzure + 2, 0),
]

ONLY_VALID_DPSI = False          # True: so jatos com dpsi12 valido
NBINS, XMIN, XMAX = 100, 0, 200  # ajusta se a distribuicao ficar cortada

# Numero de constituintes de cada jato (um valor por jato)
nconst_all = """
ROOT::RVecI n;
for (const auto &c : const_pt) n.push_back(c.size());
return n;
"""
nconst_valid = """
ROOT::RVecI n;
for (size_t i = 0; i < const_pt.size(); ++i)
    if (i < lund_psi12_sd.size() && lund_psi12_sd[i] == lund_psi12_sd[i])
        n.push_back(const_pt[i].size());
return n;
"""

hists = []
for i, (path, label, color, marker) in enumerate(samples):
    df = ROOT.RDataFrame("jetTree", path)
    df = df.Define("nconst", nconst_valid if ONLY_VALID_DPSI else nconst_all)
    h = df.Histo1D((f"h{i}", "", NBINS, XMIN, XMAX), "nconst").GetValue()
    h = h.Clone(f"hc{i}")
    h.SetDirectory(0)
    print(f"{label:45s} jatos = {h.GetEntries():12.0f}   media = {h.GetMean():6.1f}")
    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())
    h.SetLineColor(color)
    h.SetLineWidth(2)
    h.SetMarkerColor(color)
    h.SetMarkerStyle(marker if marker else 1)
    hists.append((h, label, marker))

c = ROOT.TCanvas("c", "", 800, 600)
c.SetLeftMargin(0.13)
ymax = 1.25 * max(h.GetMaximum() for h, _, _ in hists)

first = True
for h, label, marker in hists:
    h.SetMaximum(ymax)
    h.SetMinimum(0)
    h.GetXaxis().SetTitle("Number of constituents")
    h.GetYaxis().SetTitle("Fraction of jets")
    opt = "E1" if marker else "HIST"
    h.Draw(opt if first else opt + " SAME")
    first = False

leg = ROOT.TLegend(0.48, 0.70, 0.88, 0.88)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
for h, label, marker in hists:
    leg.AddEntry(h, label, "lep" if marker else "l")
leg.Draw()

c.SaveAs("nconst_jets.png")
