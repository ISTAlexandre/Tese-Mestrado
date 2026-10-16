import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kBird)
ROOT.gStyle.SetNumberContours(100)
ROOT.EnableImplicitMT()

samples = {
    "data": "../root/merged.root",
    "sim":  "../root_ML/merged_ML-600to800.root",
}

ONLY_VALID_DPSI = True          # True: so jatos com dpsi12 valido (os da analise)
NX, XMIN, XMAX = 50, 0.0, 6.0   # ln(1/DeltaR)
NY, YMIN, YMAX = 50, -3.0, 7.0  # ln(kT/GeV)
MAKE_RATIO = False              # True: tambem mapas de razao dados/simulacao

# Achata os vetores por jato num unico vetor de pontos, mantendo o alinhamento
# entre x e y, e opcionalmente so para jatos com dpsi12 valido.
def flatten(branch):
    cond = ("i < lund_psi12_sd.size() && lund_psi12_sd[i] == lund_psi12_sd[i]"
            if ONLY_VALID_DPSI else "true")
    return f"""
    ROOT::RVecD out;
    for (size_t i = 0; i < {branch}.size(); ++i)
        if ({cond})
            for (auto v : {branch}[i]) out.push_back(v);
    return out;
    """

if ONLY_VALID_DPSI:
    count_jets = ("int n = 0; for (size_t i = 0; i < jet_pt.size(); ++i) "
                  "if (i < lund_psi12_sd.size() && lund_psi12_sd[i] == lund_psi12_sd[i]) ++n; "
                  "return n;")
else:
    count_jets = "return (int) jet_pt.size();"

results = {}
for name, path in samples.items():
    df = (ROOT.RDataFrame("jetTree", path)
          .Define("px", flatten("lund_coords_x_sd"))
          .Define("py", flatten("lund_coords_y_sd"))
          .Define("sx", flatten("lund_coords_secondary_x_sd"))
          .Define("sy", flatten("lund_coords_secondary_y_sd"))
          .Define("njet", count_jets))
    # reservar todas as acoes antes de correr, para ler o ficheiro uma so vez
    hp = df.Histo2D((f"hp_{name}", "", NX, XMIN, XMAX, NY, YMIN, YMAX), "px", "py")
    hs = df.Histo2D((f"hs_{name}", "", NX, XMIN, XMAX, NY, YMIN, YMAX), "sx", "sy")
    nj = df.Sum("njet")

    njets = float(nj.GetValue())
    out = {}
    for key, h in (("primary", hp), ("secondary", hs)):
        hc = h.GetValue().Clone(f"{key}_{name}")
        hc.SetDirectory(0)
        bin_area = hc.GetXaxis().GetBinWidth(1) * hc.GetYaxis().GetBinWidth(1)
        if njets > 0:
            hc.Scale(1.0 / (njets * bin_area))   # densidade rho por jato e por area
        out[key] = hc
    results[name] = out
    print(f"{name:5s}  jatos = {njets:12.0f}   "
          f"emissoes primarias/jato = {out['primary'].Integral('width'):.2f}   "
          f"secundarias/jato = {out['secondary'].Integral('width'):.2f}")

def draw(h, fname, zmax, ztitle="#rho", zmin=None):
    c = ROOT.TCanvas("c_" + fname, "", 700, 600)
    c.SetRightMargin(0.16)
    c.SetLeftMargin(0.12)
    h.GetXaxis().SetTitle("ln(1/#DeltaR)")
    h.GetYaxis().SetTitle("ln(k_{T}/GeV)")
    h.GetZaxis().SetTitle(ztitle)
    h.GetZaxis().SetTitleOffset(1.2)
    h.SetMaximum(zmax)
    h.SetMinimum(zmin if zmin is not None else 1e-9)  # bins vazios ficam brancos
    h.Draw("COLZ")
    c.SaveAs(fname)

# Mesma escala de cor para dados e simulacao em cada linha da figura
for key in ("primary", "secondary"):
    zmax = max(results["data"][key].GetMaximum(), results["sim"][key].GetMaximum())
    for name in ("data", "sim"):
        draw(results[name][key], f"lund_{key}_{name}.png", zmax)
    if MAKE_RATIO:
        r = results["data"][key].Clone(f"ratio_{key}")
        r.Divide(results["sim"][key])
        draw(r, f"lund_{key}_ratio.png", 1.5, "data / sim", 0.5)
