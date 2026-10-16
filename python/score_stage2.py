import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.EnableImplicitMT()

SIM_FILE = "../root_ML/merged_ML-600to800.root"
APPLY_STAGE1 = False   # True: so jatos com matched score > STAGE1_CUT
STAGE1_CUT = 0.8
NBINS_PLOT = 50        # binagem da figura
NBINS_AUC = 2000       # binagem fina para as areas sob as curvas
LOGY = False

def select(cond):
    """Scores qq dos jatos com dpsi12 valido, score valido e etiqueta em cond."""
    stage1 = f" && lund_ML_matched[i] > {STAGE1_CUT}" if APPLY_STAGE1 else ""
    return f"""
    ROOT::RVecD out;
    const size_t n = std::min({{lund_psi12_sd.size(), lund_secondary_idx_sd.size(),
                               lund_ML_qq.size(), lund_ML_matched.size()}});
    for (size_t i = 0; i < n; ++i) {{
        const int l = lund_secondary_idx_sd[i];
        const double s = lund_ML_qq[i];
        if (lund_psi12_sd[i] == lund_psi12_sd[i] && s >= 0.0 && ({cond}){stage1})
            out.push_back(s);
    }}
    return out;
    """

df = (ROOT.RDataFrame("jetTree", SIM_FILE)
      .Define("s_sig", select("l == 1"))
      .Define("s_bkg", select("l == 2 || l == 3")))

h_sig_r = df.Histo1D(("h_sig", "", NBINS_PLOT, 0.0, 1.0), "s_sig")
h_bkg_r = df.Histo1D(("h_bkg", "", NBINS_PLOT, 0.0, 1.0), "s_bkg")
f_sig_r = df.Histo1D(("f_sig", "", NBINS_AUC, 0.0, 1.0), "s_sig")
f_bkg_r = df.Histo1D(("f_bkg", "", NBINS_AUC, 0.0, 1.0), "s_bkg")

# ---- areas sob as curvas, a partir dos histogramas finos --------------------
fs, fb = f_sig_r.GetValue(), f_bkg_r.GetValue()
S, B = fs.Integral(), fb.Integral()
auc_roc, ap = 0.0, 0.0
tpr_prev, fpr_prev = 0.0, 0.0
s_cum, b_cum = 0.0, 0.0
for b in range(NBINS_AUC, 0, -1):          # do score mais alto para o mais baixo
    s_cum += fs.GetBinContent(b)
    b_cum += fb.GetBinContent(b)
    tpr = s_cum / S if S > 0 else 0.0
    fpr = b_cum / B if B > 0 else 0.0
    auc_roc += (fpr - fpr_prev) * 0.5 * (tpr + tpr_prev)
    if s_cum + b_cum > 0:
        precision = s_cum / (s_cum + b_cum)
        ap += (tpr - tpr_prev) * precision
    tpr_prev, fpr_prev = tpr, fpr
baseline = S / (S + B) if (S + B) > 0 else 0.0
print(f"jatos g->qqbar = {S:,.0f}   jatos q->qg + g->gg = {B:,.0f}")
print(f"area sob a curva ROC = {auc_roc:.3f}")
print(f"area sob a curva PR  = {ap:.3f}   (classificador aleatorio = {baseline:.3f})")

# ---- figura -----------------------------------------------------------------
def prep(hr, name, color, marker):
    h = hr.GetValue().Clone(name)
    h.SetDirectory(0)
    h.Sumw2()
    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())
    h.SetLineColor(color)
    h.SetLineWidth(2)
    h.SetMarkerColor(color)
    h.SetMarkerStyle(marker if marker else 1)
    h.SetMarkerSize(0.8)
    return h

hs = prep(h_sig_r, "hs", ROOT.kAzure + 2, 20)
hb = prep(h_bkg_r, "hb", ROOT.kRed + 1, 0)

c = ROOT.TCanvas("c", "", 800, 600)
c.SetLeftMargin(0.13)
if LOGY:
    c.SetLogy()
ymax = max(hs.GetMaximum(), hb.GetMaximum())
for h in (hb, hs):
    h.SetMaximum(ymax * (10.0 if LOGY else 1.3))
    h.SetMinimum(1e-5 if LOGY else 0.0)
    h.GetXaxis().SetTitle("g#rightarrowq#bar{q} score")
    h.GetYaxis().SetTitle("Fraction of jets")
hb.Draw("HIST")
hs.Draw("E1 SAME")

leg = ROOT.TLegend(0.50, 0.74, 0.88, 0.88)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.AddEntry(hs, "g#rightarrowq#bar{q} label", "lep")
leg.AddEntry(hb, "q#rightarrowqg + g#rightarrowgg labels", "l")
leg.Draw()

c.SaveAs("score_stage2.png")