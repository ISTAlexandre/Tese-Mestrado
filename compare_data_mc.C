// compare_data_mc.C
// Compara dados e simulacao nas distribuicoes dos scores da rede e das
// variaveis de entrada, normalizadas a area 1, com painel de razao.
//
// Uso:
//   root -l -b -q 'compare_data_mc.C("dados.root","root_ML/merged_ML-600to800.root",700,900)'
//
// Os dois ultimos argumentos definem uma janela em pT do jato aplicada as duas
// amostras, para comparar jatos com a mesma cinematica. Produz compare_data_mc.pdf

#include <cstdio>
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TLegend.h"
#include "TString.h"
#include "TDirectory.h"
#include "TMath.h"

// Desenha expr com binagem fixa e devolve um histograma desligado do ficheiro
static TH1D* fill(TTree* t, const char* expr, const char* name,
                  int nb, double lo, double hi, const TString& cut)
{
  t->GetDirectory()->cd();
  t->Draw(Form("%s>>%s(%d,%g,%g)", expr, name, nb, lo, hi), cut.Data(), "goff");
  TH1D* h = (TH1D*)gDirectory->Get(name);
  if (h) h->SetDirectory(0);
  return h;
}

// Sobrepoe dados e simulacao normalizados, com razao em baixo, e grava uma pagina
static void overlay(TCanvas* c, TH1D* hd, TH1D* hm, const char* xtitle, const char* pdf)
{
  if (!hd || !hm || hd->Integral() <= 0 || hm->Integral() <= 0) {
    printf("  [%s] sem entradas, ignorado\n", xtitle);
    return;
  }
  hd->Scale(1.0 / hd->Integral());
  hm->Scale(1.0 / hm->Integral());

  c->Clear();
  TPad* p1 = new TPad("p1", "", 0, 0.30, 1, 1);
  TPad* p2 = new TPad("p2", "", 0, 0.00, 1, 0.30);
  p1->SetBottomMargin(0.02);
  p2->SetTopMargin(0.02);
  p2->SetBottomMargin(0.32);
  p1->Draw();
  p2->Draw();

  p1->cd();
  hm->SetStats(0);
  hm->SetTitle("");
  hm->SetLineColor(kRed + 1);
  hm->SetLineWidth(2);
  hm->SetMaximum(1.3 * TMath::Max(hd->GetMaximum(), hm->GetMaximum()));
  hm->SetMinimum(0);
  hm->GetYaxis()->SetTitle("normalised");
  hm->GetXaxis()->SetLabelSize(0);
  hd->SetStats(0);
  hd->SetMarkerStyle(20);
  hd->SetMarkerSize(0.7);
  hm->Draw("hist");
  hd->Draw("E same");
  TLegend* leg = new TLegend(0.62, 0.74, 0.88, 0.88);
  leg->SetBorderSize(0);
  leg->AddEntry(hd, "data", "lep");
  leg->AddEntry(hm, "simulation", "l");
  leg->Draw();

  p2->cd();
  TH1D* r = (TH1D*)hd->Clone(Form("ratio_%s", hd->GetName()));
  r->Divide(hm);
  r->SetTitle("");
  r->SetMinimum(0.5);
  r->SetMaximum(1.5);
  r->GetXaxis()->SetTitle(xtitle);
  r->GetYaxis()->SetTitle("data / sim");
  r->GetXaxis()->SetTitleSize(0.12);
  r->GetXaxis()->SetLabelSize(0.10);
  r->GetYaxis()->SetTitleSize(0.10);
  r->GetYaxis()->SetTitleOffset(0.45);
  r->GetYaxis()->SetLabelSize(0.09);
  r->GetYaxis()->SetNdivisions(505);
  r->Draw("E");

  c->Print(pdf);
}

void compare_data_mc(const char* fdata, const char* fmc,
                     double ptmin = 700., double ptmax = 900.)
{
  TFile* fd = TFile::Open(fdata);
  TFile* fm = TFile::Open(fmc);
  if (!fd || fd->IsZombie() || !fm || fm->IsZombie()) { printf("Erro a abrir ficheiros\n"); return; }
  TTree* td = (TTree*)fd->Get("jetTree");
  TTree* tm = (TTree*)fm->Get("jetTree");
  if (!td || !tm) { printf("jetTree em falta\n"); return; }

  const TString valid = Form("lund_psi12_sd==lund_psi12_sd && jet_pt>%g && jet_pt<%g", ptmin, ptmax);
  const TString stage2 = valid + " && lund_ML_matched>0.8";
  const char* pdf = "compare_data_mc.pdf";

  TCanvas* c = new TCanvas("c", "", 800, 700);
  c->Print(Form("%s[", pdf));

  // 1) Score do stage 1 (todos os jatos validos)
  overlay(c, fill(td, "lund_ML_matched", "hd_m", 50, 0, 1, valid),
             fill(tm, "lund_ML_matched", "hm_m", 50, 0, 1, valid),
          "stage-1 matched score", pdf);

  // 2) Score do stage 2 (jatos com matched > 0.8)
  TH1D* hdq = fill(td, "lund_ML_qq", "hd_q", 100, 0, 1, stage2);
  TH1D* hmq = fill(tm, "lund_ML_qq", "hm_q", 100, 0, 1, stage2);

  // 3) Fracao de jatos acima de cada threshold (distribuicao cumulativa)
  printf("\n threshold   fracao dados   fracao simulacao   razao\n");
  if (hdq && hmq && hdq->Integral() > 0 && hmq->Integral() > 0) {
    const double nd = hdq->Integral(), nm = hmq->Integral();
    for (int i = 6; i <= 14; ++i) {
      const double thr = 0.05 * i;
      const int b = hdq->FindBin(thr + 1e-9);
      const double fd_ = hdq->Integral(b, hdq->GetNbinsX()) / nd;
      const double fm_ = hmq->Integral(b, hmq->GetNbinsX()) / nm;
      printf("   %.2f       %.4f          %.4f          %.2f\n",
             thr, fd_, fm_, fm_ > 0 ? fd_ / fm_ : 0.);
    }
  }
  overlay(c, hdq, hmq, "stage-2 g#rightarrowq#bar{q} score", pdf);

  // 4) Variaveis de entrada dos ramos do segundo splitting
  struct Var { const char* br; int nb; double lo, hi; const char* title; };
  Var vars[] = {
    {"lund_p3_n_all",     60, 0,  60,   "n_{all}^{(3)}"},
    {"lund_p4_n_all",     60, 0,  60,   "n_{all}^{(4)}"},
    {"lund_p3_n_charged", 40, 0,  40,   "n_{ch}^{(3)}"},
    {"lund_p4_n_charged", 40, 0,  40,   "n_{ch}^{(4)}"},
    {"lund_p3_sigma",     50, 0,  0.1,  "#sigma^{(3)}"},
    {"lund_p4_sigma",     50, 0,  0.1,  "#sigma^{(4)}"},
    {"log(1+lund_p3_ptD)",50, 0,  8,    "ln(1+#LTp_{T}#GT^{(3)})"},
    {"log(1+lund_p4_ptD)",50, 0,  8,    "ln(1+#LTp_{T}#GT^{(4)})"},
  };
  for (int k = 0; k < 8; ++k) {
    overlay(c, fill(td, vars[k].br, Form("hd_v%d", k), vars[k].nb, vars[k].lo, vars[k].hi, valid),
               fill(tm, vars[k].br, Form("hm_v%d", k), vars[k].nb, vars[k].lo, vars[k].hi, valid),
            vars[k].title, pdf);
  }

  c->Print(Form("%s]", pdf));
  printf("\nFeito. Figuras em %s\n", pdf);
}
