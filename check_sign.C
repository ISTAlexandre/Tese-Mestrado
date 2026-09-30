// check_sign.C
// Verifica o sinal de <cos(2 dpsi12)> por canal verdadeiro e ao longo do scan
// no score qq do classificador.
//
// Uso:
//   root -l -b -q 'check_sign.C("root_ML/merged_ML-600to800.root")'
//
// Requer os ramos escritos pelo lund_gen.cpp (lund_psi12_sd,
// lund_secondary_idx_sd) e pelo apply_ML.py (lund_ML_qq, lund_ML_matched).

#include <cstdio>
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TString.h"
#include "TDirectory.h"

static TH1* fill(TTree* t, const char* hname, const TString& cut)
{
  t->Draw(Form("cos(2*lund_psi12_sd)>>%s(200,-1.01,1.01)", hname), cut.Data(), "goff");
  return (TH1*)gDirectory->Get(hname);
}

static void report(TTree* t, const char* label, const TString& cut)
{
  TH1* h = fill(t, "hcheck", cut);
  if (!h || h->GetEntries() == 0) { printf("%-26s  sem entradas\n", label); return; }
  printf("%-26s  <cos2dpsi> = %+.4f +- %.4f   N = %.0f\n",
         label, h->GetMean(), h->GetMeanError(), h->GetEntries());
}

void check_sign(const char* fname)
{
  TFile* f = TFile::Open(fname);
  if (!f || f->IsZombie()) { printf("Nao consegui abrir %s\n", fname); return; }
  TTree* t = (TTree*)f->Get("jetTree");
  if (!t) { printf("Nao encontrei jetTree\n"); return; }

  const TString valid = "lund_psi12_sd==lund_psi12_sd";   // exclui NaN

  // 1) Por canal verdadeiro do segundo splitting, sem classificador
  printf("\n=== 1) Por canal verdadeiro, sem classificador ===\n");
  const char* names[5] = {"unmatched (0)", "g->qqbar (1)", "q->qg (2)", "g->gg (3)", "rest (4)"};
  for (int c = 0; c <= 4; ++c)
    report(t, names[c], valid + Form(" && lund_secondary_idx_sd==%d", c));
  report(t, "todos com etiqueta", valid + " && lund_secondary_idx_sd>=0");

  // 2) Scan no score qq, com o corte do stage 1 (matched > 0.8)
  printf("\n=== 2) Scan no score qq (matched > 0.8) ===\n");
  printf("thr    N         pureza   <cos2> selecao        <cos2> qqbar verdadeiros\n");
  for (int i = 0; i <= 19; ++i) {
    const double thr = 0.05 * i;
    const TString sel = valid + Form(" && lund_ML_matched>0.8 && lund_ML_qq>%.2f", thr);
    TH1* hall = fill(t, "hall", sel);
    TH1* hqq  = fill(t, "hqq",  sel + " && lund_secondary_idx_sd==1");
    const double nall = hall ? hall->GetEntries() : 0.;
    const double nqq  = hqq  ? hqq->GetEntries()  : 0.;
    if (nall == 0) break;
    printf("%.2f  %8.0f   %.3f   %+.4f +- %.4f      %+.4f +- %.4f\n",
           thr, nall, nqq / nall,
           hall->GetMean(), hall->GetMeanError(),
           nqq > 0 ? hqq->GetMean() : 0., nqq > 0 ? hqq->GetMeanError() : 0.);
  }
  f->Close();
}