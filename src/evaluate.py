"""Ocena modelu. Dwa zbiory, dwie miary — ZADNA nie przewiduje struktury.

ZBIORY
  TEST NATURALNY    20% puli, rodziny nieobecne w treningu. Ocena GLOWNA.
  ETERNA <= 200 nt  zagadki projektowe ludzi. Pomocnicza, zewnetrzna wzgledem naszych danych.

MIARY
  identycznosc_nt   ulamek pozycji, na ktorych trafilismy w litere sekwencji referencyjnej.
  dE/nt    [E(cel | nasza) - E(cel | referencyjna)] / dlugosc. Ujemne = nasza stabilizuje cel
           LEPIEJ niz sekwencja referencyjna.

Nie odpowiadamy na pytanie "czy ta sekwencja faktycznie sie zwinie" — wymagaloby to przewidywania
struktury RNAfoldem, ktory ma wlasny sufit dokladnosci. Mierzymy podobienstwo do natury
(identycznosc_nt, identycznosc_par) i stabilnosc zadanej struktury (dE/nt).

Uzycie:
    python -m src.evaluate --ckpt checkpoints/e1.pt
    python -m src.evaluate --ckpt checkpoints/e1.pt --na val
    python -m src.evaluate --baseline                             # losowy kanoniczny
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import RNA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.dataset import koduj, losowa_kanoniczna, parse_pairs, BASES
from src.model import NARDesigner
from src.prepare import wczytaj
from src.split import wczytaj_split

# Kubelki NARASTAJACE, nie rozlaczne: "<= 100 nt" zawiera w sobie "<= 50 nt".
# Kazdy wiersz odpowiada na pytanie "jak model radzi sobie na strukturach do tej dlugosci".
KUBELKI = [50, 100, 200]
ANALIZA = ROOT / "experiments" / "analysis"


def wczytaj_model(ckpt, device):
    d = torch.load(ROOT / ckpt, map_location=device, weights_only=False)
    a = d["args"]
    m = NARDesigner(d_model=a["d_model"], num_layers=a["warstwy"],
                    max_len=d["max_len"] + 1, dropout=0.0).to(device)
    m.load_state_dict(d["model"])
    m.eval()
    return m, a


@torch.no_grad()
def generuj(model, structs, device, bs=32, dekodowanie="argmax", seed=0):
    """Zamiana rozkladow modelu na sekwencje. DWA SPOSOBY, ta sama siec.

    argmax        na kazdej pozycji litera o najwiekszym prawdopodobienstwie
    probkowanie   losowanie zgodnie z rozkladem

    DLACZEGO TO NIE JEST DROBIAZG. Model prawie nie korzysta z wejscia i produkuje niemal TEN SAM
    plaski rozklad na kazdej pozycji: na glowie par pewnosc zwyciezcy 0,34-0,43 przy 0,167 dla
    jednostajnego, a mimo to rodzina G:C wygrywa na 91-99,9% pozycji. `argmax` bierze zwyciezce, wiec
    skladem wyjscia staje sie punkt, a nie rozklad. Probkowanie odtwarza rozklad, bo prawdopodobienstwo
    wylosowania klasy c ROWNA SIE p(c), czyli oczekiwany udzial c w sekwencji to srednia p(c).

    Architektura sie przy tym nie zmienia: to nadal jeden przebieg enkodera, wszystkie pozycje naraz,
    a para (i,j) nadal jest JEDNA decyzja — losujemy jedna z szesciu klas kanonicznych, wiec para
    niekanoniczna pozostaje niemozliwa.

    Probkowanie jest losowe, wiec ZIARNO jest czescia wyniku; bez niego nic nie da sie odtworzyc.
    """
    if dekodowanie == "probkowanie":
        torch.manual_seed(seed)
    out = []
    for s in range(0, len(structs), bs):
        cs = structs[s : s + bs]
        sid, pad, par, _, _, _ = koduj(cs, None, device)
        out += model.generate(sid, pad, par, [len(x) for x in cs],
                              sample=(dekodowanie == "probkowanie"))
    return out


def eterna(max_len=200):
    """Zagadki Eterny po tym samym filtrze przewagi sparowanych co dane naturalne.

    Czytamy z `data/eterna_working.parquet`, ktory buduje `src/prepare.py`, a NIE z surowego pliku
    zrodlowego — inaczej ocenialibysmy takze struktury bez przewagi sparowanych. Eterna NIE przechodzi
    przez cd-hit: nie ma jej w treningu, wiec nie ma tu przecieku, ktoremu odsiewanie mialoby
    zapobiec, a uszczuplenie zbioru czyniloby nasze liczby nieporownywalnymi z cudzymi.
    """
    from src.prepare import wczytaj_eterna
    d = wczytaj_eterna()
    d = d[d.secondary_structure.str.len() <= max_len]
    return d.secondary_structure.tolist(), d.sequence.tolist()


TYP_PARY = {"GC": "GC", "CG": "GC", "AU": "AU", "UA": "AU", "GU": "GU", "UG": "GU"}


def identycznosci(struct: str, wygen: str, wzor: str) -> tuple[float, float]:
    """Dwa wspolczynniki identycznosci. UWAGA: liczone na ROZNYCH mianownikach.

    identycznosc_nt    ulamek WSZYSTKICH POZYCJI, na ktorych model postawil te sama zasade
                       co sekwencja referencyjna

    identycznosc_par   ulamek PAR (a nie pozycji), w ktorych model trafil w TYP pary, niezaleznie
                       od orientacji: gdy referencja ma G-C, a model C-G, to TRAFIENIE.
                       Pozycje NIESPAROWANE sa tu calkowicie pomijane.

    DLACZEGO ROZNE MIANOWNIKI. Architektura wymusza pare wszedzie tam, gdzie struktura jej zada —
    model wybiera jedna z szesciu klas kanonicznych dla kazdej pary. Nie moze wiec postawic pary
    w zlym miejscu ani jej pominac, a sprawdzanie rozmieszczenia par nie mialoby sensu. Jedyne, co
    da sie ocenic, to KTORY TYP wybral, i naturalna jednostka jest tu para, a nie pozycja.

    Zwraca NaN dla `identycznosc_par`, gdy struktura nie ma ani jednej pary.
    """
    n = len(struct)
    tnt = sum(a == b for a, b in zip(wygen, wzor)) / max(n, 1)

    pary = parse_pairs(struct)
    if not pary:
        return tnt, float("nan")
    trafione = sum(1 for i, j in pary
                   if TYP_PARY.get(wygen[i] + wygen[j]) is not None
                   and TYP_PARY.get(wygen[i] + wygen[j]) == TYP_PARY.get(wzor[i] + wzor[j]))
    return tnt, trafione / len(pary)


def _udzialy(struct, seq):
    """Udzialy zasad w calej sekwencji, typow par, i zasad w pozycjach niesparowanych."""
    from collections import Counter
    n = len(seq)
    zas = Counter(seq)
    typy = Counter()
    for i, j in parse_pairs(struct):
        t = TYP_PARY.get(seq[i] + seq[j])
        if t:
            typy[t] += 1
    sparowane = {k for para in parse_pairs(struct) for k in para}
    petle = Counter(seq[k] for k in range(n) if k not in sparowane)
    return zas, n, typy, sum(typy.values()), petle, sum(petle.values())


def kara_tv(struct, seq):
    """NASZA kara za sklad, policzona na WYGENEROWANEJ sekwencji: odleglosc TV od celu naturalnego.

    Per sekwencja, tak samo jak w treningu. Roznica wobec logu treningowego jest wylacznie taka,
    ze tam kara liczy sie na rozkladach prawdopodobienstwa, a tu na gotowej sekwencji. Przy
    `--dekodowanie probkowanie` obie wielkosci sie pokrywaja, bo oczekiwany sklad wylosowanej
    sekwencji ROWNA SIE skladowi rozkladu; przy `argmax` rozjezdzaja sie drastycznie.
    """
    from src.loss import NATURAL_LOOP, NATURAL_PAIR
    _, _, typy, nt, petle, npe = _udzialy(struct, seq)
    d = 0.0
    if nt:
        d += 0.5 * sum(abs(typy[k] / nt - NATURAL_PAIR[k]) for k in ("GC", "AU", "GU"))
    if npe:
        d += 0.5 * sum(abs(petle[b] / npe - NATURAL_LOOP[b]) for b in "ACGU")
    return d


def kara_progi(struct, seq):
    """Kara promotora na WYGENEROWANEJ sekwencji: DistribLoss + DistribLoss3 + DistribLoss4."""
    from src.loss import PROG_ZASADY, PROG_PARY
    zas, n, typy, nt, _, _ = _udzialy(struct, seq)
    x = [max(p - zas[b] / n, 0) / p for b, p in PROG_ZASADY.items()]
    wynik = sum(x) / 4
    if nt:
        v = [max(p - typy[k] / nt, 0) / p for k, p in PROG_PARY.items()]
        wynik += sum(v) / 3 + max(sum(v) - 1, 0)
    return wynik


def wlasna_kara(args_ckpt):
    """Ktora kara nalezy do tego modelu — odczytane z zapisanych hiperparametrow checkpointu."""
    # Kara TV ma dwa czlony z osobnymi wagami. Starsze checkpointy maja tylko `w_sklad`,
    # nowsze — `w_sklad_tv_pary` i `w_sklad_tv_petle`; czytamy jedno i drugie.
    tv_pary = args_ckpt.get("w_sklad_tv_pary", args_ckpt.get("w_sklad", 0)) or 0
    tv_petle = args_ckpt.get("w_sklad_tv_petle", args_ckpt.get("w_sklad", 0)) or 0
    if tv_pary and tv_petle:
        return "kara TV (nasza), pary + petle", kara_tv
    if tv_petle:
        return "kara TV (nasza), TYLKO petle", kara_tv
    if tv_pary:
        return "kara TV (nasza), TYLKO pary", kara_tv
    if args_ckpt.get("w_sklad_zasad", 0) or args_ckpt.get("w_sklad_par", 0):
        return "kara progi (promotor)", kara_progi
    return "brak kary za sklad", None


def recall_klas(structs, gen, refs, indeksy):
    """CZULOSC i SPECYFICZNOSC osobno dla kazdej klasy, plus srednie zbalansowane.

    DLACZEGO TO JEST POTRZEBNE OBOK ZWYKLEJ IDENTYCZNOSCI. Klasy sa bardzo nierownoliczne: pary G:C
    to 48% wszystkich, A:U 37%, G:U 14%. Zwykla identycznosc jest wiec MAKSYMALIZOWANA przez stale
    wskazywanie klasy najczestszej — sekwencja zlozona z samych par G:C dostaje 48,4%, czyli wiecej
    niz uczciwy baseline. To nie jest podatnosc na oszustwo, tylko wlasnosc matematyczna.

    CZULOSC (recall)        z tych, ktore NAPRAWDE sa klasy c, ile model rozpoznal
    SPECYFICZNOSC           z tych, ktore NIE sa klasy c, ilu model nie przypisal do c

    Te dwie razem wykrywaja nadprodukcje. Predyktor "zawsze G:C" ma czulosc G:C rowna 1,0
    i specyficznosc G:C rowna 0,0 — czyli rozpoznaje wszystkie prawdziwe G:C, ale za cene wrzucenia
    do nich rowniez wszystkiego innego. Sama czulosc tego nie pokaze.

    Srednia zbalansowana usrednia czulosci BEZ WAGI, wiec za klasy pominiete placi sie pelna cene:
    predyktor staly spada do 1/3 (pary) i 1/4 (zasady), czyli dokladnie do poziomu losowego.
    """
    tp_p, n_p, fp_p, neg_p = Counter(), Counter(), Counter(), Counter()
    tp_z, n_z, fp_z, neg_z = Counter(), Counter(), Counter(), Counter()
    for i in indeksy:
        st, g, r = structs[i], gen[i], refs[i]
        pary = parse_pairs(st)
        sparowane = {k for p in pary for k in p}
        for a, b in pary:
            wzor = TYP_PARY.get(r[a] + r[b])
            model = TYP_PARY.get(g[a] + g[b])
            if wzor is None:
                continue
            n_p[wzor] += 1
            if model == wzor:
                tp_p[wzor] += 1
            for c in ("GC", "AU", "GU"):
                if wzor != c:
                    neg_p[c] += 1
                    if model == c:
                        fp_p[c] += 1
        for k in range(len(st)):
            if k in sparowane:
                continue
            n_z[r[k]] += 1
            if g[k] == r[k]:
                tp_z[r[k]] += 1
            for b in BASES:
                if r[k] != b:
                    neg_z[b] += 1
                    if g[k] == b:
                        fp_z[b] += 1
    r_p = {k: tp_p[k] / n_p[k] for k in ("GC", "AU", "GU") if n_p[k]}
    r_z = {b: tp_z[b] / n_z[b] for b in BASES if n_z[b]}
    s_p = {k: 1 - fp_p[k] / neg_p[k] for k in ("GC", "AU", "GU") if neg_p[k]}
    s_z = {b: 1 - fp_z[b] / neg_z[b] for b in BASES if neg_z[b]}
    return r_p, r_z, s_p, s_z


def ocen(structs, gen, refs, etykieta, kubelki=KUBELKI, kara=None):
    """Miary jakosci, wszystkie BEZ przewidywania struktury.

    KUBELKI SA NARASTAJACE: wiersz "<= 100 nt" obejmuje takze struktury krotsze niz 50 nt.

    `kara` (opcjonalnie) to funkcja licząca WLASNA kare danego modelu na wygenerowanych sekwencjach.
    Kazdy model oceniamy tylko jego wlasnym kryterium — wartosci roznych kar NIE SA porownywalne
    miedzy modelami i nie wolno ich zestawiac w jednej kolumnie.
    """
    rows = []
    for hi in kubelki:
        sel = [i for i, t in enumerate(structs) if len(t) <= hi]
        if not sel:
            continue
        i_nt, i_par, de = [], [], []
        for i in sel:
            t, q, r = structs[i], gen[i], refs[i]
            a, b = identycznosci(t, q, r)
            i_nt.append(a)
            if b == b:                      # pomijamy struktury bez ani jednej pary
                i_par.append(b)
            de.append((RNA.energy_of_struct(q, t) - RNA.energy_of_struct(r, t)) / len(t))
        r_p, r_z, s_p, s_z = recall_klas(structs, gen, refs, sel)
        w = {"zbior": etykieta, "dlugosc": f"<= {hi} nt", "n": len(sel),
             "identycznosc_nt": 100 * float(np.mean(i_nt)),
             "identycznosc_par": 100 * float(np.mean(i_par)) if i_par else float("nan"),
             "zbal_par": 100 * float(np.mean(list(r_p.values()))) if r_p else float("nan"),
             "zbal_zasady": 100 * float(np.mean(list(r_z.values()))) if r_z else float("nan"),
             "dE_nt": float(np.mean(de))}
        for k in ("GC", "AU", "GU"):
            w[f"recall_{k}"] = 100 * r_p[k] if k in r_p else float("nan")
            w[f"spec_{k}"] = 100 * s_p[k] if k in s_p else float("nan")
        for b in "ACGU":
            w[f"recall_petle_{b}"] = 100 * r_z[b] if b in r_z else float("nan")
            w[f"spec_petle_{b}"] = 100 * s_z[b] if b in s_z else float("nan")
        if kara is not None:
            w["kara_wlasna"] = float(np.mean([kara(structs[i], gen[i]) for i in sel]))
        rows.append(w)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--baseline", action="store_true", help="losowa sekwencja kanoniczna zamiast modelu")
    ap.add_argument("--tryb-podzialu", choices=["rodzinowy"], default="rodzinowy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--na", choices=["test", "val"], default="test",
                    help="'val' = do strojenia (wolno wielokrotnie), 'test' = do raportu (raz)")
    ap.add_argument("--eterna-max", type=int, default=200,
                    help="gorny limit dlugosci zagadek Eterny; musi byc <= limitu z src/cdhit.py")
    ap.add_argument("--dekodowanie", choices=["argmax", "probkowanie"], default="probkowanie",
                    help="jak zamienic rozklady modelu na sekwencje; MUSI zgadzac sie z trybem "
                         "uzytym w walidacji podczas treningu. `argmax` zostawiony wylacznie do "
                         "odtworzenia diagnozy degeneracji — patrz docstring `generuj`")
    ap.add_argument("--seed-dekodowania", type=int, default=0,
                    help="ziarno losowania przy --dekodowanie probkowanie; czesc wyniku")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    ANALIZA.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = wczytaj()
    idx = wczytaj_split(args.tryb_podzialu, args.seed)[args.na]
    S = df.secondary_structure.iloc[idx].tolist()
    Q = df.sequence.iloc[idx].tolist()
    ET, ER = eterna(args.eterna_max)

    if args.baseline:
        nazwa, nazwa_kary, kara = "baseline losowy kanoniczny", "brak kary za sklad", None
        G = [losowa_kanoniczna(t) for t in S]
        GE = [losowa_kanoniczna(t) for t in ET]
    else:
        nazwa = args.ckpt
        model, a_ck = wczytaj_model(args.ckpt, dev)
        nazwa_kary, kara = wlasna_kara(a_ck)
        G = generuj(model, S, dev, dekodowanie=args.dekodowanie, seed=args.seed_dekodowania)
        GE = generuj(model, ET, dev, dekodowanie=args.dekodowanie, seed=args.seed_dekodowania)

    print(f"{nazwa}")
    print(f"zbior {args.na.upper()}: {len(S)} struktur | Eterna <= {args.eterna_max} nt: {len(ET)}")
    print(f"dekodowanie: {args.dekodowanie}"
          + (f" (ziarno {args.seed_dekodowania})" if args.dekodowanie == "probkowanie" else ""))
    print(f"wlasna kara za sklad: {nazwa_kary}")
    print()

    w = ocen(S, G, Q, args.na, kara=kara) + ocen(ET, GE, ER, "eterna", kara=kara)
    d = pd.DataFrame(w)
    naglowek = (f"{'zbior':<10}{'dlugosc':>11}{'n':>6}{'ident_nt':>10}{'ident_par':>11}"
                f"{'zbal_par':>10}{'zbal_zas':>10}{'dE/nt':>9}")
    if kara is not None:
        naglowek += f"{'kara_wlasna':>13}"
    print(naglowek)
    for _, r in d.iterrows():
        wiersz = (f"{r.zbior:<10}{r.dlugosc:>11}{r.n:>6}{r.identycznosc_nt:>9.1f}%"
                  f"{r.identycznosc_par:>10.1f}%{r.zbal_par:>9.1f}%{r.zbal_zasady:>9.1f}%"
                  f"{r.dE_nt:>9.4f}")
        if kara is not None:
            wiersz += f"{r.kara_wlasna:>13.4f}"
        print(wiersz)

    print(f"\nCZULOSC OSOBNO DLA KAZDEJ KLASY (kubelek <= {KUBELKI[-1]} nt)")
    print(f"  {'zbior':<10}" + "".join(f"{'par ' + k:>10}" for k in ("GC", "AU", "GU"))
          + "   " + "".join(f"{'petle ' + b:>11}" for b in "ACGU"))
    for _, r in d[d.dlugosc == f"<= {KUBELKI[-1]} nt"].iterrows():
        print(f"  {r.zbior:<10}" + "".join(f"{r['recall_' + k]:>9.1f}%" for k in ("GC", "AU", "GU"))
              + "   " + "".join(f"{r['recall_petle_' + b]:>10.1f}%" for b in "ACGU"))
    print(f"  {'poziom losowy':<10}" + "".join(f"{100/3:>9.1f}%" for _ in range(3))
          + "   " + "".join(f"{25.0:>10.1f}%" for _ in range(4)))

    print(f"\nSPECYFICZNOSC OSOBNO DLA KAZDEJ KLASY (kubelek <= {KUBELKI[-1]} nt)")
    print(f"  {'zbior':<10}" + "".join(f"{'par ' + k:>10}" for k in ("GC", "AU", "GU"))
          + "   " + "".join(f"{'petle ' + b:>11}" for b in "ACGU"))
    for _, r in d[d.dlugosc == f"<= {KUBELKI[-1]} nt"].iterrows():
        print(f"  {r.zbior:<10}" + "".join(f"{r['spec_' + k]:>9.1f}%" for k in ("GC", "AU", "GU"))
              + "   " + "".join(f"{r['spec_petle_' + b]:>10.1f}%" for b in "ACGU"))

    print("\nKubelki NARASTAJACE: wiersz '<= 100 nt' obejmuje takze struktury krotsze niz 50 nt.")
    print("zbal_par / zbal_zas to srednie czulosci BEZ WAGI. Predyktor staly (np. zawsze G:C) spada")
    print("tam do poziomu losowego, mimo ze na zwyklej identycznosci wypada lepiej niz baseline.")

    if args.csv:
        d.insert(0, "model", nazwa)
        d.insert(1, "kara", nazwa_kary)
        d.to_csv(ANALIZA / args.csv, index=False, encoding="utf-8")
        print(f"\nzapisano: experiments/analysis/{args.csv}")


if __name__ == "__main__":
    main()
