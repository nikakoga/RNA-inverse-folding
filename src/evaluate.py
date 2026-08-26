"""Ocena modelu. Dwa zbiory, dwie miary — ZADNA nie przewiduje struktury.

ZBIORY
  TEST NATURALNY    20% puli, rodziny nieobecne w treningu. Ocena GLOWNA.
  ETERNA <= 200 nt  zagadki projektowe ludzi. Pomocnicza, zewnetrzna wzgledem naszych danych.

MIARY
  identycznosc_nt   ulamek pozycji, na ktorych trafilismy w litere prawdziwej sekwencji.
  dE/nt    [E(cel | nasza) - E(cel | prawdziwa)] / dlugosc. Ujemne = nasza stabilizuje cel
           LEPIEJ niz prawdziwa sekwencja.

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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import RNA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.dataset import koduj, losowa_kanoniczna, parse_pairs
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
def generuj(model, structs, device, bs=32):
    out = []
    for s in range(0, len(structs), bs):
        cs = structs[s : s + bs]
        sid, pad, par, _, _, _ = koduj(cs, None, device)
        out += model.generate(sid, pad, par, [len(x) for x in cs])
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
                       co sekwencja wzorcowa

    identycznosc_par   ulamek PAR (a nie pozycji), w ktorych model trafil w TYP pary, niezaleznie
                       od orientacji: gdy wzorzec ma G-C, a model C-G, to TRAFIENIE.
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
    ze tam kara liczy sie na rozkladach prawdopodobienstwa, a tu na gotowej sekwencji po argmax.
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


@torch.no_grad()
def perplexity(model, structs, seqs, device, bs=32) -> float:
    """exp(srednia cross-entropia) — "z ilu zasad model srednio wybiera na kazdej pozycji".

    Zakres od 1 (model calkowicie pewny) do 4 (rozklad jednostajny, zero informacji). To jedyna
    wielkosc liczona na ROZKLADACH, nie na wyjsciu.

    Dlaczego wlasnie ona, a nie surowa wartosc straty: perplexity zalezy WYLACZNIE od cross-entropii,
    ktora jest identyczna we wszystkich naszych modelach. Sumy strat nie da sie porownywac miedzy
    modelami o roznych funkcjach kary.
    """
    import torch.nn.functional as F
    suma, ile = 0.0, 0
    for s in range(0, len(structs), bs):
        cs, cq = structs[s : s + bs], seqs[s : s + bs]
        sid, pad, par, cp, cz, _ = koduj(cs, cq, device)
        lp, lz, _ = model(sid, pad, par)
        for logity, cel in ((lp, cp), (lz, cz)):
            m = cel != -100
            if m.any():
                suma += float(F.cross_entropy(logity[m], cel[m], reduction="sum"))
                ile += int(m.sum())
    return float(np.exp(suma / max(ile, 1)))


def wlasna_kara(args_ckpt):
    """Ktora kara nalezy do tego modelu — odczytane z zapisanych hiperparametrow checkpointu."""
    if args_ckpt.get("w_sklad", 0):
        return "kara TV (nasza)", kara_tv
    if args_ckpt.get("w_sklad_zasad", 0) or args_ckpt.get("w_sklad_par", 0):
        return "kara progi (promotor)", kara_progi
    return "brak kary za sklad", None


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
        w = {"zbior": etykieta, "dlugosc": f"<= {hi} nt", "n": len(sel),
             "identycznosc_nt": 100 * float(np.mean(i_nt)),
             "identycznosc_par": 100 * float(np.mean(i_par)) if i_par else float("nan"),
             "dE_nt": float(np.mean(de))}
        if kara is not None:
            w["kara_wlasna"] = float(np.mean([kara(structs[i], gen[i]) for i in sel]))
        rows.append(w)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--baseline", action="store_true", help="losowa sekwencja kanoniczna zamiast modelu")
    ap.add_argument("--tryb-podzialu", choices=["rodzinowy", "losowy"], default="rodzinowy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--na", choices=["test", "val"], default="test",
                    help="'val' = do strojenia (wolno wielokrotnie), 'test' = do raportu (raz)")
    ap.add_argument("--eterna-max", type=int, default=200,
                    help="gorny limit dlugosci zagadek Eterny; musi byc <= limitu z src/cdhit.py")
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
        ppl = float("nan")
    else:
        nazwa = args.ckpt
        model, a_ck = wczytaj_model(args.ckpt, dev)
        nazwa_kary, kara = wlasna_kara(a_ck)
        G, GE = generuj(model, S, dev), generuj(model, ET, dev)
        ppl = perplexity(model, S, Q, dev)

    print(f"{nazwa}")
    print(f"zbior {args.na.upper()}: {len(S)} struktur | Eterna <= {args.eterna_max} nt: {len(ET)}")
    print(f"wlasna kara za sklad: {nazwa_kary}")
    if ppl == ppl:
        print(f"perplexity na zbiorze {args.na}: {ppl:.3f}   (1 = pewny, 4 = calkiem niezdecydowany)")
    print()

    w = ocen(S, G, Q, args.na, kara=kara) + ocen(ET, GE, ER, "eterna", kara=kara)
    d = pd.DataFrame(w)
    naglowek = (f"{'zbior':<10}{'dlugosc':>11}{'n':>6}{'identycznosc_nt':>17}"
                f"{'identycznosc_par':>18}{'dE/nt':>9}")
    if kara is not None:
        naglowek += f"{'kara_wlasna':>13}"
    print(naglowek)
    for _, r in d.iterrows():
        wiersz = (f"{r.zbior:<10}{r.dlugosc:>11}{r.n:>6}{r.identycznosc_nt:>16.1f}%"
                  f"{r.identycznosc_par:>17.1f}%{r.dE_nt:>9.4f}")
        if kara is not None:
            wiersz += f"{r.kara_wlasna:>13.4f}"
        print(wiersz)

    print("\nKubelki NARASTAJACE: wiersz '<= 100 nt' obejmuje takze struktury krotsze niz 50 nt.")
    print("Kary liczone na WYGENEROWANYCH sekwencjach, a w treningu na rozkladach prawdopodobienstwa")
    print("— liczby nie zgadzaja sie z logiem treningu i to nie jest blad.")

    if args.csv:
        d.insert(0, "model", nazwa)
        d.insert(1, "kara", nazwa_kary)
        d["perplexity"] = ppl
        d.to_csv(ANALIZA / args.csv, index=False, encoding="utf-8")
        print(f"\nzapisano: experiments/analysis/{args.csv}")


if __name__ == "__main__":
    main()
