"""Podzial 60/20/20 — losowy albo FAMILY-AWARE.

DLACZEGO FAMILY-AWARE. Przy podziale losowym po strukturach ta sama rodzina Rfam trafia i do treningu,
i do testu. Model moze wtedy odtwarzac zapamietany wzorzec rodziny zamiast generalizowac. Zmierzona
roznica jest duza: identycznosc sekwencyjna spada z 0,49 do 0,31, gdy rodziny sa rozdzielone.

PUŁAPKA, KTORA TRZEBA OBEJSC. Naiwne pakowanie rodzin po samej liczebnosci wprowadza PRZESUNIECIE
ROZKLADU DLUGOSCI (train mediana 77 nt wobec val 119 nt) i oddaje 85% walidacji jednej rodzinie.
Test mierzylby wtedy roznice dlugosci, a nie generalizacje miedzy rodzinami.

ROZWIAZANIE. Pakujemy rodziny minimalizujac koszt laczony z trzech skladnikow: odchylenie liczebnosci
od 60/20/20, odchylenie rozkladu dlugosci od calej puli, oraz dominacje pojedynczej rodziny w zbiorze.
Losujemy kolejnosc rodzin wielokrotnie i bierzemy najlepszy uklad. Deterministyczne przy danym seedzie.

Uzycie:
    python -m src.split --tryb rodzinowy
    python -m src.split --tryb losowy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.prepare import wczytaj

KUBELKI = [0, 50, 80, 110, 150, 10**9]
CELE = {"train": 0.6, "val": 0.2, "test": 0.2}
SPLITS = ROOT / "data" / "splits"


def sciezka(tryb: str, seed: int = 0) -> Path:
    return SPLITS / f"split_{tryb}_seed{seed}.json"


def _hist(lens: np.ndarray) -> np.ndarray:
    h = np.histogram(lens, bins=KUBELKI)[0].astype(float)
    return h / max(h.sum(), 1)


def _koszt(przydzial, fam_n, fam_h, cel_h, N) -> float:
    k = 0.0
    for s, cel in CELE.items():
        fams = [f for f, b in przydzial.items() if b == s]
        n = sum(fam_n[f] for f in fams)
        if n == 0:
            return 1e9
        k += 3.0 * abs(n / N - cel) / cel
        h = sum(fam_h[f] * fam_n[f] for f in fams) / n
        k += 1.0 * np.abs(h - cel_h).sum()
        k += 2.0 * max(0.0, max(fam_n[f] for f in fams) / n - 0.40)
    return k


def rodzinowy(df: pd.DataFrame, seed: int = 0, prob: int = 400):
    fam_n = df.family.value_counts().to_dict()
    fam_h = {f: _hist(df.loc[df.family == f, "secondary_structure"].str.len().values) for f in fam_n}
    cel_h = _hist(df.secondary_structure.str.len().values)
    N = len(df)
    rng = np.random.RandomState(seed)
    rodziny = list(fam_n)
    best, best_k = None, float("inf")
    for _ in range(prob):
        kolejnosc = sorted(rodziny, key=lambda f: -fam_n[f] * (1 + 0.35 * rng.randn()))
        przyd, tot = {}, {s: 0 for s in CELE}
        for f in kolejnosc:
            s = min(CELE, key=lambda s: (tot[s] + fam_n[f]) / (CELE[s] * N))
            przyd[f] = s
            tot[s] += fam_n[f]
        k = _koszt(przyd, fam_n, fam_h, cel_h, N)
        if k < best_k:
            best, best_k = przyd, k
    return {s: df.index[df.family.map(best) == s].tolist() for s in CELE}, best_k


def losowy(df: pd.DataFrame, seed: int = 0):
    idx = np.random.RandomState(seed).permutation(len(df))
    a, b = int(0.6 * len(df)), int(0.8 * len(df))
    return {"train": idx[:a].tolist(), "val": idx[a:b].tolist(), "test": idx[b:].tolist()}, 0.0


def zbuduj(tryb: str = "rodzinowy", seed: int = 0, gadaj: bool = True) -> dict:
    df = wczytaj()
    spl, k = (rodzinowy(df, seed) if tryb == "rodzinowy" else losowy(df, seed))
    SPLITS.mkdir(parents=True, exist_ok=True)
    json.dump(spl, open(sciezka(tryb, seed), "w"))

    if gadaj:
        print(f"podzial {tryb.upper()}, seed {seed}, struktur {len(df)}, rodzin {df.family.nunique()}"
              + (f", koszt ukladu {k:.4f}" if tryb == "rodzinowy" else ""))
        print(f"\n{'zbior':7}{'n':>6}{'udzial':>8}{'rodzin':>8}{'mediana L':>11}{'<=50 nt':>9}"
              f"{'max rodzina':>13}   najwieksze")
        for s in CELE:
            d = df.iloc[spl[s]]
            vc = d.family.value_counts()
            print(f"{s:7}{len(d):>6}{len(d)/len(df)*100:>7.1f}%{d.family.nunique():>8}"
                  f"{int(d.secondary_structure.str.len().median()):>11}"
                  f"{int((d.secondary_structure.str.len() <= 50).sum()):>9}"
                  f"{vc.iloc[0]/len(d)*100:>12.0f}%   {', '.join(vc.index[:3])}")
        wsp = set()
        for a in CELE:
            for b in CELE:
                if a < b:
                    wsp |= set(df.iloc[spl[a]].family) & set(df.iloc[spl[b]].family)
        print(f"\nrodziny wspolne miedzy zbiorami: {len(wsp)}"
              + ("  (musi byc 0)" if tryb == "rodzinowy" else "  (przy podziale losowym to normalne)"))
        print(f"zapisano: {sciezka(tryb, seed).relative_to(ROOT)}")
    return spl


def wczytaj_split(tryb: str = "rodzinowy", seed: int = 0) -> dict:
    p = sciezka(tryb, seed)
    return json.load(open(p)) if p.exists() else zbuduj(tryb, seed, gadaj=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tryb", choices=["rodzinowy", "losowy"], default="rodzinowy")
    ap.add_argument("--seed", type=int, default=0)
    zbuduj(**vars(ap.parse_args()))
