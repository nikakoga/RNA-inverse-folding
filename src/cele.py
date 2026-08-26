"""Pomiar SKLADU NATURALNEGO RNA — celu kary za sklad w E1.

SKAD BIERZEMY LICZBY. Z trzech opublikowanych baz struktur drugorzedowych RNA, ktore razem tworza
`data/raw/rna_raw.parquet`:

    RNAStrAlign   Tan i wsp., Nucleic Acids Research 2017
    bpRNA         Danaee i wsp., Nucleic Acids Research 2018
    ArchiveII     Sloma i Mathews

CZEGO NIE ROBIMY. Nie mierzymy tego na naszym zbiorze roboczym. Pula surowa zawiera 1455 struktur,
ktore trafily potem do naszej walidacji albo testu — gdyby zostaly w pomiarze, cel kary nioslby
informacje ze zbioru testowego. Wykluczamy je po parze (sekwencja, struktura).

CZEGO TEZ NIE ROBIMY. Nie przepisujemy liczb z NEMO (Portela, bioRxiv 345587). NEMO wypelnia pary
rozkladem 60/33/7, a pozycje niesparowane rozkladem 93% adeniny — to sa HEURYSTYKI PROJEKTOWE
nastawione na niezawodne zwijanie, a nie opis natury. Uzycie ich jako celu popchneloby model wprost
ku degeneracji poli-A, ktorej ta kara ma zapobiegac.

Zmierzone przez nas G:C = 0,599 pokrywa sie za to z priorem 0,593 z NEMO, co jest niezaleznym
potwierdzeniem. Rozbieznosc na A:U i G:U wynika z tego, ze NEMO celowo tlumi pary chwiejne.

Wynik jest WPISANY NA SZTYWNO w `src/loss.py`, zeby cel nie zmienial sie wraz z podzialem danych.
Ten modul sluzy do jego odtworzenia i sprawdzenia:

    python -m src.cele
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.dataset import parse_pairs

RAW = ROOT / "data" / "raw" / "rna_raw.parquet"
TYP_PARY = {"GC": "GC", "CG": "GC", "AU": "AU", "UA": "AU", "GU": "GU", "UG": "GU"}


def zmierz(df: pd.DataFrame) -> tuple[dict, dict]:
    """Udzialy zasad na pozycjach NIESPAROWANYCH oraz udzialy TYPOW par."""
    petle, typy = Counter(), Counter()
    for st, sq in zip(df.secondary_structure, df.sequence):
        st, sq = str(st), str(sq)
        pary = parse_pairs(st)
        sparowane = {k for p in pary for k in p}
        for i, j in pary:
            t = TYP_PARY.get(sq[i] + sq[j])
            if t:
                typy[t] += 1
        petle.update(sq[k] for k in range(len(sq)) if k not in sparowane)
    n_p = sum(petle.values()) or 1
    n_t = sum(typy.values()) or 1
    return ({b: petle[b] / n_p for b in "ACGU"},
            {k: typy[k] / n_t for k in ("GC", "AU", "GU")})


def cele() -> tuple[dict, dict]:
    """Skład naturalny z baz publikowanych, BEZ struktur obecnych w naszej walidacji i tescie."""
    from src.prepare import wczytaj
    from src.split import wczytaj_split

    raw = pd.read_parquet(RAW)
    raw = raw[raw.sequence.str.fullmatch(r"[ACGU]+")].reset_index(drop=True)

    work = wczytaj()
    spl = wczytaj_split("rodzinowy")
    trefne = set(zip(work.sequence.iloc[spl["val"] + spl["test"]],
                     work.secondary_structure.iloc[spl["val"] + spl["test"]]))
    czyste = pd.Series([k not in trefne for k in zip(raw.sequence, raw.secondary_structure)],
                       index=raw.index)
    return zmierz(raw[czyste]), (len(raw), int(czyste.sum()))


def main():
    (petle, pary), (n_all, n_ok) = cele()
    print(f"pula surowa (alfabet ACGU): {n_all}")
    print(f"  po wykluczeniu naszej walidacji i testu: {n_ok}   (-{n_all - n_ok})\n")
    print("NATURAL_LOOP = {" + ", ".join(f'"{b}": {petle[b]:.3f}' for b in "ACGU") + "}")
    print("NATURAL_PAIR = {" + ", ".join(f'"{k}": {pary[k]:.3f}' for k in ("GC", "AU", "GU")) + "}")

    from src.loss import NATURAL_LOOP, NATURAL_PAIR
    zgodne = (all(abs(petle[b] - NATURAL_LOOP[b]) < 5e-4 for b in "ACGU")
              and all(abs(pary[k] - NATURAL_PAIR[k]) < 5e-4 for k in ("GC", "AU", "GU")))
    print("\n" + ("ZGODNE ze stalymi w src/loss.py"
                  if zgodne else "ROZBIEZNE ze stalymi w src/loss.py — zaktualizuj je"))


if __name__ == "__main__":
    main()
