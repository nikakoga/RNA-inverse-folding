"""Pomiar SKLADU celu kary za sklad w E1.

SKAD BIERZEMY LICZBY. Z NASZEGO WLASNEGO zbioru roboczego `data/working.parquet` — wszystkich 3640
sekwencji, czyli train + val + test razem. Cel ma opisywac sklad tych danych, na ktorych model
pracuje, a nie innej populacji.

CO TO ZNACZY DLA INTERPRETACJI, i trzeba to zapisac w pracy. Cel obejmuje takze zbior testowy, wiec
zdanie "model trafil w sklad testu" przestaje byc dowodem, ze nauczyl sie go z danych — czesc tej
informacji dostal wprost w celu kary. Dotyczy to WYLACZNIE miar skladu i wylacznie modeli z kara E1.
Na trafnosc (`zbal_par`, wskaznik Youdena) nie ma wplywu: globalne proporcje nie mowia, ktora para
stoi w ktorym miejscu.

DLACZEGO ZMIENILISMY. Poprzedni cel pochodzil z trzech baz zewnetrznych (RNAStrAlign, bpRNA,
ArchiveII; n = 29 571, z wykluczeniem struktur obecnych w naszej walidacji i tescie) i wynosil
G:C 0,599. Pokrywal sie z naszym TRENINGIEM (0,600), ale nie z walidacja ani testem (obie 0,484):

    train   G:C 0,600   A:U 0,297   G:U 0,103
    val     G:C 0,484   A:U 0,386   G:U 0,130
    test    G:C 0,484   A:U 0,371   G:U 0,145

Kara prowadzila wiec model do wartosci o 11,5 punktu procentowego za wysokiej dla zbiorow, na
ktorych go oceniamy, i tym samym WZMACNIALA przesuniecie rozkladu wynikajace z podzialu rodzinowego,
zamiast je korygowac.

CZEGO NIE ROBIMY. Nie przepisujemy liczb z NEMO (Portela, bioRxiv 345587). NEMO wypelnia pary
rozkladem 60/33/7, a pozycje niesparowane rozkladem 93% adeniny — to sa HEURYSTYKI PROJEKTOWE
nastawione na niezawodne zwijanie, a nie opis natury. Uzycie ich jako celu popchneloby model wprost
ku degeneracji poli-A, ktorej ta kara ma zapobiegac.

Wynik jest WPISANY NA SZTYWNO w `src/dataset.py`. Ten modul sluzy do jego odtworzenia i sprawdzenia:

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

TYP_PARY = {"GC": "GC", "CG": "GC", "AU": "AU", "UA": "AU", "GU": "GU", "UG": "GU"}


def zmierz(df: pd.DataFrame) -> tuple[dict, dict, int, int]:
    """Udzialy zasad na pozycjach NIESPAROWANYCH oraz udzialy TYPOW par.

    Liczymy dokladnie tak, jak mierzy je kara w `src/loss.py`: pary po TYPIE (orientacja bez
    znaczenia), zasady tylko na pozycjach niesparowanych.
    """
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
            {k: typy[k] / n_t for k in ("GC", "AU", "GU")}, n_t, n_p)


def main():
    from src.prepare import wczytaj
    from src.split import wczytaj_split

    work = wczytaj()
    spl = wczytaj_split("rodzinowy")

    print(f"{'zbior':<24}{'par':>9}{'petli':>9}   |{'G:C':>7}{'A:U':>7}{'G:U':>7}"
          f"   |{'A':>7}{'C':>7}{'G':>7}{'U':>7}")
    print("-" * 92)
    for nazwa in ("train", "val", "test"):
        p, t, n_t, n_p = zmierz(work.iloc[spl[nazwa]])
        print(f"{nazwa:<24}{n_t:>9}{n_p:>9}   |"
              + "".join(f"{t[k]:>7.3f}" for k in ("GC", "AU", "GU"))
              + "   |" + "".join(f"{p[b]:>7.3f}" for b in "ACGU"))

    petle, pary, n_t, n_p = zmierz(work)
    print("-" * 92)
    print(f"{'CALY ZBIOR = CEL':<24}{n_t:>9}{n_p:>9}   |"
          + "".join(f"{pary[k]:>7.3f}" for k in ("GC", "AU", "GU"))
          + "   |" + "".join(f"{petle[b]:>7.3f}" for b in "ACGU"))

    print(f"\nsekwencji: {len(work)}\n")
    print("NATURAL_LOOP = {" + ", ".join(f'"{b}": {petle[b]:.3f}' for b in "ACGU") + "}")
    print("NATURAL_PAIR = {" + ", ".join(f'"{k}": {pary[k]:.3f}' for k in ("GC", "AU", "GU")) + "}")

    from src.dataset import NATURAL_LOOP, NATURAL_PAIR
    zgodne = (all(abs(petle[b] - NATURAL_LOOP[b]) < 5e-4 for b in "ACGU")
              and all(abs(pary[k] - NATURAL_PAIR[k]) < 5e-4 for k in ("GC", "AU", "GU")))
    print("\n" + ("ZGODNE ze stalymi w src/dataset.py"
                  if zgodne else "ROZBIEZNE ze stalymi w src/dataset.py — zaktualizuj je"))


if __name__ == "__main__":
    main()
