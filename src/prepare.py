"""Przygotowanie zbioru roboczego z puli naturalnej.

WEJSCIE   data/cdhit/naturalne_cdhit.parquet   — powstaje przez `python -m src.cdhit`

TRZY FILTRY:

  1. PRZEWAGA SPAROWANYCH (paired_fraction >= 0.5)
     Usuwa struktury, w ktorych wiecej pozycji jest niesparowanych niz sparowanych. To w duzej mierze
     dlugie nieustrukturyzowane ogony, na ktorych zadanie projektowe jest zle postawione — nie ma
     czego projektowac.

  2. POPRAWNOSC
     Co najmniej jedna para w strukturze i sekwencja w alfabecie ACGU (bez N i kodow IUPAC
     dla zasad niejednoznacznych).

  3. WYKONALNE PETLE SPINKI (kazda >= 3 nt)
     Szkielet cukrowo-fosforanowy nie zawraca na mniej niz trzech nukleotydach, wiec petla krotsza
     nie istnieje fizycznie i ZADNA sekwencja nie zwinie sie w taka strukture. ViennaRNA zwraca dla
     nich nieskonczonosc (wartownik 1e5), przez co dE przestaje cokolwiek mierzyc. Zrodlem sa
     struktury KONSENSUSOWE Rfam rzutowane na pojedyncze sekwencje: para (najczesciej wobble G:U)
     laduje o jeden krok za daleko i polyka dwa nukleotydy stabilnej tetrapetli.
     Ten sam powod co filtr 1: zadanie projektowe jest zle postawione.

Ograniczenie dlugosci do 200 nt naklada `src.cdhit` — odsiewamy redundancje juz w tej populacji,
ktorej faktycznie uzywamy.

WYJSCIE   data/working.parquet   — zbior, na ktorym liczone sa WSZYSTKIE eksperymenty

Uzycie:
    python -m src.prepare
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.dataset import paired_fraction, spinki_mozliwe

CDHIT = ROOT / "data" / "cdhit" / "naturalne_cdhit.parquet"
OUT = ROOT / "data" / "working.parquet"


def _filtruj(df: pd.DataFrame, nazwa: str, gadaj: bool) -> pd.DataFrame:
    n0 = len(df)
    df = df[df.secondary_structure.map(lambda s: paired_fraction(str(s)) >= 0.5)].reset_index(drop=True)
    n1 = len(df)
    ok = df.secondary_structure.str.contains(r"\(") & df.sequence.str.fullmatch(r"[ACGU]+")
    df = df[ok].reset_index(drop=True)
    n2 = len(df)
    df = df[df.secondary_structure.map(lambda s: spinki_mozliwe(str(s)))].reset_index(drop=True)
    n3 = len(df)
    if gadaj:
        print(f"{nazwa:<34}{n0:>6}")
        print(f"{'  po przewadze sparowanych':<34}{n1:>6}   (-{n0-n1})")
        print(f"{'  po kontroli poprawnosci':<34}{n2:>6}   (-{n1-n2})")
        print(f"{'  po kontroli petli spinki':<34}{n3:>6}   (-{n2-n3})")
    return df


def przygotuj(gadaj: bool = True) -> pd.DataFrame:
    if not CDHIT.exists():
        raise SystemExit(f"Brak {CDHIT.relative_to(ROOT)} — najpierw uruchom `python -m src.cdhit`")

    df = _filtruj(pd.read_parquet(CDHIT), "pula naturalna po cd-hit-est", gadaj)
    df.to_parquet(OUT, index=False)

    if gadaj:
        L = df.secondary_structure.str.len()
        print(f"\nrodzin Rfam: {df.family.nunique()}")
        print(f"dlugosc: min {L.min()}, mediana {int(L.median())}, max {L.max()}")
        print(f"kubelki  <=50: {(L<=50).sum()}   51-100: {((L>50)&(L<=100)).sum()}   "
              f"101-200: {(L>100).sum()}")
        print(f"\nzapisano: {OUT.relative_to(ROOT)}")
    return df


def wczytaj() -> pd.DataFrame:
    if not OUT.exists():
        return przygotuj(gadaj=False)
    return pd.read_parquet(OUT)


if __name__ == "__main__":
    przygotuj()
