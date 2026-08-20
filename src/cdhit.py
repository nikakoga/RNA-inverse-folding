"""Odsiewanie redundancji sekwencyjnej narzedziem cd-hit-est.

DLACZEGO NATYWNE NARZEDZIE, A NIE WLASNA IMPLEMENTACJA. Reimplementacja w Pythonie (pokrycie
8-merow >= 0.8) usuwala 2,4x mniej sekwencji niz `cd-hit-est` na tej samej puli. Redundancja, ktora
zostawala, trafiala potem i do treningu, i do testu — czyli zawyzala wyniki. Tutaj caly ten krok
wykonuje wylacznie cd-hit-est; jesli narzedzie jest niedostepne, skrypt sie zatrzymuje i nie ma
zadnej sciezki zastepczej.

CO ROBI CD-HIT. Grupuje sekwencje o podobienstwie >= progu i z kazdej grupy zostawia JEDNA,
reprezentatywna. Parametry `-c 0.8 -n 5` sa standardem dla kwasow nukleinowych przy tym progu:
`-n` to dlugosc slowa uzywanego do wstepnego filtrowania i dla `-c` w przedziale 0,75-0,8
dokumentacja zaleca 5.

DWIE PULE ODSIEWANE OSOBNO:
  * sekwencje naturalne, ograniczone do <= 200 nt,
  * zagadki Eterny z rozwiazaniami graczy, ograniczone do <= 50 nt.
To sa rozne zbiory o roznym pochodzeniu — laczenie ich zaburzyloby grupowanie.

WYMAGANIA. cd-hit-est jest napisany w C++ i nie ma wersji dla Windows; uruchamiamy go przez WSL.
Instalacja opisana w README.

Uzycie:
    python -m src.cdhit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAW = ROOT / "data" / "raw" / "rna_raw.parquet"
ETERNA = ROOT / "data" / "raw" / "eterna100.tsv"
CDHIT_DIR = ROOT / "data" / "cdhit"
PROG, SLOWO = 0.8, 5

BRAK_NARZEDZIA = """cd-hit-est jest niedostepny.

Narzedzie jest napisane w C++ i nie ma wersji dla Windows — uruchamiamy je przez WSL.
Instalacja, raz:

    PowerShell JAKO ADMINISTRATOR:   wsl --install
    (restart komputera)
    zwykly terminal:                 wsl -- sudo apt-get update
                                     wsl -- sudo apt-get install -y cd-hit
"""


def _wsl(polecenie: str) -> tuple[int, str]:
    """Uruchamia polecenie w WSL. Zwraca (kod_wyjscia, polaczone wyjscie).

    `wsl.exe` pisze wlasne komunikaty bledow w UTF-16LE, wiec dekodujemy je osobno — inaczej
    zamiast czytelnej tresci dostajemy krzaki.
    """
    try:
        r = subprocess.run(["wsl", "--", "bash", "-lc", polecenie], capture_output=True)
    except FileNotFoundError:
        raise SystemExit(BRAK_NARZEDZIA)
    surowe = (r.stdout or b"") + (r.stderr or b"")
    # wsl.exe pisze w UTF-16LE, cd-hit-est w UTF-8. Rozpoznajemy po bajtach zerowych.
    kod = "utf-16-le" if surowe.count(bytes([0])) > len(surowe) // 4 else "utf-8"
    return r.returncode, surowe.decode(kod, "replace").replace(chr(0), "").strip()


def _win2wsl(p: Path) -> str:
    """Sciezka Windows -> sciezka widziana z WSL: C:\\Users\\... -> /mnt/c/Users/..."""
    s = str(p.resolve()).replace("\\", "/")
    return f"/mnt/{s[0].lower()}{s[2:]}"


def odsiej(df: pd.DataFrame, nazwa: str, prog: float = PROG, slowo: int = SLOWO) -> pd.DataFrame:
    """Zapisuje FASTA, uruchamia cd-hit-est, zwraca wiersze, ktore narzedzie zachowalo."""
    CDHIT_DIR.mkdir(parents=True, exist_ok=True)
    fa = CDHIT_DIR / f"{nazwa}.fasta"
    out = CDHIT_DIR / f"{nazwa}_c{prog}.fasta"

    with open(fa, "w", encoding="utf-8") as f:
        for i, s in enumerate(df.sequence):
            f.write(f">s{i}\n{s}\n")

    kod, komunikat = _wsl(f"cd-hit-est -i {_win2wsl(fa)} -o {_win2wsl(out)} "
                          f"-c {prog} -n {slowo} -M 2000 -T 4 -d 0")
    if kod != 0 or not out.exists():
        niski = komunikat.lower()
        if not komunikat or "not found" in niski or "podsystem" in niski or "subsystem" in niski:
            raise SystemExit(BRAK_NARZEDZIA)
        raise SystemExit("cd-hit-est zwrocil blad:\n" + komunikat)

    # Ktore sekwencje zostaly, czytamy z pliku wyjsciowego narzedzia — decyzje podejmuje cd-hit.
    zostaly = {int(l[2:].split()[0]) for l in open(out, encoding="utf-8") if l.startswith(">")}
    kept = df.iloc[sorted(zostaly)].reset_index(drop=True)
    print(f"{nazwa:<22}{len(df):>6} -> {len(kept):>6}   "
          f"(-{len(df) - len(kept)}, {100 * (1 - len(kept) / max(len(df), 1)):.0f}%)")
    return kept


def eterna_surowa(max_len: int = 50) -> pd.DataFrame:
    """Zagadki Eterny z rozwiazaniem gracza, ograniczone do zadanej dlugosci."""
    E = pd.read_csv(ETERNA, sep="\t")
    col = "Secondary Structure V2"
    rows = []
    for st, sq in zip(E[col], E["Sample Solution (V2/Vienna2)"]):
        st, sq = str(st).strip(), str(sq).strip()
        if (len(st) <= max_len and len(st) == len(sq) and "(" in st
                and set(st) <= set(".()") and set(sq) <= set("ACGU")):
            rows.append({"sequence": sq, "secondary_structure": st})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-len", type=int, default=200,
                    help="gorny limit dlugosci dla puli naturalnej")
    ap.add_argument("--eterna-max", type=int, default=50)
    ap.add_argument("--prog", type=float, default=PROG,
                    help="prog podobienstwa sekwencyjnego dla cd-hit-est")
    args = ap.parse_args()

    print(f"cd-hit-est, prog podobienstwa {args.prog}, dlugosc slowa {SLOWO}\n")

    df = pd.read_parquet(RAW)
    df = df[df.secondary_structure.str.len() <= args.max_len].reset_index(drop=True)
    odsiej(df, "naturalne", args.prog).to_parquet(CDHIT_DIR / "naturalne_cdhit.parquet", index=False)

    et = eterna_surowa(args.eterna_max)
    odsiej(et, "eterna", args.prog).to_parquet(CDHIT_DIR / "eterna_cdhit.parquet", index=False)

    print(f"\nzapisano do data/cdhit/. Nastepny krok: python -m src.prepare")


if __name__ == "__main__":
    main()
