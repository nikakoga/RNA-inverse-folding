"""Ile z roznicy miedzy konfiguracjami jest realne, a ile to szum losowej inicjalizacji?

NA WALIDACJI — testu nie dotykamy.

PO CO. Kazde zdanie w stylu "roznica 0,7 pp jest istotna" wymaga progu, ponizej ktorego roznicy
nie ma. Ten skrypt ten prog mierzy: trenuje TE SAMA konfiguracje kilka razy, zmieniajac wylacznie
ziarno inicjalizacji wag i kolejnosci partii, i patrzy, jak bardzo rozjezdzaja sie wyniki.

DWA OSOBNE ZIARNA, i to jest kluczowe. `--seed` wyznacza PODZIAL danych, wiec jego zmiana zmienia
zbior walidacyjny i wyniki przestaja byc porownywalne. `--seed-modelu` rusza tylko inicjalizacje
wag i kolejnosc partii — i tylko to chcemy tu zmieniac.

DLACZEGO NA NOWO. Poprzedni pomiar (`experiments/analysis/szum_wagi_klas.csv`) powstal w innym
rezimie: kryterium `zlozony` i cierpliwosc 10. Prog stamtad (±0,26 pp) byl potem cytowany przy
wynikach liczonych juz obecnym protokolem, co jest niescisloscia. Tu mierzymy go w tych samych
warunkach, w ktorych zapadaja wnioski.

TRZY KONFIGURACJE, wszystkie na tych samych ziarnach:

    E1   energia + parowania + kara TV (oba czlony)
    CE   bez zadnych kar
    E3   energia + parowania + kara TV TYLKO na petle

E1 i E3 rozni wylacznie zdjecie czlonu par, a chodza po tych samych ziarnach, wiec ich porownanie
jest PAROWANE — mocniejsze niz zestawienie dwoch niezaleznych sredniej.

PROTOKOL identyczny z run.py: 60 epok bez wczesnego zatrzymania, wybor epoki po `zbal_par`,
probkowanie z ziarnem 0.

UWAGA PRZY CZYTANIU. Raportujemy dwie wielkosci na przebieg:

    zbal_par w zapisanej epoce   to, co trafia do tabel — ale jest to MAKSIMUM z 60 szumiacych
                                 pomiarow, wiec zawyzone
    zbal_par srednia z 10 epok   wielkosc bez tego obciazenia, do oceny samego rozrzutu

Odchylenie standardowe liczymy z obu; do progu istotnosci uzywamy tego dla zapisanej epoki, bo to
tej wielkosci uzywamy w porownaniach.

Uzycie:
    python -m src.szum                   # 3 konfiguracje x 3 ziarna, ok. 54 min
                                         # wznawialny: gotowe przebiegi czyta z CSV
    python -m src.szum --ziarna 5        # dokladniej
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WYNIK = ROOT / "experiments" / "analysis" / "szum_ziaren.csv"
TMP = "checkpoints/_szum_tmp.pt"

KONFIGURACJE = [
    ("E1  kara TV", ["--w-energia", "1.0", "--w-parowania", "1.0", "--w-sklad", "1.0"]),
    ("CE  bez kar", []),
    # E3 na TYCH SAMYCH ziarnach co E1 — porownanie jest wtedy PAROWANE, czyli mocniejsze:
    # w kazdej parze rozni je wylacznie zdjecie czlonu par, a nie losowa inicjalizacja.
    ("E3  tylko petle", ["--w-energia", "1.0", "--w-parowania", "1.0",
                         "--w-sklad-tv-petle", "1.0"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epoki", type=int, default=60)
    ap.add_argument("--ziarna", type=int, default=3)
    args = ap.parse_args()

    wiersze, t0 = [], time.time()
    if WYNIK.exists():
        wiersze = pd.read_csv(WYNIK).to_dict("records")
        print(f"wznawiam: {len(wiersze)} przebiegow juz w pliku")
    zrobione = {(w["konfiguracja"], w["ziarno"]) for w in wiersze}

    print(f"Szum ziarna: {len(KONFIGURACJE)} konfiguracje x {args.ziarna} ziaren, "
          f"po {args.epoki} epok. Podzial NIEZMIENIONY.\n")

    for opis, extra in KONFIGURACJE:
        for z in range(args.ziarna):
            if (opis, z) in zrobione:
                continue
            cmd = [sys.executable, "-m", "src.train", "--epoki", str(args.epoki),
                   "--tryb-podzialu", "rodzinowy", "--wybor", "zbal_par",
                   "--dekodowanie", "probkowanie", "--seed-dekodowania", "0",
                   "--cierpliwosc", str(args.epoki), "--seed-modelu", str(z),
                   *extra, "--out", TMP]
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                print(r.stdout[-900:], r.stderr[-600:])
                raise SystemExit(f"trening padl: {opis} ziarno {z}")

            # linia zapisanej epoki
            L = [l for l in r.stdout.splitlines() if "*najlepszy" in l][-1]
            # wszystkie epoki, do sredniej z ogona
            wszystkie = [float(x) for x in re.findall(r"zbal_par ([\d.]+)", r.stdout)]
            rek = {"konfiguracja": opis, "ziarno": z,
                   "epoka": int(re.match(r"\[(\d+)/", L).group(1)),
                   "zbal_par": 100 * float(re.search(r"zbal_par ([\d.]+)", L).group(1)),
                   "zbal_par_ogon": 100 * float(np.mean(wszystkie[-10:])),
                   "youden_par": float(re.search(r"j_par ([+-][\d.]+)", L).group(1)),
                   "identycznosc_nt": 100 * float(re.search(r"val ident ([\d.]+)", L).group(1)),
                   "dE_nt": float(re.search(r"dE/nt ([+-][\d.]+)", L).group(1))}
            wiersze.append(rek)
            print(f"  {opis:<14} ziarno {z}  epoka {rek['epoka']:>2}  "
                  f"zbal_par {rek['zbal_par']:.2f}%  (ogon {rek['zbal_par_ogon']:.2f}%)  "
                  f"J {rek['youden_par']:+.4f}  ident {rek['identycznosc_nt']:.2f}%   "
                  f"{time.time()-t0:.0f}s", flush=True)
            WYNIK.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(wiersze).to_csv(WYNIK, index=False, encoding="utf-8")

    Path(ROOT / TMP).unlink(missing_ok=True)
    d = pd.DataFrame(wiersze)

    print(f"\n{'konfiguracja':<14}{'zbal_par (zapisana epoka)':>28}{'zbal_par (ogon 10 epok)':>26}"
          f"{'Youden':>18}")
    for opis, _ in KONFIGURACJE:
        g = d[d.konfiguracja == opis]
        if g.empty:
            continue
        print(f"{opis:<14}{g.zbal_par.mean():>20.2f}% ±{g.zbal_par.std():.2f}"
              f"{g.zbal_par_ogon.mean():>19.2f}% ±{g.zbal_par_ogon.std():.2f}"
              f"{g.youden_par.mean():>+13.4f} ±{g.youden_par.std():.4f}")
        print(f"{'':<14}  ziarna: " + "  ".join(f"{v:.2f}" for v in g.zbal_par))

    sd = d.groupby("konfiguracja").zbal_par.std().mean()
    print(f"\n  PROG ISTOTNOSCI")
    print(f"  odchylenie standardowe pojedynczego przebiegu:  ±{sd:.2f} pp")
    print(f"  odchylenie ROZNICY dwoch przebiegow:            ±{sd * np.sqrt(2):.2f} pp")
    print(f"  -> roznice ponizej ok. {2 * sd * np.sqrt(2):.2f} pp traktujemy jako brak roznicy")
    sd_j = d.groupby("konfiguracja").youden_par.std().mean()
    print(f"  to samo dla Youdena: ±{sd_j:.4f} na przebieg, "
          f"±{sd_j * np.sqrt(2):.4f} na roznice")
    print(f"\nlacznie {time.time()-t0:.0f}s   zapisano: {WYNIK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
