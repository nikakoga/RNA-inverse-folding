"""Uruchamianie eksperymentow. Kazdy krok logowany do experiments/logs/.

    python run.py lista
    python run.py dane        przygotowanie zbioru + podzial + analiza obrazowa
    python run.py E1
    python run.py E2
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "experiments" / "logs"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CK = {
    "e1": "checkpoints/e1_kary.pt",
    "e2_w0": "checkpoints/e2_sklad0.pt",
    "e2_w40": "checkpoints/e2_sklad40.pt",
}
SPLIT = ["--tryb-podzialu", "rodzinowy"]

EKSPERYMENTY: list[tuple[str, str, list]] = [

    ("dane", "Odsianie redundancji natywnym cd-hit-est, filtry i podzial rodzinowy", [
        # cd-hit-est wymaga WSL. Sprawdzenie gotowosci: python -m src.cdhit --sprawdz
        ("cdhit", ["-m", "src.cdhit"]),
        ("przygotowanie", ["-m", "src.prepare"]),
        ("podzial", ["-m", "src.split", "--tryb", "rodzinowy"]),
    ]),

    ("E1", "Transformer nieautoregresyjny z trzema komponentami w stracie (energia 1,0 : parowania 6,0 "
           ": sklad 1,7), podzial rodzinowy. Punkt wyjscia — te wagi nie byly strojone", [
        # Podzial rodzinowy: kazda rodzina Rfam w DOKLADNIE jednym zbiorze, wiec test to rodziny,
        # ktorych model nie widzial ani razu. Bez tego ta sama rodzina bywa i w treningu, i w tescie,
        # a model odtwarza zapamietany wzorzec zamiast generalizowac (odzysk 0,49 wobec 0,31).
        ("trening", ["-m", "src.train", "--epoki", "60", *SPLIT,
                     "--w-energia", "1.0", "--w-parowania", "6.0", "--w-sklad", "1.7",
                     "--out", CK["e1"]]),
        ("ocena_test", ["-m", "src.evaluate", "--ckpt", CK["e1"], *SPLIT,
                        "--na", "test", "--csv", "e1_test.csv"]),
        ("baseline", ["-m", "src.evaluate", "--baseline", *SPLIT,
                      "--na", "test", "--csv", "baseline_test.csv"]),
    ]),

    ("E2", "Strojenie wagi kary za sklad. Sweep na WALIDACJI, ocena wybranych wag na tescie", [
        # STROIMY NA WALIDACJI. Zbior testowy wolno obejrzec RAZ, na koncu; strojenie to wielokrotne
        # zagladanie. Dopiero wybrane konfiguracje ida na test — i to jest jedyna liczba do raportu.
        ("sweep_w0", ["-m", "src.train", "--epoki", "60", *SPLIT,
                      "--w-energia", "1.0", "--w-parowania", "6.0", "--w-sklad", "0",
                      "--out", CK["e2_w0"]]),
        ("sweep_w40", ["-m", "src.train", "--epoki", "60", *SPLIT,
                       "--w-energia", "1.0", "--w-parowania", "6.0", "--w-sklad", "40",
                       "--out", CK["e2_w40"]]),
        ("val_w0", ["-m", "src.evaluate", "--ckpt", CK["e2_w0"], *SPLIT, "--na", "val"]),
        ("val_w40", ["-m", "src.evaluate", "--ckpt", CK["e2_w40"], *SPLIT, "--na", "val"]),
        ("ocena_test_w0", ["-m", "src.evaluate", "--ckpt", CK["e2_w0"], *SPLIT,
                           "--na", "test", "--csv", "e2_w0_test.csv"]),
        ("ocena_test_w40", ["-m", "src.evaluate", "--ckpt", CK["e2_w40"], *SPLIT,
                            "--na", "test", "--csv", "e2_w40_test.csv"]),
    ]),
]


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("lista", "-h", "--help"):
        for nazwa, opis, kroki in EKSPERYMENTY:
            print(f"  python run.py {nazwa:<6} # {opis}  ({len(kroki)} krokow)")
        return
    cel = sys.argv[1]
    for nazwa, opis, kroki in EKSPERYMENTY:
        if nazwa != cel:
            continue
        LOGS.mkdir(parents=True, exist_ok=True)
        print(f"=== {nazwa}: {opis} ===\n")
        for krok, cmd in kroki:
            log = LOGS / f"{nazwa}_{krok}.log"
            print(f"--- {krok} -> {log.relative_to(ROOT)}", flush=True)
            t0 = time.time()
            with open(log, "w", encoding="utf-8") as f:
                r = subprocess.run([sys.executable, *cmd], cwd=ROOT, stdout=f,
                                   stderr=subprocess.STDOUT)
            print(f"    {'OK' if r.returncode == 0 else 'BLAD'}  {time.time()-t0:.0f}s", flush=True)
            if r.returncode != 0:
                print(open(log, encoding="utf-8").read()[-2000:])
                sys.exit(1)
        print(f"\n=== {nazwa} zakonczone ===")
        return
    print(f"nieznany eksperyment: {cel}")


if __name__ == "__main__":
    main()
