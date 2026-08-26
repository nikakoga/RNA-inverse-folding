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
    "e1": "checkpoints/e1_sklad_tv.pt",        # nasza kara za sklad (odleglosc TV)
    "e2": "checkpoints/e2_sklad_progi.pt",     # kara za sklad wg specyfikacji promotora
}
SPLIT = ["--tryb-podzialu", "rodzinowy"]
# Kryterium wyboru epoki JAWNIE, i TAKIE SAMO w obu eksperymentach: identycznosc sekwencyjna jako
# klucz glowny, dE/nt jako rozstrzygacz remisow. Oba sa zewnetrzne wobec obu kar za sklad.
WYBOR = ["--wybor", "zlozony"]

EKSPERYMENTY: list[tuple[str, str, list]] = [

    ("dane", "Odsianie redundancji natywnym cd-hit-est, filtry i podzial rodzinowy", [
        # cd-hit-est wymaga WSL (instalacja w README). Odsiewana jest WYLACZNIE pula naturalna —
        # Eterny nie ma w treningu, wiec nie ma tam przecieku, ktoremu odsiewanie mialoby zapobiec.
        ("cdhit", ["-m", "src.cdhit"]),
        ("przygotowanie", ["-m", "src.prepare"]),
        ("podzial", ["-m", "src.split", "--tryb", "rodzinowy"]),
    ]),

    ("E1", "Kara za sklad: NASZA (odleglosc TV od celu naturalnego), per sekwencja, waga 1,0", [
        # WAGA 1,0 — taka sama jak w E2. Specyfikacja promotora wnosi swoja kare wprost
        # (`loss = loss + DistribLoss`), wiec dajac nasza rowniez z waga 1,0 nie wprowadzamy zadnej
        # dobranej stalej. Jedyna roznica miedzy E1 i E2 zostaje wtedy KSZTALT kary.
        ("trening", ["-m", "src.train", "--epoki", "60", *SPLIT, *WYBOR,
                     "--w-energia", "1.0", "--w-parowania", "6.0", "--w-sklad", "1.0",
                     "--out", CK["e1"]]),
        ("ocena_test", ["-m", "src.evaluate", "--ckpt", CK["e1"], *SPLIT,
                        "--na", "test", "--csv", "e1_test.csv"]),
        ("baseline", ["-m", "src.evaluate", "--baseline", *SPLIT,
                      "--na", "test", "--csv", "baseline_test.csv"]),
    ]),

    ("E2", "Kara za sklad: wg specyfikacji promotora (progi dolne, per sekwencja). "
           "Jedyna roznica wobec E1 to KONSTRUKCJA tej kary", [
        #   E1  --w-sklad 1          odleglosc TV od celu; DWUSTRONNA — karze takze nadmiar
        #   E2  --w-sklad-zasad 1    progi DOLNE udzialow A/C/G/U
        #       --w-sklad-par   1    progi DOLNE udzialow typow par + eskalacja DistribLoss4
        #                            JEDNOSTRONNA — nadmiar bezkarny
        #
        # Obie liczone PER SEKWENCJA i obie z waga 1,0, wiec jedyna zmienna jest ksztalt kary.
        # Energia i parowania identyczne jak w E1.
        ("trening", ["-m", "src.train", "--epoki", "60", *SPLIT, *WYBOR,
                     "--w-energia", "1.0", "--w-parowania", "6.0",
                     "--w-sklad-zasad", "1.0", "--w-sklad-par", "1.0",
                     "--out", CK["e2"]]),
        ("ocena_test", ["-m", "src.evaluate", "--ckpt", CK["e2"], *SPLIT,
                        "--na", "test", "--csv", "e2_test.csv"]),
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
