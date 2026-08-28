"""Sprawdzenie wlasnosci, na ktorej opiera sie cale kryterium:
wskaznik Youdena = 0 dla KAZDEGO predyktora niezaleznego od pozycji, bez wzgledu na sklad wyjscia.

Testujemy na sztucznych danych, gdzie znamy odpowiedz — nie na modelu.
"""
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
rng = np.random.RandomState(0)

KLASY = ["GC", "AU", "GU"]
PRAWDA = [0.484, 0.371, 0.145]          # rozklad referencyjny (jak w tescie)
N = 400_000


def zmierz(y, pred):
    """Te same wzory co w src/train.py: czulosc, specyficznosc, Youden."""
    n, tp, wyd = Counter(y), Counter(), Counter()
    for a, b in zip(y, pred):
        wyd[b] += 1
        if a == b:
            tp[a] += 1
    NT = len(y)
    out = {}
    for k in KLASY:
        czul = tp[k] / n[k]
        neg = NT - n[k]
        fp = wyd[k] - tp[k]
        spec = (neg - fp) / neg
        out[k] = (czul, spec, czul + spec - 1.0)
    return out


y = rng.choice(KLASY, N, p=PRAWDA)

print("PREDYKTORY NIEZALEZNE OD POZYCJI — Youden powinien wyjsc 0 w kazdym wierszu\n")
print(f"{'co robi predyktor':<34}{'czulGC':>8}{'specGC':>8}{'J_GC':>9}{'zbal':>8}{'J_sred':>9}")
for opis, q in [
    ("jednostajnie 1/3",              [1/3, 1/3, 1/3]),
    ("wg czestosci referencyjnych",   PRAWDA),
    ("jak nasze E1 (0,62/0,26/0,11)", [0.622, 0.266, 0.112]),
    ("jak nasze E2 (0,69/0,23/0,08)", [0.693, 0.232, 0.075]),
    ("SAME G:C",                      [1.0, 0.0, 0.0]),
    ("ZERO G:C, reszta po polowie",   [0.0, 0.5, 0.5]),
]:
    pred = rng.choice(KLASY, N, p=q)
    r = zmierz(y, pred)
    zbal = np.mean([r[k][0] for k in KLASY])
    jsr = np.mean([r[k][2] for k in KLASY])
    print(f"{opis:<34}{r['GC'][0]:>8.3f}{r['GC'][1]:>8.3f}{r['GC'][2]:>+9.4f}"
          f"{zbal:>8.3f}{jsr:>+9.4f}")

print("\n  ^ specyficznosc G:C skacze od 0,00 do 1,00, identycznosc tez sie zmienia,")
print("    a Youden stoi na zerze. O to wlasnie chodzi.\n")

print("PREDYKTORY, KTORE COS WIEDZA — Youden musi byc dodatni\n")
print(f"{'co robi predyktor':<34}{'czulGC':>8}{'specGC':>8}{'J_GC':>9}{'zbal':>8}{'J_sred':>9}")
for opis, p_ok in [("trafia w 50% przypadkow", 0.5),
                   ("trafia w 80% przypadkow", 0.8),
                   ("trafia zawsze", 1.0)]:
    pred = [a if rng.rand() < p_ok else rng.choice(KLASY) for a in y]
    r = zmierz(y, pred)
    zbal = np.mean([r[k][0] for k in KLASY])
    jsr = np.mean([r[k][2] for k in KLASY])
    print(f"{opis:<34}{r['GC'][0]:>8.3f}{r['GC'][1]:>8.3f}{r['GC'][2]:>+9.4f}"
          f"{zbal:>8.3f}{jsr:>+9.4f}")
