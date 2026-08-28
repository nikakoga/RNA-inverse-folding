"""Przeglad wag funkcji straty. WSZYSTKO NA WALIDACJI — zbioru testowego nie dotykamy.

PO CO. Wagi 1,0 : 6,0 : 1,0 zostaly odziedziczone i nigdzie nie sa uzasadnione. Ablacja pokazala
tylko tyle, ze wyzerowanie WSZYSTKICH TRZECH naraz niczego nie psuje — nie umiemy z niej odczytac,
co robi kazdy czlon z osobna ani czy istnieje lepsze ustawienie.

CO ROBIMY. Trenujemy ten sam model dla kazdego ustawienia wag i odczytujemy miary walidacyjne
usrednione po OSTATNICH 10 EPOKACH. Trening loguje komplet miar w kazdej epoce, wiec nie potrzeba
osobnej oceny — czytamy je wprost z logu.

DLACZEGO SREDNIA, A NIE MAKSIMUM. Maksimum z kilkudziesieciu epok szumiacej miary jest obciazone
w gore i potrafi wskazac epoke 1, w ktorej model jeszcze niczego sie nie nauczyl. Wraz z wynikiem
raportujemy odchylenie standardowe `zbal_par` po epokach — jesli rozstep MIEDZY ustawieniami nie
przekracza wyraznie tego szumu, roznice sa nieistotne i tak nalezy je opisac.

DLACZEGO TO NIE JEST PRZECIEK. Walidacja istnieje wlasnie po to, zeby wybierac konfiguracje. Testu
nie ogladamy ani razu; dopiero po wybraniu jednego ustawienia wolno na niego spojrzec, raz.

CZEGO SIE SPODZIEWAC. Trafnosc zbalansowana stoi na poziomie losowym we wszystkich dotychczasowych
przebiegach, a trzy czlony sa NIENADZOROWANE — nie widza referencji, wiec zadna waga nie sprawi, ze
naucza model, ktora pare postawic w danym miejscu. Waga steruje sila, nie rodzajem informacji.
Realnie spodziewam sie ruchu na `kara_TV` i `dE/nt`, a nie na trafnosci.

Uzycie:
    python -m src.przeglad                  # pelny przeglad
    python -m src.przeglad --epoki 15       # szybciej, zgrubnie
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WYNIK = ROOT / "experiments" / "analysis" / "przeglad_wag.csv"
TYMCZASOWY = ROOT / "checkpoints" / "_przeglad_tmp.pt"

# (energia, parowania, sklad_TV, sklad_zasad, sklad_par, etykieta)
#
# Blok 1  KAZDY CZLON OSOBNO — czego brakowalo w ablacji zbiorczej.
# Blok 2  WAGA PAROWAN, jedyna liczba w projekcie bez uzasadnienia (6,0 z poprzedniego projektu).
# Blok 3  WAGA SKLADU przy reszcie ustalonej.
# Blok 4  KARA PROMOTORA zamiast naszej, dla porownania w tych samych warunkach.
SIATKA = [
    (0.0, 0.0, 0.0, 0.0, 0.0, "nic (sama CE)"),
    (1.0, 0.0, 0.0, 0.0, 0.0, "sama energia"),
    (0.0, 6.0, 0.0, 0.0, 0.0, "same parowania"),
    (0.0, 0.0, 1.0, 0.0, 0.0, "sam sklad TV"),

    (1.0, 0.0, 1.0, 0.0, 0.0, "parowania 0"),
    (1.0, 1.0, 1.0, 0.0, 0.0, "parowania 1"),
    (1.0, 3.0, 1.0, 0.0, 0.0, "parowania 3"),
    (1.0, 6.0, 1.0, 0.0, 0.0, "parowania 6  (obecne)"),
    (1.0, 12.0, 1.0, 0.0, 0.0, "parowania 12"),

    (1.0, 6.0, 0.0, 0.0, 0.0, "sklad 0"),
    (1.0, 6.0, 3.0, 0.0, 0.0, "sklad 3"),
    (1.0, 6.0, 10.0, 0.0, 0.0, "sklad 10"),

    (1.0, 6.0, 0.0, 1.0, 1.0, "kara promotora"),
]

WZOR = re.compile(
    r"\[(\d+)/\d+\].*?val ident ([\d.]+) zbal_par ([\d.na]+) zbal_zas ([\d.nan]+) "
    r"CE ([\d.]+) loss ([\d.nan]+) dE/nt ([+-][\d.]+)")


def jeden(wagi, epoki, seed_dek=0):
    """Jeden trening; zwraca miary walidacyjne usrednione po ostatnich 10 epokach."""
    e, a, c, sz, sp, opis = wagi
    cmd = [sys.executable, "-m", "src.train", "--epoki", str(epoki),
           "--tryb-podzialu", "rodzinowy", "--wybor", "zbal_par",
           "--dekodowanie", "probkowanie", "--seed-dekodowania", str(seed_dek),
           "--cierpliwosc", str(epoki + 1),          # bez wczesnego zatrzymania
           "--w-energia", str(e), "--w-parowania", str(a), "--w-sklad", str(c),
           "--w-sklad-zasad", str(sz), "--w-sklad-par", str(sp),
           "--out", str(TYMCZASOWY.relative_to(ROOT))]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-800:])
        raise SystemExit(f"trening padl dla: {opis}")

    epoki_dane = [m.groups() for m in WZOR.finditer(r.stdout)]
    if not epoki_dane:
        raise SystemExit(f"nie odczytalem miar z logu dla: {opis}")
    kolumny = ["epoka", "ident_nt", "zbal_par", "zbal_zas", "CE_val", "loss_val", "dE_nt"]
    d = pd.DataFrame(epoki_dane, columns=kolumny).astype(float)

    # RAPORTUJEMY SREDNIA Z OSTATNICH 10 EPOK, nie maksimum. Maksimum z 25 epok szumiacej miary jest
    # obciazone w gore i potrafi wskazac epoke 1, w ktorej model jeszcze niczego sie nie nauczyl.
    # Zmierzone wahanie `zbal_par` z epoki na epoke w obrebie JEDNEGO przebiegu: odch.std 0,003-0,006,
    # rozstep min-max 0,011-0,020 — czyli wiecej niz roznice miedzy ustawieniami.
    ogon = d.tail(10)
    return {"opis": opis, "w_energia": e, "w_parowania": a, "w_sklad": c,
            "w_sklad_zasad": sz, "w_sklad_par": sp,
            "zbal_par": ogon.zbal_par.mean(), "zbal_par_std": d.zbal_par.std(),
            "zbal_zas": ogon.zbal_zas.mean(), "ident_nt": ogon.ident_nt.mean(),
            "CE_val": ogon.CE_val.mean(), "dE_nt": ogon.dE_nt.mean(),
            "zbal_par_max": d.zbal_par.max(), "epoka_max": int(d.loc[d.zbal_par.idxmax()].epoka)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epoki", type=int, default=25,
                    help="dlugosc kazdego treningu; miary walidacyjne wyplaszczaja sie okolo 15")
    args = ap.parse_args()

    print(f"Przeglad {len(SIATKA)} ustawien po {args.epoki} epok. "
          f"Wszystko na WALIDACJI, test nietkniety.\n")
    wiersze, t0 = [], time.time()
    for i, w in enumerate(SIATKA, 1):
        t = time.time()
        wiersze.append(jeden(w, args.epoki))
        r = wiersze[-1]
        print(f"[{i:>2}/{len(SIATKA)}] {r['opis']:<24} "
              f"zbal_par {100*r['zbal_par']:>5.1f}% (+-{100*r['zbal_par_std']:.1f})  "
              f"zbal_zas {100*r['zbal_zas']:>5.1f}%  ident {100*r['ident_nt']:>5.1f}%  "
              f"CE {r['CE_val']:.4f}  dE/nt {r['dE_nt']:+.4f}   {time.time()-t:.0f}s", flush=True)

    d = pd.DataFrame(wiersze).sort_values("zbal_par", ascending=False)
    WYNIK.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(WYNIK, index=False, encoding="utf-8")
    TYMCZASOWY.unlink(missing_ok=True)

    print(f"\nUPORZADKOWANE wg trafnosci zbalansowanej par (poziom losowy 33,3%)")
    print(f"  {'ustawienie':<24}{'zbal_par':>9}{'szum':>7}{'zbal_zas':>10}{'ident_nt':>10}"
          f"{'CE_val':>9}{'dE/nt':>9}")
    for _, r in d.iterrows():
        print(f"  {r.opis:<24}{100*r.zbal_par:>8.1f}%{100*r.zbal_par_std:>7.1f}"
              f"{100*r.zbal_zas:>9.1f}%{100*r.ident_nt:>9.1f}%{r.CE_val:>9.4f}{r.dE_nt:>9.4f}")

    rozstep = 100 * (d.zbal_par.max() - d.zbal_par.min())
    szum = 100 * d.zbal_par_std.mean()
    print(f"\n  rozstep miedzy ustawieniami:          {rozstep:.2f} pp")
    print(f"  typowy szum W OBREBIE jednego przebiegu: {szum:.2f} pp (odch.std po epokach)")
    print("  Jesli rozstep nie przekracza wyraznie szumu, roznice miedzy ustawieniami sa NIEISTOTNE.")
    print(f"\nlacznie {time.time()-t0:.0f}s   zapisano: {WYNIK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
