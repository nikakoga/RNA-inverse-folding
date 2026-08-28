"""Ablacja: wylaczamy po JEDNEJ karze i patrzymy, co sie psuje. NA WALIDACJI, test nietkniety.

PYTANIE. E1 ma trzy kary naraz i wypada gorzej niz CE, ktore nie ma zadnej. Nie wiemy jednak,
KTORA z trzech za to odpowiada — a to jest pytanie, na ktore da sie odpowiedziec tylko wylaczajac
je pojedynczo.

CZTERY PRZEBIEGI:

    energia  parowania  sklad
       1         1        1     wszystkie trzy  (punkt odniesienia, = E1)
       0         1        1     BEZ energii
       1         0        1     BEZ parowan
       1         1        0     BEZ skladu

Kazdy porownujemy z pierwszym. Jesli wylaczenie kary niczego nie pogorszylo, ta kara nic nie robila.

CO PATRZYMY (reszta miar jest w tabeli, ale te dwie rozstrzygaja):

    zbal_par   czy model wie, ktora para gdzie stoi.  Wyzej lepiej, poziom losowy 33,3%.
               Model ignorujacy wejscie dostaje tam 33,3% NIEZALEZNIE od tego, co produkuje,
               wiec tej miary nie da sie podbic nadprodukcja klasy najczestszej.
    TV         jak daleko sklad wyjscia od PRAWDZIWYCH sekwencji walidacji. Nizej lepiej, 0 = idealnie.
               Miara neutralna: liczona wzgledem natury, nie wzgledem celu ktorejkolwiek kary.

DWIE HIPOTEZY, ktore ta ablacja rozdziela. Nadmiar par G:C moze pochodzic od kary za SKLAD (jej cel
to G:C 0,599) albo od kary ENERGETYCZNEJ (G:C ma trzy wiazania wodorowe, wiec jest para
najstabilniejsza i czlon energetyczny nagradza wstawianie jej wszedzie). W E1 dzialaja obie naraz.

PROTOKOL identyczny z run.py: 60 epok bez wczesnego zatrzymania, wybor epoki po `zbal_par`,
probkowanie z ziarnem 0 — zeby wyniki dawaly sie zestawiac z tabela w EKSPERYMENTY.md.

JAK CZYTAC. Rozrzut `zbal_par` miedzy ziarnami inicjalizacji to ±0,26 pp
(`experiments/analysis/szum_wagi_klas.csv`), wiec roznica dwoch przebiegow ma odchylenie ok. 0,37 pp
i ponizej tego roznicy nie ma. Odleglosci TV roznia sie natomiast kilkukrotnie, wiec tam wnioski
sa duzo pewniejsze.

Uzycie:
    python -m src.siatka                 # 4 przebiegi, ok. 26 min; wznawialny
    python -m src.siatka --wazona        # to samo na CE WAZONEJ (--wagi-klas)
    python -m src.siatka --epoki 25      # szybciej, zgrubnie
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WYNIK = ROOT / "experiments" / "analysis" / "ablacja_kar.csv"
TMP = "checkpoints/_ablacja_tmp.pt"
TYP = {"GC": "G:C", "CG": "G:C", "AU": "A:U", "UA": "A:U", "GU": "G:U", "UG": "G:U"}

# (etykieta, w_energia, w_parowania, w_sklad)
USTAWIENIA = [
    ("wszystkie trzy", 1.0, 1.0, 1.0),
    ("BEZ energii",    0.0, 1.0, 1.0),
    ("BEZ parowan",    1.0, 0.0, 1.0),
    ("BEZ skladu",     1.0, 1.0, 0.0),
]

POLA = [("zbal_par", r"zbal_par ([\d.]+)", 100),
        ("zbal_zasady", r"zbal_zas ([\d.]+)", 100),
        ("youden_par", r"j_par ([+-][\d.]+)", 1),
        ("youden_GC", r"jGC ([+-][\d.]+)", 1),
        ("identycznosc_nt", r"val ident ([\d.]+)", 100),
        ("ce", r"CE ([\d.]+) loss", 1),
        ("dE_nt", r"dE/nt ([+-][\d.]+)", 1)]


def udzialy(structs, seqs, parse_pairs, motyw_pozycji, BASES):
    """Udzialy typow par oraz zasad w pozycjach niesparowanych."""
    cp, cz = Counter(), Counter()
    for st, s in zip(structs, seqs):
        for i, j in parse_pairs(st):
            k = TYP.get(s[i] + s[j])
            if k:
                cp[k] += 1
        for m, ch in zip(motyw_pozycji(st), s):
            if m in ("spinka", "wybrzuszenie", "multipetla", "regiony-zewnetrzne"):
                cz[ch] += 1
    np_, nz = sum(cp.values()) or 1, sum(cz.values()) or 1
    return (np.array([cp[k] / np_ for k in ("G:C", "A:U", "G:U")]),
            np.array([cz[b] / nz for b in BASES]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epoki", type=int, default=60)
    ap.add_argument("--wazona", action="store_true",
                    help="ta sama ablacja, ale na cross-entropii WAZONEJ (--wagi-klas)")
    args = ap.parse_args()

    import torch
    from src.evaluate import wczytaj_model, generuj
    from src.prepare import wczytaj
    from src.split import wczytaj_split
    from src.dataset import parse_pairs, motyw_pozycji, BASES

    pod = "CEW" if args.wazona else "CE"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baza = wczytaj()
    spl = wczytaj_split("rodzinowy")
    S = baza.secondary_structure.iloc[spl["val"]].tolist()
    Q = baza.sequence.iloc[spl["val"]].tolist()
    REF_P, REF_Z = udzialy(S, Q, parse_pairs, motyw_pozycji, BASES)

    wiersze, t0 = [], time.time()
    if WYNIK.exists():
        wiersze = pd.read_csv(WYNIK).to_dict("records")
        print(f"wznawiam: {len(wiersze)} przebiegow juz w pliku")
    zrobione = {(w["podstawa"], w["ustawienie"]) for w in wiersze}

    print(f"Ablacja na podstawie {pod}: {len(USTAWIENIA)} przebiegow po {args.epoki} epok.")
    print(f"WALIDACJA (727 struktur), test nietkniety.")
    print(f"Referencja: G:C {REF_P[0]:.3f}  A:U {REF_P[1]:.3f}  G:U {REF_P[2]:.3f}\n")

    for opis, we, wa, ws in USTAWIENIA:
        if (pod, opis) in zrobione:
            continue
        cmd = [sys.executable, "-m", "src.train", "--epoki", str(args.epoki),
               "--tryb-podzialu", "rodzinowy", "--wybor", "zbal_par",
               "--dekodowanie", "probkowanie", "--seed-dekodowania", "0",
               "--cierpliwosc", str(args.epoki),
               "--w-energia", str(we), "--w-parowania", str(wa), "--w-sklad", str(ws),
               "--out", TMP]
        if args.wazona:
            cmd.append("--wagi-klas")
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            print(r.stdout[-900:], r.stderr[-600:])
            raise SystemExit(f"trening padl dla: {pod} / {opis}")

        L = [l for l in r.stdout.splitlines() if "*najlepszy" in l][-1]
        rek = {"podstawa": pod, "ustawienie": opis,
               "w_energia": we, "w_parowania": wa, "w_sklad": ws,
               "epoka": int(re.match(r"\[(\d+)/", L).group(1))}
        for nazwa, wzor, skala in POLA:
            m = re.search(wzor, L)
            rek[nazwa] = skala * float(m.group(1)) if m else float("nan")

        m_, _ = wczytaj_model(TMP, dev)
        g = generuj(m_, S, dev, dekodowanie="probkowanie", seed=0)
        up, uz = udzialy(S, g, parse_pairs, motyw_pozycji, BASES)
        del m_
        torch.cuda.empty_cache()
        rek["GC_wyjscie"] = up[0]
        rek["TV_pary"] = 0.5 * np.abs(up - REF_P).sum()
        rek["TV_petle"] = 0.5 * np.abs(uz - REF_Z).sum()
        rek["TV_razem"] = rek["TV_pary"] + rek["TV_petle"]
        wiersze.append(rek)

        print(f"  {opis:<16} zbal_par {rek['zbal_par']:>5.2f}%  J_par {rek['youden_par']:>+7.4f}  "
              f"ident {rek['identycznosc_nt']:>5.2f}%  dE {rek['dE_nt']:>+7.4f}  "
              f"G:C {rek['GC_wyjscie']:.3f}  TV {rek['TV_razem']:.3f}   "
              f"{time.time()-t0:.0f}s", flush=True)
        WYNIK.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(wiersze).to_csv(WYNIK, index=False, encoding="utf-8")

    Path(ROOT / TMP).unlink(missing_ok=True)
    d = pd.DataFrame(wiersze)
    cz = d[d.podstawa == pod]

    print(f"\n\nPODSTAWA {pod} — pelna tabela")
    print(f"  {'ustawienie':<16}{'zbal_par':>10}{'zbal_zas':>10}{'J_par':>10}{'J_GC':>10}"
          f"{'ident_nt':>10}{'CE':>9}{'dE/nt':>9}{'G:C':>8}{'TVpary':>8}{'TVpetle':>9}{'ep':>5}")
    for _, r in cz.iterrows():
        print(f"  {r.ustawienie:<16}{r.zbal_par:>9.2f}%{r.zbal_zasady:>9.2f}%{r.youden_par:>+10.4f}"
              f"{r.youden_GC:>+10.4f}{r.identycznosc_nt:>9.2f}%{r.ce:>9.4f}{r.dE_nt:>+9.4f}"
              f"{r.GC_wyjscie:>8.3f}{r.TV_pary:>8.3f}{r.TV_petle:>9.3f}{int(r.epoka):>5}")

    pelne = cz[cz.ustawienie == "wszystkie trzy"]
    if not pelne.empty:
        p = pelne.iloc[0]
        print(f"\n  EFEKT WYLACZENIA (roznica wobec 'wszystkie trzy'):")
        print(f"  {'co wylaczono':<16}{'zbal_par':>11}{'J_par':>10}{'ident_nt':>11}"
              f"{'TVpary':>9}{'TVpetle':>9}")
        for _, r in cz[cz.ustawienie != "wszystkie trzy"].iterrows():
            print(f"  {r.ustawienie:<16}{r.zbal_par - p.zbal_par:>+10.2f}pp"
                  f"{r.youden_par - p.youden_par:>+10.4f}"
                  f"{r.identycznosc_nt - p.identycznosc_nt:>+10.2f}pp"
                  f"{r.TV_pary - p.TV_pary:>+9.3f}{r.TV_petle - p.TV_petle:>+9.3f}")
        print(f"\n  zbal_par: DODATNIA roznica = wylaczenie POMOGLO")
        print(f"  TV:       UJEMNA roznica = wylaczenie POMOGLO (blizej natury)")

    print(f"\n  poziom losowy: zbal_par 33,33%  zbal_zas 25,00%  Youden 0")
    print(f"  szum miedzy ziarnami: ±0,26 pp na zbal_par (roznica dwoch przebiegow ±0,37 pp)")
    print(f"\nlacznie {time.time()-t0:.0f}s   zapisano: {WYNIK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
