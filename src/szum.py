"""Prog istotnosci: ile z roznicy miedzy konfiguracjami jest realne, a ile to szum.

PO CO. Kazde zdanie w stylu "E3 bije E1 o 0,23 punktu" wymaga progu, ponizej ktorego roznicy nie ma.
Ten skrypt go mierzy: trenuje KAZDA konfiguracje po kilka razy, zmieniajac wylacznie ziarno losowej
inicjalizacji wag i kolejnosci partii, i patrzy, jak bardzo rozjezdzaja sie wyniki.

DWA OSOBNE ZIARNA, i to jest kluczowe. `--seed` wyznacza PODZIAL danych, wiec jego zmiana zmienia
zbiory i wyniki przestaja byc porownywalne. `--seed-modelu` rusza tylko inicjalizacje wag
i kolejnosc partii — i tylko to chcemy tu zmieniac.

MIERZYMY NA TESCIE I NA WALIDACJI, kazdy przebieg na obu. To nie jest przeciek: przeciek to
sytuacja, w ktorej zbior testowy wplywa na WYBOR modelu, a tu niczego nie wybieramy — mierzymy
tylko, jak bardzo wynik skacze przy przelosowaniu ziarna. Modele z tych przebiegow maja zreszta
inne ziarna niz siedem raportowanych i nigdzie ich nie zastepuja.

Wynik na tescie jest wiazacy, bo to na tescie raportujemy roznice; walidacja zostaje w pliku jako
kontrola, ze rozrzut na obu zbiorach jest podobny.

CO Z TEGO WYCHODZI. Dla kazdej miary:

    odchylenie POJEDYNCZEGO przebiegu      s
    odchylenie ROZNICY dwoch przebiegow    s * sqrt(2)
    prog istotnosci (2 sigma)              2 * s * sqrt(2)

Do tego porownania PAROWANE: konfiguracje chodza po tych samych ziarnach, wiec roznice liczymy
w obrebie ziarna. To wycina szum wspolny obu przebiegom i jest mocniejsze niz zestawianie srednich.

PROTOKOL identyczny z run.py: 60 epok bez wczesnego zatrzymania, wybor epoki po `zbal_par`
na walidacji, dekodowanie `argmax` ze straight-through.

Uzycie:
    python -m src.szum                   # 7 konfiguracji x 3 ziarna, ok. 2h15
    python -m src.szum --ziarna 2        # szybciej, zgrubnie
    python -m src.szum --tylko E1 E3 CE  # wybrane konfiguracje
                                         # wznawialny: gotowe przebiegi czyta z CSV
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd  # PRZED torch — inaczej read_parquet potrafi wywrocic proces
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WYNIK = ROOT / "experiments" / "analysis" / "szum_ziaren.csv"
TMP = "checkpoints/_szum_tmp.pt"
TYP = {"GC": "GC", "CG": "GC", "AU": "AU", "UA": "AU", "GU": "GU", "UG": "GU"}
PETLE = ("spinka", "wybrzuszenie", "multipetla", "regiony-zewnetrzne")
ZBIORY = ("test", "val")

TV = ["--w-energia", "1.0", "--w-parowania", "1.0", "--w-sklad", "1.0"]
PROGI = ["--w-energia", "1.0", "--w-parowania", "1.0",
         "--w-sklad-zasad", "1.0", "--w-sklad-par", "1.0"]
ENER = ["--w-energia", "1.0", "--w-parowania", "1.0"]

KONFIGURACJE = {
    "E1":  TV,
    "E2":  PROGI,
    "E3":  ENER,
    "CE":  [],
    "E1W": TV + ["--wagi-klas"],
    "E2W": PROGI + ["--wagi-klas"],
    "CEW": ["--wagi-klas"],
}

MIARY = ["zbal_par", "zbal_zasady", "youden_par", "youden_zasady",
         "identycznosc_nt", "identycznosc_par", "dE_nt", "GC_wyjscie", "G_petle"]
GLOWNE = ("zbal_par", "youden_par", "identycznosc_nt", "GC_wyjscie", "G_petle")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epoki", type=int, default=60)
    ap.add_argument("--ziarna", type=int, default=3)
    ap.add_argument("--tylko", nargs="*", default=None,
                    help="ogranicz do wybranych konfiguracji, np. --tylko E1 E3 CE")
    args = ap.parse_args()

    import RNA
    import torch
    from src.evaluate import wczytaj_model, generuj, identycznosci, sprawdz_zgodnosc
    from src.prepare import wczytaj
    from src.split import wczytaj_split
    from src.dataset import parse_pairs, motyw_pozycji

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baza = wczytaj()
    spl = wczytaj_split("rodzinowy")

    ZB = {}
    for z in ZBIORY:
        S = baza.secondary_structure.iloc[spl[z]].tolist()
        Q = baza.sequence.iloc[spl[z]].tolist()
        ZB[z] = (S, Q, [RNA.energy_of_struct(q, s) for s, q in zip(S, Q)])

    def miary(S, Q, E_REF, gen):
        """Wszystkie miary naraz, liczone na wygenerowanych sekwencjach."""
        idn = [identycznosci(s, g, q) for s, g, q in zip(S, gen, Q)]

        n, tp, wyd = Counter(), Counter(), Counter()
        nz, tz, wz = Counter(), Counter(), Counter()
        cp, petle = Counter(), Counter()
        for st, g, q in zip(S, gen, Q):
            sparowane = set()
            for i, j in parse_pairs(st):
                sparowane.update((i, j))
                w, p_ = TYP.get(q[i] + q[j]), TYP.get(g[i] + g[j])
                if p_:
                    cp[p_] += 1
                if w is None:
                    continue
                n[w] += 1
                if p_:
                    wyd[p_] += 1
                if p_ == w:
                    tp[w] += 1
            for i in range(len(st)):
                if i not in sparowane:
                    nz[q[i]] += 1
                    wz[g[i]] += 1
                    if g[i] == q[i]:
                        tz[q[i]] += 1
            for m, ch in zip(motyw_pozycji(st), g):
                if m in PETLE:
                    petle[ch] += 1

        def zb_j(nn, ttp, ww, klasy):
            """Trafnosc zbalansowana i wskaznik Youdena, usrednione po klasach bez wagi."""
            N = sum(nn.values())
            cz, jj = [], []
            for k in klasy:
                if not nn[k]:
                    continue
                c = ttp[k] / nn[k]
                neg = N - nn[k]
                cz.append(c)
                jj.append(c + (neg - (ww[k] - ttp[k])) / neg - 1)
            return 100 * float(np.mean(cz)), float(np.mean(jj))

        zbal_par, j_par = zb_j(n, tp, wyd, ("GC", "AU", "GU"))
        zbal_zas, j_zas = zb_j(nz, tz, wz, tuple("ACGU"))
        return {
            "zbal_par": zbal_par, "zbal_zasady": zbal_zas,
            "youden_par": j_par, "youden_zasady": j_zas,
            "identycznosc_nt": 100 * float(np.mean([a for a, _ in idn])),
            "identycznosc_par": 100 * float(np.nanmean([b for _, b in idn])),
            "dE_nt": float(np.mean([(RNA.energy_of_struct(g, s) - e) / len(s)
                                    for s, g, e in zip(S, gen, E_REF)])),
            "GC_wyjscie": cp["GC"] / max(sum(cp.values()), 1),
            "G_petle": petle["G"] / max(sum(petle.values()), 1),
        }

    wybrane = {k: v for k, v in KONFIGURACJE.items()
               if args.tylko is None or k in args.tylko}

    wiersze, t0 = [], time.time()
    if WYNIK.exists():
        stare = pd.read_csv(WYNIK)
        if "zbior" in stare.columns and not set(MIARY) - set(stare.columns):
            wiersze = stare.to_dict("records")
            print(f"wznawiam: {len(wiersze)} wierszy juz w pliku")
        else:
            print("plik z poprzedniej wersji skryptu (brak kolumny 'zbior' albo miar) — "
                  "liczymy od zera")
    zrobione = {(w["konfiguracja"], w["ziarno"]) for w in wiersze}

    ile = sum(1 for k in wybrane for z in range(args.ziarna) if (k, z) not in zrobione)
    print(f"Prog istotnosci: {len(wybrane)} konfiguracji x {args.ziarna} ziaren = {ile} "
          f"przebiegow. Kazdy oceniany na: {', '.join(ZBIORY)}. Podzial NIEZMIENIONY.\n")

    for opis, extra in wybrane.items():
        for z in range(args.ziarna):
            if (opis, z) in zrobione:
                continue
            cmd = [sys.executable, "-m", "src.train", "--epoki", str(args.epoki),
                   "--tryb-podzialu", "rodzinowy", "--wybor", "zbal_par",
                   "--dekodowanie", "argmax", "--kary-na-argmax",
                   "--cierpliwosc", str(args.epoki), "--seed-modelu", str(z),
                   *extra, "--out", TMP]
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                print(r.stdout[-900:], r.stderr[-600:])
                raise SystemExit(f"trening padl: {opis} ziarno {z}")
            L = [l for l in r.stdout.splitlines() if "*najlepszy" in l][-1]
            epoka = int(re.match(r"\[(\d+)/", L).group(1))

            mm, _ = wczytaj_model(TMP, dev)
            for zb, (S, Q, E) in ZB.items():
                g = generuj(mm, S, dev, dekodowanie="argmax")
                if sprawdz_zgodnosc(S, g, f"{opis} ziarno {z} / {zb}"):
                    raise SystemExit("kontrola zgodnosci nie przeszla")
                wiersze.append({"konfiguracja": opis, "ziarno": z, "epoka": epoka,
                                "zbior": zb, **miary(S, Q, E, g)})
            del mm
            torch.cuda.empty_cache()

            t = wiersze[-2]   # test jest pierwszy w ZBIORY
            print(f"  {opis:<5} ziarno {z}  ep {epoka:>2}  [test] "
                  f"zbal_par {t['zbal_par']:.2f}%  J {t['youden_par']:+.4f}  "
                  f"G:C {t['GC_wyjscie']:.3f}  G_petle {t['G_petle']:.3f}   "
                  f"{time.time()-t0:.0f}s", flush=True)
            WYNIK.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(wiersze).to_csv(WYNIK, index=False, encoding="utf-8")

    Path(ROOT / TMP).unlink(missing_ok=True)
    d = pd.DataFrame(wiersze)

    def prog(zb, m):
        """2 sigma dla ROZNICY. Odchylenie liczone W OBREBIE konfiguracji, potem usredniane —
        policzone na wszystkich przebiegach naraz zmierzyloby roznice MIEDZY konfiguracjami,
        czyli dokladnie to, co chcemy testowac."""
        g = d[d.zbior == zb]
        return 2 * float(g.groupby("konfiguracja")[m].std().mean()) * np.sqrt(2)

    for zb in ZBIORY:
        g = d[d.zbior == zb]
        print(f"\n\nSREDNIE I ROZRZUT PO ZIARNACH — {zb.upper()}\n")
        print(f"{'konf':<6}" + "".join(f"{m[:12]:>16}" for m in GLOWNE))
        for k in wybrane:
            gg = g[g.konfiguracja == k]
            if not gg.empty:
                print(f"{k:<6}" + "".join(f"{gg[m].mean():>10.4f}±{gg[m].std():.4f}"
                                          for m in GLOWNE))

    print("\n\nPROG ISTOTNOSCI\n")
    print(f"{'miara':<18}" + "".join(f"{'odch. ' + z:>14}{'prog ' + z:>14}" for z in ZBIORY))
    for m in MIARY:
        w = ""
        for zb in ZBIORY:
            s = float(d[d.zbior == zb].groupby("konfiguracja")[m].std().mean())
            w += f"{s:>14.4f}{2 * s * np.sqrt(2):>14.4f}"
        print(f"{m:<18}{w}")

    print("\n\nPOROWNANIA PAROWANE NA TESCIE (te same ziarna, szum wspolny sie znosi)\n")
    print(f"{'porownanie':<16}{'miara':<18}{'srednia roznica':>17}{'prog':>10}{'znaki':>8}   werdykt")
    PARY = [("E1", "E3"), ("E3", "CE"), ("E1", "CE"),
            ("E1", "E1W"), ("E2", "E2W"), ("CE", "CEW"), ("E1", "E2")]
    t = d[d.zbior == "test"]
    for a, b in PARY:
        ga, gb = t[t.konfiguracja == a], t[t.konfiguracja == b]
        if ga.empty or gb.empty:
            continue
        for m in GLOWNE:
            wsp = sorted(set(ga.ziarno) & set(gb.ziarno))
            roz = [float(gb[gb.ziarno == z][m].iloc[0]) - float(ga[ga.ziarno == z][m].iloc[0])
                   for z in wsp]
            sr, dod = float(np.mean(roz)), sum(1 for x in roz if x > 0)
            p = prog("test", m)
            zgodne = dod in (0, len(roz))     # ten sam kierunek na kazdym ziarnie
            werdykt = ("REALNA" if abs(sr) > p and zgodne
                       else "zgodny kierunek" if zgodne else "w szumie")
            print(f"{a + ' -> ' + b:<16}{m:<18}{sr:>+17.4f}{p:>10.4f}"
                  f"{f'{dod}/{len(roz)}':>8}   {werdykt}")
        print()

    print(f"lacznie {time.time()-t0:.0f}s   zapisano: {WYNIK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
