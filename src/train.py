"""Trening nieautoregresyjnego transformera.

FUNKCJA STRATY
    CE                    uczy sie z sekwencji referencyjnych: co w jakim motywie wystepuje
  + w_e  * energia        czlony sekwencyjne z tablic Turnera
  + w_a  * parowania      liczba MOZLIWYCH parowan G*C + A*U + G*U, na nt^2
  + w_c  * sklad          KARA ZA SKLAD, wariant E1: odleglosc TV od celu, DWUSTRONNA
  + w_sz * sklad_zasad    KARA ZA SKLAD, wariant E2: progi dolne A/C/G/U, JEDNOSTRONNA
  + w_sp * sklad_par      KARA ZA SKLAD, wariant E2: progi dolne typow par, JEDNOSTRONNA

Obie kary za sklad licza sie PER SEKWENCJA, potem srednia po partii. Sa alternatywne — wlacza sie
jedna albo druga, nie obie naraz. Wagi domyslnie zerowe: czysta CE jest punktem odniesienia.

WYBOR EPOKI. Wszystkie kryteria licza sie na WALIDACJI i zadne nie przewiduje struktury. W logu
pojawiaja sie WSZYSTKIE w kazdej epoce; flaga `--wybor` decyduje tylko, ktore zapisuje checkpoint.

  zbal_par         DOMYSLNE. Srednia czulosc po TYPACH par, bez wagi liczebnoscia
  zbal_zasady      to samo dla zasad w petlach
                   Obu nie da sie podbic stalym wskazywaniem klasy najczestszej: predyktor
                   niezalezny od pozycji dostaje tam 1/k niezaleznie od tego, co produkuje.
  youden_GC        czulosc + specyficznosc - 1 dla typu G:C
  youden_par       to samo, usrednione po 3 typach par
  youden_zasady    to samo, usrednione po 4 zasadach w petlach
                   Poziom odniesienia = 0, nie 1/k, wiec czyta sie wprost jako ulamek pozycji,
                   na ktorych model zna odpowiedz. Wybiera prawie te same epoki co `zbal_par`
                   (korelacja 0,995-0,999), wiec do WYBORU jest wymienne; do RAPORTOWANIA lepsze.
  identycznosc_nt  ulamek pozycji trafionych wzgledem sekwencji referencyjnej
  ce               srednia cross-entropia; liczona na ROZKLADACH, wiec widzi tez pewnosc modelu
  energia          o ile stabilizujemy cel lepiej niz sekwencja referencyjna
  loss             PELNA strata tego modelu, z jego wlasnymi karami i wagami
  zlozony          identycznosc jako klucz glowny, dE/nt jako rozstrzygacz remisow

DLACZEGO DOMYSLNE PRZESTALO BYC `zlozony`. Oba jego czlony premiuja nadprodukcje klasy najczestszej,
czyli awarie, ktora eksperyment ma zmierzyc: identycznosc jest maksymalizowana przez staly predyktor
(sama para G:C daje 48,4%), a dE/nt spada, gdy udzial G:C rosnie, bo G:C jest para najstabilniejsza.

Wybor musi byc TAKI SAM we wszystkich porownywanych eksperymentach, inaczej porownanie traci sens —
dlatego `loss`, ktory w E1 i E2 obejmuje inna kare, sie do tego nie nadaje.

DEKODOWANIE (`--dekodowanie`, domyslnie `probkowanie`). Przy `argmax` plaskie rozklady daja te sama
litere na kazdej pozycji, wiec sklad wyjscia degeneruje sie mimo poprawnego skladu rozkladu.
`probkowanie` odtwarza rozklad. Ta sama opcja musi byc uzyta w src/evaluate.py, inaczej wybieramy
epoke pod inne dekodowanie niz raportujemy.

WCZESNE ZATRZYMANIE jest domyslnie WYLACZONE (`--cierpliwosc` = `--epoki`), bo `zbal_par` szumi tuz
przy poziomie losowym i jego maksimum potrafi wypasc w epoce 1. Ochrone przed przeuczeniem daje
wybor epoki, ktory oglada wszystkie epoki, a nie zatrzymanie sie na pierwszym plaskowyzu.

Uzycie:
    python -m src.train --epoki 60 --out checkpoints/e1.pt \\
        --w-energia 1.0 --w-parowania 1.0 --w-sklad 1.0
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.dataset import koduj, parse_pairs, BASES, PAIR_TO_CLASS, N_PAIR_CLASSES
from src.model import NARDesigner
from src.loss import KomponentyNAR, NATURAL_LOOP, NATURAL_PAIR, PROG_ZASADY, PROG_PARY
from src.prepare import wczytaj
from src.split import wczytaj_split


def energie_referencyjne(structs, seqs):
    """E(struktura | sekwencja referencyjna) dla calego zbioru, liczone RAZ.

    Sekwencje referencyjne sie nie zmieniaja, wiec liczenie tego w kazdej epoce byloby czysta strata.
    """
    import RNA
    return [RNA.energy_of_struct(q, s) for s, q in zip(structs, seqs)]


def zlozony_score(ident: float, dE_nt: float) -> int:
    """Kryterium zlozone: identycznosc jako klucz GLOWNY, delta energii jako rozstrzygacz remisow.

    Porzadek leksykograficzny zapisany jedna liczba, wg pomyslu promotora (`1000 * B + A`):

        score = round(identycznosc% ) * 1000  +  round(-dE_nt * 1000)

    Identycznosc idzie w PELNYCH PROCENTACH, bo roznice ponizej punktu procentowego na zbiorze
    walidacyjnej wielkosci to najpewniej szum; delta energii rozstrzyga dopiero wtedy, gdy dwie epoki
    wypadaja w tym samym procencie.

    DLACZEGO IDENTYCZNOSC JEST KLUCZEM GLOWNYM, a nie odwrotnie: sekwencja calkowicie zdegenerowana
    (same pary G:C, petle z adeniny) ma energie -0,538 kcal/mol/nt wobec -0,295 dla sekwencji
    naturalnych, czyli na samej energii wygrywa z ogromna przewaga. Gdyby to ona byla kluczem
    glownym, wybor epoki systematycznie wskazywalby epoke najbardziej zdegenerowana — czyli dokladnie
    ta awarie, ktora eksperyment ma zmierzyc.

    Warunek poprawnosci porzadku: |dE_nt| < 1, inaczej rozstrzygacz przebilby klucz glowny.
    Obserwowany zakres to okolo +-0,4, a na wszelki wypadek przycinamy.
    """
    return round(ident * 100) * 1000 + round(-max(-0.999, min(0.999, dE_nt)) * 1000)


# Trzy warianty wskaznika Youdena, bo mamy DWIE glowice i jedna klase szczegolnie problematyczna:
#   youden_GC       tylko typ G:C — klasa, ktora model nadprodukuje
#   youden_par      srednia po 3 typach par           (glowica par, odpowiednik zbal_par)
#   youden_zasady   srednia po 4 zasadach w petlach   (glowica zasad, odpowiednik zbal_zasady)
KRYTERIA = ("identycznosc_nt", "zbal_par", "zbal_zasady", "youden_GC", "youden_par",
            "youden_zasady", "ce", "loss", "energia", "zlozony")


def wagi_klas(structs, seqs, device):
    """Wagi ODWROTNIE PROPORCJONALNE do czestosci klas, policzone na TRENINGU.

    PO CO. Zwykla CE optymalizuje trafienia wazone liczebnoscia, wiec klasa G:C (60% treningu) wnosi
    do gradientu osiem razy wiecej niz G:U (7%). Model oplaca sie wtedy zbudowac wokol klasy
    najczestszej. Waga 1/czestosc zrownuje wklad kazdej klasy — to jest dokladnie ta zmiana, ktorej
    odpowiednikiem po stronie POMIARU jest trafnosc zbalansowana.

    NORMALIZACJA do sredniej 1 sprawia, ze skala CE sie nie zmienia, wiec wagi pozostalych czlonow
    straty zachowuja swoje znaczenie.

    TYLKO Z TRENINGU. Czestosci walidacji ani testu nie ogladamy — to byloby zagladanie w odpowiedz.
    """
    lp = torch.zeros(N_PAIR_CLASSES)
    lz = torch.zeros(len(BASES))
    for st, q in zip(structs, seqs):
        for i, j in parse_pairs(st):
            k = PAIR_TO_CLASS.get((q[i], q[j]))
            if k is not None:
                lp[k] += 1
        for i, c in enumerate(st):
            if c == "." and q[i] in BASES:
                lz[BASES.index(q[i])] += 1
    wp = (1.0 / lp.clamp_min(1.0))
    wz = (1.0 / lz.clamp_min(1.0))
    return (wp / wp.mean()).to(device), (wz / wz.mean()).to(device), lp, lz


def kryterium(w: dict, tryb: str) -> float:
    """Wybrane kryterium w konwencji "WIECEJ = LEPIEJ" (CE, loss i dE wchodza ze zmienionym znakiem).

    NIE MA TU SAMEJ SPECYFICZNOSCI, i to jest decyzja, a nie przeoczenie. Specyficznosc G:C rosnie
    do 100%, gdy model PRZESTAJE wystawiac G:C — model produkujacy same A:U mialby ja idealna.
    Jest wiec dokladnie tak samo podatna na predyktor staly jak identycznosc, tylko z drugiej strony.
    Zamiast niej wchodzi WSKAZNIK YOUDENA (czulosc + specyficznosc - 1), ktory laczy obie polowy
    i zeruje sie dla kazdego predyktora niezaleznego od pozycji, bez wzgledu na sklad wyjscia.
    """
    return {"identycznosc_nt": w["identycznosc_nt"], "zbal_par": w["zbal_par"],
            "zbal_zasady": w["zbal_zasady"], "youden_GC": w["youden_GC"],
            "youden_par": w["youden_par"], "youden_zasady": w["youden_zasady"],
            "ce": -w["ce"], "loss": -w["loss"],
            "energia": -w["dE_nt"], "zlozony": w["zlozony"]}[tryb]


@torch.no_grad()
def waliduj(model, structs, seqs, device, bs=64, komp=None, args=None, e_ref=None,
            dekodowanie="argmax", seed_dekodowania=0):
    """Liczy WSZYSTKIE kryteria naraz i zwraca je slownikiem.

    Sekwencje generujemy RAZ i z tych samych sekwencji liczymy identycznosc, czulosci klasowe oraz
    delte energii, na CALYM zbiorze walidacyjnym. Energie referencyjne przychodza gotowe.

    ZIARNO JEST STALE MIEDZY EPOKAMI. Przy `--dekodowanie probkowanie` generowanie jest losowe, wiec
    bez ustalonego ziarna kryterium walidacyjne szumialoby i wybor epoki bylby po czesci przypadkowy.
    To samo ziarno w kazdej epoce sprawia, ze porownujemy epoki, a nie losowania.
    """
    import RNA
    from src.evaluate import TYP_PARY
    model.eval()
    if dekodowanie == "probkowanie":
        torch.manual_seed(seed_dekodowania)
    traf = tot = 0
    ce_sum = ce_n = 0.0
    loss_sum = loss_n = 0.0
    dE = []
    tp_p, n_p, tp_z, n_z = Counter(), Counter(), Counter(), Counter()
    wyd_p, wyd_z = Counter(), Counter()          # ile razy model WYSTAWIL dana klase
    for s in range(0, len(structs), bs):
        cs, cq = structs[s : s + bs], seqs[s : s + bs]
        sid, pad, par, cp, cz, realne = koduj(cs, cq, device)
        lp, lz, otw = model(sid, pad, par)

        ce_partia = lp.new_zeros(())
        for logity, cel in ((lp, cp), (lz, cz)):
            m = cel != -100
            if m.any():
                ce_sum += float(nn.functional.cross_entropy(logity[m], cel[m], reduction="sum"))
                ce_n += int(m.sum())
                ce_partia = ce_partia + nn.functional.cross_entropy(logity[m], cel[m])

        if komp is not None and args is not None:
            loss_sum += float(skladaj_loss(komp, args, lp, lz, par, otw, realne, ce_partia)[0])
            loss_n += 1

        gen = model.generate(sid, pad, par, [len(x) for x in cs],
                             sample=(dekodowanie == "probkowanie"))
        for k, (g, q, t) in enumerate(zip(gen, cq, cs)):
            traf += sum(a == b for a, b in zip(g, q))
            tot += len(q)
            if e_ref is not None:
                dE.append((RNA.energy_of_struct(g, t) - e_ref[s + k]) / len(t))
            pary = parse_pairs(t)
            sparowane = {x for p in pary for x in p}
            for a, b in pary:
                wzor = TYP_PARY.get(q[a] + q[b])
                if wzor is None:
                    continue
                n_p[wzor] += 1
                pred = TYP_PARY.get(g[a] + g[b])
                if pred is not None:
                    wyd_p[pred] += 1          # ile RAZY model wystawil dana klase — do specyficznosci
                if pred == wzor:
                    tp_p[wzor] += 1
            for i in range(len(t)):
                if i in sparowane:
                    continue
                n_z[q[i]] += 1
                wyd_z[g[i]] += 1
                if g[i] == q[i]:
                    tp_z[q[i]] += 1

    r_p = [tp_p[k] / n_p[k] for k in ("GC", "AU", "GU") if n_p[k]]
    r_z = [tp_z[b] / n_z[b] for b in "ACGU" if n_z[b]]

    def specyficznosc(k, tp, n, wyd, N):
        """Z tych, ktore NIE sa klasy k, ilu model NIE wrzucil do k."""
        neg = N - n[k]
        if neg <= 0:
            return float("nan")
        fp = wyd[k] - tp[k]                    # wystawione jako k, a nie bedace k
        return (neg - fp) / neg

    N_p, N_z = sum(n_p.values()), sum(n_z.values())
    s_p = {k: specyficznosc(k, tp_p, n_p, wyd_p, N_p) for k in ("GC", "AU", "GU") if n_p[k]}
    s_z = {b: specyficznosc(b, tp_z, n_z, wyd_z, N_z) for b in "ACGU" if n_z[b]}

    # WSKAZNIK YOUDENA = czulosc + specyficznosc - 1. Jego poziom odniesienia to DOKLADNIE ZERO
    # dla kazdego modelu, ktory przypisuje klasy niezaleznie od pozycji — bo taki model ma
    # czulosc_k = q_k oraz specyficznosc_k = 1 - q_k, wiec suma zawsze wychodzi 1. Nie da sie go
    # podbic przesunieciem skladu wyjscia, w przeciwiescieństwie i do identycznosci, i do samej
    # specyficznosci (ta ostatnia rosnie do 100%, gdy model PRZESTAJE wystawiac dana klase).
    j_p = {k: (tp_p[k] / n_p[k]) + s_p[k] - 1.0 for k in s_p}
    j_z = {b: (tp_z[b] / n_z[b]) + s_z[b] - 1.0 for b in s_z}

    w = {"identycznosc_nt": traf / max(tot, 1),
         "zbal_par": float(np.mean(r_p)) if r_p else float("nan"),
         "zbal_zasady": float(np.mean(r_z)) if r_z else float("nan"),
         "spec_GC": s_p.get("GC", float("nan")),
         "youden_GC": j_p.get("GC", float("nan")),
         "youden_par": float(np.mean(list(j_p.values()))) if j_p else float("nan"),
         "youden_zasady": float(np.mean(list(j_z.values()))) if j_z else float("nan"),
         "ce": ce_sum / max(ce_n, 1),
         "loss": loss_sum / max(loss_n, 1) if loss_n else float("nan"),
         "dE_nt": float(np.mean(dE)) if dE else float("nan")}
    w["zlozony"] = zlozony_score(w["identycznosc_nt"], w["dE_nt"]) if dE else float("nan")

    model.train()
    return w


def skladaj_loss(komp, args, lp, lz, par, otw, realne, ce):
    """Pelna strata: CE plus wlaczone komponenty, kazdy ze swoja waga.

    Uzywana i w treningu, i w walidacji — dzieki temu `--wybor loss` porownuje DOKLADNIE te wielkosc,
    ktora model minimalizuje, a nie jej przyblizenie.
    """
    p_par, p_zas = lp.softmax(-1), lz.softmax(-1)
    z = lp.new_zeros(())
    e = komp.energia(p_par, p_zas, par, otw, realne) if args.w_energia else z
    a = komp.parowania(p_par, p_zas, par, otw, realne) if args.w_parowania else z
    # Kara TV ma DWA czlony z osobnymi wagami: pary i petle. Ablacja pokazala, ze ciagna
    # w przeciwne strony, wiec E3 wlacza tylko petle.
    if args.w_sklad_tv_pary or args.w_sklad_tv_petle:
        c_par, c_pet = komp.sklad(p_par, p_zas, par, otw, realne)
    else:
        c_par = c_pet = z
    sz = komp.sklad_zasad(p_par, p_zas, par, otw, realne) if args.w_sklad_zasad else z
    sp = komp.sklad_par(p_par, otw) if args.w_sklad_par else z
    loss = (ce + args.w_energia * e + args.w_parowania * a
            + args.w_sklad_tv_pary * c_par + args.w_sklad_tv_petle * c_pet
            + args.w_sklad_zasad * sz + args.w_sklad_par * sp)
    return loss, (e, a, c_par, c_pet, sz, sp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tryb-podzialu", choices=["rodzinowy"], default="rodzinowy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoki", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warstwy", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--w-energia", type=float, default=0.0)
    ap.add_argument("--w-parowania", type=float, default=0.0)
    ap.add_argument("--w-sklad", type=float, default=0.0,
                    help="NASZA kara za sklad (odleglosc TV, dwustronna, per sekwencja) — OBA czlony, "
                         "pary i petle, z ta sama waga. To jest ustawienie E1")
    ap.add_argument("--w-sklad-tv-pary", type=float, default=None,
                    help="tylko czlon PAR kary TV; nadpisuje --w-sklad. Cel G:C 0,599 pasuje do "
                         "treningu, a nie do nowych rodzin, wiec ten czlon zwykle szkodzi")
    ap.add_argument("--w-sklad-tv-petle", type=float, default=None,
                    help="tylko czlon PETLI kary TV; nadpisuje --w-sklad. Cel jest tu trafny, bo "
                         "sklad petli nie rozni sie miedzy rodzinami. E3 = same petle")
    ap.add_argument("--w-sklad-zasad", type=float, default=0.0,
                    help="kara promotora, DistribLoss: progi dolne udzialow A/C/G/U, per sekwencja")
    ap.add_argument("--w-sklad-par", type=float, default=0.0,
                    help="kara promotora, DistribLoss3+4: progi dolne udzialow typow par G:C/A:U/G:U")
    ap.add_argument("--wybor", choices=list(KRYTERIA), default="zbal_par",
                    help="czym wybierac najlepsza epoke na walidacji; zadna opcja nie przewiduje "
                         "struktury. Domyslne 'zbal_par' to srednia czulosc po typach par, bez wagi "
                         "liczebnoscia — predyktor niezalezny od pozycji dostaje tam 1/3 niezaleznie "
                         "od tego, co produkuje. UWAGA: 'loss' to wlasna strata modelu, wiec E1 i E2 "
                         "wybieraja wtedy epoke roznymi miarami")
    ap.add_argument("--cierpliwosc", type=int, default=None,
                    help="ile epok bez poprawy konczy trening. DOMYSLNIE = --epoki, czyli wczesnego "
                         "zatrzymania NIE MA. Powod: `zbal_par` stoi tuz przy poziomie losowym "
                         "i szumi, wiec jego maksimum potrafi wypasc w epoce 1 — przy cierpliwosci "
                         "10 zapisywal sie model po JEDNEJ epoce. Do tego cosine schodzi do zera "
                         "dopiero w ostatniej epoce, wiec przerwanie zostawia model nierozstrojony"),
    ap.add_argument("--dekodowanie", choices=["argmax", "probkowanie"], default="probkowanie",
                    help="jak zamienic rozklady na sekwencje w WALIDACJI; ta sama opcja musi byc "
                         "uzyta pozniej w src/evaluate.py, inaczej wybieramy epoke pod inne "
                         "dekodowanie niz raportujemy. `argmax` zostawiony wylacznie do odtworzenia "
                         "diagnozy degeneracji — nie uzywac do nowych wynikow"),
    ap.add_argument("--seed-modelu", type=int, default=None,
                    help="ziarno inicjalizacji wag i kolejnosci partii, BEZ ruszania podzialu "
                         "danych; sluzy do mierzenia szumu miedzy przebiegami. Domyslnie = --seed")
    ap.add_argument("--wagi-klas", action="store_true",
                    help="WAZONA CE: kazda klasa wnosi do gradientu tyle samo, wagi 1/czestosc "
                         "policzone na TRENINGU. Odpowiednik trafnosci zbalansowanej po stronie "
                         "uczenia. CE w logu pozostaje niewazona, zeby dalo sie ja porownywac")
    ap.add_argument("--seed-dekodowania", type=int, default=0,
                    help="ziarno losowania; STALE miedzy epokami, zeby kryterium nie szumialo")
    ap.add_argument("--out", default="checkpoints/model.pt")
    args = ap.parse_args()
    # Brak wartosci = brak wczesnego zatrzymania, niezaleznie od tego, ile epok zamowiono.
    if args.cierpliwosc is None:
        args.cierpliwosc = args.epoki
    # `--w-sklad` ustawia oba czlony kary TV; osobne flagi go nadpisuja.
    if args.w_sklad_tv_pary is None:
        args.w_sklad_tv_pary = args.w_sklad
    if args.w_sklad_tv_petle is None:
        args.w_sklad_tv_petle = args.w_sklad

    # DWA OSOBNE ZIARNA. `--seed` wyznacza PODZIAL danych, wiec zmiana go zmienia zbior testowy
    # i wyniki przestaja byc porownywalne. `--seed-modelu` rusza tylko losowa inicjalizacje wag
    # i kolejnosc partii — to jest wlasciwe ziarno do sprawdzania, ile z roznicy miedzy przebiegami
    # jest szumem. Domyslnie rowne `--seed`, wiec dotychczasowe komendy dzialaja jak przedtem.
    sm = args.seed if args.seed_modelu is None else args.seed_modelu
    torch.manual_seed(sm)
    np.random.seed(sm)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = wczytaj()
    spl = wczytaj_split(args.tryb_podzialu, args.seed)
    S = {k: df.secondary_structure.iloc[v].tolist() for k, v in spl.items()}
    Q = {k: df.sequence.iloc[v].tolist() for k, v in spl.items()}
    max_len = int(df.secondary_structure.str.len().max())

    print(f"urzadzenie {dev} | struktur {len(df)} | podzial {args.tryb_podzialu}")
    print(f"train {len(S['train'])}  val {len(S['val'])}  test {len(S['test'])}")
    print(f"wagi kar: energia {args.w_energia}  parowania {args.w_parowania}")
    if args.w_sklad_tv_pary or args.w_sklad_tv_petle:
        print(f"  sklad TV (nasz):        pary {args.w_sklad_tv_pary}  "
              f"petle {args.w_sklad_tv_petle}  "
              f"cel petle {NATURAL_LOOP}  cel pary {NATURAL_PAIR}")
    if args.w_sklad_zasad:
        print(f"  sklad zasad (promotor): waga {args.w_sklad_zasad}  progi {PROG_ZASADY}")
    if args.w_sklad_par:
        print(f"  sklad par (promotor):   waga {args.w_sklad_par}  progi {PROG_PARY}")
    print(f"wybor epoki: {args.wybor}")

    w_par = w_zas = None
    if args.wagi_klas:
        w_par, w_zas, lp_, lz_ = wagi_klas(S["train"], Q["train"], dev)
        print("WAZONA CE, wagi 1/czestosc z TRENINGU, znormalizowane do sredniej 1:")
        print("  pary  " + "  ".join(
            f"{a}-{b} {lp_[i]/lp_.sum():.3f}->{w_par[i]:.2f}"
            for i, (a, b) in enumerate([("G", "C"), ("C", "G"), ("A", "U"),
                                        ("U", "A"), ("G", "U"), ("U", "G")])))
        print("  petle " + "  ".join(
            f"{b} {lz_[i]/lz_.sum():.3f}->{w_zas[i]:.2f}" for i, b in enumerate(BASES)))

    model = NARDesigner(d_model=args.d_model, num_layers=args.warstwy,
                        max_len=max_len + 1, dropout=args.dropout).to(dev)
    print(f"parametry: {sum(p.numel() for p in model.parameters())/1e6:.2f} mln")
    komp = KomponentyNAR(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoki)

    # Sekwencje referencyjne sie nie zmieniaja, wiec ich energie liczymy raz na caly trening.
    t_ref = time.time()
    e_ref = energie_referencyjne(S["val"], Q["val"])
    print(f"energie referencyjne walidacji: {len(e_ref)} struktur, {time.time()-t_ref:.1f}s")

    order = np.arange(len(S["train"]))
    best, bad = -1e9, 0
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    for ep in range(1, args.epoki + 1):
        np.random.shuffle(order)
        t0 = time.time()
        agg = {"ce": 0.0, "e": 0.0, "a": 0.0, "cp": 0.0, "cl": 0.0, "sz": 0.0,
               "sp": 0.0, "n": 0,
               "ce_sum": 0.0, "ce_n": 0}
        for s in range(0, len(order), args.batch):
            b = order[s : s + args.batch]
            cs = [S["train"][i] for i in b]
            cq = [Q["train"][i] for i in b]
            sid, pad, par, cp, cz, realne = koduj(cs, cq, dev)
            lp, lz, otw = model(sid, pad, par)

            # CE W STRACIE to SUMA dwoch srednich (osobno glowica par, osobno zasad) — obie
            # glowice maja wtedy rowny wplyw na gradient, niezaleznie od tego, ile pozycji obsluguja.
            ce = lp.new_zeros(())
            for logity, cel, wagi in ((lp, cp, w_par), (lz, cz, w_zas)):
                m = cel != -100
                if m.any():
                    ce = ce + nn.functional.cross_entropy(logity[m], cel[m], weight=wagi)
                    # CE DO LOGU zostaje NIEWAZONA — inaczej nie dalaby sie zestawic z walidacyjna
                    # ani z CE innych przebiegow. Wagi zmieniaja to, co model optymalizuje, a nie to,
                    # czym go mierzymy.
                    # CE DO LOGU liczymy INACZEJ: jedna srednia po wszystkich pozycjach. Musi byc
                    # tak samo jak w `waliduj`, inaczej zestawienie "CE treningowa obok walidacyjnej"
                    # porownywaloby wielkosci w roznej skali (suma dwoch srednich jest ok. 2x wieksza)
                    # i przeuczenie wygladaloby na duzo lagodniejsze, niz jest.
                    with torch.no_grad():
                        agg["ce_sum"] += float(nn.functional.cross_entropy(
                            logity[m], cel[m], reduction="sum"))
                        agg["ce_n"] += int(m.sum())

            loss, (e, a, c_par, c_pet, sz, sp) = skladaj_loss(komp, args, lp, lz, par, otw,
                                                              realne, ce)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in zip(("ce", "e", "a", "cp", "cl", "sz", "sp"),
                            (ce, e, a, c_par, c_pet, sz, sp)):
                agg[k] += float(v)
            agg["n"] += 1
        sched.step()

        w = waliduj(model, S["val"], Q["val"], dev, komp=komp, args=args, e_ref=e_ref,
                    dekodowanie=args.dekodowanie, seed_dekodowania=args.seed_dekodowania)
        kryt = kryterium(w, args.wybor)
        n = max(agg["n"], 1)
        ce_tr = agg["ce_sum"] / max(agg["ce_n"], 1)
        msg = (f"[{ep}/{args.epoki}] CE {ce_tr:.4f} | energia {agg['e']/n:+.4f} "
               f"parowania {agg['a']/n:.4f}")
        for klucz, etyk, waga in (("cp", "sklad_pary", args.w_sklad_tv_pary),
                                  ("cl", "sklad_petle", args.w_sklad_tv_petle),
                                  ("sz", "zasady", args.w_sklad_zasad),
                                  ("sp", "pary", args.w_sklad_par)):
            if waga:
                msg += f" {etyk} {agg[klucz]/n:.4f}"
        # WSZYSTKIE kryteria w kazdej epoce, niezaleznie od tego, ktore wybieramy. Kosztuje to zero,
        # a pozwala pozniej powiedziec, ktora epoke wskazalby kazdy z nich. To sa liczby z WALIDACJI,
        # wiec patrzenie na nie nie jest przeciekiem; przeciekiem byloby ocenianie ich na tescie.
        msg += (f" | val ident {w['identycznosc_nt']:.4f} zbal_par {w['zbal_par']:.4f} "
                f"zbal_zas {w['zbal_zasady']:.4f} specGC {w['spec_GC']:.4f} "
                f"jGC {w['youden_GC']:+.4f} j_par {w['youden_par']:+.4f} "
                f"j_zas {w['youden_zasady']:+.4f} CE {w['ce']:.4f} loss {w['loss']:.4f} "
                f"dE/nt {w['dE_nt']:+.4f} zlozony {w['zlozony']}")
        msg += f" | {time.time()-t0:.0f}s"
        if kryt > best:
            best, bad = kryt, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "max_len": max_len},
                       ROOT / args.out)
            msg += "  *najlepszy -> zapis"
        else:
            bad += 1
            msg += f"  (bez poprawy {bad}/{args.cierpliwosc})"
        print(msg, flush=True)
        if bad >= args.cierpliwosc:
            print("early stopping")
            break

    print(f"\nnajlepszy ({args.wybor}) na walidacji: {best:.4f} | {args.out}")


if __name__ == "__main__":
    main()
