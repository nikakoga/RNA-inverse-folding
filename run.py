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
    "ce": "checkpoints/ce_sama_ce.pt",         # sama cross-entropia, bez zadnych kar
    "cew": "checkpoints/cew_wazona_ce.pt",     # jak CE, ale CE wazona odwrotnie do czestosci klas
    "e1w": "checkpoints/e1w_wazona_ce.pt",
    "e2w": "checkpoints/e2w_wazona_ce.pt",     # jak E2, ale CE wazona
    "e3":  "checkpoints/e3_bez_skladu.pt",     # energia + parowania, BEZ kary za sklad
}
SPLIT = ["--tryb-podzialu", "rodzinowy"]
# KRYTERIUM WYBORU EPOKI, jawnie i TAKIE SAMO we wszystkich eksperymentach.
#
# POPRZEDNIO bylo `zlozony`: identycznosc jako klucz glowny, dE/nt jako rozstrzygacz remisow. Oba
# czlony okazaly sie premiowac dokladnie te awarie, ktora badamy. Identycznosc jest maksymalizowana
# przez stale wskazywanie klasy najczestszej (sama para G:C daje 48,4%, wiecej niz ktorykolwiek
# z naszych modeli), a dE/nt spada, gdy rosnie udzial G:C, bo to najstabilniejsza para. Wybieralismy
# wiec epoke miara, ktora nagradza nadprodukcje G:C.
#
# `zbal_par` to srednia czulosc po trzech typach par, BEZ wagi liczebnoscia. Predyktor niezalezny od
# pozycji dostaje tam dokladnie 1/3, niezaleznie od tego, co produkuje — przesuniecie skladu wyjscia
# nie podbija tej miary.
WYBOR = ["--wybor", "zbal_par"]
# BEZ WCZESNEGO ZATRZYMANIA, i to jest konieczne wlasnie przy `zbal_par`.
#
# `zbal_par` stoi tuz przy poziomie losowym i szumi z epoki na epoke o +-0,3 pp, wiec jego maksimum
# potrafi wypasc w epoce 1 — zanim model czegokolwiek sie nauczy. Przy cierpliwosci 10 trening E1
# zatrzymal sie wtedy w epoce 11 i zapisal epoke 1, podczas gdy CE przeszlo pelne 60 epok. Modele
# roznily sie czasem uczenia szescdziesieciokrotnie, wiec porownanie kar nie mierzylo juz kar.
#
# Przy cierpliwosci 60 kazdy przebieg widzi TYLE SAMO epok i konczy z rozstrojonym krokiem uczenia
# (cosine schodzi do zera dopiero w ostatniej epoce). Wybor epoki dalej nalezy do `--wybor`.
CIERPLIWOSC = ["--cierpliwosc", "60"]
# DEKODOWANIE: ARGMAX, czyli na kazdej pozycji litera o najwiekszym prawdopodobienstwie.
# Zaleta wobec probkowania: generowanie jest DETERMINISTYCZNE, nie trzeba ziarna i wynik da sie
# odtworzyc co do litery.
#
# ARGMAX WYMAGA `--kary-na-argmax`, i to nie jest opcjonalne. Kary licza sie normalnie na
# rozkladach, a argmax patrzy tylko, kto jest na szczycie — nie o ile wygrywa. Zmierzylismy model
# o miekkim skladzie `G:C 0,622`, ktory po argmaxie dawal 0,984: kara byla spelniona, wyjscie
# zdegenerowane. Straight-through pokazuje karom twarde wyjscie, wiec maja co karac.
#
# Architektura sie przy tym nie zmienia: nadal jeden przebieg enkodera, a para (i,j) nadal jest
# JEDNA decyzja z szesciu klas kanonicznych.
DEKOD = ["--dekodowanie", "argmax", "--kary-na-argmax"]
# src/evaluate.py nie zna flagi `--kary-na-argmax` (nie liczy strat), wiec do oceny
# przekazujemy samo dekodowanie. Musi byc TAKIE SAMO jak w walidacji.
DEKOD_OCENA = ["--dekodowanie", "argmax"]

EKSPERYMENTY: list[tuple[str, str, list]] = [

    ("dane", "Odsianie redundancji natywnym cd-hit-est, filtry i podzial rodzinowy", [
        # cd-hit-est wymaga WSL (instalacja w README). Odsiewana jest WYLACZNIE pula naturalna —
        # Eterny nie ma w treningu, wiec nie ma tam przecieku, ktoremu odsiewanie mialoby zapobiec.
        ("cdhit", ["-m", "src.cdhit"]),
        ("przygotowanie", ["-m", "src.prepare"]),
        ("podzial", ["-m", "src.split", "--tryb", "rodzinowy"]),
    ]),

    ("E1", "Kara za sklad: NASZA (odleglosc TV od celu naturalnego). Wszystkie trzy komponenty z waga 1,0", [
        # WSZYSTKIE TRZY KOMPONENTY Z WAGA 1,0. Zadna z tych liczb nie jest dobrana pod wynik:
        # specyfikacja promotora wnosi swoja kare wprost (`loss = loss + DistribLoss`), a my
        # przyjmujemy te sama konwencje dla pozostalych czlonow. Dzieki temu w calej pracy nie ma
        # ani jednej wagi, ktorej nie da sie uzasadnic jednym zdaniem.
        #
        # POPRZEDNIO parowania mialy 6,0 — liczbe odziedziczona z wczesniejszego projektu, ktorej
        # nigdzie nie umielismy uzasadnic. Przeglad wag (`python -m src.przeglad`) pokazal, ze na
        # trafnosc nie wplywa ona w ogole: caly zakres 0..12 miesci sie w szumie pojedynczego
        # przebiegu. Skoro nie zmienia wyniku, a wymaga tlumaczenia, zdejmujemy ja.
        ("trening", ["-m", "src.train", "--epoki", "60", *SPLIT, *WYBOR, *DEKOD, *CIERPLIWOSC,
                     "--w-energia", "1.0", "--w-parowania", "1.0", "--w-sklad", "1.0",
                     "--out", CK["e1"]]),
        # Dekodowanie w ocenie MUSI byc takie samo jak w walidacji.
        ("ocena_test", ["-m", "src.evaluate", "--ckpt", CK["e1"], *SPLIT, *DEKOD_OCENA,
                        "--na", "test", "--csv", "e1_test.csv"]),
        ("baseline", ["-m", "src.evaluate", "--baseline", *SPLIT,
                      "--na", "test", "--csv", "baseline_test.csv"]),
    ]),

    ("E2", "Kara za sklad: wg specyfikacji promotora (progi dolne, per sekwencja). "
           "Jedyna roznica wobec E1 to KONSTRUKCJA tej kary", [
        #   E1  --w-sklad 1          odleglosc TV od celu; DWUSTRONNA — karze takze nadmiar
        #   E2  --w-sklad-zasad 1    progi DOLNE udzialow A/C/G/U
        #       --w-sklad-par   1    progi DOLNE udzialow typow par (DistribLoss3)
        #                            JEDNOSTRONNA — nadmiar bezkarny
        #
        # Obie liczone PER SEKWENCJA, wszystkie komponenty z waga 1,0, wiec jedyna zmienna
        # jest KSZTALT kary za sklad. Energia i parowania identyczne jak w E1.
        ("trening", ["-m", "src.train", "--epoki", "60", *SPLIT, *WYBOR, *DEKOD, *CIERPLIWOSC,
                     "--w-energia", "1.0", "--w-parowania", "1.0",
                     "--w-sklad-zasad", "1.0", "--w-sklad-par", "1.0",
                     "--out", CK["e2"]]),
        ("ocena_test", ["-m", "src.evaluate", "--ckpt", CK["e2"], *SPLIT, *DEKOD_OCENA,
                        "--na", "test", "--csv", "e2_test.csv"]),
    ]),

    ("CE", "SAMA CROSS-ENTROPIA: to samo co E1, ale wszystkie trzy kary wylaczone", [
        # ABLACJA. Pytanie: czy czlony ze specyfikacji promotora w ogole ucza model trafiac
        # w pare? Skoro kary sa JEDYNA roznica wobec E1, kazda roznica w wyniku pochodzi od nich.
        #
        # Podzial rol w naszej stracie jest asymetryczny:
        #   CE          JEDYNY czlon ogladajacy sekwencje referencyjna — tylko on moze nauczyc,
        #               ze W TYM MIEJSCU ma byc A:U, a nie G:C
        #   energia     patrzy na strukture i wlasna sekwencje; mowi "badz stabilny"
        #   parowania   to samo; mowi "badz jednoznaczny"
        #   sklad       mowi "miej naturalne proporcje", nie "postaw je we wlasciwych miejscach"
        #
        # Trzy ostatnie sa NIENADZOROWANE: da sie je policzyc bez znajomosci referencji, wiec nie maja
        # jak przekazac informacji o pozycji. Energia moze wrecz przeszkadzac, bo para G:C jest
        # najstabilniejsza, wiec czlon energetyczny nagradza wstawianie jej wszedzie — czyli
        # dokladnie zachowanie predyktora stalego.
        #
        # Wszystko poza wagami kar identyczne jak w E1, wiec kary sa JEDYNA zmienna.
        ("trening", ["-m", "src.train", "--epoki", "60", *SPLIT, *WYBOR, *DEKOD, *CIERPLIWOSC,
                     "--out", CK["ce"]]),
        ("ocena_test", ["-m", "src.evaluate", "--ckpt", CK["ce"], *SPLIT, *DEKOD_OCENA,
                        "--na", "test", "--csv", "ce_test.csv"]),
    ]),

    ("CEW", "WAZONA CROSS-ENTROPIA: to samo co CE, ale kazda klasa wnosi do gradientu tyle samo", [
        # PYTANIE. Trafnosc zbalansowana stoi na poziomie losowym we wszystkich przebiegach. Jedna
        # z hipotez mowi, ze winna jest NIEROWNOWAGA KLAS: G:C to 60% par treningowych, G:U 5%, wiec
        # zwykla CE oplaca sie modelowi zaspokoic przez klase najczestsza. Wagi 1/czestosc usuwaja
        # te zachete — to jest odpowiednik trafnosci zbalansowanej po stronie UCZENIA.
        #
        # CZEGO SIE SPODZIEWAC, i dlaczego mimo to warto zmierzyc. Jesli rozklad modelu nie zalezy
        # od pozycji, to jego czulosc dla klasy c rowna sie po prostu udzialowi c na wyjsciu, a
        # trafnosc zbalansowana wynosi wtedy 1/k NIEZALEZNIE od tego, jaki ten udzial jest. Wagi
        # zmieniaja udzialy, wiec same z siebie nie moga ruszyc trafnosci zbalansowanej. Ale zmieniaja
        # tez gradient, wiec MOGA pomoc modelowi rozroznic klasy — i tego wlasnie nie da sie
        # przewidziec zza biurka.
        ("trening", ["-m", "src.train", "--epoki", "60", *SPLIT, *WYBOR, *DEKOD, *CIERPLIWOSC,
                     "--wagi-klas", "--out", CK["cew"]]),
        ("ocena_test", ["-m", "src.evaluate", "--ckpt", CK["cew"], *SPLIT, *DEKOD_OCENA,
                        "--na", "test", "--csv", "cew_test.csv"]),
    ]),

    ("E1W", "E1 z wazona CE: kara TV za sklad ORAZ wyrownany wklad klas do gradientu", [
        # Czy wagi klas i kara za sklad sie sumuja, czy sobie przeszkadzaja. Obie ruszaja udzialy
        # klas, ale w inna strone: kara ciagnie ku skladowi NATURALNEMU (G:C 0,599), a wagi ku
        # ROWNEMU wkladowi klas, czyli w praktyce ku wyrownaniu udzialow.
        ("trening", ["-m", "src.train", "--epoki", "60", *SPLIT, *WYBOR, *DEKOD, *CIERPLIWOSC,
                     "--w-energia", "1.0", "--w-parowania", "1.0", "--w-sklad", "1.0",
                     "--wagi-klas", "--out", CK["e1w"]]),
        ("ocena_test", ["-m", "src.evaluate", "--ckpt", CK["e1w"], *SPLIT, *DEKOD_OCENA,
                        "--na", "test", "--csv", "e1w_test.csv"]),
    ]),

    ("E2W", "E2 z wazona CE: kara progowa promotora ORAZ wyrownany wklad klas do gradientu", [
        # Domkniecie kwadratu. Pierwszy zestaw eksperymentow to E1 / E2 / CE, czyli dwie konstrukcje
        # kary i ablacja. Drugi to E1W / E2W / CEW — te same trzy, ale z CE wazona odwrotnie do
        # czestosci klas. Dzieki temu roznica "kara TV kontra kara progowa" da sie odczytac
        # OSOBNO w kazdym z dwoch rezimow, zamiast mieszac sie ze zmiana wazenia.
        ("trening", ["-m", "src.train", "--epoki", "60", *SPLIT, *WYBOR, *DEKOD, *CIERPLIWOSC,
                     "--w-energia", "1.0", "--w-parowania", "1.0",
                     "--w-sklad-zasad", "1.0", "--w-sklad-par", "1.0",
                     "--wagi-klas", "--out", CK["e2w"]]),
        ("ocena_test", ["-m", "src.evaluate", "--ckpt", CK["e2w"], *SPLIT, *DEKOD_OCENA,
                        "--na", "test", "--csv", "e2w_test.csv"]),
    ]),

    ("E3", "KONTROLA: energia i parowania z waga 1,0, ale BEZ kary za sklad", [
        # PO CO. E1 ma trzy kary i wypada gorzej niz CE, ktore nie ma zadnej:
        #
        #     E1  zbal_par 34,43%   Youden +0,0169   G:C 0,654
        #     CE  zbal_par 35,22%   Youden +0,0287   G:C 0,592
        #
        # Ale te dwa przebiegi roznia sie TRZEMA rzeczami naraz, wiec nie wiadomo, ktora odpowiada
        # za roznice. E3 rozdziela to na dwa kroki:
        #
        #     E1 -> E3   zdejmujemy KARE ZA SKLAD          (zostaja energia i parowania)
        #     E3 -> CE   zdejmujemy ENERGIE i PAROWANIA    (nie zostaje nic)
        #
        # Kazdy krok zmienia dokladnie jedna rzecz, wiec roznice da sie przypisac.
        ("trening", ["-m", "src.train", "--epoki", "60", *SPLIT, *WYBOR, *DEKOD, *CIERPLIWOSC,
                     "--w-energia", "1.0", "--w-parowania", "1.0",
                     "--out", CK["e3"]]),
        ("ocena_test", ["-m", "src.evaluate", "--ckpt", CK["e3"], *SPLIT, *DEKOD_OCENA,
                        "--na", "test", "--csv", "e3_test.csv"]),
    ]),]


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
