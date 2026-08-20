"""Ocena modelu. Dwa zbiory, trzy miary.

ZBIORY
  TEST NATURALNY   20% puli, rodziny nieobecne w treningu. Ocena GLOWNA.
  ETERNA <= 50 nt  zagadki projektowe ludzi. Pomocnicza, zewnetrzna wzgledem naszych danych.

MIARY
  rozwiazane   czy nasza sekwencja ZWIJA SIE w zadana strukture (RNAfold). Zero-jedynkowa.
               UWAGA: to NIE jest porownanie z prawdziwa sekwencja — sekwencja calkiem inna od
               wzorcowej moze rozwiazac zagadke, jesli tylko zwija sie poprawnie.
  odzysk       ulamek pozycji, na ktorych trafilismy w litere prawdziwej sekwencji. Bez zwijania.
  dE/nt        [E(cel | nasza) - E(cel | prawdziwa)] / dlugosc. Ujemne = nasza stabilizuje cel
               LEPIEJ niz prawdziwa. Jedyna miara, ktora nie uzywa zwijania w ogole.

F1 celowo pomijamy: rozdaje punkty czesciowe za pojedyncze pary, wiec dziedziczy blad RNAfolda
na kazdej z nich i jest trudne do odczytu.

Uzycie:
    python -m src.evaluate --ckpt checkpoints/e1.pt
    python -m src.evaluate --ckpt checkpoints/e1.pt --na val      # do strojenia
    python -m src.evaluate --baseline                             # losowy kanoniczny
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import RNA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.dataset import koduj, losowa_kanoniczna, parse_pairs
from src.model import NARDesigner
from src.prepare import wczytaj
from src.split import wczytaj_split

KUBELKI = [(0, 50), (51, 100), (101, 200)]
ANALIZA = ROOT / "experiments" / "analysis"


def wczytaj_model(ckpt, device):
    d = torch.load(ROOT / ckpt, map_location=device, weights_only=False)
    a = d["args"]
    m = NARDesigner(d_model=a["d_model"], num_layers=a["warstwy"],
                    max_len=d["max_len"] + 1, dropout=0.0).to(device)
    m.load_state_dict(d["model"])
    m.eval()
    return m


@torch.no_grad()
def generuj(model, structs, device, bs=32):
    out = []
    for s in range(0, len(structs), bs):
        cs = structs[s : s + bs]
        sid, pad, par, _, _, _ = koduj(cs, None, device)
        out += model.generate(sid, pad, par, [len(x) for x in cs])
    return out


def eterna(max_len=50):
    E = pd.read_csv(ROOT / "data" / "raw" / "eterna100.tsv", sep="\t")
    col = "Secondary Structure V2"
    par = []
    for st, sq in zip(E[col], E["Sample Solution (V2/Vienna2)"]):
        st, sq = str(st).strip(), str(sq).strip()
        if (len(st) <= max_len and len(st) == len(sq) and "(" in st
                and set(st) <= set(".()") and set(sq) <= set("ACGU")):
            par.append((st, sq))
    return [a for a, _ in par], [b for _, b in par]


def ocen(structs, gen, refs, etykieta, kubelki=KUBELKI):
    rows = []
    for lo, hi in kubelki:
        sel = [i for i, t in enumerate(structs) if lo <= len(t) <= hi]
        if not sel:
            continue
        sol, odz, de = [], [], []
        for i in sel:
            t, q, r = structs[i], gen[i], refs[i]
            sol.append(float(RNA.fold(q)[0] == t))
            odz.append(float(np.mean([a == b for a, b in zip(q, r)])))
            de.append((RNA.energy_of_struct(q, t) - RNA.energy_of_struct(r, t)) / len(t))
        rows.append({"zbior": etykieta, "dlugosc": f"{lo}-{hi}", "n": len(sel),
                     "rozwiazane": int(np.sum(sol)), "odzysk": float(np.mean(odz)),
                     "dE_nt": float(np.mean(de))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--baseline", action="store_true", help="losowa sekwencja kanoniczna zamiast modelu")
    ap.add_argument("--tryb-podzialu", choices=["rodzinowy", "losowy"], default="rodzinowy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--na", choices=["test", "val"], default="test",
                    help="'val' = do strojenia (wolno wielokrotnie), 'test' = do raportu (raz)")
    ap.add_argument("--eterna-max", type=int, default=50)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    ANALIZA.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = wczytaj()
    idx = wczytaj_split(args.tryb_podzialu, args.seed)[args.na]
    S = df.secondary_structure.iloc[idx].tolist()
    Q = df.sequence.iloc[idx].tolist()
    ET, ER = eterna(args.eterna_max)

    if args.baseline:
        nazwa = "baseline losowy kanoniczny"
        G = [losowa_kanoniczna(t) for t in S]
        GE = [losowa_kanoniczna(t) for t in ET]
    else:
        nazwa = args.ckpt
        model = wczytaj_model(args.ckpt, dev)
        G, GE = generuj(model, S, dev), generuj(model, ET, dev)

    print(f"{nazwa}\nzbior {args.na.upper()}: {len(S)} struktur | Eterna <= {args.eterna_max} nt: {len(ET)}\n")
    w = ocen(S, G, Q, args.na) + ocen(ET, GE, ER, f"eterna<={args.eterna_max}",
                                      [(0, args.eterna_max)])
    d = pd.DataFrame(w)
    print(f"{'zbior':<12}{'dlugosc':>10}{'n':>6}{'rozwiazane':>12}{'odzysk':>9}{'dE/nt':>9}")
    for _, r in d.iterrows():
        print(f"{r.zbior:<12}{r.dlugosc:>10}{r.n:>6}"
              f"{str(r.rozwiazane)+'/'+str(r.n):>12}{r.odzysk:>9.3f}{r.dE_nt:>9.4f}")
    if args.csv:
        d.insert(0, "model", nazwa)
        d.to_csv(ANALIZA / args.csv, index=False, encoding="utf-8")
        print(f"\nzapisano: experiments/analysis/{args.csv}")


if __name__ == "__main__":
    main()
