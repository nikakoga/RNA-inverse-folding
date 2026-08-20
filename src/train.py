"""Trening nieautoregresyjnego transformera. Bez uczenia ze wzmocnieniem.

FUNKCJA STRATY
    CE                    uczy sie z prawdziwych sekwencji: co w jakim motywie faktycznie wystepuje
  + w_e * energia         czlony sekwencyjne z tablic Turnera
  + w_a * parowania       liczba MOZLIWYCH parowan G*C + A*U + G*U, na nt^2
  + w_c * sklad           odleglosc od skladu naturalnego, osobno TYPY PAR i osobno petle

Wagi domyslnie zerowe — czysta CE jest punktem odniesienia. Komponenty wlacza sie flagami.

WYBOR EPOKI. Domyslnie po ODZYSKU (ulamek pozycji trafionych wzgledem prawdziwej sekwencji), bo ta
miara nie wymaga zwijania. UWAGA: odzysk i liczba rozwiazanych sa PRZECIWSTAWNE — korelacja -0,79
na sweepie wag. Wybieranie po odzysku systematycznie zapisuje model, ktory ROZWIAZUJE NAJGORZEJ.
Alternatywy: `--wybor solved` (uzywa RNAfolda, tylko do wyboru epoki) albo `--wybor energia`
(nie uzywa zwijania w ogole, korelacja z rozwiazanymi -0,63).

Uzycie:
    python -m src.train --epoki 60 --out checkpoints/e1.pt --w-energia 1.0 --w-parowania 6.0 --w-sklad 1.7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.dataset import koduj, BASES
from src.model import NARDesigner
from src.loss import KomponentyNAR, NATURAL_LOOP, NATURAL_PAIR
from src.prepare import wczytaj
from src.split import wczytaj_split


@torch.no_grad()
def waliduj(model, structs, seqs, device, bs=64, tryb="odzysk", n_prob=150):
    """Zwraca (odzysk, CE, kryterium_wyboru)."""
    model.eval()
    traf = tot = 0
    ce_sum = ce_n = 0.0
    for s in range(0, len(structs), bs):
        cs, cq = structs[s : s + bs], seqs[s : s + bs]
        sid, pad, par, cp, cz, _ = koduj(cs, cq, device)
        lp, lz, _ = model(sid, pad, par)
        for logity, cel in ((lp, cp), (lz, cz)):
            m = cel != -100
            if m.any():
                ce_sum += float(nn.functional.cross_entropy(logity[m], cel[m], reduction="sum"))
                ce_n += int(m.sum())
        gen = model.generate(sid, pad, par, [len(x) for x in cs])
        for g, q in zip(gen, cq):
            traf += sum(a == b for a, b in zip(g, q))
            tot += len(q)
    odz = traf / max(tot, 1)

    kryt = odz
    if tryb in ("solved", "energia"):
        import RNA
        sub_s, sub_q = structs[:n_prob], seqs[:n_prob]
        g = []
        for s in range(0, len(sub_s), bs):
            cs = sub_s[s : s + bs]
            sid, pad, par, _, _, _ = koduj(cs, None, device)
            g += model.generate(sid, pad, par, [len(x) for x in cs])
        if tryb == "solved":
            kryt = float(np.mean([RNA.fold(q)[0] == t for t, q in zip(sub_s, g)]))
        else:   # energia: im NIZSZA delta, tym lepiej -> zmieniamy znak, by "wiecej = lepiej"
            kryt = -float(np.mean([(RNA.energy_of_struct(q, t) - RNA.energy_of_struct(r, t)) / len(t)
                                   for t, r, q in zip(sub_s, sub_q, g)]))
    model.train()
    return odz, ce_sum / max(ce_n, 1), kryt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tryb-podzialu", choices=["rodzinowy", "losowy"], default="rodzinowy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoki", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warstwy", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--w-energia", type=float, default=0.0)
    ap.add_argument("--w-parowania", type=float, default=0.0)
    ap.add_argument("--w-sklad", type=float, default=0.0)
    ap.add_argument("--wybor", choices=["odzysk", "solved", "energia"], default="odzysk")
    ap.add_argument("--cierpliwosc", type=int, default=10)
    ap.add_argument("--out", default="checkpoints/model.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = wczytaj()
    spl = wczytaj_split(args.tryb_podzialu, args.seed)
    S = {k: df.secondary_structure.iloc[v].tolist() for k, v in spl.items()}
    Q = {k: df.sequence.iloc[v].tolist() for k, v in spl.items()}
    max_len = int(df.secondary_structure.str.len().max())

    print(f"urzadzenie {dev} | struktur {len(df)} | podzial {args.tryb_podzialu}")
    print(f"train {len(S['train'])}  val {len(S['val'])}  test {len(S['test'])}")
    print(f"cel skladu: petle {NATURAL_LOOP}  pary {NATURAL_PAIR}")
    print(f"wagi kar: energia {args.w_energia}  parowania {args.w_parowania}  sklad {args.w_sklad}"
          f" | wybor epoki: {args.wybor}")

    model = NARDesigner(d_model=args.d_model, num_layers=args.warstwy,
                        max_len=max_len + 1, dropout=args.dropout).to(dev)
    print(f"parametry: {sum(p.numel() for p in model.parameters())/1e6:.2f} mln")
    komp = KomponentyNAR(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoki)

    order = np.arange(len(S["train"]))
    best, bad = -1e9, 0
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    for ep in range(1, args.epoki + 1):
        np.random.shuffle(order)
        t0 = time.time()
        agg = {"ce": 0.0, "e": 0.0, "a": 0.0, "c": 0.0, "n": 0}
        for s in range(0, len(order), args.batch):
            b = order[s : s + args.batch]
            cs = [S["train"][i] for i in b]
            cq = [Q["train"][i] for i in b]
            sid, pad, par, cp, cz, realne = koduj(cs, cq, dev)
            lp, lz, otw = model(sid, pad, par)

            ce = lp.new_zeros(())
            for logity, cel in ((lp, cp), (lz, cz)):
                m = cel != -100
                if m.any():
                    ce = ce + nn.functional.cross_entropy(logity[m], cel[m])

            p_par, p_zas = lp.softmax(-1), lz.softmax(-1)
            e = komp.energia(p_par, p_zas, par, otw, realne) if args.w_energia else lp.new_zeros(())
            a = komp.parowania(p_par, p_zas, par, otw, realne) if args.w_parowania else lp.new_zeros(())
            c = komp.sklad(p_par, p_zas, par, otw, realne) if args.w_sklad else lp.new_zeros(())

            loss = ce + args.w_energia * e + args.w_parowania * a + args.w_sklad * c
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in zip(("ce", "e", "a", "c"), (ce, e, a, c)):
                agg[k] += float(v)
            agg["n"] += 1
        sched.step()

        odz, ce_val, kryt = waliduj(model, S["val"], Q["val"], dev, tryb=args.wybor)
        n = max(agg["n"], 1)
        msg = (f"[{ep}/{args.epoki}] CE {agg['ce']/n:.4f} | energia {agg['e']/n:+.4f} "
               f"parowania {agg['a']/n:.4f} sklad {agg['c']/n:.4f} | "
               f"val odzysk {odz:.4f} CE {ce_val:.4f}")
        if args.wybor != "odzysk":
            msg += f" {args.wybor} {kryt:+.4f}"
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
