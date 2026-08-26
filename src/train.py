"""Trening nieautoregresyjnego transformera.

FUNKCJA STRATY
    CE                    uczy sie z prawdziwych sekwencji: co w jakim motywie faktycznie wystepuje
  + w_e  * energia        czlony sekwencyjne z tablic Turnera
  + w_a  * parowania      liczba MOZLIWYCH parowan G*C + A*U + G*U, na nt^2
  + w_c  * sklad          KARA ZA SKLAD, wariant E1: odleglosc TV od celu, DWUSTRONNA
  + w_sz * sklad_zasad    KARA ZA SKLAD, wariant E2: progi dolne A/C/G/U, JEDNOSTRONNA
  + w_sp * sklad_par      KARA ZA SKLAD, wariant E2: progi dolne typow par, JEDNOSTRONNA

Obie kary za sklad licza sie PER SEKWENCJA, potem srednia po partii. Sa alternatywne — wlacza sie
jedna albo druga, nie obie naraz. Wagi domyslnie zerowe: czysta CE jest punktem odniesienia.

WYBOR EPOKI. Wszystkie kryteria licza sie na WALIDACJI i zadne nie przewiduje struktury. W logu
pojawiaja sie WSZYSTKIE w kazdej epoce; flaga `--wybor` decyduje tylko, ktore zapisuje checkpoint.

  zlozony          DOMYSLNE. Porzadek leksykograficzny: identycznosc w pelnych procentach jako klucz
                   glowny, dE/nt jako rozstrzygacz remisow. Patrz `zlozony_score`
  identycznosc_nt  ulamek pozycji trafionych wzgledem prawdziwej sekwencji
  ce               srednia cross-entropia; liczona na ROZKLADACH, wiec widzi tez pewnosc modelu
  energia          o ile stabilizujemy cel lepiej niz prawdziwa sekwencja
  loss             PELNA strata tego modelu, z jego wlasnymi karami i wagami

Kryteria ciagna w rozne strony: identycznosc i ce premiuja podobienstwo do biologii, energia mocne
i jednoznaczne helisy. Wybor musi byc TAKI SAM w E1 i E2, inaczej porownanie kar traci sens —
dlatego `loss`, ktory w E1 i E2 obejmuje inna kare, nie nadaje sie do tego porownania.

Uzycie:
    python -m src.train --epoki 60 --out checkpoints/e1.pt --w-energia 1.0 --w-parowania 6.0 --w-sklad 1.0
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
from src.loss import KomponentyNAR, NATURAL_LOOP, NATURAL_PAIR, PROG_ZASADY, PROG_PARY
from src.prepare import wczytaj
from src.split import wczytaj_split


def energie_referencyjne(structs, seqs):
    """E(struktura | sekwencja wzorcowa) dla calego zbioru, liczone RAZ.

    Sekwencje wzorcowe sie nie zmieniaja, wiec liczenie tego w kazdej epoce byloby czysta strata.
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


KRYTERIA = ("identycznosc_nt", "ce", "loss", "energia", "zlozony")


def kryterium(w: dict, tryb: str) -> float:
    """Wybrane kryterium w konwencji "WIECEJ = LEPIEJ" (CE, loss i dE wchodza ze zmienionym znakiem)."""
    return {"identycznosc_nt": w["identycznosc_nt"], "ce": -w["ce"], "loss": -w["loss"],
            "energia": -w["dE_nt"], "zlozony": w["zlozony"]}[tryb]


@torch.no_grad()
def waliduj(model, structs, seqs, device, bs=64, komp=None, args=None, e_ref=None):
    """Liczy WSZYSTKIE kryteria naraz i zwraca je slownikiem.

    Sekwencje generujemy RAZ i z tych samych sekwencji liczymy identycznosc oraz delte energii,
    na CALYM zbiorze walidacyjnym. Energie referencyjne przychodza gotowe z `energie_referencyjne`.
    """
    import RNA
    model.eval()
    traf = tot = 0
    ce_sum = ce_n = 0.0
    loss_sum = loss_n = 0.0
    dE = []
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

        gen = model.generate(sid, pad, par, [len(x) for x in cs])
        for k, (g, q, t) in enumerate(zip(gen, cq, cs)):
            traf += sum(a == b for a, b in zip(g, q))
            tot += len(q)
            if e_ref is not None:
                dE.append((RNA.energy_of_struct(g, t) - e_ref[s + k]) / len(t))

    w = {"identycznosc_nt": traf / max(tot, 1),
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
    c = komp.sklad(p_par, p_zas, par, otw, realne) if args.w_sklad else z
    sz = komp.sklad_zasad(p_par, p_zas, par, otw, realne) if args.w_sklad_zasad else z
    sp = komp.sklad_par(p_par, otw) if args.w_sklad_par else z
    loss = (ce + args.w_energia * e + args.w_parowania * a + args.w_sklad * c
            + args.w_sklad_zasad * sz + args.w_sklad_par * sp)
    return loss, (e, a, c, sz, sp)


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
    ap.add_argument("--w-sklad", type=float, default=0.0,
                    help="NASZA kara za sklad: odleglosc TV od celu, dwustronna, per sekwencja")
    ap.add_argument("--w-sklad-zasad", type=float, default=0.0,
                    help="kara promotora, DistribLoss: progi dolne udzialow A/C/G/U, per sekwencja")
    ap.add_argument("--w-sklad-par", type=float, default=0.0,
                    help="kara promotora, DistribLoss3+4: progi dolne udzialow typow par G:C/A:U/G:U")
    ap.add_argument("--wybor", choices=list(KRYTERIA), default="zlozony",
                    help="czym wybierac najlepsza epoke na walidacji; zadna opcja nie przewiduje "
                         "struktury. Domyslne 'zlozony' to identycznosc jako klucz glowny "
                         "i dE/nt jako rozstrzygacz remisow. UWAGA: 'loss' to wlasna strata modelu, "
                         "wiec E1 i E2 wybieraja wtedy epoke roznymi miarami")
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
    print(f"wagi kar: energia {args.w_energia}  parowania {args.w_parowania}")
    if args.w_sklad:
        print(f"  sklad TV (nasz):        waga {args.w_sklad}  "
              f"cel petle {NATURAL_LOOP}  cel pary {NATURAL_PAIR}")
    if args.w_sklad_zasad:
        print(f"  sklad zasad (promotor): waga {args.w_sklad_zasad}  progi {PROG_ZASADY}")
    if args.w_sklad_par:
        print(f"  sklad par (promotor):   waga {args.w_sklad_par}  progi {PROG_PARY}")
    print(f"wybor epoki: {args.wybor}")

    model = NARDesigner(d_model=args.d_model, num_layers=args.warstwy,
                        max_len=max_len + 1, dropout=args.dropout).to(dev)
    print(f"parametry: {sum(p.numel() for p in model.parameters())/1e6:.2f} mln")
    komp = KomponentyNAR(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoki)

    # Sekwencje wzorcowe sie nie zmieniaja, wiec ich energie liczymy raz na caly trening.
    t_ref = time.time()
    e_ref = energie_referencyjne(S["val"], Q["val"])
    print(f"energie referencyjne walidacji: {len(e_ref)} struktur, {time.time()-t_ref:.1f}s")

    order = np.arange(len(S["train"]))
    best, bad = -1e9, 0
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    for ep in range(1, args.epoki + 1):
        np.random.shuffle(order)
        t0 = time.time()
        agg = {"ce": 0.0, "e": 0.0, "a": 0.0, "c": 0.0, "sz": 0.0, "sp": 0.0, "n": 0}
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

            loss, (e, a, c, sz, sp) = skladaj_loss(komp, args, lp, lz, par, otw, realne, ce)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in zip(("ce", "e", "a", "c", "sz", "sp"), (ce, e, a, c, sz, sp)):
                agg[k] += float(v)
            agg["n"] += 1
        sched.step()

        w = waliduj(model, S["val"], Q["val"], dev, komp=komp, args=args, e_ref=e_ref)
        kryt = kryterium(w, args.wybor)
        n = max(agg["n"], 1)
        msg = (f"[{ep}/{args.epoki}] CE {agg['ce']/n:.4f} | energia {agg['e']/n:+.4f} "
               f"parowania {agg['a']/n:.4f}")
        for klucz, etyk, waga in (("c", "sklad", args.w_sklad),
                                  ("sz", "zasady", args.w_sklad_zasad),
                                  ("sp", "pary", args.w_sklad_par)):
            if waga:
                msg += f" {etyk} {agg[klucz]/n:.4f}"
        # WSZYSTKIE kryteria w kazdej epoce, niezaleznie od tego, ktore wybieramy. Kosztuje to zero,
        # a pozwala pozniej powiedziec, ktora epoke wskazalby kazdy z nich. To sa liczby z WALIDACJI,
        # wiec patrzenie na nie nie jest przeciekiem; przeciekiem byloby ocenianie ich na tescie.
        msg += (f" | val ident {w['identycznosc_nt']:.4f} CE {w['ce']:.4f} "
                f"loss {w['loss']:.4f} dE/nt {w['dE_nt']:+.4f} zlozony {w['zlozony']}")
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
