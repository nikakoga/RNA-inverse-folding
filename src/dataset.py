"""Wspólne operacje na strukturach i sekwencjach RNA. Bez zależności od ViennaRNA."""

from __future__ import annotations

import numpy as np
import torch

# Notacja kropkowo-nawiasowa. 0 = padding.
STRUCT_TO_IDX = {".": 1, "(": 2, ")": 3}
STRUCT_VOCAB_SIZE = 4

BASES = ["A", "C", "G", "U"]
BASE_TO_IDX = {b: i + 1 for i, b in enumerate(BASES)}          # 1..4, 0 = padding
IDX_TO_BASE = {v: k for k, v in BASE_TO_IDX.items()}

# Sześć UPORZĄDKOWANYCH par kanonicznych — kolejność klas głowicy par.
PAIRS = [("G", "C"), ("C", "G"), ("A", "U"), ("U", "A"), ("G", "U"), ("U", "G")]
PAIR_TO_CLASS = {p: i for i, p in enumerate(PAIRS)}
N_PAIR_CLASSES = len(PAIRS)
PAIR_TO_BASE_IDX = torch.tensor([[BASE_TO_IDX[a], BASE_TO_IDX[b]] for a, b in PAIRS], dtype=torch.long)


def parse_pairs(struct: str) -> list[tuple[int, int]]:
    """Lista par (i, j), i < j, z notacji kropkowo-nawiasowej."""
    stos, out = [], []
    for i, c in enumerate(struct):
        if c == "(":
            stos.append(i)
        elif c == ")" and stos:
            out.append((stos.pop(), i))
    return out


def partner_array(struct: str) -> np.ndarray:
    """Dla każdej pozycji indeks partnera, albo -1 gdy niesparowana."""
    p = np.full(len(struct), -1, dtype=np.int64)
    for i, j in parse_pairs(struct):
        p[i], p[j] = j, i
    return p


def paired_fraction(struct: str) -> float:
    """Ułamek pozycji SPAROWANYCH. Używane w filtrze „przewaga sparowanych"."""
    n = len(struct)
    return 0.0 if n == 0 else 2 * len(parse_pairs(struct)) / n


def motyw_pozycji(struct: str) -> list[str]:
    """Etykieta motywu dla każdej pozycji.

    helisa-wnetrze  para z sąsiadami po obu stronach (ciągła helisa)
    helisa-koniec   para na końcu helisy
    spinka          niesparowana, w pętli domykanej JEDNĄ helisą
    wybrzuszenie    niesparowana, w pętli domykanej dwiema helisami
    multipetla      niesparowana, w pętli domykanej trzema lub więcej
    zewnetrzna      niesparowana, bez pary domykającej
    """
    n = len(struct)
    par = partner_array(struct)
    lab = [""] * n

    for i in range(n):
        if par[i] < 0:
            continue
        j = int(par[i])
        a, b = (i, j) if i < j else (j, i)
        zew = a > 0 and par[a - 1] == b + 1
        wew = a < n - 1 and par[a + 1] == b - 1
        lab[i] = "helisa-wnetrze" if (zew and wew) else "helisa-koniec"

    for i in range(n):
        if par[i] >= 0:
            continue
        zamk = -1
        for k in range(i - 1, -1, -1):
            if par[k] > i:
                zamk = k
                break
        if zamk < 0:
            lab[i] = "zewnetrzna"
            continue
        j = int(par[zamk])
        helisy, k = 1, zamk + 1
        while k < j:
            if par[k] > k:
                helisy += 1
                k = int(par[k]) + 1
            else:
                k += 1
        lab[i] = "spinka" if helisy == 1 else ("wybrzuszenie" if helisy == 2 else "multipetla")
    return lab


MOTYWY = ["helisa-wnetrze", "helisa-koniec", "spinka", "wybrzuszenie", "multipetla", "zewnetrzna"]


def koduj(structs: list[str], seqs: list[str] | None, device):
    """Wejście modelu + cele uczenia.

    Zwraca (struct_ids, pad_mask, partner, cel_par, cel_zasad, realne). Cele mają -100 tam,
    gdzie nie ma czego uczyć — cross-entropia to ignoruje.
    """
    B = len(structs)
    L = max(len(s) for s in structs)
    sid = torch.zeros(B, L, dtype=torch.long)
    par = torch.full((B, L), -1, dtype=torch.long)
    pad = torch.ones(B, L, dtype=torch.bool)
    cel_par = torch.full((B, L), -100, dtype=torch.long)
    cel_zas = torch.full((B, L), -100, dtype=torch.long)
    realne = torch.zeros(B, L)

    for b, st in enumerate(structs):
        n = len(st)
        sid[b, :n] = torch.tensor([STRUCT_TO_IDX[c] for c in st])
        par[b, :n] = torch.tensor(partner_array(st))
        pad[b, :n] = False
        realne[b, :n] = 1.0
        if seqs is None:
            continue
        q = seqs[b]
        for i, j in parse_pairs(st):
            kl = PAIR_TO_CLASS.get((q[i], q[j]))
            if kl is not None:
                cel_par[b, i] = kl
        for i, c in enumerate(st):
            if c == "." and q[i] in BASES:
                cel_zas[b, i] = BASES.index(q[i])

    return (sid.to(device), pad.to(device), par.to(device),
            cel_par.to(device), cel_zas.to(device), realne.to(device))


def losowa_kanoniczna(struct: str, czestosci: dict | None = None, rng=None) -> str:
    """BASELINE: losowa sekwencja z zachowaniem kanoniczności par.

    Mierzy, ile da się ugrać SAMĄ komplementarnością zasad, bez żadnego uczenia.
    """
    import random
    rng = rng or random
    cz = czestosci or {"loop": [0.330, 0.198, 0.213, 0.259], "pair": [0.555, 0.321, 0.125]}
    s = [""] * len(struct)
    typy = [("G", "C"), ("A", "U"), ("G", "U")]
    for i, j in parse_pairs(struct):
        t = rng.choices(typy, weights=cz["pair"])[0]
        if rng.random() < 0.5:
            t = (t[1], t[0])
        s[i], s[j] = t
    for i, c in enumerate(struct):
        if c == ".":
            s[i] = rng.choices(BASES, weights=cz["loop"])[0]
    return "".join(s)
