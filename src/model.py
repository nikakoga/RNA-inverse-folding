"""Enkoder-only, NIEAUTOREGRESYJNY projektant RNA.

PO CO TA ARCHITEKTURA — dwa problemy starego modelu, które usuwa u źródła:

1. STARY MODEL PISAŁ LEWO-DO-PRAWA. poprzedni model autoregresyjny to enkoder-dekoder z maską przyczynową
   i pętlą token po tokenie. Pozycja 30 nie wiedziała, co stanie na pozycji 3, dopóki tam nie doszła,
   choć obie tworzą PARĘ i ich wybór jest z definicji wspólny. Tutaj nie ma dekodera: enkoder patrzy
   na całą strukturę naraz, a wszystkie pozycje przewidujemy w JEDNYM przebiegu.

2. ODCHYLENIE EKSPOZYCJI ZNIKA. Skoro nie ma autoregresji, nie ma teacher forcingu — model podczas
   uczenia i podczas generowania robi DOKŁADNIE to samo. To dlatego trzy komponenty ze specyfikacji
   promotora (energia, alternatywne parowania, skład) można tu wstawić wprost do funkcji straty
   i nie potrzeba RL, które w E24/E25 służyło wyłącznie obejściu tego problemu.

DWIE GŁOWICE:

    pozycje SPAROWANE    -> jedna z 6 klas par kanonicznych (G-C, C-G, A-U, U-A, G-U, U-G)
                            para (i,j) rozstrzygana WSPÓLNIE z połączonych reprezentacji h_i i h_j
    pozycje NIESPAROWANE -> jedna z 4 zasad

Kanoniczność jest tu własnością WYJŚCIA, nie maski nakładanej po fakcie: klasy "A-C" po prostu nie ma.
Stary model wybierał literę osobno na i oraz na j i dopiero maska pilnowała zgodności — co w E23
doprowadziło do błędu, gdzie strata liczyła się na rozkładzie surowym, a generowanie na maskowanym.

Konwencja kierunku: głowica par działa na pozycji OTWIERAJĄCEJ (i < j) i zwraca rozkład dla
uporządkowanej pary (zasada na i, zasada na j). Pozycja zamykająca bierze literę stąd, nie z osobnej
predykcji.

Literatura: RNAinformer (bioRxiv 2024.03.09.584209) — transformer z axial attention na macierzy
sąsiedztwa, generowanie nieautoregresyjne; oraz arXiv 2312.02447 — ta sama diagnoza autoregresji jako
wąskiego gardła w projektowaniu sekwencji.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.dataset import (BASE_TO_IDX as NUC_TO_IDX, STRUCT_VOCAB_SIZE,
                         BASES, PAIRS, PAIR_TO_BASE_IDX as PAIR_TO_IDX, N_PAIR_CLASSES)

# Definicje BASES / PAIRS / PAIR_TO_IDX zyja w src/dataset.py — jedno zrodlo prawdy.
BASE_TO_COL = {b: i for i, b in enumerate(BASES)}
PAIR_TO_COL = __import__('torch').tensor(
    [[BASE_TO_COL[a], BASE_TO_COL[b]] for a, b in PAIRS], dtype=__import__('torch').long)


class PositionalEncoding(nn.Module):
    """Klasyczne kodowanie sinusoidalne (Vaswani i in. 2017)."""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class NARDesigner(nn.Module):
    """Struktura 2D -> sekwencja, jednym przebiegiem, bez dekodera."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model, self.max_len = d_model, max_len

        self.struct_embedding = nn.Embedding(STRUCT_VOCAB_SIZE, d_model, padding_idx=0)
        # Embedding ZNAKOWANEJ odległości do partnera: mówi wprost "z kim się parujesz i jak daleko",
        # zamiast zmuszać sieć do dopasowywania nawiasów na dystans.
        # Indeksy: 0..2*max_len-1 = pary (offset + max_len); 2*max_len = pozycja niesparowana.
        self.partner_embedding = nn.Embedding(2 * max_len + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len + 1, dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(d_model))

        # Głowica PAR: wejściem jest złączenie reprezentacji obu końców pary -> jedna z 6 klas.
        self.pair_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, N_PAIR_CLASSES)
        )
        # Głowica ZASAD dla pozycji niesparowanych -> jedna z 4.
        self.base_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, len(BASES))
        )

    def encode(self, struct_ids, struct_pad_mask, partner):
        """Reprezentacje wszystkich pozycji naraz. (B,L) -> (B,L,d)."""
        B, L = struct_ids.shape
        pos = torch.arange(L, device=struct_ids.device).unsqueeze(0).expand(B, L)
        off = torch.where(partner >= 0, partner - pos + self.max_len,
                          torch.full_like(partner, 2 * self.max_len))
        off = off.clamp(0, 2 * self.max_len)
        x = self.struct_embedding(struct_ids) + self.partner_embedding(off)
        return self.encoder(self.pos_encoder(x), src_key_padding_mask=struct_pad_mask)

    def forward(self, struct_ids, struct_pad_mask, partner):
        """Zwraca (logits_par (B,L,6), logits_zasad (B,L,4), otwierajace (B,L) bool).

        `logits_par[b,i]` ma sens TYLKO dla pozycji otwierających (i < partner[i]); pozostałe
        ignorujemy przy stracie. `logits_zasad[b,i]` ma sens dla pozycji niesparowanych.
        """
        h = self.encode(struct_ids, struct_pad_mask, partner)
        L = h.size(1)
        pos = torch.arange(L, device=h.device).unsqueeze(0)
        otw = (partner > pos) & (partner >= 0)      # każda para liczona RAZ, od strony otwierającej

        h_j = torch.gather(h, 1, partner.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.d_model))
        logits_par = self.pair_head(torch.cat([h, h_j], dim=-1))
        logits_zasad = self.base_head(h)
        return logits_par, logits_zasad, otw

    @torch.no_grad()
    def generate(self, struct_ids, struct_pad_mask, partner, lengths, sample: bool = False,
                 temperature: float = 1.0) -> list[str]:
        """Jeden przebieg, wszystkie pozycje naraz. Pary kanoniczne Z KONSTRUKCJI."""
        logits_par, logits_zasad, otw = self(struct_ids, struct_pad_mask, partner)
        if sample:
            kl = torch.distributions.Categorical(logits=logits_par / temperature).sample()
            zb = torch.distributions.Categorical(logits=logits_zasad / temperature).sample()
        else:
            kl, zb = logits_par.argmax(-1), logits_zasad.argmax(-1)

        p2i = PAIR_TO_IDX.to(kl.device)
        idx = torch.zeros_like(kl)                          # 0 = brak / padding
        idx = torch.where(partner < 0, zb + 1, idx)         # +1, bo NUC_TO_IDX zaczyna od 1
        idx = torch.where(otw, p2i[kl][..., 0], idx)        # zasada na pozycji otwierającej
        # pozycja zamykająca bierze DRUGĄ literę z klasy wybranej przez swojego partnera
        zamk = (partner >= 0) & ~otw
        kl_part = torch.gather(kl, 1, partner.clamp(min=0))
        idx = torch.where(zamk, p2i[kl_part][..., 1], idx)

        inv = {v: k for k, v in NUC_TO_IDX.items()}
        out = []
        for b in range(idx.size(0)):
            n = int(lengths[b])
            out.append("".join(inv.get(int(v), "A") for v in idx[b, :n]))
        return out
