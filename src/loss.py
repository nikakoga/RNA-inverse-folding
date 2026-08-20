"""Trzy komponenty ze specyfikacji promotora, liczone WPROST na rozkładach modelu NAR.

DLACZEGO TU JEST PROŚCIEJ NIŻ W E23. Model nieautoregresyjny przewiduje parę (i,j) jako jedną z SZEŚCIU
klas kanonicznych, więc rozkład łączny na parze mamy bezpośrednio z softmaxu — nie trzeba go sklejać
z dwóch niezależnych rozkładów ani maskować po fakcie. 
Nie ma też teacher forcingu, więc rozkłady, na których liczymy kary, są DOKŁADNIE tymi, z których
model próbkuje przy generowaniu. To dlatego nie potrzeba RL.

VIENNARNA UŻYWAMY WYŁĄCZNIE DO ODCZYTU TABLIC TURNERA (`RNA.param`), raz przy starcie. Żadnego
`RNA.fold` ani `RNA.pf`.

CO LICZYMY W ENERGII (człony zależne od SEKWENCJI; człony zależne wyłącznie od kształtu struktury
skracają się w różnicy wobec referencji, więc ich nie potrzebujemy):
  * stosy par            — tablica `stack`, 6x6 kombinacji klas
  * kary terminalne AU/GU na końcach helis — `TerminalAU`
  * niedopasowanie pętli spinki — `mismatchH`; TEGO CZŁONU BRAKOWAŁO W E23, a odpowiada za medianę
    26% pełnej delty energii (pomiar na Eternie /61)
"""

from __future__ import annotations

import torch
import RNA

from src.dataset import PAIRS, BASES

# Kodowanie typów par w tablicach ViennaRNA. 0 = para niedozwolona.
VRNA_PT = {("C", "G"): 1, ("G", "C"): 2, ("G", "U"): 3, ("U", "G"): 4, ("A", "U"): 5, ("U", "A"): 6}
# Kodowanie zasad w ViennaRNA: A=1, C=2, G=3, U=4 — zgodne z naszym NUC_TO_IDX.
VRNA_BASE = {"A": 1, "C": 2, "G": 3, "U": 4}
_P = RNA.param(RNA.md())
_EPS = 1e-9

# Sklad NATURALNY — cel komponentu skladu. Zmierzony na tej samej puli po natywnym cd-hit,
# ktora jest wejsciem do src/prepare.py. To samo zrodlo, z ktorego losuje baseline.
NATURAL_LOOP = {"A": 0.330, "C": 0.198, "G": 0.213, "U": 0.259}   # zasady na pozycjach NIESPAROWANYCH
NATURAL_PAIR = {"GC": 0.555, "AU": 0.321, "GU": 0.125}            # udzialy TYPOW par


def tabela_stosow(device) -> torch.Tensor:
    """(6,6): energia stosu klasy zewnętrznej c1 leżącej na klasie wewnętrznej c2, kcal/mol.

    Konwencja ViennaRNA: energia = stack[type(i,j)][type(j-1,i+1)] — para wewnętrzna wchodzi ODWRÓCONA.
    Tablice trzymają dekakalorie, stąd /100.
    """
    t = torch.zeros(6, 6)
    for a, (x1, y1) in enumerate(PAIRS):
        for b, (x2, y2) in enumerate(PAIRS):
            t[a, b] = _P.stack[VRNA_PT[(x1, y1)]][VRNA_PT[(y2, x2)]] / 100.0
    return t.to(device)


def tabela_terminalna(device) -> torch.Tensor:
    """(6,): kara za zakończenie helisy daną klasą pary. G:C = 0, reszta = TerminalAU."""
    kara = _P.TerminalAU / 100.0
    return torch.tensor([0.0 if {x, y} == {"G", "C"} else kara for x, y in PAIRS]).to(device)


def tabela_spinki(device) -> torch.Tensor:
    """(6,4,4): niedopasowanie petli spinki — klasa pary zamykajacej x pierwsza x ostatnia zasada petli.

    Uzywamy OFICJALNEGO API `RNA.E_Hairpin(size, type, si1, sj1, string, P)`. Samej tablicy
    `mismatchH` nie da sie indeksowac z Pythona (jest SwigPyObject), ale E_Hairpin jest udokumentowana
    funkcja biblioteki i czyta te sama tablice — nie reimplementujemy wiec niczego wlasnego.

    Petla ma 6 niesparowanych, a nie 4, CELOWO: dla rozmiarow 3, 4 i 6 model Turnera ma osobne tablice
    dla "unusually stable" tri-, tetra- i heksapetli (patrz docstring E_Hairpin). Rozmiar 6 z wnetrzem
    AAAA nie trafia w zadna z nich, wiec odczyt dotyczy samego niedopasowania.

    Wartosci zawieraja stala kare za ROZMIAR petli, identyczna dla wszystkich 96 kombinacji; znika ona
    przy odejmowaniu od referencji.

    PRZYBLIZENIE, ktore zostaje: dla petli o rozmiarze 4 pomijamy premie za tetrapetle (GAAA daje ok.
    1,4 kcal/mol wzgledem AAAA). Model uczy sie tego motywu z cross-entropii na prawdziwych sekwencjach.
    """
    t = torch.zeros(6, 4, 4)
    P = RNA.param(RNA.md())
    for c, (x, y) in enumerate(PAIRS):
        for i, bi in enumerate(BASES):
            for j, bj in enumerate(BASES):
                t[c, i, j] = RNA.E_Hairpin(6, VRNA_PT[(x, y)], VRNA_BASE[bi], VRNA_BASE[bj],
                                           bi + "AAAA" + bj, P) / 100.0
    return t.to(device)


def tabela_par_na_kolumny(device) -> torch.Tensor:
    """(6,4,2): rozkład klasy pary na liczebności zasad — [klasa, zasada, koniec(0=i,1=j)]."""
    t = torch.zeros(6, 4, 2)
    for c, (x, y) in enumerate(PAIRS):
        t[c, BASES.index(x), 0] = 1.0
        t[c, BASES.index(y), 1] = 1.0
    return t.to(device)


def _sasiedztwo(partner):
    """Maski: czy para (i,j) ma sąsiada od zewnątrz (i-1,j+1) i od wewnątrz (i+1,j-1)."""
    poprz = torch.roll(partner, 1, 1).clone(); poprz[:, 0] = -1
    nast = torch.roll(partner, -1, 1).clone(); nast[:, -1] = -1
    return (poprz == partner + 1), (nast == partner - 1)


class KomponentyNAR:
    """Trzy człony kary. Tablice budowane raz, przy tworzeniu obiektu."""

    def __init__(self, device, cel_pary: torch.Tensor | None = None,
                 cel_petle: torch.Tensor | None = None):
        self.stos = tabela_stosow(device)
        self.term = tabela_terminalna(device)
        self.spinka = tabela_spinki(device)
        self.par2kol = tabela_par_na_kolumny(device)
        # Cele ze stalych "naturalnych" — tego samego zrodla, ktorego uzywa baseline NEMO.
        dom = torch.tensor([NATURAL_PAIR[k] for k in ("GC", "AU", "GU")])
        dop = torch.tensor([NATURAL_LOOP[b] for b in BASES])
        self.cel_pary = (cel_pary if cel_pary is not None else dom).to(device)   # (3,) G:C/A:U/G:U
        self.cel_petle = (cel_petle if cel_petle is not None else dop).to(device)  # (4,) A,C,G,U

    # ---------------------------------------------------------------- energia
    def energia(self, p_par, p_zasad, partner, otw, realne) -> torch.Tensor:
        """Oczekiwana energia członów SEKWENCYJNYCH, kcal/mol na nukleotyd. Niższa = lepsza."""
        if otw.sum() == 0:
            return p_par.new_zeros(())
        ma_zewn, ma_wewn = _sasiedztwo(partner)

        # stosy: para (i,j) leżąca na parze (i+1,j-1)
        p_wewn = torch.roll(p_par, -1, 1).clone()
        p_wewn[:, -1] = 0.0
        e_stos = torch.einsum("bic,bid,cd->bi", p_par, p_wewn, self.stos)
        suma = (e_stos * (otw & ma_wewn).float()).sum(1)

        # kara terminalna TYLKO na ZEWNETRZNYM koncu helisy. Koniec wewnetrzny, gdy helisa domyka
        # spinke, jest juz wliczony w tablice `spinka` (eval_hp_loop zawiera kare terminalna) —
        # naliczanie go tu drugi raz dawalo blad rowny dokladnie TerminalAU (0,50 kcal/mol).
        koniec = otw & ~ma_zewn
        suma = suma + ((p_par @ self.term) * koniec.float()).sum(1)

        # spinka: para zamykająca (i,j) bez sąsiada wewnętrznego.
        # TRIPETLE (3 niesparowane) model Turnera traktuje ODDZIELNIE — nie dostaja czlonu
        # niedopasowania, tylko kare terminalna. Bez tego rozroznienia blad na tripetlach wynosil
        # 0,28 kcal/mol (pomiar); z nim energia zgadza sie z ViennaRNA co do zera.
        L_pos = torch.arange(partner.size(1), device=partner.device).unsqueeze(0)
        rozmiar = partner - L_pos - 1                      # liczba niesparowanych w spince
        spinka = otw & ~ma_wewn & (rozmiar > 3)
        tri = otw & ~ma_wewn & (rozmiar == 3)
        suma = suma + ((p_par @ self.term) * tri.float()).sum(1)
        p_i1 = torch.roll(p_zasad, -1, 1).clone(); p_i1[:, -1] = 0.0     # zasada na i+1
        j_idx = (partner - 1).clamp(min=0)
        p_j1 = torch.gather(p_zasad, 1, j_idx.unsqueeze(-1).expand(-1, -1, 4))
        e_hp = torch.einsum("bic,bia,bid,cad->bi", p_par, p_i1, p_j1, self.spinka)
        suma = suma + (e_hp * spinka.float()).sum(1)

        return (suma / realne.sum(1).clamp(min=1)).mean()

    # ------------------------------------------------------ alternatywne pary
    def parowania(self, p_par, p_zasad, partner, otw, realne) -> torch.Tensor:
        """Oczekiwana liczba MOŻLIWYCH parowań (G*C + A*U + G*U) na nukleotyd^2. Niższa = lepsza.

        Liczebności zasad składamy z obu głowic: pary wnoszą DWIE zasady (oba końce), pozycje
        niesparowane po jednej. Dzielimy przez kwadrat długości, bo iloczyn liczebności rośnie
        kwadratowo z długością.
        """
        n_par = torch.einsum("bic,cke->bk", p_par * otw.unsqueeze(-1).float(),
                             self.par2kol.sum(-1).unsqueeze(-1)).squeeze(-1) \
            if False else torch.einsum("bic,ck->bk", p_par * otw.unsqueeze(-1).float(),
                                       self.par2kol.sum(-1))
        niespar = (partner < 0) & realne.bool()
        n_nsp = (p_zasad * niespar.unsqueeze(-1).float()).sum(1)
        n = n_par + n_nsp                                     # (B,4) w kolejności A,C,G,U
        A, C, G, U = 0, 1, 2, 3
        komb = n[:, G] * n[:, C] + n[:, A] * n[:, U] + n[:, G] * n[:, U]
        return (komb / realne.sum(1).clamp(min=1) ** 2).mean()

    # ------------------------------------------------------------------ skład
    def sklad(self, p_par, p_zasad, partner, otw, realne) -> torch.Tensor:
        """Odleglosc skladu od NATURALNEGO, osobno dla par i osobno dla petli.

        PARY mierzymy jako TYPY (G:C / A:U / G:U), nie jako liczebnosci pojedynczych zasad.
        To jest ta sama jednostka, ktorej uzywaly wczesniejsze eksperymenty (`NATURAL_PAIR`
        w `energy_metrics.py`) — dzieki temu liczby sa porownywalne miedzy faza stara i nowa,
        a kara trafia wprost w to, co chcemy ograniczyc: nadmiar par G:C.

        Cel bierzemy ze STALYCH `NATURAL_LOOP` / `NATURAL_PAIR`, czyli z tego samego zrodla co
        baseline NEMO, a nie przeliczamy go z wlasnego zbioru treningowego. Inaczej cel zmienialby
        sie z podzialem danych i wyniki przestalyby byc porownywalne miedzy eksperymentami.
        """
        # typy par: klasy 0,1 = G:C; 2,3 = A:U; 4,5 = G:U
        wagi = p_par * otw.unsqueeze(-1).float()
        typy = torch.stack([wagi[..., 0] + wagi[..., 1],
                            wagi[..., 2] + wagi[..., 3],
                            wagi[..., 4] + wagi[..., 5]], dim=-1).sum(dim=(0, 1))
        r_par = typy / typy.sum().clamp_min(_EPS)

        niespar = (partner < 0) & realne.bool()
        n_nsp = (p_zasad * niespar.unsqueeze(-1).float()).sum(dim=(0, 1))
        r_nsp = n_nsp / n_nsp.sum().clamp_min(_EPS)

        d_par = 0.5 * (r_par - self.cel_pary).abs().sum()
        d_nsp = 0.5 * (r_nsp - self.cel_petle).abs().sum()
        return d_par + d_nsp
