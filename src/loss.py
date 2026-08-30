"""Komponenty straty, liczone WPROST na rozkładach modelu.

Model przewiduje parę (i,j) jako jedną z SZEŚCIU klas kanonicznych, więc rozkład łączny na parze
mamy bezpośrednio z softmaxu. Nie ma teacher forcingu, więc rozkłady, na których liczymy kary, są
dokładnie tymi, z których model korzysta przy generowaniu.

VIENNARNA UŻYWAMY WYŁĄCZNIE DO ODCZYTU TABLIC TURNERA (`RNA.param`), raz przy starcie.

CO LICZYMY W ENERGII (człony zależne od SEKWENCJI; człony zależne wyłącznie od kształtu struktury
skracają się w różnicy wobec referencji, więc ich nie potrzebujemy):
  * stosy par                              — tablica `stack`, 6x6 kombinacji klas
  * kary terminalne AU/GU na koncach helis — `TerminalAU`
  * niedopasowanie petli spinki            — `mismatchH`

KARY ZA SKLAD — dwie konkurencyjne konstrukcje, ktore porownuja E1 i E2. Obie licza sie PER
SEKWENCJA, potem srednia po partii; roznia sie wylacznie KSZTALTEM:
  * `sklad`                    odleglosc TV od celu naturalnego — DWUSTRONNA (karze tez nadmiar)
  * `sklad_zasad`, `sklad_par` progi dolne udzialow            — JEDNOSTRONNA (tylko niedobor)
"""

from __future__ import annotations

import torch
import RNA

from src.dataset import PAIRS, BASES, NATURAL_LOOP, NATURAL_PAIR

# Kodowanie typów par w tablicach ViennaRNA. 0 = para niedozwolona.
VRNA_PT = {("C", "G"): 1, ("G", "C"): 2, ("G", "U"): 3, ("U", "G"): 4, ("A", "U"): 5, ("U", "A"): 6}
# Kodowanie zasad w ViennaRNA: A=1, C=2, G=3, U=4 — zgodne z naszym NUC_TO_IDX.
VRNA_BASE = {"A": 1, "C": 2, "G": 3, "U": 4}
_P = RNA.param(RNA.md())
_EPS = 1e-9

# Sklad NATURALNY (cel kary E1) i MARTWA STREFA sa zdefiniowane w `src/dataset.py`, zeby kara
# i baseline korzystaly z jednej stalej. Odtworzenie: `python -m src.cele`.

# PROGI DOLNE ze specyfikacji promotora — uzywane przez `sklad_zasad` i `sklad_par`.
# To NIE sa cele, tylko MINIMA: kara pojawia sie dopiero ponizej progu, nadmiar jest bezkarny.
# Sumy progow musza byc mniejsze od 1, inaczej zadna sekwencja nie moglaby ich wszystkich spelnic
# (tu 0,90 i 0,75).
PROG_ZASADY = {"A": 0.15, "C": 0.30, "G": 0.30, "U": 0.15}
PROG_PARY = {"GC": 0.50, "AU": 0.20, "GU": 0.05}


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
        # Progi dolne dla kary ze specyfikacji promotora.
        self.prog_zasady = torch.tensor([PROG_ZASADY[b] for b in BASES]).to(device)      # (4,)
        self.prog_pary = torch.tensor([PROG_PARY[k] for k in ("GC", "AU", "GU")]).to(device)  # (3,)

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
        n_par = torch.einsum("bic,ck->bk", p_par * otw.unsqueeze(-1).float(), self.par2kol.sum(-1))
        niespar = (partner < 0) & realne.bool()
        n_nsp = (p_zasad * niespar.unsqueeze(-1).float()).sum(1)
        n = n_par + n_nsp                                     # (B,4) w kolejności A,C,G,U
        A, C, G, U = 0, 1, 2, 3
        komb = n[:, G] * n[:, C] + n[:, A] * n[:, U] + n[:, G] * n[:, U]
        return (komb / realne.sum(1).clamp(min=1) ** 2).mean()

    # ------------------------------------------------------------------ skład
    def sklad(self, p_par, p_zasad, partner, otw, realne,
              tol_par: float = 0.0, tol_petle: float = 0.0) -> torch.Tensor:
        """Odleglosc skladu od NATURALNEGO, osobno dla par i osobno dla petli.

        PARY mierzymy jako TYPY (G:C / A:U / G:U), nie jako liczebnosci pojedynczych zasad — kara
        trafia wtedy wprost w to, co chcemy ograniczyc: nadmiar par G:C.

        Cel bierzemy ze STALYCH `NATURAL_LOOP` / `NATURAL_PAIR`, a nie przeliczamy go z wlasnego
        zbioru treningowego. Inaczej cel zmienialby sie z podzialem danych i wyniki przestalyby byc
        porownywalne miedzy eksperymentami.

        LICZONE PER SEKWENCJA, potem srednia po partii — tak samo jak kara ze specyfikacji promotora.
        Gdyby liczyc na rozkladach zagregowanych po calej partii, kara bylaby spelniona "w sredniej":
        pojedyncza sekwencja moglaby byc calkiem zdegenerowana, dopoki inne kompensuja jej odchylenie.

        Kazda sekwencja ma i pary, i pozycje niesparowane: filtr `paired_fraction >= 0.5` wymusza to
        pierwsze, a domkniecie helisy petla — to drugie. Oba czlony sa wiec zawsze okreslone.

        ZWRACAMY OBA CZLONY OSOBNO, bo maja rozne wlasnosci i moga wymagac roznych wag:

          PETLE  cel A 0,311 C 0,203 G 0,205 U 0,280 — sklad petli praktycznie nie rozni sie miedzy
                 rodzinami (train 0,316/0,198/0,208/0,278 wobec test 0,309/0,212/0,205/0,274),
                 wiec ten cel pasuje do kazdego podzbioru.
          PARY   cel G:C 0,551 to srednia wazona po calym zbiorze, ale trening ma 0,600, a walidacja
                 i test po 0,484. Cel jest wiec blizej treningu niz zbiorow, na ktorych oceniamy.

        Wywolujacy nadaje kazdemu czlonowi wlasna wage. `--w-sklad 1.0` ustawia oba, co odtwarza
        zachowanie E1; osobne flagi pozwalaja wazyc je niezaleznie.

        MARTWA STREFA (`tol_par`, `tol_petle`). Cel jest srednia POPULACYJNA, a kara dziala na
        POJEDYNCZEJ sekwencji — a pojedyncza czasteczka nie ma skladu rownego sredniej i nie powinna
        miec. Zmierzone na naszym zbiorze, odleglosc PRAWDZIWEJ sekwencji od celu:

            czlon     srednia   mediana   75%     90%
            pary        0,147     0,132   0,199   0,265
            petle       0,124     0,114   0,158   0,211

        Bez martwej strefy kara ma wiec nieusuwalna podloge i karze za bycie normalnym. Widac to
        w wynikach: model z ta kara osiaga 0,327 przy 0,314 dla sekwencji referencyjnych, czyli jest
        juz tam, gdzie natura — a mimo to gradient dalej go sciska. Skutek uboczny: odchylenie
        standardowe udzialu G:C miedzy sekwencjami spada do 0,114 wobec 0,146 w naturze, czyli kara
        wygladza zmiennosc biologiczna.

        Odejmujemy wiec tolerancje i przycinamy do zera. Sekwencja miesczaca sie w typowym rozrzucie
        placi ZERO; kara wlacza sie dopiero dla odstajacych. Degeneracja typu poli-A ma odleglosc
        rzedu 0,6-0,8, wiec nadal jest karana mocno.

        Domyslne progi to 75. percentyl naturalnego rozrzutu (`TOLERANCJA_*` w src/dataset.py):
        przepuszczaja trzy czwarte prawdziwych sekwencji. `tol = 0` odtwarza zachowanie sprzed
        tej zmiany.
        """
        # typy par: klasy 0,1 = G:C; 2,3 = A:U; 4,5 = G:U            -> (B,3)
        r_par = self._udzialy_typow_par(p_par, otw)

        niespar = (partner < 0) & realne.bool()
        n_nsp = (p_zasad * niespar.unsqueeze(-1).float()).sum(1)                    # (B,4)
        r_nsp = n_nsp / n_nsp.sum(-1, keepdim=True).clamp_min(_EPS)

        d_par = 0.5 * (r_par - self.cel_pary).abs().sum(-1)                         # (B,)
        d_nsp = 0.5 * (r_nsp - self.cel_petle).abs().sum(-1)                        # (B,)
        # MARTWA STREFA: odchylenie mieszczace sie w naturalnym rozrzucie nie jest karane.
        d_par = (d_par - tol_par).clamp(min=0)
        d_nsp = (d_nsp - tol_petle).clamp(min=0)
        return d_par.mean(), d_nsp.mean()

    # -------------------------------------------------- kara skladu wg specyfikacji promotora
    def _udzialy_zasad(self, p_par, p_zasad, partner, otw, realne):
        """Oczekiwane udzialy A/C/G/U w CALEJ sekwencji, per sekwencja. (B,4)."""
        n_par = torch.einsum("bic,ck->bk", p_par * otw.unsqueeze(-1).float(), self.par2kol.sum(-1))
        niespar = (partner < 0) & realne.bool()
        n_nsp = (p_zasad * niespar.unsqueeze(-1).float()).sum(1)
        return (n_par + n_nsp) / realne.sum(1, keepdim=True).clamp(min=1)

    def _udzialy_typow_par(self, p_par, otw):
        """Oczekiwane udzialy typow par G:C / A:U / G:U, per sekwencja. (B,3)."""
        w = p_par * otw.unsqueeze(-1).float()
        typy = torch.stack([w[..., 0] + w[..., 1],      # G-C i C-G
                            w[..., 2] + w[..., 3],      # A-U i U-A
                            w[..., 4] + w[..., 5]],     # G-U i U-G
                           dim=-1).sum(1)
        return typy / typy.sum(-1, keepdim=True).clamp_min(_EPS)

    def sklad_zasad(self, p_par, p_zasad, partner, otw, realne) -> torch.Tensor:
        """DistribLoss ze specyfikacji promotora — udzialy zasad w calej sekwencji.

            x = max(prog - udzial, 0) / prog        dla kazdej z czterech zasad
            DistribLoss = (x_A + x_C + x_G + x_U) / 4

        JEDNOSTRONNA: karzemy wylacznie NIEDOBOR. Nadmiar zostaje bez kary, wiec kara nie ciagnie
        kazdej sekwencji do sredniej populacyjnej i nie zabija zmiennosci biologicznej. Brak zasady
        calkowicie daje x = 1, czyli maksimum.

        Dzielenie przez prog wyrownuje wklad zasad o roznych progach: brak cytozyny przy progu 0,30
        i brak adeniny przy progu 0,15 daja te sama jedynke.

        Liczone PER SEKWENCJA, potem srednia po partii — tak samo jak `sklad`. Roznica miedzy
        obiema karami lezy wylacznie w KSZTALCIE: tu prog dolny (kara tylko za niedobor), tam
        odleglosc TV od celu (kara takze za nadmiar).
        """
        u = self._udzialy_zasad(p_par, p_zasad, partner, otw, realne)
        x = (self.prog_zasady - u).clamp(min=0) / self.prog_zasady
        return (x.sum(-1) / 4).mean()

    def sklad_par(self, p_par, otw) -> torch.Tensor:
        """DistribLoss3 ze specyfikacji promotora — udzialy TYPOW par.

            a, b, c        = max(prog - udzial, 0) / prog   dla G:C, A:U, G:U
            DistribLoss3   = (a + b + c) / 3

        JEDNOSTRONNA: karzemy wylacznie NIEDOBOR. Nadmiar zostaje bez kary.
        """
        u = self._udzialy_typow_par(p_par, otw)
        v = (self.prog_pary - u).clamp(min=0) / self.prog_pary
        return (v.sum(-1) / 3).mean()
