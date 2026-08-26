# Eksperymenty i wyniki

Architektura i funkcja straty: [MODEL.md](MODEL.md).
Analiza danych: [notebooks/01_dane.ipynb](notebooks/01_dane.ipynb).
**Analiza wyników: [notebooks/02_wyniki.ipynb](notebooks/02_wyniki.ipynb)** — tam są wykresy i omówienie.

```
python run.py dane     # cd-hit-est + filtry + podzial rodzinowy
python run.py E1
python run.py E2
```

---

## Pytanie badawcze

**Która konstrukcja kary za skład nukleotydowy działa lepiej.** E1 i E2 różnią się wyłącznie tym
jednym członem straty; architektura, dane, podział, człon energetyczny i człon parowań są identyczne.

**Obie kary liczą się PER SEKWENCJA**, potem uśredniamy po partii. Różnica leży wyłącznie
w **kształcie** kary:

```
E1   kara TV        odleglosc od skladu NATURALNEGO
                    DWUSTRONNA: karze kazde odchylenie, takze nadmiar
                    waga 1,0

E2   kara progowa   progi DOLNE udzialow
                    JEDNOSTRONNA: karze wylacznie niedobor, nadmiar bezkarny
                    wagi 1,0 + 1,0, wprost ze specyfikacji promotora
```

**Wagi nie są strojone i w obu eksperymentach wynoszą 1,0.** Specyfikacja promotora wnosi swoją karę
wprost (`loss = loss + DistribLoss`), więc dając naszą również z wagą 1,0 nie wprowadzamy żadnej
dobranej stałej. Na niewytrenowanym modelu daje to porównywalny wkład do straty — 0,33 dla E1 wobec
0,22 dla E2 — więc różnica wyników nie będzie efektem tego, że jedna kara jest po prostu silniejsza.

### Co liczy każda z nich

```
E1   TV = ½ · Σ |udzial_modelu − cel|      osobno dla typow par i dla petli, sumowane
     cel: petle A 0,324  C 0,208  G 0,217  U 0,252
          pary  G:C 0,599   A:U 0,276   G:U 0,124

E2   DistribLoss   = (x_A + x_C + x_G + x_U) / 4     x = max(prog − udzial, 0) / prog
                     progi A 0,15   C 0,30   G 0,30   U 0,15
     DistribLoss3  = (a + b + c) / 3                 dla typow par G:C, A:U, G:U
     DistribLoss4  = max((a + b + c) − 1, 0)         eskalacja przy dwoch zawodzacych typach
                     progi G:C 0,50   A:U 0,20   G:U 0,05
```

**Odstępstwo od dosłownego zapisu specyfikacji.** W bloku o parach zdefiniowane są `gc`, `au`, `gu`,
ale po prawej stronie użyto potem `a`, `c` i `g` z poprzedniego bloku, czyli udziałów adeniny,
cytozyny i guaniny w całej sekwencji; pierwsza linijka jest dodatkowo sama do siebie
(`a = max(0.50-a,0)/0.50`). Wzięte dosłownie porównywałoby próg 50% dla par G:C z udziałem adeniny.
Skoro `gc`/`au`/`gu` są zdefiniowane linijkę wyżej, a progi 50/20/5 dotyczą par, zaimplementowano:

```
a = max(0,50 − gc, 0) / 0,50      b = max(0,20 − au, 0) / 0,20      c = max(0,05 − gu, 0) / 0,05
```

**Skąd cel w E1.** Zmierzony na trzech opublikowanych bazach struktur drugorzędowych RNA — bpRNA,
RNAStrAlign i ArchiveII, razem `data/raw/rna_raw.parquet` — z **wykluczeniem 1455 struktur obecnych
w naszej walidacji albo teście**, czyli n = 29 571. Odtworzenie: `python -m src.cele`.

Zmierzony udział par G:C wynosi 0,599 i pokrywa się z priorem 0,593 opublikowanym niezależnie przez
Portelę (NEMO, bioRxiv 345587). Nie przepisujemy natomiast pozostałych liczb NEMO: ma on A:U 0,333
i G:U 0,074, bo celowo tłumi pary chwiejne dla niezawodności zwijania, a pętle wypełnia w **93%
adeniną**. To są decyzje projektowe, nie opis natury — jako cel popchnęłyby model wprost ku
degeneracji poli-A, której ta kara ma zapobiegać.

**Progi promotora to więzy projektowe, nie opis natury.** Ta sama konwencja jest domyślna w DesiRNA
(*Nucleic Acids Research*, 2025): udziały A/C/G/U ograniczane od dołu, żeby projekt nie zdegenerował
się do jednego nukleotydu. Kara nie zejdzie więc do zera nawet dla modelu wiernie naśladującego
naturę — prawdziwe sekwencje same bywają poniżej progów. W logu widać to jako stałe tło.

---

## Przygotowanie danych

```
data/raw/rna_raw.parquet          data/raw/eterna100.tsv
31 026 sekwencji, 896 rodzin      71 zagadek <= 200 nt
  |                                 |
  |  src/cdhit.py  <= 200 nt,       |   BEZ cd-hit
  |  cd-hit-est 4.8.1               |
  |  -c 0.8 -n 5   -> 8 500         |
  v                                 v
data/cdhit/naturalne_cdhit.parquet  |
  |                                 |
  |  src/prepare.py   przewaga sparowanych (paired_fraction >= 0.5)
  |                   poprawnosc: >= 1 para, alfabet ACGU
  |                   wykonalne petle spinki (kazda >= 3 nt)
  v                                 v
data/working.parquet  3 640       data/eterna_working.parquet  39
  |
  |  src/split.py     podzial 60/20/20 RODZINOWY
  v
data/splits/split_rodzinowy_seed0.json   2185 / 727 / 728, 324 rodziny
```

**Dlaczego filtr pętli spinki.** Szkielet cukrowo-fosforanowy nie zawraca na mniej niż trzech
nukleotydach, więc pętla krótsza nie istnieje fizycznie i **żadna sekwencja nie zwinie się w taką
strukturę**. Znalazło się ich 59 (1,6%): 43 o rozmiarze 2 nt, 12 o rozmiarze 1 nt i 8 o rozmiarze 0,
czyli z nukleotydem sparowanym z bezpośrednim sąsiadem. Źródłem są struktury konsensusowe Rfam
rzutowane na pojedyncze sekwencje — 30 z 59 par zamykających to wobble G:U, a po usunięciu spornej
pary w 15 przypadkach wychodzi `UUCG`, najstabilniejsza tetrapętla w RNA.

Konsekwencja, dla której to usuwamy: ViennaRNA zwraca dla nich wartownik nieskończoności, przez co
`dE` wynosi zero **niezależnie od tego, co model wypisze**. Na zbiorze testowym zaniżało to
raportowane `dE/nt` o 4%. Motyw spinki nie ucierpiał: zostaje 8557 z 8712 pętli (98,2%), a każda
pozostała struktura ma co najmniej jedną.

**Eterna omija cd-hit**, bo nie ma jej w treningu ani razu — odsianie jej przeciwko samej sobie nie
zapobiegłoby żadnemu przeciekowi, a uszczupliłoby zbiór testowy (39 → 35) i uczyniło liczby
nieporównywalnymi z cudzymi.

**Podział rodzinowy:** każda rodzina Rfam trafia do dokładnie jednego podzbioru, więc test składa się
z rodzin niewidzianych w treningu. [`src/split.py`](src/split.py) minimalizuje przy tym trzy rzeczy
naraz — odchylenie liczebności od 60/20/20, odchylenie rozkładu długości od całej puli i dominację
pojedynczej rodziny — inaczej test mierzyłby różnicę długości, a nie generalizację.

Ceną jest to, że największe rodziny dominują trening: rodziny nie wolno rozdzielić między zbiory.

---
### Ile czego zostało

```
pula naturalna <= 200 nt        28 678
  po cd-hit-est                  8 500   (-70%)
  po przewadze sparowanych       3 699   (-4801)
  po kontroli poprawnosci        3 699   (-0)
  po kontroli petli spinki       3 640   (-59)

Eterna <= 200 nt                    71
  po przewadze sparowanych          39   (-32)
  po kontroli poprawnosci           39   (-0)
  po kontroli petli spinki          39   (-0)
```

## Rola każdego zbioru

```
train   uczy wagi modelu — jedyny zbior, ktory zmienia wagi
val     wybiera, ktora epoke zapisac i kiedy przerwac trening; zaden gradient stad nie plynie
test    raport koncowy, ogladany RAZ
```

Wybór epoki po każdej z 60 epok, na pełnym zbiorze walidacyjnym. Wczesne zatrzymanie po 10 epokach
bez poprawy. **Wszystkie kryteria trafiają do logu w każdej epoce**; flaga `--wybor` decyduje tylko,
które zapisuje checkpoint.

| kryterium | wspólne dla E1 i E2 | wolne od kar | uwaga |
|---|---|---|---|
| **`zlozony`** (domyślne) | tak | tak | identyczność jako klucz główny, `dE/nt` jako rozstrzygacz remisów |
| `identycznosc_nt` | tak | tak | liczone na `argmax`, blisko tego, co raportujemy na teście |
| `ce` | tak | tak | liczone na rozkładach, więc widzi też pewność modelu |
| `energia` | tak | tak | premiuje mocne helisy, nie podobieństwo do natury |
| `loss` | **nie** | nie | pełna strata danego modelu, z jego własnymi karami |

**Kryterium musi być takie samo w E1 i E2**, inaczej porównanie kar traci sens.

Kryterium złożone, zaproponowane przez promotora, to porządek leksykograficzny zapisany jedną liczbą:

```
score = round(identycznosc% ) * 1000  +  round(-dE_nt * 1000)
```

**Identyczność jest kluczem głównym, a nie delta energii**, i to jest decyzja mierzona, nie estetyczna.
Sekwencja całkowicie zdegenerowana (same pary G:C, pętle z adeniny) ma energię −0,538 kcal/mol/nt
wobec −0,295 dla sekwencji naturalnych, czyli na samej energii wygrywa z ogromną przewagą. Gdyby to
ona rozstrzygała, wybór epoki systematycznie wskazywałby epokę najbardziej zdegenerowaną.

Z tego samego powodu odrzucamy `loss`: człon parowań ma trywialne minimum, bo pętle z samej adeniny
zerują `A·U`, `G·C` i `G·U` naraz. Do tego `loss` znaczy co innego w E1 i w E2, bo obejmuje inną karę.

Hiperparametrów nie stroimy — każdy eksperyment ma jedną konfigurację. Konsekwencja: na test wolno
spojrzeć raz.

---

## Co raportujemy

Kubełki są **narastające**: `<= 100 nt` zawiera także struktury krótsze niż 50 nt.

| rola | miara | co mierzy |
|---|---|---|
| **rozstrzyga** | `identycznosc_nt %` | ile pozycji ma tę samą zasadę co wzorzec |
| **rozstrzyga** | `identycznosc_par %` | to samo, ale para liczy się po TYPIE — `G-C` i `C-G` to trafienie w G:C |
| kontrola | `dE/nt` | o ile stabilizujemy cel lepiej niż wzorzec; ujemne = lepiej |
| kontrola | `perplexity` | z ilu zasad model średnio wybiera (1 = pewny, 4 = niezdecydowany) |
| diagnostyka | `kara_wlasna` | czy kara zadziałała |

Rozstrzygają identyczności, bo są zewnętrzne wobec obu kar — żaden model nie optymalizował ich
bezpośrednio. `dE/nt` i `perplexity` są porównywalne, bo człon energetyczny ma identyczną wagę
w obu, a perplexity zależy wyłącznie od cross-entropii.

**`kara_wlasna` to inna wielkość w E1 i w E2 — tych liczb nie wolno zestawiać ze sobą.** Odpowiada
na pytanie „czy moja kara zrobiła to, co obiecywała", a nie „która kara jest lepsza".

Różnica `identycznosc_par − identycznosc_nt` mówi, ile błędów to samo odwrócenie pary: model wie,
jaka para ma być, ale myli kierunek. Z definicji `identycznosc_par >= identycznosc_nt`.

Nie raportujemy `solved` ani F1 — obie wymagają przewidywania struktury RNAfoldem, który ma własny
sufit dokładności. Nie odpowiadamy więc na pytanie „czy ta sekwencja się zwinie"; mierzymy
podobieństwo do natury i stabilność zadanej struktury.

**Baseline:** sekwencja losowana z naturalnych częstości, z zachowaniem komplementarności par.
Pokazuje, ile daje sama komplementarność, bez uczenia.

---

## Wyniki

*Do uzupełnienia po uruchomieniu eksperymentów.*

Kary w tabeli liczone są na **wygenerowanych sekwencjach**, a w treningu na rozkładach
prawdopodobieństwa — te liczby nie będą się zgadzać z logiem i nie jest to błąd.

---

## Znane ograniczenia

**Obie kary działają na miękkich rozkładach.** Sekwencja powstaje przez `argmax`, więc model może
mieć rozkład wyglądający naturalnie i przy tym na każdej pozycji wskazywać tę samą zasadę. Dotyczy
to E1 i E2 tak samo.

**Kara E1 nie schodzi do zera i nie powinna.** Prawdziwe sekwencje treningowe mają TV średnio 0,249
(mediana 0,228), bo pojedyncza cząsteczka nie ma składu równego średniej populacyjnej. Kara dwustronna
ciągnie więc każdą sekwencję ku średniej nawet wtedy, gdy model już odtwarza naturę poprawnie — i to
jest cena, jaką E1 płaci za karanie nadmiaru. E2 tego nie robi, bo jest jednostronna.

**Człon parowań ma trywialne minimum.** Wypełnienie pętli adeniną zeruje `A·U`, `G·C` i `G·U`
w części niesparowanej, bo znikają U, G i C. Kara za skład jest jedyną przeciwwagą — i to jest
dokładnie to, co porównują E1 i E2.

**Wszystkie typy pętli dzielą jeden cel składu**, mimo że w naturze się różnią. Rozdzielenie ich per
sekwencja jest niewykonalne: mediana liczby pozycji w wybrzuszeniu i multipętli to sześć, a jedna
trzecia sekwencji nie ma ich wcale.
