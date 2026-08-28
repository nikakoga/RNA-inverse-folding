# Eksperymenty i wyniki

Architektura i funkcja straty: [MODEL.md](MODEL.md).
Analiza danych: [notebooks/01_dane.ipynb](notebooks/01_dane.ipynb).
**Analiza wyników: [notebooks/02_wyniki.ipynb](notebooks/02_wyniki.ipynb)** — tam są wykresy i omówienie.

```
python run.py dane       # cd-hit-est + filtry + podzial rodzinowy
python run.py E1         # nasza kara za sklad (odleglosc TV)
python run.py E2         # kara promotora (progi dolne)
python run.py CE         # BEZ zadnych kar — ablacja trzech czlonow
python run.py E1W        # jak E1, ale cross-entropia WAZONA odwrotnie do czestosci klas
python run.py E2W        # jak E2, ale wazona
python run.py CEW        # jak CE, ale wazona
python -m src.przeglad   # przeglad 13 ustawien wag, na WALIDACJI (~38 min)
python -m src.test_youden # weryfikacja wskaznika Youdena na danych sztucznych
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

**Wszystkie komponenty straty mają wagę 1,0** — energia, parowania i skład. Żadna z tych liczb nie
jest dobrana pod wynik: specyfikacja promotora wnosi swoją karę wprost (`loss = loss + DistribLoss`),
a my przyjmujemy tę samą konwencję dla pozostałych członów. Dzięki temu w całej pracy nie ma ani
jednej wagi, której nie da się uzasadnić jednym zdaniem.


**Sekwencje powstają przez losowanie z rozkładu modelu** (`--dekodowanie probkowanie
--seed-dekodowania 0`), tak samo w walidacji i w ocenie końcowej. Uzasadnienie w sekcji „Dlaczego nie
`argmax`". Ziarno jest częścią wyniku.

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
naturę — sekwencje referencyjne same bywają poniżej progów. W logu widać to jako stałe tło.

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

### Cena podziału rodzinowego: trening to praktycznie dwie rodziny

Rodziny nie wolno rozdzielić między zbiory, a dwie największe w puli są ogromne. Skutek:

```
train     tRNA 64%,  5S 29%,  mir-9 1%,  mir-124 1%
val       16S 31%,   U2 4%,   SSU_rRNA_bacteria 3%
test      SRP 14%,   RNaseP 8%,  Cobalamin 3%
```

**93% zbioru treningowego to tRNA i 5S.** Na pozostałe 80 rodzin przypada 153 struktury z 2185.

To przekłada się wprost na skład. W pętlach podzbiory są niemal identyczne, ale **w parach jest
przesunięcie o 11,6 punktu procentowego**, bo tRNA i 5S są bogate w G:C:

```
                    A        C        G        U   |     G:C      A:U      G:U
train          0,3163   0,1977   0,2075   0,2785   |  0,6005   0,2969   0,1026
val            0,2997   0,2121   0,1989   0,2892   |  0,4839   0,3862   0,1299
test           0,3091   0,2116   0,2051   0,2742   |  0,4841   0,3711   0,1448
               roznica train-test najwyzej 1,4 pp     roznica train-test 11,6 pp
```

I model dokładnie to odtwarza:

```
udzial par G:C
  w TRENINGU (czego sie uczyl)     0,600
  model E1 wypisuje                0,622
  w TESCIE  (czego oczekujemy)     0,484
```

Model nauczył się rozkładu treningowego i stosuje go do zbioru, w którym ten rozkład jest zły. To nie
jest wada funkcji straty ani dekodowania, tylko **przesunięcie rozkładu wbudowane w podział**.

Wykresy i pełne liczby: [notebooks/01_dane.ipynb](notebooks/01_dane.ipynb), sekcja 5.

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
val     wybiera, ktora epoke zapisac
test    raport koncowy, ogladany RAZ
```

**Wszystkie kryteria trafiają do logu w każdej epoce**; flaga `--wybor` decyduje tylko, które
zapisuje checkpoint.

| kryterium | wspólne dla eksperymentów | wolne od kar | poziom losowy | uwaga |
|---|---|---|---|---|
| **`zbal_par`** (domyślne) | tak | tak | 33,3% | średnia czułość po 3 typach par, bez wagi |
| `zbal_zasady` | tak | tak | 25,0% | to samo dla 4 zasad w pętlach |
| `youden_GC` | tak | tak | **0** | czułość + specyficzność − 1 dla G:C |
| `youden_par`, `youden_zasady` | tak | tak | **0** | to samo, uśrednione po klasach |
| `identycznosc_nt` | tak | tak | — | podbija ją stały predyktor — patrz „Miary odporne…" niżej |
| `ce` | tak | tak | — | liczone na rozkładach, więc widzi też pewność modelu |
| `energia` | tak | tak | — | premiuje mocne helisy, nie podobieństwo do natury |
| `zlozony` | tak | tak | — | identyczność jako klucz główny, `dE/nt` jako rozstrzygacz remisów |
| `loss` | **nie** | nie | — | pełna strata danego modelu, z jego własnymi karami |

**Kryterium musi być takie samo we wszystkich porównywanych przebiegach**, inaczej porównanie traci
sens. Wszystkie eksperymenty używają `zbal_par`.

### Dlaczego domyślne przestało być `zlozony`

Oba jego człony premiują nadprodukcję klasy najczęstszej, czyli **dokładnie tę awarię, którą
eksperyment ma zmierzyć**. Identyczność jest maksymalizowana przez stały predyktor — sekwencja
z samych par G:C dostaje 48,4%, więcej niż którykolwiek z naszych modeli. A `dE/nt` spada wraz ze
wzrostem udziału G:C, bo G:C jest parą najstabilniejszą.

`zbal_par` tej wady nie ma: predyktor niezależny od pozycji dostaje tam dokładnie 1/3, **niezależnie
od tego, co produkuje**. Przesunięcie składu wyjścia nie podbija tej miary ani o punkt.

### Dlaczego nie ma samej specyficzności GC jako kryterium?

Specyficzność G:C rośnie do 100%, gdy model **przestaje** wystawiać G:C — model produkujący same A:U
miałby ją idealną. Jest więc podatna na stały predyktor tak samo jak identyczność, tylko z drugiej
strony. Zamiast niej wchodzi **wskaźnik Youdena**, łączący obie połowy:

```
J = czulosc + specyficznosc - 1
```

Poziom odniesienia to **dokładnie zero** dla każdego modelu przypisującego klasy niezależnie od
pozycji, bez względu na skład wyjścia: taki model ma czułość `q` i specyficzność `1 − q`, więc suma
zawsze wynosi 1. Nie da się go podbić przesunięciem składu w żadną stronę.

Wskaźnik ma jeszcze jedną własność, wygodną przy pisaniu: **czyta się go wprost jako ułamek
pozycji, na których model naprawdę zna odpowiedź.** Predyktor, który z prawdopodobieństwem `p` trafia,
a poza tym zgaduje losowo, ma `J = p` dokładnie. Sprawdzone symulacją na 400 000 pozycji:

```
co robi predyktor                 czul G:C   spec G:C     J_GC    zbal_par
--- NIEZALEZNE OD POZYCJI, J musi byc 0 ---
jednostajnie 1/3                     0,331      0,667   -0,0023      0,332
wg czestosci referencyjnych          0,485      0,517   +0,0019      0,334
jak nasze E1                         0,621      0,379   -0,0002      0,333
SAME G:C                             1,000      0,000   +0,0000      0,333
ZERO G:C                             0,000      1,000   +0,0000      0,333
--- COS WIEDZA ---
trafia w 50% przypadkow              0,669      0,834   +0,5024      0,668
trafia w 80% przypadkow              0,867      0,933   +0,8002      0,866
trafia zawsze                        1,000      1,000   +1,0000      1,000
```

Zwróć uwagę na dwa skrajne wiersze pierwszej grupy: specyficzność G:C skacze od 0,00 do 1,00,
a wskaźnik Youdena ani drgnie. Dokładnie tego chcemy od miary, która ma być odporna na przesunięcie
składu wyjścia. Odtworzenie: `python -m src.test_youden`.

Nasze modele mają `J` rzędu **+0,01**, czyli **znają odpowiedź na jakimś jednym procencie pozycji**.
Ta sama diagnoza wychodzi niezależnie z porównania z predyktorem marginalnym niżej.

Kryterium złożone, zaproponowane przez promotora, to porządek leksykograficzny zapisany jedną liczbą:

```
score = round(identycznosc% ) * 1000  +  round(-dE_nt * 1000)
```

**Identyczność jest kluczem głównym, a nie delta energii**. 
Sekwencja całkowicie zdegenerowana (same pary G:C, pętle z adeniny) ma energię −0,538 kcal/mol/nt
wobec −0,295 dla sekwencji naturalnych, czyli na samej energii wygrywa z ogromną przewagą. Gdyby to
ona rozstrzygała, wybór epoki systematycznie wskazywałby epokę najbardziej zdegenerowaną.

**Strojenie wag odbywa się WYŁĄCZNIE na walidacji** (`python -m src.przeglad`), a na test patrzymy
raz, po wybraniu konfiguracji. Wynik przeglądu jest negatywny — patrz „Przegląd wag" niżej — więc
wszystkie sześć przebiegów zachowuje wagi ustalone z góry, bez dobierania pod wynik.

---

## Co raportujemy

Kubełki są **narastające**: `<= 100 nt` zawiera także struktury krótsze niż 50 nt.

| rola | miara | co mierzy |
|---|---|---|
| **rozstrzyga** | `identycznosc_nt %` | ile pozycji ma tę samą zasadę co referencja |
| **rozstrzyga** | `identycznosc_par %` | to samo, ale para liczy się po TYPIE — `G-C` i `C-G` to trafienie w G:C |
| kontrola | `dE/nt` | o ile stabilizujemy cel lepiej niż referencja; ujemne = lepiej |
| diagnostyka | `kara_wlasna` | czy kara zadziałała |

Rozstrzygają identyczności, bo są zewnętrzne wobec obu kar — żaden model nie optymalizował ich
bezpośrednio. `dE/nt` jest porównywalne, bo człon energetyczny ma identyczną wagę we wszystkich
przebiegach.

**`kara_wlasna` to inna wielkość w E1 i w E2 — tych liczb nie wolno zestawiać ze sobą.** Odpowiada
na pytanie „czy moja kara zrobiła to, co obiecywała", a nie „która kara jest lepsza".

Różnica `identycznosc_par − identycznosc_nt` mówi, ile błędów to samo odwrócenie pary: model wie,
jaka para ma być, ale myli kierunek. Z definicji `identycznosc_par >= identycznosc_nt`.

Nie raportujemy `solved` ani F1 — obie wymagają przewidywania struktury RNAfoldem, który ma własny
sufit dokładności. Nie odpowiadamy więc na pytanie „czy ta sekwencja się zwinie"; mierzymy
podobieństwo do natury i stabilność zadanej struktury.

**Baseline:** sekwencja losowana **jednostajnie**, z zachowaniem komplementarności par — w miejscu
pary dowolny z trzech typów po 1/3, na pozycji niesparowanej dowolna zasada po 1/4. Celowo **nie** losuje z częstości naturalnych. Taki wariant dostawałby za darmo wiedzę o składzie RNA,
czyli dokładnie to, czego model musi się dopiero nauczyć z danych — i przez to byłby punktem
odniesienia zawyżonym. Baseline ma reprezentować zero wiedzy.

Skutek uboczny, wygodny przy czytaniu wykresów: jego czułość wynosi 1/3 dla każdego typu pary i 1/4
dla każdej zasady, więc leży dokładnie na poziomie losowym.

---

## Miary odporne na model, który zawsze wskazuje tę samą klasę

Zwykła identyczność ma wadę, którą trzeba znać: **jest maksymalizowana przez stałe wskazywanie klasy
najczęstszej.** Sekwencja złożona z samych par G:C dostaje `identycznosc_par` 48,4%, czyli dokładnie
tyle, ile wynosi udział par G:C w referencji — i więcej niż uczciwy baseline (33,7%), a także więcej
niż którykolwiek z naszych modeli.

Dlatego raportujemy też **czułość osobno dla każdej klasy** i ich średnią **bez wagi**:

```
zbal_par = (czulosc G:C + czulosc A:U + czulosc G:U) / 3
zbal_zas = (czulosc A + czulosc C + czulosc G + czulosc U) / 4
```

Predyktor stały spada tam do 1/3 (33,3%) i 1/4 (25,0%), niezależnie od tego, którą klasę wybierze.
Orientacja pary nie ma znaczenia: `A-U` i `U-A` to ten sam typ A:U.

---

## Wyniki

Wszystko na zbiorze testowym, kubełek `<= 200 nt` (728 struktur), podział rodzinowy. Sześć
przebiegów w **identycznych warunkach** — te same dane, wagi 1,0, próbkowanie z ziarnem 0, pełne
60 epok, epoka wybrana po `zbal_par` na walidacji. Różnią się wyłącznie karą za skład i tym, czy
cross-entropia jest ważona.

| model | ident_nt | ident_par | zbal_par | zbal_zas | dE/nt | Youden |
|---|---|---|---|---|---|---|
| baseline losowy | 25,95% | 33,70% | 33,46% | 25,03% | +0,1273 | +0,0028 |
| E1 kara TV | 27,37% | 42,61% | 34,49% | 25,60% | −0,0571 | +0,0170 |
| E2 kara promotora | 27,40% | **43,16%** | 34,38% | 25,98% | −0,0869 | +0,0152 |
| CE bez kar | **27,76%** | 41,69% | **35,00%** | **26,27%** | +0,0017 | **+0,0256** |
| E1W TV + ważona CE | 26,71% | 41,18% | 34,33% | 25,43% | −0,0293 | +0,0135 |
| E2W progi + ważona CE | 26,81% | 42,11% | 34,66% | 25,89% | −0,0439 | +0,0179 |
| CEW ważona CE | 26,30% | 35,78% | 34,46% | 25,65% | +0,1406 | +0,0152 |
| **poziom losowy** | | | **33,33%** | **25,00%** | | **0** |

Wykresy: [notebooks/02_wyniki.ipynb](notebooks/02_wyniki.ipynb) — zestaw I (bez ważenia), zestaw II
(z ważeniem), zestaw III (pary), zestaw IV (Youden na trzech podzbiorach).

### Modele biją poziom losowy, ale między sobą są nierozróżnialne

`zbal_par` mieści się w przedziale 34,33–35,00% przy poziomie losowym 33,33%. Wszystkie sześć
przekracza go o 1,0–1,7 pp i potwierdzają to trzy niezależne miary:

```
                    zbal_par     Youden    nadwyzka identycznosci nad samym skladem
baseline             33,46%      +0,003              +0,25 pp
modele            34,3-35,0%  +0,014..+0,026      +0,98..+1,86 pp
```

Ostatnia kolumna to identyczność ponad to, co dostałby predyktor o **tym samym składzie wyjścia**,
losujący bez patrzenia na pozycję (`Σ udzial_k · czestosc_k`). Baseline ma tam 0,25 pp, czyli zero
w granicach szumu; modele 1–1,9 pp.

**Ale różnice MIĘDZY modelami są w granicach szumu.** Rozstęp na `zbal_par` wynosi 0,67 pp, a
zmierzony rozrzut między ziarnami inicjalizacji to ±0,26 pp (12 przebiegów,
`experiments/analysis/szum_wagi_klas.csv`) — różnica dwóch niezależnych przebiegów ma więc
odchylenie 0,37 pp. Na tej mierze **nie da się uszeregować sześciu konfiguracji.**

### Identyczność mierzy skład, nie wiedzę o pozycji

Dla predyktora przypisującego klasy niezależnie od pozycji identyczność par wynosi dokładnie
`Σ udzial_k · czestosc_k`. Podstawiając sam skład wyjścia, bez żadnej wiedzy o modelu:

```
model       zmierzona   przewidziana z samego skladu   nadwyzka
E1             42,61%                        41,42%     +1,19 pp
E2             43,16%                        42,04%     +1,12 pp
CE             41,69%                        39,84%     +1,86 pp
E1W            41,18%                        40,19%     +0,98 pp
E2W            42,11%                        40,98%     +1,13 pp
CEW            35,78%                        34,73%     +1,05 pp
```

Zgodność do 1–2 pp. **Przewaga E2 nad E1 na `identycznosc_par` jest w całości skutkiem większego
udziału G:C** (0,645 wobec 0,595), a nie lepszej wiedzy o tym, gdzie która para stoi. Dlatego ta
miara nie rozstrzyga u nas niczego, mimo że jest najbardziej intuicyjna.

### Skład: kara za skład go POGARSZA

Miara neutralna — odległość TV od składu **referencyjnego**:

```
model                        G:C     A:U     G:U   |     A      C      G      U   | TVpary TVpetle RAZEM
E1   kara TV               0,595   0,299   0,106   | 0,326  0,208  0,209  0,258   |  0,111  0,020  0,131
E2   kara promotora        0,645   0,251   0,104   | 0,295  0,249  0,213  0,243   |  0,161  0,045  0,207
CE   bez kar               0,505   0,363   0,132   | 0,304  0,231  0,173  0,292   |  0,021  0,038  0,059
E1W  TV + wazona CE        0,552   0,309   0,139   | 0,321  0,220  0,209  0,250   |  0,068  0,024  0,092
E2W  progi + wazona CE     0,577   0,306   0,117   | 0,233  0,314  0,217  0,236   |  0,093  0,114  0,207
CEW  wazona CE             0,297   0,449   0,254   | 0,234  0,306  0,193  0,267   |  0,187  0,094  0,281
baseline                   0,336   0,334   0,330   | 0,252  0,246  0,250  0,251   |  0,185  0,080  0,264
REFERENCJA                 0,484   0,371   0,145   | 0,309  0,212  0,205  0,274   |
```

Tu różnice są kilkukrotne, więc **wykraczają poza szum** — wynik:

**Model bez żadnej kary za skład ma skład najbliższy naturze** (0,059), a na samych parach
praktycznie trafia w referencję (0,021 przy `G:C 0,505` wobec 0,484). Uczy się składu z danych
i uogólnia go na nowe rodziny — mimo że trening ma `G:C 0,600`.

**Kara ciągnie go z powrotem od natury.** E1 dochodzi do 0,595, E2 do 0,645, bo cel kary wynosi
`G:C 0,599` — zmierzony na bazach zewnętrznych, zgodny z naszym **treningiem**, ale o dwanaście
punktów wyższy niż zbiór testowy. Kara sumiennie prowadzi model do celu, który dla rodzin testowych
jest zły, i tym samym **wzmacnia przesunięcie rozkładu** opisane wyżej, zamiast je korygować.

Nie da się tego naprawić strojeniem: cel dopasowany do treningu zawsze będzie za wysoki dla testu,
a dopasowanie go do testu byłoby przeciekiem.

**E1 wypada lepiej niż E2** (0,131 wobec 0,207), i to jest odpowiedź na pytanie badawcze. Kara
dwustronna karze nadmiar, kara promotora ma próg dolny `G:C >= 0,50`, więc nadmiaru nie widzi.

### Ważenie klas: pomaga tam, gdzie jest kara; szkodzi tam, gdzie jej nie ma

Zestawienie parami, w których jedyną różnicą jest ważona cross-entropia:

```
para          zbal_par        TV pary            TV razem
E1  -> E1W   34,49 -> 34,33   0,111 -> 0,068     0,131 -> 0,092
E2  -> E2W   34,38 -> 34,66   0,161 -> 0,093     0,207 -> 0,207
CE  -> CEW   35,00 -> 34,46   0,021 -> 0,187     0,059 -> 0,281
```

Na trafności zmiany wynoszą −0,16, +0,28 i −0,54 pp — bez zgodnego znaku i w granicach szumu.

Na składzie działa mechanizm, który da się opisać jednym zdaniem: **ważenie przesuwa wyjście ku
klasom rzadkim, czyli dokładnie przeciwnie niż kara za skład.** Tam, gdzie kara przestrzeliła w górę,
ważenie ją równoważy (TV par 0,111 → 0,068 i 0,161 → 0,093). Tam, gdzie kary nie ma, nie ma czego
równoważyć i samo ważenie przestrzeliwuje w drugą stronę — CEW schodzi do `G:C 0,297` przy
referencji 0,484, czyli dalej od natury niż losowy baseline.

### Youden na treningu kontra rodziny nowe: to nie jest problem straty

Ten sam wskaźnik policzony na podzbiorach. Podział jest rodzinowy, więc walidacja i test składają się
z rodzin **nieobecnych w treningu**:

```
model     J train     J val    J test   spadek
E1        +0,3222   +0,0202   +0,0170     19x
E2        +0,3279   +0,0158   +0,0152     22x
CE        +0,3964   +0,0284   +0,0256     16x
E1W       +0,3589   +0,0244   +0,0135     27x
E2W       +0,4060   +0,0327   +0,0179     23x
CEW       +0,3684   +0,0363   +0,0152     24x
```

`J` czyta się wprost jako ułamek pozycji, na których model zna odpowiedź. **Na rodzinach widzianych
jest to 32–41%. Na nowych — 1,4–3,6%.** Spadek 16–27-krotny, identyczny we wszystkich sześciu
konfiguracjach, więc nie jest właściwością żadnej kary ani ważenia.


```
Zastrzeżenie: wynik na treningu jest częściowo zapamiętaniem konkretnych sekwencji, więc **nie jest sufitem osiągalnym na nowych rodzinach** Prawdziwy sufit jest zresztą poniżej 1,0, bo odwrotne zwijanie jest jeden-do-wielu.
```

Konsekwencja praktyczna: **wąskim gardłem nie jest funkcja straty, tylko skład zbioru treningowego**
(92% to tRNA i 5S). Żadna waga ani nowy człon straty tego nie ruszy — waga steruje siłą, a nie
rodzajem informacji, a informacji o nowych rodzinach w treningu po prostu nie ma.

### Dlaczego nie `argmax`

Wcześniejsze wersje E1 i E2 dekodowały przez `argmax` i **zapadały się**: 96% adeniny w pętlach,
99,9% par G:C. Rozkłady były przy tym w porządku — E2 miał miękko `G:C 0,693`, a nie 0,999.

Przyczyna jest jedna i prosta: **rozkład jest niemal ten sam na każdej pozycji.** Model prawie nie
korzysta z wejścia, więc wszędzie produkuje ten sam płaski rozkład z tym samym drobnym przechyłem ku
G:C. `argmax` bierze zwycięzcę, a zwycięzca jest wszędzie ten sam — punkt zamiast rozkładu.

Zmierzone na 20 691 pozycjach otwierających parę w zbiorze testowym:

```
model   pewnosc zwyciezcy   przewaga nad drugim   pozycji wygranych przez G:C
E1              0,384              0,137                    98,4%
E2              0,432              0,169                    99,9%
CE              0,338              0,109                    91,5%
                (jednostajny 0,167)
```

Zwycięzca ma średnio 0,38–0,43 prawdopodobieństwa — czyli w ponad połowie przypadków model stawia na
klasę, w którą sam nie wierzy. Ale wygrywa, i to wystarczy.

Warto od razu odrzucić wyjaśnienie, które się samo nasuwa: że winny jest podział na **sześć** klas,
w którym typ A:U musi przebić obie orientacje G:C osobno. Sprawdziliśmy to — `argmax` liczony po
trzech typach (z masami orientacji zsumowanymi) daje `G:C` 0,990 / 0,999 / 0,922, czyli tyle samo
albo więcej. Sześć klas nic tu nie psuje; psuje sam `argmax` na rozkładzie, który nie zależy od
pozycji.

**Losowanie z rozkładu tego problemu nie ma**, bo oczekiwany skład wylosowanej sekwencji równa się
składowi rozkładu. Zgodność jest dokładna do trzeciego miejsca po przecinku: E1 ma miękko
`G:C 0,622`, a na wygenerowanych sekwencjach 0,625. Wartość kary raportowana w treningu opisuje więc
to samo, co widać na gotowej sekwencji.

### Przegląd wag: optymalnego ustawienia nie ma

`python -m src.przeglad` — 13 ustawień wag, każde trenowane osobno, wszystko **na walidacji**.

```
rozstep miedzy 13 ustawieniami:                   0,93 pp
rozrzut miedzy ziarnami dla tej samej konfiguracji: ±0,26 pp
najwyzej wypada:                                  "nic (sama CE)", 34,51%
```

**Różnice między konfiguracjami są rzędu szumu**, więc żadnego ustawienia nie da się uznać za lepsze.
Waga parowań przebiega cały zakres 0 → 1 → 3 → 6 → 12 i nie wynika z tego nic — to była przesłanka,
przez którą zdjęliśmy odziedziczoną wagę 6,0.

Negatywny wynik przeglądu zwalnia nas z dobierania: wszystkie wagi stoją na 1,0 i da się to
uzasadnić jednym zdaniem, zamiast bronić liczby dopasowanej pod wynik.

Pełny wynik: `experiments/analysis/przeglad_wag.csv`.

---

Kary w tabelach liczone są na **wygenerowanych sekwencjach**.

## Znane ograniczenia

**Cel składu par jest za wysoki dla zbioru testowego.** Mierzymy `G:C 0,599` na bazach zewnętrznych,
a test ma 0,484 — kara prowadzi więc model do celu przesuniętego o dwanaście punktów. To nie jest
wada implementacji, tylko konsekwencja podziału rodzinowego: cel pokrywa się ze składem treningu
(0,600), a test zawiera inne rodziny. Dopasowanie celu do testu byłoby przeciekiem, więc zostawiamy
go takim, jaki jest, i raportujemy skutek.

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
