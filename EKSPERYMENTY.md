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
python run.py E3         # KONTROLA: energia i parowania, ale BEZ kary za sklad
python -m src.szum       # prog istotnosci: 7 konfiguracji x 3 ziarna (ok. 2h15)
python -m src.test_youden # weryfikacja wskaznika Youdena na danych sztucznych
```

---

## Pytanie badawcze

**Zbudować model oparty na transformerze, który jak najlepiej przewiduje sekwencję nukleotydową dla
zadanej struktury drugorzędowej.**

Eksperymenty oznaczone literą `E` nie są celem samym w sobie — to **droga do tego modelu**. Każdy
sprawdza jedną decyzję projektową, przy wszystkich pozostałych zamrożonych:

```
E1 kontra E2      konstrukcja kary za sklad: dwustronna czy jednostronna
E3               czy energia i parowania w ogole pomagaja
CE               czy jakiekolwiek kary poza cross-entropia pomagaja
warianty W       czy wazenie klas w cross-entropii pomaga
argmax / losowanie  sposob dekodowania
```

Wnioski z tych porównań są rozproszone po sekcjach niżej; **nie ma tu jednego werdyktu „ta
konfiguracja jest najlepsza"**, bo żadna nie wygrywa na wszystkich miarach naraz, a różnice są
w większości małe wobec progu istotności.

### Dwie konstrukcje kary za skład

E1 i E2 różnią się wyłącznie tym jednym członem straty; architektura, dane, podział, człon
energetyczny i człon parowań są identyczne.

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


**Sekwencje powstają przez `argmax`** — na każdej pozycji litera o największym
prawdopodobieństwie, tak samo w walidacji i w ocenie końcowej. Generowanie jest przez to
**deterministyczne**: nie trzeba ziarna, a wynik da się odtworzyć co do litery.

Wymaga to jednak, żeby **kary też patrzyły na twarde wyjście** (`--kary-na-argmax`, estymator
straight-through). Kary liczą się normalnie na rozkładach, a `argmax` patrzy tylko, kto jest na
szczycie — nie o ile wygrywa. Bez tego model może mieć rozkład o poprawnym składzie i zdegenerowane
wyjście: zmierzyliśmy taki o miękkim `G:C 0,622`, który po `argmax` dawał 0,984. Szczegóły
w sekcji „Dekodowanie".

### Co liczy każda z nich

```
E1   TV = ½ · Σ |udzial_modelu − cel|      osobno dla typow par i dla petli, sumowane
     cel: petle A 0,311  C 0,203  G 0,205  U 0,280
          pary  G:C 0,551   A:U 0,332   G:U 0,117

E2   DistribLoss   = (x_A + x_C + x_G + x_U) / 4     x = max(prog − udzial, 0) / prog
                     progi A 0,15   C 0,30   G 0,30   U 0,15
     DistribLoss3  = (a + b + c) / 3                 dla typow par G:C, A:U, G:U
                     progi G:C 0,50   A:U 0,20   G:U 0,05
```



```
a = max(0,50 − gc, 0) / 0,50      b = max(0,20 − au, 0) / 0,20      c = max(0,05 − gu, 0) / 0,05
```

**Skąd cel w E1.** Zmierzony na **naszym własnym zbiorze** `data/working.parquet` — wszystkich
3640 sekwencjach, czyli train + val + test razem (95 790 par, 147 092 pozycje niesparowane).
Odtworzenie: `python -m src.cele`.

```
train   G:C 0,600      <- 57% wszystkich par, wiec ciagnie srednia w gore
val     G:C 0,484
test    G:C 0,484
-----------------
razem   G:C 0,551      <- cel
```

Poprzednio cel pochodził z trzech baz zewnętrznych (bpRNA, RNAStrAlign, ArchiveII; n = 29 571) i
wynosił `G:C 0,599`. Pokrywał się z naszym **treningiem** (0,600), ale nie z walidacją ani testem
(obie 0,484), więc kara prowadziła model ku składowi złemu dla zbiorów, na których go oceniamy.

⚠️ **Konsekwencja dla interpretacji.** Cel obejmuje teraz także test, więc zdanie „model trafił
w skład testu" przestaje być dowodem, że nauczył się go z danych — część tej informacji dostał wprost
w celu kary. Dotyczy to wyłącznie miar składu i wyłącznie modeli z karą E1; na trafność (`zbal_par`,
Youden) nie ma wpływu, bo globalne proporcje nie mówią, która para stoi w którym miejscu.

Dla porównania: pomiar na tych samych bazach zewnętrznych dawał `G:C 0,599`, co pokrywało się
z priorem 0,593 opublikowanym niezależnie przez Portelę (NEMO, bioRxiv 345587). Nasz zbiór jest
uboższy w G:C (0,551), bo obejmuje inny dobór rodzin.

Nie przepisujemy natomiast pozostałych liczb NEMO: ma on A:U 0,333 i G:U 0,074, bo celowo tłumi pary
chwiejne dla niezawodności zwijania, a pętle wypełnia w **93% adeniną**. To są decyzje projektowe,
nie opis natury — jako cel popchnęłyby model wprost ku degeneracji poli-A, której ta kara ma
zapobiegać.

**Progi promotora to więzy projektowe, nie opis natury.** Ta sama konwencja jest domyślna w DesiRNA
(*Nucleic Acids Research*, 2025): udziały A/C/G/U ograniczane od dołu, żeby projekt nie zdegenerował
się do jednego nukleotydu. Kara nie zejdzie więc do zera nawet dla modelu wiernie naśladującego
naturę — sekwencje referencyjne same bywają poniżej progów. W logu widać to jako stałe tło.

---

## Przygotowanie danych

```
data/raw/rna_raw.parquet
31 026 sekwencji, 896 rodzin
  |
  |  src/cdhit.py     <= 200 nt, cd-hit-est 4.8.1, -c 0.8 -n 5   -> 8 500
  v
data/cdhit/naturalne_cdhit.parquet
  |
  |  src/prepare.py   przewaga sparowanych (paired_fraction >= 0.5)
  |                   poprawnosc: >= 1 para, alfabet ACGU
  |                   wykonalne petle spinki (kazda >= 3 nt)
  v
data/working.parquet  3 640
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

## Próg istotności

Żadnej różnicy w tabelach niżej nie wolno czytać bez tej sekcji.

Każda z siedmiu konfiguracji została wytrenowana **trzy razy**, przy zmienionym wyłącznie ziarnie
inicjalizacji wag i kolejności partii (`--seed-modelu`). Podział danych pozostaje nietknięty — steruje
nim osobna flaga `--seed`, a jej zmiana zmieniłaby zbiór walidacyjny i wyniki przestałyby być
porównywalne. Pomiar jest **na walidacji**: to kwestia metodologiczna, nie wynik do raportu.
Odtworzenie: `python -m src.szum`, wyniki w `experiments/analysis/szum_ziaren.csv`.

```
miara               odch. TEST   prog TEST   odch. walidacji   ile razy wiekszy na tescie
zbal_par                 0,535       1,513             0,125             4,3
zbal_zasady              0,384       1,086             0,208             1,8
youden_par              0,0079      0,0223            0,0017             4,6
youden_zasady           0,0053      0,0150            0,0028             1,9
identycznosc_nt          0,294       0,830             0,263             1,1
identycznosc_par         1,274       3,604             1,018             1,3
dE/nt                   0,0299      0,0847            0,0267             1,1
udzial G:C              0,0449      0,1269            0,0413             1,1
guanina w petlach       0,0323      0,0914            0,0300             1,1
```

Próg dla różnicy jest większy niż rozrzut pojedynczego przebiegu, bo różnica zbiera szum z obu
modeli: `2 · s · √2`. Odchylenie liczymy **w obrębie konfiguracji**, a dopiero potem uśredniamy po
konfiguracjach — policzone na wszystkich przebiegach naraz zmierzyłoby różnice *między*
konfiguracjami, czyli dokładnie to, co ma być testowane.

### Szum na teście jest cztery razy większy niż na walidacji

Dotyczy to **wyłącznie miar, po których wybieramy epokę** — `zbal_par` i wskaźnik Youdena dla par.
Pozostałe miary mają na obu zbiorach ten sam rozrzut.

Przyczyna jest systematyczna: epokę wybieramy po `zbal_par` **na walidacji**, więc wynik walidacyjny
to maksimum z 60 pomiarów, a maksimum z serii szumiących liczb jest z natury stabilniejsze niż
pojedynczy pomiar. Test nie bierze udziału w żadnej selekcji i pokazuje prawdziwą zmienność
procedury.

**Wniosek metodologiczny: progu nie wolno mierzyć na zbiorze, na którym dokonuje się selekcji** —
zaniża go tam kilkukrotnie. Pierwsza wersja tego pomiaru była liczona na walidacji i dawała próg
`0,35` dla `zbal_par` zamiast `1,51`; kilka różnic wyglądało wtedy na istotne, choć nie są.

Porównania raportujemy **parowane**: różnicę liczymy w obrębie ziarna, bo obie konfiguracje przeszły
przez te same trzy ziarna. To znosi szum wspólny obu przebiegom i jest mocniejsze niż zestawianie
samych średnich. Podajemy przy tym zgodność znaku — różnica wychodząca w tę samą stronę na wszystkich
trzech ziarnach coś znaczy nawet wtedy, gdy jest mała.

⚠️ **Trzy ziarna to mało**; sam próg jest obarczony sporą niepewnością. Sześć ziaren zwęziłoby go
o około 30%, kosztem kolejnych dwóch godzin.

---

## Wyniki

Wszystko na zbiorze testowym, kubełek `<= 200 nt` (728 struktur), podział rodzinowy. Sześć
przebiegów w **identycznych warunkach** — te same dane, wagi 1,0, dekodowanie `argmax` ze
straight-through, pełne 60 epok, epoka wybrana po `zbal_par` na walidacji. Różnią się wyłącznie
karą za skład i tym, czy cross-entropia jest ważona.

| model | ident_nt | ident_par | zbal_par | zbal_zas | dE/nt | Youden | G:C na wyjściu |
|---|---|---|---|---|---|---|---|
| baseline losowy | 25,95% | 33,70% | 33,46% | 25,03% | +0,1273 | +0,0028 | 0,336 |
| E1 kara TV | 28,79% | **44,18%** | 34,43% | 26,49% | −0,0977 | +0,0169 | 0,654 |
| E2 kara promotora | 28,76% | 43,57% | 34,43% | 26,20% | −0,0940 | +0,0161 | 0,653 |
| CE bez kar | **29,88%** | 43,98% | 35,22% | **27,33%** | −0,0563 | +0,0287 | 0,592 |
| E1W TV + ważona CE | 28,11% | 40,68% | 35,30% | 26,42% | +0,0094 | +0,0283 | **0,488** |
| E2W progi + ważona CE | 27,82% | 38,72% | 35,41% | 26,67% | +0,0246 | +0,0292 | **0,488** |
| CEW ważona CE | 27,62% | 38,00% | **35,70%** | 27,26% | +0,1212 | **+0,0361** | 0,333 |
| **poziom losowy** | | | **33,33%** | **25,00%** | | **0** | |
| **REFERENCJA** | | | | | | | **0,484** |
| **próg istotności** | 0,83 | 3,60 | **1,51** | 1,09 | 0,085 | **0,022** | 0,127 |

Zapisane epoki: 45, 54, 41, 30, 29, 29 — wszystkie późne, żadna nie ostatnia.

⚠️ **Ostatni wiersz zmienia czytanie całej tabeli.** Pogrubienia oznaczają najwyższą wartość
w kolumnie, ale na `zbal_par` i na wskaźniku Youdena **cały rozstrzał między modelami mieści się
w progu** — 34,43% i 35,70% to ten sam wynik w granicach zmienności treningu. Różnice
przekraczające próg są tylko na identyczności nukleotydowej i na składzie wyjścia. Szczegóły
w sekcji „Rozkład kar na składniki" niżej.

Wykresy: [notebooks/02_wyniki.ipynb](notebooks/02_wyniki.ipynb).

### Argmax ze straight-through wypadł lepiej niż próbkowanie

Wcześniejsza wersja tych eksperymentów dekodowała przez losowanie z rozkładu. Przejście na `argmax`
z karami liczonymi na twardym wyjściu poprawiło każdy model na każdej mierze:

```
                 probkowanie     argmax    roznica     prog
CE   ident_nt       27,76%       29,88%     +2,12      0,83   <- powyzej progu
CE   zbal_par       35,00%       35,22%     +0,22      1,51
CE   Youden        +0,0256      +0,0287    +0,0031    0,0223
CEW  Youden        +0,0152      +0,0361    +0,0209    0,0223
```

⚠️ Powyżej progu jest tylko **identyczność nukleotydowa**. Pozostałe różnice idą w dobrą stronę, ale
mieszczą się w zmienności przebiegów — a ponieważ pochodzą z pojedynczych przebiegów w dwóch różnych
reżimach, nie są nawet porównaniem parowanym.

Główny argument za `argmax` jest zresztą inny i niezależny od tych liczb: **determinizm**. Nie trzeba
ziarna, a wynik da się odtworzyć co do litery. Do tego dochodzi zniknięcie degeneracji, opisane niżej
— i to jest efekt ogromny, nie subtelny.

### Degeneracja zniknęła

To był główny powód, dla którego wcześniej porzuciliśmy `argmax`:

```
                          G:C na wyjsciu   adenina w petlach
argmax BEZ straight-through     0,984            0,96
argmax ZE straight-through      0,488-0,654      0,15-0,44
REFERENCJA                      0,484            0,309
```

Kara widzi teraz twarde wyjście, więc ma co karać. Widać to w przebiegu treningu `E1`:

```
epoka  1    sklad_pary 0,2875   sklad_petle 0,3795
epoka  3    sklad_pary 0,1816   sklad_petle 0,1067
epoka 60    sklad_pary 0,1381   sklad_petle 0,0652
```

Kara startuje wysoko, bo na starcie wyjście faktycznie jest zdegenerowane, i systematycznie spada.

### Rozkład kar na składniki — kontrola E3

`E1` ma trzy kary i wypada gorzej niż `CE`, które nie ma żadnej:

```
             zbal_par   Youden
E1  z karami  34,43%   +0,0169
CE  bez kar   35,22%   +0,0287
```

Te dwa przebiegi różnią się jednak **trzema rzeczami naraz**, więc z samego ich zestawienia nie
wynika, która odpowiada za różnicę. Kontrola `E3` rozcina to na dwa kroki, z których każdy zmienia
dokładnie jedną rzecz:

```
E1 -> E3    zdejmujemy KARE ZA SKLAD          (zostaja energia i parowania)
E3 -> CE    zdejmujemy ENERGIE i PAROWANIA    (nie zostaje nic)
```

| model | kara za skład | energia + parowania | zbal_par | Youden | ident_nt | dE/nt | G:C |
|---|---|---|---|---|---|---|---|
| E1 | tak | tak | 34,43% | +0,0169 | 28,79% | −0,0977 | 0,654 |
| E3 | — | tak | 34,66% | +0,0183 | 29,44% | −0,0997 | 0,647 |
| CE | — | — | 35,22% | +0,0287 | 29,88% | −0,0563 | 0,592 |

Porównania parowane na teście, po trzy ziarna na konfigurację (`python -m src.szum`):

```
krok                        zbal_par  youden_par  ident_nt      G:C  G w petlach
E1 -> E3   bez kary            +0,74     +0,0113    +1,14     +0,062      -0,039
                             w szumie   w szumie   REALNA    kierunek    kierunek
E3 -> CE   bez energii/parow.  +0,27     +0,0062    -0,08     -0,124      -0,015
                             kierunek   kierunek  w szumie   kierunek    w szumie
                       prog     1,513     0,0223     0,830     0,127       0,091
```

**Na trafności zbalansowanej i na wskaźniku Youdena żaden z tych kroków nie przekracza progu.**
Różnice idą wprawdzie w stronę „bez kar jest lepiej", ale przy rzeczywistym rozrzucie przebiegów nie
da się tego odróżnić od przypadku.

Jedyna różnica istotna to **identyczność nukleotydowa: zdjęcie kary za skład podnosi ją o 1,14
punktu**, zgodnie na wszystkich trzech ziarnach. Kara za skład kosztuje więc identyczność — i to
jest jedyny koszt, jaki udało się zmierzyć.

⚠️ **Dwa sprostowania wobec wcześniejszych wersji tego dokumentu.**

Pierwsze: figurowało tu zdanie „kara za skład szkodzi trafności", oparte na zestawieniu `E1` z `CE`,
które różnią się trzema rzeczami naraz. Kontrola `E3` pokazała, że efektu nie da się przypisać temu
członowi.

Drugie: po kontroli `E3` napisaliśmy, że za spadek trafności odpowiadają energia i parowania. Było to
oparte na progu zmierzonym na **walidacji** (0,47). Przy progu zmierzonym na teście (1,51) ten krok
też jest w szumie. **Na trafności nie mamy istotnego efektu żadnego z członów straty.**

### Ważona cross-entropia zmienia skład, nie trafność

```
para          zbal_par   youden_par   ident_nt      G:C   G w petlach
E1 -> E1W       +0,42       +0,0050      -0,71    -0,135      -0,006
E2 -> E2W       +0,67       +0,0083      -0,83    -0,157      +0,127
CE -> CEW       +0,19       +0,0016      -2,18    -0,292      +0,076
       prog      1,513       0,0223       0,830     0,127       0,091
```

Ważenie **niezawodnie obniża identyczność i udział G:C** — obie te zmiany przekraczają próg w dwóch
albo trzech parach, przy zgodnym kierunku. `CEW` jako jedyny model nie nadprodukuje par G:C (0,333
przy referencji 0,484 — tym razem produkuje ich za mało).

**Wzrost trafności zbalansowanej nie przeżył pomiaru na teście.** Kierunek jest dodatni we wszystkich
trzech parach, ale największa różnica (`+0,67`) to niecała połowa progu. Wcześniejsza wersja tego
dokumentu podawała ten efekt jako najmocniejszy w całym zestawie — przy progu z walidacji tak
wyglądał.

### E1 i E2 są nierozróżnialne

```
miara              roznica     prog
zbal_par            -0,09     1,513
youden_par        -0,0001    0,0223
identycznosc_nt     +0,13     0,830
G:C                +0,034     0,127
```

Żadna różnica nie zbliża się do progu, mimo że obie kary są zbudowane zupełnie inaczej — nasza
dwustronna wobec jednostronnej promotora. **Konstrukcja kary za skład nie ma mierzalnego wpływu na
nic**, co raportujemy.

### Podsumowanie: różnice między konfiguracjami są w większości szumem

Zestawienie wszystkiego, co przekroczyło próg na teście:

```
zmiana                                     co przekroczylo prog
zdjecie kary za sklad (E1 -> E3)           identycznosc +1,14
zdjecie wszystkich kar  (E1 -> CE)         identycznosc +1,06
wazenie klas w CE                          identycznosc -0,7 do -2,2
                                           udzial G:C   -0,14 do -0,29
                                           guanina w petlach +0,13 (E2W)
```

**Na trafności zbalansowanej i na wskaźniku Youdena — czyli na miarach, które miały rozstrzygać —
nie przeżyła ani jedna różnica.** Wszystkie człony straty, obie konstrukcje kary i ważenie klas dają
wyniki nieodróżnialne od siebie w granicach zmienności samego treningu.

To jest główny wynik negatywny pracy i jest zgodny z diagnozą z sekcji o podziale rodzinowym: skoro
model prawie nie ma wiedzy pozycyjnej na nowych rodzinach, nie ma czego poprawiać funkcją straty.
Człony straty zmieniają **skład** wyjścia — i tu ich efekt jest realny i mierzalny — ale nie
zmieniają tego, czy model wie, co postawić na konkretnej pozycji.

### Skład wyjścia

```
model            G:C     A:U     G:U   |       A       C       G       U
E1             0,654   0,272   0,074   |   0,388   0,212   0,141   0,260
E2             0,653   0,260   0,087   |   0,358   0,223   0,117   0,302
CE             0,592   0,319   0,089   |   0,444   0,148   0,069   0,339
E1W            0,488   0,336   0,176   |   0,277   0,278   0,146   0,299
E2W            0,488   0,263   0,248   |   0,237   0,441   0,159   0,163
CEW            0,333   0,422   0,244   |   0,146   0,478   0,159   0,217
baseline       0,336   0,334   0,330   |   0,252   0,246   0,250   0,251
REFERENCJA     0,484   0,371   0,145   |   0,309   0,212   0,205   0,274
```

**`E1W` i `E2W` trafiają w pary niemal idealnie** — obie 0,488 przy referencji 0,484. To ważenie
cross-entropii, nie kara, odpowiada za tę zgodność: `E1` bez ważenia daje 0,654.

**E1 i E2 wypadają teraz niemal identycznie** — 0,654 wobec 0,653 na parach, `zbal_par` obie 34,43%,
Youden +0,0169 wobec +0,0161. Po przejściu na `argmax` ze straight-through różnica między dwiema
konstrukcjami kary **zniknęła**; wcześniej, przy próbkowaniu, E1 wypadał wyraźnie lepiej. Pomiar
szumu to potwierdza: żadna z tych różnic nie przekracza progu — patrz „E1 i E2 są nierozróżnialne"
wyżej.

Konsekwencja praktyczna: przy tym dekodowaniu **kształt kary za skład przestaje mieć znaczenie**. Obie prowadzą do tego samego nadmiaru G:C (0,65 przy referencji 0,484) i do tej samej
trafności.

⚠️ **Guanina jest teraz największym problemem, we wszystkich modelach.** Przy referencji 0,205:

```
E1 0,141    E2 0,117    CE 0,069    E1W 0,146    E2W 0,159    CEW 0,159
```

`CE` ma jej trzykrotnie za mało. To pogorszyło się wraz z przejściem na `argmax` — guanina rzadko
wygrywa w pętlach, a przy losowaniu dostawała swoją część proporcjonalnie do prawdopodobieństwa.

### Kara promotora ma szeroką strefę bezkarną

Progi par sumują się do `0,50 + 0,20 + 0,05 = 0,75`, więc udział G:C może rosnąć aż do 0,75, zanim
A:U spadnie poniżej swojego progu. Policzone wprost ze wzoru (`scratchpad/kiedy_gryzie.py`):

```
G:C na wyjsciu   0,484   0,550   0,645   0,700   0,750   0,850
kara za pary     0,011   0,000   0,000   0,000   0,034   0,206
                         ^^^^^^^^^^^^^^^^^^^^^^ tu kara jest ZEREM
```

Nasze `E2` produkuje 0,653 — siedzi w środku tej strefy i **płaci zero** za pary. Zwiększanie wagi tego nie
zmieni, bo `waga × 0 = 0`. Żeby ciągnąć model w dół od nadmiaru G:C, kara musiałaby mieć górny próg
albo być dwustronna — to drugie jest właśnie wariantem E1.

Kara za zasady też nie pomaga w tę stronę: progi `C >= 0,30` i `G >= 0,30` żądają 60% zawartości GC,
a prawdziwe sekwencje mają C 0,229 i G 0,268. **97% sekwencji testowych łamie co najmniej jeden
próg**, więc kara jest stale aktywna i stale popycha C i G w górę — a każda para G:C wnosi jedno G
i jedno C.

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

### Wagi: wszystkie 1,0, bez dobierania pod wynik

Wcześniejszy przegląd 13 ustawień wag (`python -m src.przeglad`, na walidacji) dał wynik **negatywny**:
różnice między konfiguracjami były rzędu szumu, a waga parowań nie robiła nic w całym zakresie
0 → 12. To była przesłanka, przez którą zdjęliśmy odziedziczoną wagę 6,0.

⚠️ Ten przegląd był liczony w poprzednim reżimie (dekodowanie przez losowanie) i **nie został
powtórzony po przejściu na `argmax`**. Jego wyniku nie cytujemy więc liczbowo — służy tylko jako
uzasadnienie, dlaczego wszystkie wagi stoją na 1,0 i dlaczego żadna nie jest dopasowana pod wynik.

---

Kary w tabelach liczone są na **wygenerowanych sekwencjach**.

## Znane ograniczenia

**Cel składu par jest wyższy niż skład zbioru testowego.** Cel wynosi `G:C 0,551` (średnia z całej
puli), a test ma 0,484 — kara prowadzi więc model do wartości przesuniętej o niecałe siedem punktów.
To nie jest wada implementacji, tylko konsekwencja podziału rodzinowego: trening ma 0,600 i stanowi
57% wszystkich par, więc ciągnie średnią w górę. Poprzednia wersja celu, mierzona na bazach
zewnętrznych, wynosiła 0,599 i przesunięcie było dwa razy większe.

**Kara za skład i tak nie steruje udziałem G:C** — pokazuje to kontrola `E3` wyżej — więc to
ograniczenie jest w praktyce łagodniejsze, niż na to wygląda.

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
