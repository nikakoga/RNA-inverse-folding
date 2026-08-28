# Model i sposób trenowania

## Zadanie

Odwrotne zwijanie RNA: na wejściu **struktura drugorzędowa** w notacji kropkowo-nawiasowej,
na wyjściu **sekwencja nukleotydów**, która ma się w tę strukturę zwinąć.

```
wejscie   ((((((....))))))
wyjscie   GCGCGCAAAAGCGCGC
```

## Architektura — enkoder-only, nieautoregresyjna

[`src/model.py`](src/model.py). Transformer **bez dekodera**, 5,04 mln parametrów: 6 warstw enkodera,
`d_model` 256, 8 głów uwagi, `dim_feedforward` 1024, dropout 0,1, normalizacja przed warstwą.

```
wejscie    symbole . ( ) + embedding ZNAKOWANEJ odleglosci do partnera
enkoder    6 warstw; kazda pozycja widzi wszystkie pozostale, w obie strony
wyjscie    JEDEN przebieg, wszystkie pozycje naraz

glowica PAR    dla pary (i,j): zlaczenie reprezentacji h_i i h_j -> jedna z 6 klas
               G-C, C-G, A-U, U-A, G-U, U-G
glowica ZASAD  dla pozycji niesparowanej -> jedna z 4 zasad
```

### Dlaczego bez dekodera

Model autoregresyjny pisze sekwencję lewo-do-prawa, więc pozycja 30 nie wie, co stanie na pozycji 3,
mimo że obie tworzą **parę** i ich wybór jest z definicji wspólny. Wymusza to maskowanie kanoniczności
po fakcie oraz teacher forcing, przez który funkcja straty mierzy inną sytuację niż ta, w której model
jest oceniany.

Tutaj para jest **jedną decyzją**: głowica dostaje reprezentacje obu końców naraz i wybiera jedną
z sześciu klas kanonicznych.

### Kanoniczność jest własnością wyjścia, nie maski

Klasy „A-C" po prostu nie ma w zbiorze wyjściowym, więc **niekanoniczna para jest niemożliwa do
wyprodukowania**. Nie trzeba nakładać maski po fakcie, a co za tym idzie — funkcja straty i generowanie
patrzą na dokładnie ten sam rozkład.

### Znika odchylenie ekspozycji

Nie ma autoregresji, więc nie ma teacher forcingu (ground-truth-guided decoding podczas treningu).
Model podczas uczenia i podczas generowania dostaje dokładnie to samo wejście — pełną strukturę —
więc człony energetyczne można wstawić wprost do funkcji straty i optymalizować gradientem.

Zostaje jedna, znacznie mniejsza rozbieżność: trening liczy komponenty na **rozkładach softmax**,
a generowanie musi wyprodukować konkretne litery. Wracamy do tego niżej.

### Ograniczenie: holistycznie, ale tylko raz

Każda pozycja widzi wszystkie pozostałe, w obie strony — i to jest realna przewaga. Ale jest to
**jeden przebieg, jedna decyzja, koniec**. Model nie wraca do wcześniejszych wyborów i ich nie
rewiduje; „holistycznie" znaczy tu „z pełnym kontekstem", a nie „iteracyjnie".

Konsekwencji spodziewamy się przede wszystkim na strukturach długich, gdzie jeden nietrafiony wybór
nie ma jak zostać naprawiony.

### Dekodowanie: dlaczego losujemy, a nie bierzemy maksimum

Komponenty energii, parowania i składu liczymy na **miękkich rozkładach** (softmax) podczas treningu.
Generowanie musi z nich zrobić konkretne litery i sposób, w jaki to robi, decyduje o wyniku.

`argmax` bierze klasę najbardziej prawdopodobną. To zawodzi, bo model prawie nie korzysta z wejścia
i produkuje **niemal ten sam płaski rozkład na każdej pozycji**. Zwycięzca jest wtedy wszędzie ten
sam, więc `argmax` zamienia rozkład w punkt. Zmierzone na głowie par: pewność zwycięzcy 0,34–0,43
przy 0,167 dla jednostajnego, przewaga nad drugim 0,11–0,17 — i mimo to rodzina G:C wygrywa na
91–99,9% pozycji.

Podział na **sześć** klas nie jest tu winny, choć wygląda podejrzanie. `argmax` liczony po trzech
typach, z masami orientacji zsumowanymi, daje ten sam albo większy nadmiar G:C.

**Losujemy z rozkładu** (`--dekodowanie probkowanie`), bo oczekiwany skład wylosowanej sekwencji
**równa się** składowi rozkładu. Wartość kary raportowana w treningu opisuje więc dokładnie to, co
widać na wygenerowanej sekwencji. Ceną jest utrata determinizmu — `--seed-dekodowania` jest częścią
wyniku, bez niego nic nie da się odtworzyć.

---

## Funkcja straty

```
strata = CE  (opcjonalnie WAZONA odwrotnie do czestosci klas, --wagi-klas)
       + w_energia      * energia
       + w_parowania    * parowania
       + KARA ZA SKLAD, jeden z dwoch wariantow:
           w_sklad       * sklad           odleglosc TV      (E1)
           w_sklad_zasad * sklad_zasad  \  progi dolne       (E2)
           w_sklad_par   * sklad_par    /
```

[`src/loss.py`](src/loss.py). Wagi kar zerowe domyślnie — czysta cross-entropia jest punktem
odniesienia. Warianty kary za skład są **alternatywne**: włącza się jeden albo drugi.

Dwie niezależne decyzje dają sześć eksperymentów opisanych w [EKSPERYMENTY.md](EKSPERYMENTY.md):

```
             kara TV     kara progowa    brak kary
CE zwykla       E1            E2            CE
CE wazona       E1W           E2W           CEW
```

### CE — cross-entropia

Kara za niepewność wobec prawdziwej odpowiedzi. Model dla każdej pozycji podaje prawdopodobieństwa.
Jeśli prawdziwa litera to A, a model dał jej 0,7 — kara mała; jeśli 0,1 — kara duża. To z tego członu
model uczy się, **co w jakim motywie faktycznie występuje** (np. że w pętlach spinek jest więcej
adeniny), bo widzi z kontekstu, w jakim motywie stoi.

**CE jest jedynym członem, który ogląda sekwencję referencyjną.** Energia, parowania i skład dają się
policzyć bez jej znajomości, więc nie mają jak przekazać informacji o tym, która para należy do
której pozycji. Żadna waga tego nie zmieni, bo waga steruje siłą, a nie rodzajem informacji.

Liczona jako **suma dwóch średnich** — osobno dla głowicy par i głowicy zasad — żeby obie miały równy
wpływ na gradient niezależnie od tego, ile pozycji obsługują. Do logu trafia natomiast jedna średnia
po wszystkich pozycjach, tak samo jak w walidacji; inaczej „CE treningowa obok walidacyjnej"
porównywałaby wielkości w różnej skali i przeuczenie wyglądałoby łagodniej, niż jest.

### Ważona CE — odpowiednik trafności zbalansowanej po stronie uczenia

`--wagi-klas`. Zwykła CE optymalizuje trafienia ważone liczebnością, więc klasa G:C (60% par
treningowych) wnosi do gradientu osiem razy więcej niż G:U (5%). Modelowi opłaca się wtedy zbudować
wokół klasy najczęstszej — czyli dokładnie ta awaria, którą mierzymy.

Wagi `1/częstość`, liczone **na treningu**, znormalizowane do średniej 1, więc skala straty się nie
zmienia i wagi pozostałych członów zachowują znaczenie. Zmierzone:

```
pary   G-C 0,30   C-G 0,39   A-U 0,66   U-A 0,71   G-U 2,07   U-G 1,88
petle  A 0,76   C 1,22   G 1,16   U 0,86
```

Relacja do miar: `zbal_par` wyrównuje wkład klas w **pomiarze**, ważona CE wyrównuje go
w **gradiencie**. To ta sama idea po dwóch stronach.

**CE w logu pozostaje nieważona**, żeby dało się ją zestawiać z walidacyjną i z innymi przebiegami.
Wagi zmieniają to, co model optymalizuje, a nie to, czym go mierzymy.

Częstości pochodzą wyłącznie z treningu — sięgnięcie po walidację albo test byłoby zaglądaniem
w odpowiedź.

### Energia — człony sekwencyjne modelu Turnera

Oczekiwana energia struktury docelowej, liczona na rozkładach prawdopodobieństwa modelu. Trzy składniki:

```
stosy par              tablica 6x6 klas, pelna orientacja pary
kary terminalne AU/GU  na ZEWNETRZNYM koncu helisy oraz przy tripetlach
niedopasowanie spinki  odczyt z RNA.E_Hairpin, dla petli od 4 niesparowanych wzwyz
```

Człony zależne wyłącznie od **kształtu** struktury (kara za wielkość pętli) są pomijane celowo —
w różnicy wobec sekwencji referencyjnej skracają się do zera.

**Weryfikacja.** Komponent sprawdzony na rozkładach one-hot przez porównanie różnic między dwiema
sekwencjami na tej samej strukturze. Średni błąd **0,000 kcal/mol** na trzech różnych strukturach.
Dwie pułapki znalezione i naprawione po drodze: podwójne naliczanie kary terminalnej na końcu
zamykanym spinką (błąd dokładnie `TerminalAU` = 0,50) oraz tripętle, które model Turnera traktuje
osobno (błąd 0,28).

Tablicy `mismatchH` nie da się indeksować z Pythona (to `SwigPyObject`), ale oficjalna funkcja
`RNA.E_Hairpin` czyta tę samą tablicę — nie reimplementujemy niczego.

### Parowania — ile sekwencja dopuszcza alternatyw

```
kombinacje = (ile G)*(ile C) + (ile A)*(ile U) + (ile G)*(ile U)
```

Liczba par, jakie w tej sekwencji dałoby się utworzyć, policzona z samych liczebności liter. Każde
dodatkowe możliwe parowanie to dodatkowy sposób, na jaki cząsteczka może zwinąć się **nie tak, jak
chcemy**. Dzielimy przez kwadrat długości, bo iloczyn liczebności rośnie kwadratowo.

**Ograniczenie:** wzór jest ślepy na pozycje. Traktuje guaninę z pozycji 3 jako mogącą sparować
z cytozyną z pozycji 90, choć dzieli je 87 nukleotydów.

### Skład — dwa warianty, które porównuje E1 kontra E2

Po co ten człon: energia i parowania obie premiują mocne, jednoznaczne helisy, a człon parowań ma
**trywialne minimum** — wypełnienie pętli adeniną zeruje `G·C`, `A·U` i `G·U` naraz, bo z pętli
znikają C, U i G. Kara za skład jest jedyną przeciwwagą.

Obie wersje liczą się **per sekwencja**, potem uśredniamy po partii. Różnią się kształtem.

**Wariant E1 — odległość całkowitego wahania** (`TV = ½·L1`, zakres 0–1) od składu naturalnego,
osobno dla dwóch grup:

```
TYPY PAR              cel  G:C 0,599   A:U 0,276   G:U 0,124
ZASADY W PETLACH      cel  A 0,324  C 0,208  G 0,217  U 0,252
```

Kara **dwustronna**: nadmiar boli tak samo jak niedobór. Cel pochodzi ze stałych w `src/loss.py` —
tego samego źródła, z którego losuje baseline.

**Wariant E2 — progi dolne** ze specyfikacji promotora:

```
ZASADY W CALEJ SEKWENCJI   progi  A 0,15  C 0,30  G 0,30  U 0,15
TYPY PAR                   progi  G:C 0,50  A:U 0,20  G:U 0,05  + eskalacja DistribLoss4
```

Kara **jednostronna**: `max(prog − udzial, 0) / prog`, więc nadmiar jest bezkarny. To są więzy
projektowe w konwencji DesiRNA, a nie opis natury.

**Różnica, o której łatwo zapomnieć:** wariant E1 patrzy na zasady **w pętlach**, a wariant E2
na zasady **w całej sekwencji**, razem ze sparowanymi. E2 nie potrafi więc powiedzieć „za mało
adeniny w pętlach", tylko „za mało adeniny w ogóle".

**Ograniczenie wspólne dla obu.** Kara liczy się na miękkich rozkładach, a nie na gotowej sekwencji.
Przy losowaniu z rozkładu obie wielkości się pokrywają, ale każde inne dekodowanie tę zgodność łamie
— przy `argmax` model z rozkładem wyglądającym naturalnie stawiał G-C na prawie każdej pozycji.

**Ograniczenie wspólne dla obu, drugie.** Wszystkie typy pętli dzielą jeden cel: spinka, wybrzuszenie,
multipętla i regiony zewnętrzne, mimo że w naturze różnią się składem. Rozdzielenie ich per sekwencja
jest niewykonalne — mediana liczby pozycji w wybrzuszeniu i multipętli to sześć, a jedna trzecia
sekwencji nie ma ich wcale.

---

## Trening

[`src/train.py`](src/train.py).

```
optymalizator   AdamW, lr 3e-4, weight decay 0,01
harmonogram     cosine annealing przez cala dlugosc treningu
batch           32 struktury
epoki           60, BEZ wczesnego zatrzymania
przycinanie     norma gradientu <= 1,0
```

Jedna epoka trwa około 6 sekund na RTX 4060, więc pełny trening to sześć minut.

### Dlaczego bez wczesnego zatrzymania

Domyślnie `--cierpliwosc` równa się `--epoki`, więc próg nigdy się nie uruchamia. Dwa powody:

**`zbal_par` szumi za bardzo, żeby „przestało się poprawiać" cokolwiek znaczyło.** Miara stoi tuż nad
poziomem losowym i skacze o ±0,3 pp między epokami, więc jej maksimum może wypaść zanim model
czegokolwiek się nauczy. Przy cierpliwości 10 trening E1 zatrzymał się w epoce 11 i zapisał **epokę
pierwszą** — rekord padł na starcie i przez dziesięć epok nikt go nie pobił. Przebiegi kończyły się
wtedy po 11, 26, 38, 43 i 60 epokach, więc porównanie kar mierzyło głównie czas uczenia.

**Harmonogram kroku uczenia jest rozpisany na 60 epok.** Cosine schodzi do zera dopiero w ostatniej;
przerwanie w epoce 11 zostawia krok na ~97% wartości początkowej, czyli model nigdy nie dostaje fazy
dostrajania małymi krokami.

Ochrona przed przeuczeniem nie zniknęła — przeniosła się w całości na **wybór epoki**, który jest
silniejszy, bo ogląda wszystkie 60 epok zamiast zatrzymywać się na pierwszym płaskowyżu. Widać, że
działa: w sześciu przebiegach zapisane epoki to 28–51, żadna nie jest ostatnia.

### Wybór najlepszej epoki — miejsce, w którym łatwo się pomylić

Po każdej epoce liczymy **wszystkie** kryteria na pełnym zbiorze walidacyjnym i wypisujemy je do logu;
`--wybor` decyduje tylko, które z nich zapisuje checkpoint. Żadne nie przewiduje struktury.

```
--wybor zbal_par          DOMYSLNE; srednia czulosc po 3 typach par  (poziom losowy 33,3%)
--wybor zbal_zasady       srednia czulosc po 4 zasadach w petlach    (poziom losowy 25,0%)
--wybor youden_GC         czulosc + specyficznosc - 1 dla typu G:C   (poziom losowy 0)
--wybor youden_par        to samo, usrednione po 3 typach par        (poziom losowy 0)
--wybor youden_zasady     to samo, usrednione po 4 zasadach          (poziom losowy 0)
--wybor identycznosc_nt   ulamek pozycji trafionych wobec sekwencji referencyjnej
--wybor ce                srednia cross-entropia na walidacji
--wybor loss              PELNA strata tego modelu, z jego wlasnymi karami
--wybor energia           o ile stabilizujemy cel lepiej niz sekwencja referencyjna
--wybor zlozony           identycznosc jako klucz glowny, dE/nt jako rozstrzygacz remisow
```

**Dlaczego domyslne przestalo byc `zlozony`.** Oba jego czlony premiuja nadprodukcje klasy
najczestszej. Identycznosc jest maksymalizowana przez staly predyktor — sekwencja z samych par G:C
dostaje 48,4%, wiecej niz ktorykolwiek z naszych modeli. A `dE/nt` spada, gdy udzial G:C rosnie, bo
G:C jest para najstabilniejsza. Wybieralismy wiec epoke miara, ktora nagradza dokladnie te awarie,
ktora eksperyment ma zmierzyc.

**Dlaczego nie ma samej specyficznosci.** Specyficznosc G:C rosnie do 100%, gdy model PRZESTAJE
wystawiac G:C — model z samych A:U mialby ja idealna. Jest wiec podatna na predyktor staly tak samo
jak identycznosc, tylko z drugiej strony. Wchodzi zamiast niej wskaznik Youdena, ktory laczy czulosc
ze specyficznoscia:

```
J = czulosc + specyficznosc - 1
```

Jego poziom odniesienia to DOKLADNIE ZERO dla kazdego modelu przypisujacego klasy niezaleznie od
pozycji, bez wzgledu na sklad wyjscia — taki model ma czulosc rowna q, a specyficznosc rowna 1 - q,
wiec suma zawsze wychodzi 1. Nie da sie go podbic przesunieciem skladu w zadna strone. To jedyna
nasza miara z zerem jako poziomem losowym; reszta ma 1/k, co gorzej sie czyta.

Sekwencje walidacyjne generujemy **raz na epokę**, z ustalonym ziarnem, i z nich liczymy identyczność,
czułości klasowe oraz `dE/nt`. Stałe ziarno sprawia, że porównujemy epoki, a nie losowania.

**Kryterium musi być takie samo we wszystkich porównywanych przebiegach**, inaczej
porównanie kar traci sens. Wszystkie sześć eksperymentów używa `zbal_par`.

**Youden czy `zbal_par`?** Do wyboru epoki są praktycznie wymienne — sprawdzone na wszystkich sześciu
przebiegach, korelacja `0,995–0,999`, a tam gdzie wskazują różne epoki, różnica w jakości wynosi
0,00–0,08 pp. Powód jest algebraiczny: rozpisując definicję, `J_k = (czułość_k − udział_k na
wyjściu) / (1 − częstość_k)`, a `zbal_par` to średnia z samych czułości. Youden dokłada tylko
odjęcie udziału i normalizację przez rzadkość klasy. **Do raportowania jest jednak lepszy**, bo jego
poziom odniesienia to zero, więc czyta się go bez dopowiadania „przy losowym 33,3%".

**Dlaczego `loss` nie jest domyślne**, mimo że jest standardem w uczeniu maszynowym: nasza strata
zawiera człon parowań z trywialnym minimum. Wybieranie epoki o najniższej stracie może więc
systematycznie wskazywać epokę najbardziej zdegenerowaną — czyli dokładnie tę awarię, którą
eksperyment ma zmierzyć. Do tego `loss` znaczy co innego w każdym z sześciu przebiegów, bo obejmuje
inną karę i inne ważenie CE.

`identycznosc_nt` i `ce` są zewnętrzne wobec kar i identyczne dla wszystkich eksperymentów. Różnią
się tym, że pierwsza liczy się na wygenerowanej sekwencji, a druga na rozkładach, więc widzi też
pewność modelu. `energia` premiuje mocne helisy, a nie podobieństwo do biologii — te cele ciągną
w różne strony.

---

## Ocena

[`src/evaluate.py`](src/evaluate.py). Dwa zbiory, **żadna miara nie przewiduje struktury**.

```
TEST NATURALNY     20% puli, rodziny nieobecne w treningu. Ocena glowna.
ETERNA <= 200 nt   39 zagadek projektowych ludzi. Pomocnicza, spoza naszych danych.
```

| miara | co znaczy | rola |
|---|---|---|
| `zbal_par`, `zbal_zasady` | czułość uśredniona po klasach, bez wagi | **rozstrzyga** |
| Youden | czułość + specyficzność − 1, per klasa i uśredniony | **rozstrzyga** |
| `identycznosc_nt` | ułamek **pozycji** z tą samą zasadą co referencja | pomocnicza |
| `identycznosc_par` | ułamek **par** z trafionym TYPEM; `G-C` i `C-G` to jedno trafienie | pomocnicza |
| czułość i specyficzność per klasa | razem wykrywają nadprodukcję jednej klasy | diagnostyka |
| `dE/nt` | o ile stabilizujemy cel lepiej niż sekwencja referencyjna | kontrola |
| `kara_wlasna` | wartość kary za skład tego modelu | diagnostyka |

**Identyczności zeszły do roli pomocniczej i trzeba wiedzieć dlaczego.** Dla predyktora
przypisującego klasy niezależnie od pozycji `identycznosc_par` wynosi dokładnie
`Σ udział_k · częstość_k`, czyli da się ją policzyć z samego składu wyjścia. Zmierzone na naszych
modelach: przewidziana z samego składu zgadza się z rzeczywistą do 1–2 pp. Sekwencja z samych par
G:C dostaje 48,4%, więcej niż którykolwiek z modeli. Ta miara premiuje więc trafienie w skład,
a nie wiedzę o tym, gdzie która para stoi.

**Dwie identyczności mają różne mianowniki.** Architektura wymusza parę wszędzie tam, gdzie struktura
jej żąda, więc model nie może postawić pary w złym miejscu ani jej pominąć — rozliczanie go z
rozmieszczenia par nie miałoby sensu. Da się ocenić tylko, **który typ** wybrał, a naturalną jednostką
jest tam para, nie pozycja. Z definicji `identycznosc_par >= identycznosc_nt`, a różnica między nimi
to błędy samej orientacji.

**`kara_wlasna` to inna wielkość w każdym wariancie kary** — liczb z E1 i E2 nie wolno zestawiać ze
sobą. Odpowiada na pytanie „czy moja kara zrobiła to, co obiecywała", a nie „która kara jest lepsza".
Do porównywania modeli między sobą służy odległość TV od składu **referencyjnego** — miara neutralna,
niezwiązana z celem żadnej z kar.

**Czego tu nie ma:** `rozwiazane` i `F1`. Obie wymagają przewidzenia struktury RNAfoldem, który ma
własny sufit dokładności, więc poprawnie zaprojektowana sekwencja bywa uznawana za błędną z powodu
ograniczeń narzędzia, a nie modelu. Konsekwencja, którą trzeba znać: nie odpowiadamy na pytanie
„czy ta sekwencja się zwinie".

### Baseline

Sekwencja losowana **jednostajnie**, z zachowaniem komplementarności par
(`src/dataset.py::losowa_kanoniczna`): w miejscu pary dowolny z trzech typów po 1/3, na pozycji
niesparowanej dowolna zasada po 1/4. Mierzy, ile da się ugrać **samą komplementarnością zasad**,
bez żadnej wiedzy o RNA.

Losowanie ważone naturalnymi częstościami dałoby baseline'owi za darmo wiedzę o składzie, którą model
musi wyciągnąć z danych — dlatego go nie stosujemy.

```
python -m src.evaluate --baseline --na test
```
