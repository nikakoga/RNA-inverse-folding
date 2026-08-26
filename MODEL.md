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
a generowanie wybiera **argmax**. Wracamy do tego niżej.

### Ograniczenie: holistycznie, ale tylko raz

Każda pozycja widzi wszystkie pozostałe, w obie strony — i to jest realna przewaga. Ale jest to
**jeden przebieg, jedna decyzja, koniec**. Model nie wraca do wcześniejszych wyborów i ich nie
rewiduje; „holistycznie" znaczy tu „z pełnym kontekstem", a nie „iteracyjnie".

Konsekwencji spodziewamy się przede wszystkim na strukturach długich, gdzie jeden nietrafiony wybór
nie ma jak zostać naprawiony.

### Ograniczenie: softmax w treningu, argmax w generowaniu

Komponenty energii, parowania i składu liczymy na **miękkich rozkładach** (softmax) podczas treningu.
Podczas generowania wybieramy **argmax** — twardą decyzję. Jeśli rozkład jest rozmyty (high-entropy),
oczekiwana energia może być niska, ale rzeczywista energia wygenerowanej sekwencji mogłaby być wyższa.
To jest potencjalny problem dla komponentów o silnych preferencjach.

---

## Funkcja straty

```
strata = CE
       + w_energia      * energia
       + w_parowania    * parowania
       + KARA ZA SKLAD, jeden z dwoch wariantow:
           w_sklad       * sklad           odleglosc TV      (E1)
           w_sklad_zasad * sklad_zasad  \  progi dolne       (E2)
           w_sklad_par   * sklad_par    /
```

[`src/loss.py`](src/loss.py). Wagi zerowe domyślnie — czysta cross-entropia jest punktem odniesienia.

Warianty kary za skład są **alternatywne**: włącza się jeden albo drugi. Na tym polega cała różnica
między E1 a E2, opisana w [EKSPERYMENTY.md](EKSPERYMENTY.md).

### CE — cross-entropia

Kara za niepewność wobec prawdziwej odpowiedzi. Model dla każdej pozycji podaje prawdopodobieństwa.
Jeśli prawdziwa litera to A, a model dał jej 0,7 — kara mała; jeśli 0,1 — kara duża. To z tego członu
model uczy się, **co w jakim motywie faktycznie występuje** (np. że w pętlach spinek jest więcej
adeniny), bo widzi z kontekstu, w jakim motywie stoi.

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

**Ograniczenie wspólne dla obu.** Kara liczy się na miękkich rozkładach, a sekwencja powstaje przez
`argmax`. Model może mieć rozkład wyglądający naturalnie i przy tym na każdej pozycji wskazywać G-C.

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
epoki           60, early stopping po 10 bez poprawy
przycinanie     norma gradientu <= 1,0
```

Jedna epoka trwa około 7 sekund na RTX 4060, więc pełny trening to kilka minut.

### Wybór najlepszej epoki — miejsce, w którym łatwo się pomylić

Po każdej epoce liczymy kryterium na **pełnym zbiorze walidacyjnym** i zapisujemy checkpoint tylko
wtedy, gdy się poprawiło. Żadne z czterech kryteriów nie przewiduje struktury.

```
--wybor identycznosc_nt   domyslne; ulamek pozycji trafionych wobec prawdziwej sekwencji
--wybor ce                srednia cross-entropia na walidacji
--wybor loss              PELNA strata tego modelu, z jego wlasnymi karami
--wybor energia           o ile stabilizujemy cel lepiej niz prawdziwa sekwencja
```

**Kryterium musi być takie samo w E1 i E2**, inaczej porównanie kar traci sens.

**Dlaczego `loss` nie jest domyślne**, mimo że jest standardem w uczeniu maszynowym: nasza strata
zawiera człon parowań z trywialnym minimum. Wybieranie epoki o najniższej stracie może więc
systematycznie wskazywać epokę najbardziej zdegenerowaną — czyli dokładnie tę awarię, którą
eksperyment ma zmierzyć. Do tego `loss` znaczy co innego w E1 i w E2, bo obejmuje inną karę.

`identycznosc_nt` i `ce` są zewnętrzne wobec obu kar i identyczne dla obu eksperymentów. Różnią się
tym, że pierwsza liczy się na `argmax`, a druga na rozkładach, więc widzi też pewność modelu.
`energia` premiuje mocne helisy, a nie podobieństwo do biologii — te cele ciągną w różne strony.

---

## Ocena

[`src/evaluate.py`](src/evaluate.py). Dwa zbiory, pięć miar, **żadna nie przewiduje struktury**.

```
TEST NATURALNY     20% puli, rodziny nieobecne w treningu. Ocena glowna.
ETERNA <= 200 nt   39 zagadek projektowych ludzi. Pomocnicza, spoza naszych danych.
```

| miara | co znaczy | rola |
|---|---|---|
| `identycznosc_nt` | ułamek **pozycji** z tą samą zasadą co wzorzec | rozstrzyga |
| `identycznosc_par` | ułamek **par** z trafionym TYPEM; `G-C` i `C-G` to jedno trafienie | rozstrzyga |
| `dE/nt` | o ile stabilizujemy cel lepiej niż prawdziwa sekwencja | kontrola |
| `perplexity` | z ilu zasad model średnio wybiera (1 = pewny, 4 = niezdecydowany) | kontrola |
| `kara_wlasna` | wartość kary za skład tego modelu | diagnostyka |

**Dwie identyczności mają różne mianowniki.** Architektura wymusza parę wszędzie tam, gdzie struktura
jej żąda, więc model nie może postawić pary w złym miejscu ani jej pominąć — rozliczanie go z
rozmieszczenia par nie miałoby sensu. Da się ocenić tylko, **który typ** wybrał, a naturalną jednostką
jest tam para, nie pozycja. Z definicji `identycznosc_par >= identycznosc_nt`, a różnica między nimi
to błędy samej orientacji.

**`kara_wlasna` to inna wielkość w E1 i w E2** — tych liczb nie wolno zestawiać ze sobą.

**Czego tu nie ma:** `rozwiazane` i `F1`. Obie wymagają przewidzenia struktury RNAfoldem, który ma
własny sufit dokładności, więc poprawnie zaprojektowana sekwencja bywa uznawana za błędną z powodu
ograniczeń narzędzia, a nie modelu. Konsekwencja, którą trzeba znać: nie odpowiadamy na pytanie
„czy ta sekwencja się zwinie".

### Baseline

Sekwencja losowana z naturalnych częstości, z zachowaniem komplementarności par
(`src/dataset.py::losowa_kanoniczna`). Mierzy, ile da się ugrać **samą komplementarnością zasad**,
bez żadnego uczenia.

```
python -m src.evaluate --baseline --na test
```
