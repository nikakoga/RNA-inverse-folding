# Model i sposób trenowania

## Zadanie

Odwrotne zwijanie RNA: na wejściu **struktura drugorzędowa** w notacji kropkowo-nawiasowej,
na wyjściu **sekwencja nukleotydów**, która ma się w tę strukturę zwinąć.

```
wejscie   ((((((....))))))
wyjscie   GCGCGCAAAAGCGCGC
```

## Architektura — enkoder-only, nieautoregresyjna

[`src/model.py`](src/model.py). Transformer **bez dekodera**, 5,07 mln parametrów: 6 warstw enkodera,
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

Nie ma autoregresji, więc nie ma teacher forcingu: uczenie i generowanie to ten sam tryb. Dzięki temu
człony energetyczne można wstawić wprost do funkcji straty. **Uczenie ze wzmocnieniem nie jest
potrzebne** — było wyłącznie obejściem tego problemu.

### Ograniczenie: holistycznie, ale tylko raz

Każda pozycja widzi wszystkie pozostałe, w obie strony — i to jest realna przewaga. Ale jest to
**jeden przebieg, jedna decyzja, koniec**. Model nie wraca do wcześniejszych wyborów i ich nie
rewiduje; „holistycznie" znaczy tu „z pełnym kontekstem", a nie „iteracyjnie".

Inaczej działają **modele dyfuzyjne**: generują w wielu krokach, zaczynając od szumu i stopniowo go
odszumiając, więc każda iteracja może zmienić to, co ustalono wcześniej. Tam rewizja jest wbudowana
w sposób generowania. U nas nie ma jej wcale i jest to widoczne w wynikach na strukturach powyżej
100 nt.

---

## Funkcja straty

```
strata = CE
       + w_energia   * energia
       + w_parowania * parowania
       + w_sklad     * sklad
```

[`src/loss.py`](src/loss.py). Wagi zerowe domyślnie — czysta cross-entropia jest punktem odniesienia.

### CE — cross-entropia

Kara za niepewność wobec prawdziwej odpowiedzi. Model dla każdej pozycji podaje prawdopodobieństwa.
Jeśli prawdziwa litera to A, a model dał jej 0,7 — kara mała; jeśli 0,1 — kara duża. To z tego członu
model uczy się, **co w jakim motywie faktycznie występuje** (np. że w pętlach spinek jest więcej
adeniny), bo widzi z kontekstu, w jakim motywie stoi.

### Energia — człony sekwencyjne modelu Turnera

Oczekiwana energia struktury docelowej, liczona na rozkładach prawdopodobieństwa modelu. Trzy składniki:

```
stosy par            tablica 6x6 klas, pelna orientacja pary
kary terminalne AU   na końcu helisy w REGIONACH ZEWNETRZNYCH
niedopasowanie spinki  odczyt z RNA.E_Hairpin
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

### Skład — odległość od naturalnego

Odległość całkowitego wahania (`TV = ½·L1`, zakres 0–1) między rozkładem produkowanym przez model
a naturalnym, liczona **osobno** dla dwóch grup:

```
TYPY PAR              cel  G:C 0,555   A:U 0,321   G:U 0,125
ZASADY W PETLACH      cel  A 0,330  C 0,198  G 0,213  U 0,259
```

Cel pochodzi ze stałych w `src/loss.py`, zmierzonych na naturalnych sekwencjach RNA. To samo źródło,
z którego losuje baseline — dzięki temu obie strony porównania mówią o tym samym rozkładzie.

**Znane ograniczenie tego członu.** Kara liczy się na **miękkich rozkładach prawdopodobieństwa
uśrednionych po partii**, a sekwencja powstaje przez `argmax`. Model może mieć rozkład, który średnio
wygląda naturalnie, i przy tym na każdej pozycji wskazywać G-C — kara jest wtedy spełniona, a wyjście
zdegenerowane. We wcześniejszej fazie pracy zmierzono to wprost: przy silnej wadze kara w treningu spadała
pięciokrotnie, a odległość składu w wygenerowanych sekwencjach **rosła** dwukrotnie. Do przeliczenia
na nowych danych — patrz [EKSPERYMENTY.md](EKSPERYMENTY.md).

**Wszystkie trzy pętle: również ograniczenie.** Spinka, wybrzuszenie i multipętla dzielą jeden cel,
mimo że w naturze różnią się składem.

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

Domyślnie zapisujemy epokę o najlepszym **odzysku** (ułamek pozycji trafionych wobec prawdziwej
sekwencji), bo ta miara nie wymaga zwijania.

**To jest pułapka.** Odzysk mierzy podobieństwo do sekwencji naturalnej, a rozwiązywanie premiuje
mocne, jednoznaczne helisy — te cele są ze sobą sprzeczne, więc wybieranie epoki po odzysku
systematycznie zapisuje ten wariant modelu, który rozwiązuje najgorzej.

We wcześniejszej fazie pracy, na innym zbiorze, zmierzona korelacja Pearsona wynosiła **−0,79**.
Tutaj jest do przeliczenia na nowych danych, ale sam mechanizm wynika z konstrukcji miar, nie
z konkretnego zbioru.

```
--wybor odzysk    domyslne, bez zwijania, ale sprzeczne z rozwiazywaniem
--wybor solved    najlepiej przewiduje rozwiazywanie, ale UZYWA RNAfolda do wyboru epoki
--wybor energia   ZERO zwijania, rozsadny kompromis
```

---

## Ocena

[`src/evaluate.py`](src/evaluate.py). Dwa zbiory i trzy miary.

```
TEST NATURALNY    20% puli, rodziny nieobecne w treningu. Ocena glowna.
ETERNA <= 50 nt   zagadki projektowe ludzi. Pomocnicza, spoza naszych danych.
```

| miara | co znaczy | używa RNAfolda |
|---|---|---|
| `rozwiazane` | czy nasza sekwencja **zwija się** w zadaną strukturę | tak |
| `odzysk` | ułamek pozycji trafionych wobec prawdziwej sekwencji | nie |
| `dE/nt` | o ile stabilizujemy cel lepiej niż prawdziwa sekwencja | nie |

**`rozwiazane` nie porównuje sekwencji z odpowiedzią.** Sekwencja całkiem inna od wzorcowej może
rozwiązać zagadkę, jeśli tylko zwija się poprawnie. To rozróżnienie jest kluczowe: sekwencja z samych
par G:C zwija się bardzo niezawodnie, mając może 30% liter wspólnych ze wzorcem.

`F1` celowo pomijamy — rozdaje punkty częściowe za pojedyncze pary, więc dziedziczy błąd RNAfolda
na każdej z nich i jest trudne do odczytu.

### Baseline

Sekwencja losowana z naturalnych częstości, z zachowaniem komplementarności par
(`src/dataset.py::losowa_kanoniczna`). Mierzy, ile da się ugrać **samą komplementarnością zasad**,
bez żadnego uczenia. Bez tego punktu odniesienia nie da się powiedzieć, czy model czegokolwiek się
nauczył.

```
python -m src.evaluate --baseline --na test
```
