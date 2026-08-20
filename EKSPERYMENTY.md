# Eksperymenty i wyniki

Architektura i funkcja straty: [MODEL.md](MODEL.md).
Analiza danych: [notebooks/01_dane.ipynb](notebooks/01_dane.ipynb).
Wykresy wyników: [notebooks/02_wyniki.ipynb](notebooks/02_wyniki.ipynb).

```
python run.py dane     # cd-hit-est + filtry + podzial rodzinowy
python run.py E1
python run.py E2
```

---

## Przygotowanie danych

```
data/raw/rna_raw.parquet        31 026 sekwencji, 896 rodzin Rfam — PRZED odsianiem
  |
  |  src/cdhit.py     ograniczenie dlugosci (<= 200 nt naturalne, <= 50 nt Eterna)
  |                   nastepnie NATYWNY cd-hit-est 4.8.1, -c 0.8 -n 5, obie pule OSOBNO
  v
data/cdhit/*.parquet
  |
  |  src/prepare.py   przewaga sparowanych (paired_fraction >= 0.5)
  |                   poprawnosc: >= 1 para, alfabet ACGU
  v
data/working.parquet + data/eterna_working.parquet
  |
  |  src/split.py     podzial 60/20/20 RODZINOWY
  v
data/splits/split_rodzinowy_seed0.json
```

### Dlaczego natywny cd-hit, a nie reimplementacja

Reimplementacja w Pythonie (pokrycie 8-merów ≥ 0,8) usuwała **2,4× mniej** sekwencji niż
`cd-hit-est` na tej samej puli. Redundancja, która zostawała, trafiała potem i do treningu, i do
testu — czyli zawyżała wyniki. To jest błąd, którego nie chcemy powtórzyć, i dlatego cały ten krok
liczymy sami, prawdziwym narzędziem.

### Dlaczego podział rodzinowy

Przy podziale losowym po strukturach ta sama rodzina Rfam trafia i do treningu, i do testu, więc
model może odtwarzać zapamiętany wzorzec zamiast generalizować. Przy rodzinowym każda rodzina jest
w dokładnie jednym podzbiorze.

Naiwne pakowanie rodzin po samej liczebności wprowadza jednak **przesunięcie rozkładu długości**
i oddaje większość walidacji jednej rodzinie — test mierzyłby wtedy różnicę długości, a nie
generalizację. [`src/split.py`](src/split.py) minimalizuje naraz trzy rzeczy: odchylenie liczebności
od 60/20/20, odchylenie rozkładu długości od całej puli oraz dominację pojedynczej rodziny.

### Ile czego zostało

```
pula naturalna <= 200 nt        28 678
  po cd-hit-est                  8 500   (-70%)
  po przewadze sparowanych       3 699   (-4801)
  po kontroli poprawnosci        3 699

Eterna <= 50 nt                     18
  po cd-hit-est                     15   (3 zagadki byly wzajemnymi duplikatami)
  po przewadze sparowanych          11   (-4)
```

Podział rodzinowy, 340 rodzin:

```
        n     rodzin  mediana L  <=50 nt  max rodzina   najwieksze
train  2220    110       78        12        63%        tRNA, 5S, SNORA38
val     740    117       89        23        31%        16S, mir-154, T-box
test    739    113       92        14        14%        SRP, RNaseP, U2

rodziny wspolne miedzy zbiorami: 0
```

**Ograniczenie, którego nie da się usunąć:** największe rodziny zdominują trening (63% tRNA),
bo rodziny nie wolno rozdzielić między zbiory.

**Eterna została z 11 zagadkami.** Po odsianiu duplikatów i filtrze przewagi sparowanych zbiór jest
tak mały, że każdy wynik na nim ma ogromny słupek niepewności — różnica jednej zagadki to 9 punktów
procentowych.

---

## E1 — trzy komponenty w stracie

Wagi `energia 1,0 : parowania 6,0 : skład 1,7`. Przeniesione z wcześniejszych pomiarów rozrzutu
komponentów, **nie strojone** — to punkt wyjścia, nie wynik optymalizacji.

## E2 — strojenie wagi kary za skład

Sweep po `--w-sklad` przy pozostałych wagach stałych. **Strojenie na WALIDACJI**, bo zbiór testowy
wolno obejrzeć raz, na końcu. Dopiero wybrane konfiguracje idą na test i to jest jedyna liczba
do raportu.

## Baseline

Sekwencja losowana z naturalnych częstości, z zachowaniem komplementarności par. Mierzy, ile da się
ugrać **samą komplementarnością zasad**, bez żadnego uczenia. Bez tego punktu odniesienia nie da się
powiedzieć, czy model czegokolwiek się nauczył.

---

## Wyniki

Zbiór testowy, generowanie zachłanne, jeden przebieg na strukturę.

```
model                  0-50    51-100   101-200   Eterna   odzysk   dE/nt
                       n=14     n=451     n=274     n=11   (51-100 nt)
baseline losowy        2/14     3/451     0/274     2/11    0,261   -0,047
E1  sklad 1,7          8/14   187/451    33/274     7/11    0,301   -0,224
E2  sklad 0           10/14   190/451    52/274     7/11    0,298   -0,232
E2  sklad 40          10/14   277/451    84/274     6/11    0,280   -0,289
```

### 1. Model uczy się czegoś realnego

Baseline losuje litery z naturalnych częstości, zachowując komplementarność par. Rozwiązuje
**3 na 451**. Najlepszy model **277**. To jest dziewięćdziesięciokrotna przewaga i odpowiedź na
pytanie, czy sama komplementarność wystarcza — nie wystarcza.

Odzysk sekwencji naturalnej 0,28–0,30 wobec 0,26 dla losowania. Przewaga realna, ale niewielka:
model nie odtwarza konkretnej sekwencji, tylko projektuje własną.

### 2. Kara za skład przy wadze 1,7 NIE szkodzi

```
sklad 1,7    187/451
sklad 0      190/451
```

Praktycznie tyle samo. **To jest istotna zmiana wobec wcześniejszych pomiarów** na danych odsianych
reimplementacją w Pythonie, gdzie ta sama para wag dawała 32 wobec 248. Tamten dramatyczny rozjazd
**nie powtórzył się** na danych odsianych prawdziwym cd-hit-est i należy go uznać za artefakt.

### 3. Silna kara pomaga na danych naturalnych, szkodzi na Eternie

```
                     test naturalny        Eterna
sklad 0             190/451, 52/274         7/11
sklad 40            277/451, 84/274         6/11
```

Na strukturach naturalnych waga 40 daje wyraźnie więcej rozwiązań, na Eternie o jedną mniej. Przy
n = 11 ta różnica jest w granicach szumu i nie należy z niej wyciągać wniosków.

### 4. Energia jest ujemna dla wszystkich modeli

`dE/nt` od −0,22 do −0,29, więc nasze sekwencje stabilizują strukturę docelową **lepiej niż
sekwencje naturalne**. Baseline też jest ujemny (−0,047), ale wielokrotnie słabiej.

### 5. Struktury długie pozostają problemem

Kubełek 101–200 nt: najlepiej 84 na 274, czyli niecała jedna trzecia. Model działa dobrze do około
100 nt i dalej wyraźnie słabnie.

---

Raportujemy trzy miary:

| miara | co znaczy | używa RNAfolda |
|---|---|---|
| `rozwiazane` | czy nasza sekwencja **zwija się** w zadaną strukturę | tak |
| `odzysk` | ułamek pozycji trafionych wobec prawdziwej sekwencji | nie |
| `dE/nt` | o ile stabilizujemy cel lepiej niż prawdziwa sekwencja | nie |

**`rozwiazane` nie porównuje sekwencji z odpowiedzią.** Sekwencja całkiem inna od wzorcowej może
rozwiązać zagadkę, jeśli tylko zwija się poprawnie. Sekwencja z samych par G:C zwija się bardzo
niezawodnie, mając może 30% liter wspólnych ze wzorcem — i to jest sedno problemu, który mierzymy.

`F1` celowo pomijamy: rozdaje punkty częściowe za pojedyncze pary, więc dziedziczy błąd RNAfolda
na każdej z nich.

---

## Znane wady konstrukcji, do sprawdzenia na nowych danych

Poniższe wynikają z budowy kodu, nie z konkretnego zbioru, więc dotyczą także nadchodzących
eksperymentów.

**Kara za skład liczy się na miękkich rozkładach.** Sekwencja powstaje przez `argmax`, a kara patrzy
na rozkład prawdopodobieństwa uśredniony po partii. Model może mieć rozkład, który średnio wygląda
naturalnie, i przy tym na każdej pozycji wskazywać G-C — kara jest wtedy spełniona, a wyjście
zdegenerowane. Trzy możliwe naprawy, żadna nieprzetestowana: straight-through estimator, kara za
pewność rozkładu, albo liczenie kary na próbce zamiast na średniej.

**Wybór epoki po odzysku jest pułapką.** Odzysk mierzy podobieństwo do sekwencji naturalnej,
a rozwiązywanie premiuje mocne, jednoznaczne helisy — te cele są ze sobą sprzeczne. Flaga `--wybor`
pozwala wybrać epokę po energii (bez zwijania) albo po liczbie rozwiązanych (z RNAfoldem).

**Wszystkie trzy typy pętli dzielą jeden cel składu**, mimo że w naturze się różnią.

**Brak jakiejkolwiek iteracyjnej poprawy.** Model pisze sekwencję raz i nigdy jej nie rewiduje.

---

## Kierunki dalszej pracy

1. **Iteracyjna poprawa sekwencji** — poprawianie pozycji obniżających energię celu, wykonalne
   **bez zwijania**; najprawdopodobniej najwięcej da na strukturach powyżej 100 nt.
2. **Naprawa kary za skład**, żeby dotyczyła faktycznego wyboru, a nie średniej rozkładu.
3. **Cel składu osobno dla każdego typu pętli.**
4. **Powtórzenia z różnymi ziarnami** — bez nich nie da się odróżnić efektu od szumu.
