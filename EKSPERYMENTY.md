# Eksperymenty i wyniki

> **STATUS: WYNIKI NIEPOLICZONE.** Kod jest gotowy i przetestowany, ale czeka na odsianie redundancji
> natywnym `cd-hit-est`, co wymaga WSL — instalacja opisana w [README](README.md). Liczby pojawią się tutaj po
> uruchomieniu `python run.py dane`, `E1` i `E2`.
>
> Wcześniejsze wyniki tego kodu, policzone na gotowym snapshocie po cd-hit z innej maszyny, zostały
> **usunięte razem z tym snapshotem** — nie chcemy raportować liczb, których nie da się w tym
> repozytorium odtworzyć od początku.

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

**Ograniczenie, którego nie da się usunąć:** największe rodziny zdominują trening, bo rodziny nie
wolno rozdzielić między zbiory.

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

*Do uzupełnienia po uruchomieniu eksperymentów.*

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
