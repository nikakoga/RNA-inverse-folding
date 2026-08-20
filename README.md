# RNA inverse folding — transformer nieautoregresyjny

Projektowanie sekwencji RNA pod zadaną strukturę drugorzędową. Folder jest **samowystarczalny**:
zawiera dane, kod, eksperymenty i dokumentację, nie odwołuje się do niczego na zewnątrz.

```
python run.py dane              # cd-hit-est + filtry + podzial rodzinowy
python run.py E1                # trzy komponenty w stracie, wagi 1,0 : 6,0 : 1,7
python run.py E2                # strojenie wagi kary za sklad
```

## Co gdzie jest

| plik | zawartość |
|---|---|
| [MODEL.md](MODEL.md) | architektura, funkcja straty, sposób trenowania i oceny |
| [EKSPERYMENTY.md](EKSPERYMENTY.md) | co testowano, wyniki, znane wady |
| [notebooks/01_dane.ipynb](notebooks/01_dane.ipynb) | wykresy: co jest w zbiorach treningowym, walidacyjnym, testowym i w Eternie |
| [notebooks/02_wyniki.ipynb](notebooks/02_wyniki.ipynb) | wykresy: wyniki obu eksperymentów |

## Dane

Wszystko jest w repozytorium, nic się nie pobiera.

```
data/raw/rna_raw.parquet   31 026 sekwencji, 896 rodzin Rfam — PRZED odsianiem redundancji
data/raw/eterna100.tsv     zagadki Eterna100 z rozwiazaniami graczy

data/cdhit/                po cd-hit-est          <- buduje src/cdhit.py
data/working.parquet       po filtrach            <- buduje src/prepare.py
data/splits/               podzialy 60/20/20      <- buduje src/split.py
```

Repozytorium zaczyna od puli **przed** odsianiem redundancji i liczy ten krok samo. Wcześniej leżał
tu gotowy plik po cd-hit z innej maszyny — został usunięty, bo nie dawał się odtworzyć na miejscu.

### cd-hit-est wymaga WSL

Odsiewanie redundancji robimy **natywnym** `cd-hit-est 4.8.1`, nie reimplementacją w Pythonie: ta
usuwała 2,4× mniej duplikatów, a to, co zostawało, trafiało i do treningu, i do testu — czyli
zawyżało wyniki.

Narzędzie jest w C++ i nie ma wersji dla Windows, ale działa przez WSL. Instalacja raz,
**w PowerShellu uruchomionym jako administrator**:

```powershell
wsl --install
```

Po restarcie komputera, już w zwykłym terminalu:

```powershell
wsl -- sudo apt-get update
wsl -- sudo apt-get install -y cd-hit
```

## Struktura kodu

```
src/dataset.py      parsowanie struktur, motywy, kodowanie wejscia, baseline
src/prepare.py      filtry: przewaga sparowanych, dlugosc <= 200, alfabet ACGU
src/split.py        podzial 60/20/20, rodzinowy albo losowy
src/model.py        transformer enkoder-only, dwie glowice
src/loss.py         trzy komponenty: energia, parowania, sklad
src/train.py        petla treningowa
src/evaluate.py     ocena na tescie naturalnym i Eternie
src/analyze_data.py obrazowa analiza zbioru
src/plots.py        funkcje rysujace, wspolne dla notatnikow
```

## Wymagania

```
python 3.11, torch (build CUDA), ViennaRNA, pandas, numpy, matplotlib
```

ViennaRNA używamy do dwóch rzeczy: `RNA.energy_of_struct` i tablic Turnera (obie **nie przewidują**
struktury, tylko wyceniają zadaną) oraz `RNA.fold` — wyłącznie w ocenie, nigdy w treningu.
