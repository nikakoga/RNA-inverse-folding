# RNA inverse folding — transformer nieautoregresyjny

Projektowanie sekwencji RNA pod zadaną strukturę drugorzędową. Folder jest **samowystarczalny**:
zawiera dane, kod, eksperymenty i dokumentację, nie odwołuje się do niczego na zewnątrz.

```
python run.py dane              # cd-hit-est + filtry + podzial rodzinowy
python run.py E1                # kara za sklad: odleglosc TV (dwustronna), waga 1,0
python run.py E2                # kara za sklad: progi dolne (jednostronna), waga 1,0
python run.py CE                # BEZ kary za sklad — ablacja
python run.py E1W / E2W / CEW   # to samo, ale cross-entropia WAZONA odwrotnie do czestosci klas
```

Sześć przebiegów w dwóch wymiarach — konstrukcja kary za skład i ważenie cross-entropii:

```
             kara TV     kara progowa    brak kary
CE zwykla       E1            E2            CE
CE wazona       E1W           E2W           CEW
```

W obrębie każdego przebiegu wszystko poza tymi dwiema decyzjami jest identyczne: dane, podział,
architektura, wagi 1,0, próbkowanie z ziarnem 0, 60 epok, epoka wybrana po `zbal_par` na walidacji.
Szczegóły i wyniki w [EKSPERYMENTY.md](EKSPERYMENTY.md).

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
data/eterna_working.parquet                       <- buduje src/prepare.py
data/splits/               podzialy 60/20/20      <- buduje src/split.py
```

Repozytorium zaczyna od puli **przed** odsianiem redundancji i liczy ten krok samo.

### cd-hit-est wymaga WSL

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
src/cdhit.py        odsianie redundancji w puli naturalnej
src/prepare.py      filtry: przewaga sparowanych, dlugosc <= 200, alfabet ACGU
src/split.py        podzial 60/20/20, rodzinowy albo losowy
src/model.py        transformer enkoder-only, dwie glowice
src/loss.py         komponenty straty: energia, parowania, sklad
src/train.py        petla treningowa
src/evaluate.py     ocena na tescie naturalnym i Eternie
```

Kod rysujący wykresy jest w samych notatnikach, nie w `src/`.

## Wymagania

```
python 3.11, torch (build CUDA), ViennaRNA, pandas, numpy, matplotlib
```

ViennaRNA służy wyłącznie do wyceny zadanej struktury: `RNA.energy_of_struct` i tablice Turnera.
Struktury nie przewidujemy nigdzie — `RNA.fold` nie występuje w repozytorium.
