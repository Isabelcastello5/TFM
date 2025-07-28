# TFM - Predicció de propietats mecàniques de l’os cranial humà mitjançant imatges micro-CT
Predicció de propietats mecàniques de l’os cranial humà mitjançant imatges micro-CT.

Aquest projecte forma part d’un treball de final de màster (TFM) centrat en l’anàlisi de les propietats mecàniques de l'os cranial humà a partir d'imatges de micro-tomografia computada (micro-CT) de crani. 

L’objectiu és estimar la dimensió fractal (FD) i correlacionar-la amb el mòdul d’elasticitat aparent (E) mitjançant un model exponencial.


---

## Estructura del projecte

```

TFM\_CRANI\_ANALISI/
├── analisiFD.py           # Script principal d’anàlisi fractal per imatges BMP
├── predictor.py             # Ajust del model FD vs E i prediccions interactives
├── README.md                # Descripció del projecte

````

---

## Requisits

El projecte utilitza Python 3 i les següents biblioteques:

- `numpy`
- `scipy`
- `opencv-python`
- `matplotlib`
- `pandas`


---

## Scripts principals

### `analisiFD.py`

* Llegeix imatges `.bmp` des d’una carpeta
* Permet seleccionar manualment una ROI al slice central
* Calcula la **PSD radial** i ajusta la **FD** via regressió log-log
* Desa:

  * Gràfics `log-log` per a cada slice
  * Fitxers Excel amb els vectors `PSD` (`Aout`) 
  * `FD_resultats.xlsx` 
  * Imatges retallades (`crops`) i FFT 

### `predictor.py`

* Ajusta un model exponencial: `E = E₀ * exp(-α * FD)`
* Mostra el valor de `E₀`, `α` i `R²` del model
* Permet predir E a partir de FD introduïda per l’usuari

---

## Autoria

Isabel Castelló Ortega
Universitat de Barcelona - Universitat Politècnica de Catalunya
TFM 2025 · Enginyeria Biomèdica

