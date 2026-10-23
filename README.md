# 🐟 Luseradar

Luseradar predikerer risikoen for lakselus i norske oppdrettsanlegg. Brukeren velger et anlegg og en prognoseperiode, og en maskinlæringsmodell trent på åpne data fra BarentsWatch estimerer lusenivået de kommende ukene.

**Live demo:** TODO: lenke til Streamlit Community Cloud
**Video:** TODO: lenke til video (reserve hvis live-demoen er nede)
**Rapport:** [report.md](report.md)

---

## Funksjonalitet

- TODO: Velg anlegg (navn / lokalitetsnummer / kart?)
- Velg hvor mange uker fram i tid prognosen skal gjelde
- TODO: Vis predikert lusenivå (og om det er over lusegrensen)
- TODO: Vis historikk for anlegget

## Data

Data hentes fra [BarentsWatch Fiskehelse-API](https://www.barentswatch.no/) (åpne data, NLOD-lisens).

- TODO: Hvilke endepunkter/datasett brukes (lusetellinger, sjøtemperatur, …)
- TODO: Tidsperiode som brukes til trening (f.eks. 2015–2025)
- TODO: Antall anlegg og antall rader etter rensing

For å hente data trenger du en gratis API-klient fra BarentsWatch:
1. Logg inn på BarentsWatch og opprett en API-klient under «Min side».
2. Kopier client ID og client secret inn i `.env` (se under).

## Installasjon

Python 3.14.3

```bash
git clone https://github.com/sharoy674281/luserader.git
cd luserader
python -m venv .venv
```

Aktiver det virtuelle miljøet:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

Installer avhengigheter:

```bash
pip install -r requirements.txt
```

## Konfigurasjon

Kopier malen og fyll inn verdiene:

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

| Variabel | Beskrivelse |
|---|---|
| TODO: `BW_CLIENT_ID` | Client ID fra BarentsWatch |
| TODO: `BW_CLIENT_SECRET` | Client secret fra BarentsWatch |

`.env` er lagt i `.gitignore` og skal aldri committes.

## Reprodusere resultatene

TODO: Tilpass til faktiske skript/notebooks.

```bash
# 1. Hent rådata fra BarentsWatch
python TODO_hent_data.py

# 2. Forbehandle og bygg features
python TODO_forbehandling.py

# 3. Tren og evaluer modellen (lagrer modellen til models/)
python TODO_tren_modell.py
```

## Kjøre nettsiden lokalt

```bash
streamlit run app.py
```

Åpne http://localhost:8501 i nettleseren.

## Publisering

Appen er publisert med [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Koble GitHub-repoet til Streamlit Community Cloud og velg `app.py` som hovedfil.
2. Legg inn variablene fra `.env` under *Settings → Secrets*.

## Modell og resultater

| Modell | TODO: MAE | TODO: RMSE |
|---|---|---|
| Baseline (lusenivå neste uke = denne uka) | TODO | TODO |
| TODO: modell (f.eks. RandomForestRegressor) | TODO | TODO |

- Treningsdata: TODO (eldre uker). Testdata: TODO (nyeste uker). Data er splittet på **tid**, ikke tilfeldig, slik at modellen ikke ser framtiden under trening.
- Se [report.md](report.md) for detaljer om valg av features, hyperparametersøk og evaluering.

## Begrensninger

- TODO: f.eks. anlegg med få tellinger, sesongvariasjon, ingen data om behandlinger, …

## Prosjektstruktur

```
luserader/
├── app.py              # Streamlit-nettsiden
├── requirements.txt    # Python-avhengigheter
├── .env.example        # Mal for miljøvariabler
├── report.md           # Prosjektrapport
└── TODO                # data/, notebooks/, models/, src/ …
```

## Kilder og bruk av AI-verktøy

- BarentsWatch Fiskehelse-API: TODO lenke
- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (O'Reilly)
- TODO: Andre kilder (artikler, tutorials, kodeeksempler)
- **AI-verktøy:** TODO: hvilke verktøy (f.eks. Claude, ChatGPT, Copilot) og hva de ble brukt til (f.eks. feilsøking, kodeforslag, språkvask av rapporten).

## Forfattere

- Rakhman Ismailov
- David TODO: etternavn

Prosjektet er laget som avsluttende prosjekt i DAT158 ved Høgskulen på Vestlandet, høsten 2026.
