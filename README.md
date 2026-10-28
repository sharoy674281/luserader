# 🐟 Luseradar

Luseradar predikerer risikoen for lakselus i norske oppdrettsanlegg. En maskinlæringsmodell trent på åpne lusetellinger fra BarentsWatch estimerer sannsynligheten for at hvert anlegg er over lusegrensen om to uker. Resultatet vises som et kart over anlegg farget etter risiko.

**Live demo:** TODO: lenke til Streamlit Community Cloud
**Video:** TODO: lenke til video (reserve hvis live-demoen er nede)
**Rapport:** [report.md](report.md)

---

## Funksjonalitet

- **Kart:** alle anlegg farget etter risiko for å være over lusegrensen om 2 uker, med filter på produksjonsområde og risiko
- **Anlegg:** velg et anlegg og se risiko, dagens lusetall, lusegrensen og historikk for de siste to årene
- **Prøv modellen:** legg inn egne lusetall, temperatur og uke, og få modellens prediksjon direkte
- **Om modellen:** metrikker på testsettet, precision-recall-kurve og feature importance

## Data

Data hentes fra [BarentsWatch Fiskehelse-API](https://www.barentswatch.no/) (åpne data, NLOD-lisens).

- Endepunkt: `GET /v1/geodata/download/fishhealth?reporttype=lice&filetype=csv` gir én rad per lokalitet per uke med voksne hunnlus, bevegelige og fastsittende lus, lusegrense, sjøtemperatur, posisjon og produksjonsområde.
- Tidsperiode: 2016 til i dag
- 603 221 anleggsuker fra 1 328 lokaliteter, hvorav 314 413 med lusetelling. Etter at vi krever telling både i uke *t* og uke *t*+2, gjenstår 298 623 rader med label (3,2 % over grensen).

Den rensede fila `data/lice.parquet` ligger i repoet, så du trenger ikke API-nøkler for å trene modellen eller kjøre nettsiden. Nøklene trengs bare for å hente nye data.

For å hente data trenger du en gratis API-klient fra BarentsWatch:
1. Logg inn på BarentsWatch og opprett en API-klient under «Min side».
2. Kopier client ID og client secret inn i `.env` (se under).

## Installasjon

Python 3.12–3.14. Merk: Windows på ARM har ingen ferdige `pyarrow`-pakker, så der må du bruke en x64-Python (den kjører under emulering).

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
| `BW_CLIENT_ID` | Client ID fra BarentsWatch |
| `BW_CLIENT_SECRET` | Client secret fra BarentsWatch |

`.env` er lagt i `.gitignore` og skal aldri committes.

## Reprodusere resultatene

```bash
# 1. Hent lusetellinger fra BarentsWatch -> data/lice.parquet (krever .env, kan hoppes over)
python scripts/fetch_lice.py

# 2. Bygg features, tren baselines og modeller, evaluer på testsettet og lag prediksjoner
python scripts/train.py
```

`train.py` skriver `models/model.joblib`, `models/metrics.json`, `data/predictions.parquet` og figurer i `reports/figures/`. Alle tall i rapporten kommer fra `metrics.json`.

## Kjøre nettsiden lokalt

```bash
streamlit run app.py
```

Åpne http://localhost:8501 i nettleseren.

## Publisering

Appen er publisert med [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Koble GitHub-repoet til Streamlit Community Cloud og velg `app.py` som hovedfil.
2. Nettsiden leser bare filer fra repoet, så den trenger ingen secrets.
3. Ny ukesprognose: kjør `fetch_lice.py` og `train.py` på nytt, og push.

## Modell og resultater

Oppgaven er klassifisering: er anlegget over lusegrensen om 2 uker? Klassene er ubalanserte, så vi bruker PR-AUC, recall og falske alarmer per anlegg per år i stedet for accuracy.

| Modell | PR-AUC | Recall | Precision | Falske alarmer / anlegg / år |
|---|---|---|---|---|
| Alltid nei | 0,03 | 0 % | – | 0 |
| Persistens (om to uker er det som nå) | 0,15 | 26 % | 24 % | 0,8 |
| Lineær trend | 0,13 | 48 % | 11 % | 3,7 |
| Logistisk regresjon | 0,19 | 65 % | 13 % | 3,9 |
| Random forest | 0,22 | 65 % | 14 % | 3,7 |
| **Gradient boosting (valgt)** | **0,24** | **62 %** | **15 %** | **3,3** |

- Trening: 2016–2022. Validering (modellvalg og terskel): 2023. Test: 2024 til i dag.
- Recall for *nye* overskridelser, altså anlegg som er under grensen i dag: gradient boosting 49 %, persistens 0 %. Data er splittet på **tid**, ikke tilfeldig, slik at modellen ikke ser framtiden under trening.
- Se [report.md](report.md) for detaljer om valg av features, hyperparametersøk og evaluering.

## Begrensninger

- Modellen vet ikke om avlusing som er planlagt. Mange anlegg som er over grensen i dag, blir behandlet før neste telling, og derfor fanger persistens bare 26 % av overskridelsene.
- Overskridelser er sjeldne (3 %), så precision blir lav: omtrent 1 av 7 varsler slår til.
- Anlegg med kort historikk eller få tellinger har mange manglende features.
- Inneværende uke er ofte bare delvis rapportert, så prediksjonene lages fra siste uke der de fleste anleggene har rapportert.

## Prosjektstruktur

```
luserader/
├── app.py                  # Streamlit-nettsiden
├── scripts/
│   ├── fetch_lice.py       # Henter data fra BarentsWatch
│   └── train.py            # Trener, evaluerer og lager prediksjoner
├── src/features.py         # Feature engineering (felles for trening og nettside)
├── data/
│   ├── lice.parquet        # Rensede lusetellinger
│   └── predictions.parquet # Risiko per anlegg, nyeste uke
├── models/                 # model.joblib og metrics.json
├── reports/figures/        # Figurer til rapporten
├── requirements.txt
├── .env.example            # Mal for API-nøkler
└── report.md               # Prosjektrapport
```

## Kilder og bruk av AI-verktøy

- BarentsWatch Fiskehelse-API: https://developer.barentswatch.no/docs/fishhealth
- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (O'Reilly)
- TODO: Andre kilder (artikler, tutorials, kodeeksempler)
- **AI-verktøy:** TODO: hvilke verktøy (f.eks. Claude, ChatGPT, Copilot) og hva de ble brukt til (f.eks. feilsøking, kodeforslag, språkvask av rapporten).

## Forfattere

- Rakhman Ismailov
- David Gotten

Prosjektet er laget som avsluttende prosjekt i DAT158 ved Høgskulen på Vestlandet, høsten 2026.
