# 🐟 Luseradar

Luseradar varsler om et oppdrettsanlegg kommer til å passere lusegrensen om to uker, basert kun på åpne data fra BarentsWatch. En maskinlæringsmodell gir en sannsynlighet per anlegg, og resultatet vises på en nettside som et kart over anlegg farget etter risiko.

**Live demo:** https://luseradar.streamlit.app
**Rapport:** (luseradar innlevering.pdf)

---

## Funksjonalitet

- **Radar:** kart over alle anlegg med tre visninger:
  - **Om 2 uker:** modellens sannsynlighet for å være over lusegrensen
  - **Nå:** siste lusetelling sammenlignet med grensen
  - **Historikk:** spol gjennom de siste to årene og se hvordan lusa sprer seg
- Klikk på et anlegg i kartet, i lista eller søk etter navn for å se risiko, dagens lusetall, grensen og et års historikk
- **Prøv modellen:** legg inn egne lusetall, temperatur og uke, og få modellens prediksjon direkte
- **Om modellen:** metrikker på testsettet, precision-recall-kurve og feature importance

## Problem og tilnærming

Myndighetene setter en grense for gjennomsnittlig antall voksne hunnlus per fisk (0,5, og 0,2 om våren). Går et anlegg over grensen, må fisken behandles, ofte på kort varsel. BarentsWatch publiserer lusetellingene, men bare som historikk.

Luseradar behandler dette som et **binært klassifiseringsproblem**: Vil anlegget være over lusegrensen om 2 uker? Modellen gir en sannsynlighet per anlegg, som vises som risiko på kartet.

## Data

- **Kilde:** [BarentsWatch Fish Health API](https://developer.barentswatch.no/docs/fishhealth), åpne data under Norsk lisens for offentlige data (NLOD).
- **Endepunkt:** `GET /v1/geodata/download/fishhealth?reporttype=lice&filetype=csv`. Gir én rad per lokalitet per uke med voksne hunnlus, bevegelige og fastsittende lus, lusegrense, sjøtemperatur, posisjon og produksjonsområde.
- **Periode:** 2016 til i dag.
- **Mengde:** 603 221 anleggsuker fra 1 328 lokaliteter, hvorav 314 413 med lusetelling. Etter at vi krever telling både i uke *t* og uke *t*+2, gjenstår 298 623 rader med label (3,2 % over grensen).

Dataene lastes ned én gang med `scripts/fetch_lice.py` og lagres lokalt. Modellen trenes aldri direkte mot API-et. Den rensede fila `data/lice.parquet` ligger i repoet, så du trenger ikke API-nøkler for å trene modellen eller kjøre nettsiden.

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

## API-tilgang (kun nødvendig for å laste ned data på nytt)

1. Registrer en gratis API-klient hos BarentsWatch, se [Application registration](https://developer.barentswatch.no/docs/appreg).
2. Kopier malen og fyll inn klientens ID og secret:

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

Appen kjører på [Streamlit Community Cloud](https://streamlit.io/cloud): https://luseradar.streamlit.app. Prediksjonene beregnes på forhånd, så nettsiden trenger ikke API-nøkler.

1. Koble GitHub-repoet til Streamlit Community Cloud og velg `app.py` som hovedfil.
2. Nettsiden leser bare filer fra repoet, så den trenger ingen secrets.
3. Ny ukesprognose: kjør `fetch_lice.py` og `train.py` på nytt, og push.

## Modell og resultater

- **Datasplitt:** på tid, ikke tilfeldig. Trening: 2016–2022. Validering (modellvalg og terskel): 2023. Test: 2024 til i dag. Da ser ikke modellen framtiden under trening.
- **Metrikker:** PR-AUC, recall og antall falske alarmer. Accuracy brukes ikke, fordi de fleste anleggsuker ligger under grensen, og en modell som alltid svarer «nei» ville fått høy accuracy.
- **Terskel:** valgt på valideringssettet slik at recall prioriteres, med høyst 3 falske alarmer per anlegg per år. Det er verre å gå glipp av en overskridelse enn å gi en falsk alarm.

| Modell | PR-AUC | Recall | Precision | Falske alarmer / anlegg / år |
|---|---|---|---|---|
| Alltid nei | 0,03 | 0 % | – | 0 |
| Persistens (om to uker er det som nå) | 0,15 | 26 % | 24 % | 0,8 |
| Lineær trend | 0,13 | 48 % | 11 % | 3,7 |
| Logistisk regresjon | 0,19 | 65 % | 13 % | 3,9 |
| Random forest | 0,22 | 65 % | 14 % | 3,7 |
| **Gradient boosting (valgt)** | **0,24** | **62 %** | **15 %** | **3,3** |

Recall for *nye* overskridelser, altså anlegg som er under grensen i dag: gradient boosting 49 %, persistens 0 %. Se rapporten for feature importance og feilanalyse.

## Begrensninger

- Modellen bruker bare åpne data. Interne data som behandlingsplaner og fiskehelse er ikke med.
- Modellen vet ikke om avlusing som er planlagt. Mange anlegg som er over grensen i dag, blir behandlet før neste telling, og derfor fanger persistens bare 26 % av overskridelsene.
- Overskridelser er sjeldne (3 %), så precision blir lav: omtrent 1 av 7 varsler slår til.
- Anlegg med kort historikk eller få tellinger har mange manglende features, og modellen er svakest i regioner med få overskridelser.
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
├── .streamlit/config.toml  # Farger og tema for nettsiden
├── requirements.txt
├── .env.example            # Mal for API-nøkler
└── report.md               # Prosjektrapport
```

## Kilder og bruk av AI-verktøy

- BarentsWatch. *Fish Health API.* https://developer.barentswatch.no/docs/fishhealth
- BarentsWatch. *Application registration and authentication.* https://developer.barentswatch.no/docs/appreg
- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (O'Reilly)
- Lovdata. *Forskrift om bekjempelse av lakselus i akvakulturanlegg* (lakselusforskriften), FOR-2012-12-05-1140.
- Streamlit- og scikit-learn-dokumentasjonen.
- **AI-verktøy:** Claude (Anthropic) ble brukt til idéutvikling, planlegging, kodeutkast og utkast til rapport og README, samt feilsøking. Vi har lest gjennom og kjørt koden selv, sjekket tallene mot `models/metrics.json` og testet nettsiden manuelt.

Fullstendig kildeliste står i rapporten.

## Forfattere

- David Gotten
- Rakhman Ismailov

Prosjektet er laget som avsluttende prosjekt i DAT158 ved Høgskulen på Vestlandet, høsten 2026.
