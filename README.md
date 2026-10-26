# 🐟 Luseradar

Luseradar varsler om et oppdrettsanlegg kommer til å passere lusegrensen de neste 1–3 ukene, basert kun på åpne data fra BarentsWatch. Resultatet vises på en nettside som et kart over anlegg, farget etter risiko.

**Live demo:** TODO: lenke til Streamlit Community Cloud
**Video:** TODO: lenke til video (reserve hvis live-demoen er nede)
**Rapport:** TODO: lenke til rapporten (f.eks. [report.pdf](report.pdf))

---

## Funksjonalitet

- Kart over oppdrettsanlegg, farget etter risikoen for å passere lusegrensen
- Velg prognosehorisont (TODO: 1–3 uker, eller bare 2 uker)
- TODO: Klikk på et anlegg for å se historikk og predikert risiko

## Problem og tilnærming

Myndighetene setter en grense for gjennomsnittlig antall voksne hunnlus per fisk (0,5, strengere om våren). Går et anlegg over grensen, må fisken behandles, ofte på kort varsel. BarentsWatch publiserer lusetellingene, men bare som historikk.

Luseradar behandler dette som et **binært klassifiseringsproblem**: Vil anlegget være over lusegrensen om *h* uker? Modellen gir en sannsynlighet per anlegg, som vises som risiko på kartet.

## Data

- **Kilde:** [BarentsWatch Fish Health API](https://developer.barentswatch.no/docs/fishhealth), åpne data under Norsk lisens for offentlige data (NLOD).
- **Innhold:** Ukentlige lusetellinger per lokalitet, med sjøtemperatur, behandlinger og posisjon.
- **Periode:** TODO (f.eks. 2012–2025)
- **Mengde:** TODO: antall lokaliteter og antall rader etter rensing

Dataene lastes ned én gang med `fetch_lice.py` og lagres lokalt. Modellen trenes aldri direkte mot API-et.

## Installasjon

Krever Python TODO: versjon.

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
| TODO | Client ID fra BarentsWatch |
| TODO | Client secret fra BarentsWatch |

`.env` er lagt i `.gitignore` og skal aldri committes.

## Reprodusere resultatene

TODO: Tilpass til de faktiske skriptene og notebookene.

```bash
# 1. Last ned lusetellinger fra BarentsWatch (krever .env)
python fetch_lice.py

# 2. Rens data og bygg features
python TODO_features.py

# 3. Tren modellen, evaluer på testsettet og lagre modellen
python TODO_train.py

# 4. Beregn prediksjoner for siste uke (brukes av nettsiden)
python TODO_predict.py
```

## Kjøre nettsiden lokalt

```bash
streamlit run app.py
```

Åpne http://localhost:8501 i nettleseren.

## Publisering

Appen kjører på [Streamlit Community Cloud](https://streamlit.io/cloud). Prediksjonene beregnes på forhånd, så nettsiden trenger ikke API-nøkler for å vise kartet.

1. Koble GitHub-repoet til Streamlit Community Cloud og velg `app.py` som hovedfil.
2. Trenger appen API-tilgang, legges variablene fra `.env` inn under *Settings → Secrets*.

## Modell og resultater

- **Datasplitt:** på tid, ikke tilfeldig. Trening: TODO (eldre år). Validering: TODO. Test: TODO (nyeste år). Da ser ikke modellen framtiden under trening.
- **Metrikker:** PR-AUC, recall og antall falske alarmer. Accuracy brukes ikke, fordi de fleste anleggsuker ligger under grensen, og en modell som alltid svarer «nei» ville fått høy accuracy.
- **Terskel:** valgt på valideringssettet slik at recall prioriteres. Det er verre å gå glipp av en overskridelse enn å gi en falsk alarm.

| Modell | PR-AUC | Recall | Falske alarmer per anlegg per år |
|---|---|---|---|
| Baseline: «over grensen nå → over grensen om *h* uker» | TODO | TODO | TODO |
| TODO: modell (f.eks. LightGBM) | TODO | TODO | TODO |

Se rapporten for feature importance og feilanalyse.

## Begrensninger

- Modellen bruker bare åpne data. Interne data som behandlingsplaner og fiskehelse er ikke med.
- TODO: f.eks. anlegg med få tellinger, endringer i dataformatet hos BarentsWatch, data drift ved varmere sjø

## Prosjektstruktur

```
luserader/
├── app.py              # Streamlit-nettsiden
├── fetch_lice.py       # Nedlasting fra BarentsWatch
├── requirements.txt    # Python-avhengigheter
├── .env.example        # Mal for miljøvariabler
├── TODO                # data/, notebooks/, models/ …
└── TODO                # rapporten
```

## Kilder og bruk av AI-verktøy

- BarentsWatch. *Fish Health API.* https://developer.barentswatch.no/docs/fishhealth
- BarentsWatch. *Application registration and authentication.* https://developer.barentswatch.no/docs/appreg
- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (O'Reilly)
- TODO: Lakselusforskriften (Lovdata), dokumentasjon for LightGBM og Streamlit, og andre kilder
- **AI-verktøy:** Claude (Anthropic) ble brukt til idéutvikling, planlegging, kodeutkast og utkast til rapport og README. TODO: hvordan dere har kontrollert resultatet.

Fullstendig kildeliste står i rapporten.

## Forfattere

- David Gotten
- Rakhman Ismailov

Prosjektet er laget som avsluttende prosjekt i DAT158 ved Høgskulen på Vestlandet, høsten 2026.
