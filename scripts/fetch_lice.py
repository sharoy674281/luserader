"""Laster ned ukentlige lusetellinger fra BarentsWatch Fish Health API.

Bruker eksportendepunktet /v1/geodata/download/fishhealth (reporttype=lice), som gir
én rad per lokalitet per uke med lusetall, lusegrense, sjøtemperatur og posisjon.
Vi henter ett år om gangen, sekvensielt (BarentsWatch ber om at batch-nedlasting ikke
kjøres i parallell), lagrer råfilene i data/raw/ og skriver et renset datasett til
data/lice.parquet. Den rensede fila committes, slik at man kan trene modellen og kjøre
nettsiden uten egne API-nøkler.

Bruk:
    python scripts/fetch_lice.py               # 2016 til og med inneværende år
    python scripts/fetch_lice.py --from-year 2020
"""
import argparse
import datetime as dt
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = ROOT / "data" / "lice.parquet"

TOKEN_URL = "https://id.barentswatch.no/connect/token"
EXPORT_URL = "https://www.barentswatch.no/bwapi/v1/geodata/download/fishhealth"

# Eksporten kan ha norske eller engelske kolonnenavn. Vi mapper begge til samme navn.
COLUMNS = {
    "uke": "week", "week": "week",
    "år": "year", "year": "year",
    "lokalitetsnummer": "locality_no", "localityno": "locality_no",
    "lokalitetsnavn": "locality_name", "localityname": "locality_name",
    "voksne hunnlus": "adult_female", "avgadultfemalelicerounded": "adult_female",
    "lus i bevegelige stadier": "mobile", "avgmobilelicerounded": "mobile",
    "fastsittende lus": "stationary", "avgstationarylicerounded": "stationary",
    "brakklagt": "fallow", "trolig uten fisk": "fallow", "fallowstring": "fallow",
    "har telt lakselus": "has_reported", "hasreportedlicestring": "has_reported",
    "kommunenummer": "municipality_no", "municipalityno": "municipality_no",
    "kommune": "municipality", "municipality": "municipality",
    "fylke": "county", "county": "county",
    "lat": "lat", "lon": "lon",
    "lusegrense uke": "lice_limit", "licelimit": "lice_limit",
    "over lusegrense uke": "above_limit", "abovelicelimit": "above_limit",
    "sjøtemperatur": "sea_temp", "seatemperaturerounded": "sea_temp",
    "produksjonsområdeid": "area_id", "productionareaid": "area_id",
    "produksjonsområde": "area_name", "productionareaname": "area_name",
}
NUMERIC = ["adult_female", "mobile", "stationary", "lat", "lon", "lice_limit", "sea_temp", "area_id"]
YES = {"ja", "yes", "true", "1"}


def get_token(client_id: str, client_secret: str) -> str:
    """OAuth 2.0 client credentials: bytter client id/secret mot et access token."""
    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "api",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def last_iso_week(year: int) -> int:
    """Siste ISO-uke i året (52 eller 53), eller inneværende uke for i år."""
    today = dt.date.today()
    if year == today.year:
        return today.isocalendar().week
    return dt.date(year, 12, 28).isocalendar().week


def download_year(token: str, year: int) -> bytes:
    params = {
        "reporttype": "lice",
        "filetype": "csv",
        "fromyear": year,
        "fromweek": 1,
        "toyear": year,
        "toweek": last_iso_week(year),
    }
    r = requests.get(EXPORT_URL, params=params, headers={"Authorization": f"Bearer {token}"}, timeout=300)
    r.raise_for_status()
    return r.content


def normalize(raw: bytes) -> pd.DataFrame:
    """Leser CSV-en (ukjent skilletegn/desimaltegn) og gir faste kolonnenavn og datatyper."""
    text = raw.decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str)
    df.columns = [c.strip().strip('"').lower() for c in df.columns]
    unknown = [c for c in df.columns if c not in COLUMNS]
    if unknown:
        print(f"  (ignorerer kolonner: {unknown})")
    df = df[[c for c in df.columns if c in COLUMNS]].rename(columns=COLUMNS)
    df = df.loc[:, ~df.columns.duplicated()]

    for col in NUMERIC:
        if col in df:
            df[col] = pd.to_numeric(df[col].str.replace(",", ".", regex=False), errors="coerce")
    for col in ["has_reported", "above_limit", "fallow"]:
        if col in df:
            df[col] = df[col].fillna("").str.strip().str.lower().isin(YES)
    # Feilregistreringer: lusetall og temperatur utenfor det fysisk mulige settes til NaN.
    df.loc[~df["sea_temp"].between(-2, 30), "sea_temp"] = None
    for col in ["adult_female", "mobile", "stationary"]:
        df.loc[df[col] > 50, col] = None
    for col in ["week", "year", "locality_no"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df.dropna(subset=["week", "year", "locality_no"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-year", type=int, default=2016)
    parser.add_argument("--to-year", type=int, default=dt.date.today().year)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client_id, client_secret = os.getenv("BW_CLIENT_ID"), os.getenv("BW_CLIENT_SECRET")
    if not client_id or not client_secret:
        sys.exit("Mangler BW_CLIENT_ID / BW_CLIENT_SECRET i .env (se README).")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    token = get_token(client_id, client_secret)

    frames = []
    for year in range(args.from_year, args.to_year + 1):
        raw_file = RAW_DIR / f"lice_{year}.csv"
        # Tidligere år endrer seg ikke, så vi gjenbruker dem. Inneværende år hentes alltid på nytt.
        if raw_file.exists() and year < dt.date.today().year:
            raw = raw_file.read_bytes()
            print(f"{year}: bruker lagret fil")
        else:
            print(f"{year}: laster ned ...", end=" ", flush=True)
            raw = download_year(token, year)
            raw_file.write_bytes(raw)
            print(f"{len(raw) / 1e6:.1f} MB")
            time.sleep(1)  # vær snill mot API-et
        frames.append(normalize(raw))

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["locality_no", "year", "week"]).sort_values(["locality_no", "year", "week"])
    df.to_parquet(OUT_FILE, index=False)

    reported = df["has_reported"].sum() if "has_reported" in df else "?"
    print(f"\nLagret {len(df):,} rader ({reported:,} med lusetelling), "
          f"{df['locality_no'].nunique():,} lokaliteter -> {OUT_FILE.relative_to(ROOT)}")
    print("Kolonner:", list(df.columns))


if __name__ == "__main__":
    main()
