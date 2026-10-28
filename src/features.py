"""Feature engineering for Luseradar. Brukes både av train.py og av nettsiden,
slik at modellen får nøyaktig de samme featurene i trening og i drift."""
import numpy as np
import pandas as pd

HORIZON = 2  # antall uker fram i tid vi predikerer

FEATURES = [
    "lice", "lice_lag1", "lice_lag2", "lice_lag3",
    "lice_diff1", "lice_roll4_mean", "lice_roll4_max",
    "mobile", "stationary", "mobile_lag1",
    "sea_temp", "sea_temp_roll4",
    "limit_future", "ratio_to_limit",
    "area_lice_mean", "area_lice_mean_lag1",
    "week_sin", "week_cos", "lat", "area_id",
]


def add_week_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Gir hver lokalitet én rad per uke, også uker uten telling, slik at shift(1)
    betyr «forrige uke» og ikke «forrige rapport»."""
    df = df.copy()
    df["date"] = pd.to_datetime(
        df["year"].astype(int).astype(str) + "-" + df["week"].astype(int).astype(str).str.zfill(2) + "-1",
        format="%G-%V-%u",
    )
    static = df.groupby("locality_no")[["locality_name", "lat", "lon", "area_id", "area_name"]].last()
    full = []
    for loc, g in df.groupby("locality_no", sort=False):
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="W-MON")
        full.append(g.set_index("date").reindex(idx).rename_axis("date").reset_index().assign(locality_no=loc))
    out = pd.concat(full, ignore_index=True)
    for col in static.columns:
        out[col] = out["locality_no"].map(static[col])
    iso = out["date"].dt.isocalendar()
    out["year"], out["week"] = iso["year"].astype(int), iso["week"].astype(int)
    return out


def limit_table(df: pd.DataFrame) -> pd.Series:
    """Vanligste lusegrense per (produksjonsområde, uke). Grensen er fastsatt i forskriften
    på forhånd (0,2 om våren, ellers 0,5), så den er kjent også for uker vi predikerer."""
    known = df.dropna(subset=["lice_limit", "area_id"])
    return known.groupby(["area_id", "week"])["lice_limit"].agg(lambda s: s.mode().iloc[0])


def build_features(df: pd.DataFrame, limits: pd.Series) -> pd.DataFrame:
    """Tar renset data fra fetch_lice.py og returnerer én rad per lokalitet per uke med
    features for uke t og label for uke t + HORIZON."""
    d = add_week_grid(df).sort_values(["locality_no", "date"]).reset_index(drop=True)
    reported = d["has_reported"].fillna(False).astype(bool) if "has_reported" in d else d["adult_female"].notna()
    d["lice"] = d["adult_female"].where(reported)
    d["mobile"] = d["mobile"].where(reported)
    d["stationary"] = d["stationary"].where(reported)

    g = d.groupby("locality_no", sort=False)
    for k in (1, 2, 3):
        d[f"lice_lag{k}"] = g["lice"].shift(k)
    d["mobile_lag1"] = g["mobile"].shift(1)
    d["lice_diff1"] = d["lice"] - d["lice_lag1"]
    d["lice_roll4_mean"] = g["lice"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    d["lice_roll4_max"] = g["lice"].transform(lambda s: s.rolling(4, min_periods=1).max())
    d["sea_temp_roll4"] = g["sea_temp"].transform(lambda s: s.rolling(4, min_periods=1).mean())

    # Smittepress fra naboanlegg: snittet i samme produksjonsområde samme uke.
    area = d.groupby(["area_id", "date"])["lice"].mean().rename("area_lice_mean").reset_index()
    area = area.sort_values("date")
    area["area_lice_mean_lag1"] = area.groupby("area_id")["area_lice_mean"].shift(1)
    d = d.merge(area, on=["area_id", "date"], how="left")

    # Sesong som sirkel, slik at uke 52 og uke 1 ligger nær hverandre.
    d["week_sin"] = np.sin(2 * np.pi * d["week"] / 52)
    d["week_cos"] = np.cos(2 * np.pi * d["week"] / 52)

    # Lusegrensen som gjelder i målukene.
    future_date = d["date"] + pd.Timedelta(weeks=HORIZON)
    future_week = future_date.dt.isocalendar()["week"].astype(int)
    keys = pd.MultiIndex.from_arrays([d["area_id"], future_week])
    d["limit_future"] = limits.reindex(keys).to_numpy()
    d["limit_future"] = d["limit_future"].fillna(0.5)
    d["ratio_to_limit"] = d["lice"] / d["limit_future"]

    # Label: er lokaliteten over grensen om HORIZON uker? (NaN hvis det ikke ble telt da)
    g = d.groupby("locality_no", sort=False)
    d["target_date"] = future_date
    d["lice_future"] = g["lice"].shift(-HORIZON)
    actual_limit = g["lice_limit"].shift(-HORIZON).fillna(d["limit_future"])
    d["y"] = (d["lice_future"] > actual_limit).astype(float).where(d["lice_future"].notna())
    return d
