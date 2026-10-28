"""Luseradar: Streamlit-nettside. Kjør med `streamlit run app.py`."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from src.features import FEATURES

ROOT = Path(__file__).parent

st.set_page_config(page_title="Luseradar", page_icon="🐟", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "model.joblib")


@st.cache_data
def load_data():
    preds = pd.read_parquet(ROOT / "data" / "predictions.parquet")
    history = pd.read_parquet(ROOT / "data" / "lice.parquet",
                              columns=["locality_no", "year", "week", "adult_female", "lice_limit", "sea_temp"])
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    return preds, history, metrics


bundle = load_model()
preds, history, metrics = load_data()
H = bundle["horizon"]

st.title("🐟 Luseradar")
st.write(
    f"Sannsynligheten for at et oppdrettsanlegg er **over lusegrensen om {H} uker**, "
    f"basert på åpne lusetellinger fra BarentsWatch. Siste telleuke: **{metrics['prediction_week']}**."
)

tab_map, tab_site, tab_try, tab_model = st.tabs(["Kart", "Anlegg", "Prøv modellen", "Om modellen"])

# --- Kart ---------------------------------------------------------------------------
with tab_map:
    areas = ["Alle"] + sorted(preds["area_name"].dropna().unique())
    c1, c2 = st.columns([1, 1])
    area = c1.selectbox("Produksjonsområde", areas)
    min_risk = c2.slider("Vis anlegg med risiko over", 0.0, 1.0, 0.0, 0.05)
    shown = preds if area == "Alle" else preds[preds["area_name"] == area]
    shown = shown[shown["risk"] >= min_risk].dropna(subset=["lat", "lon"]).copy()

    # Grønn (lav risiko) til rød (høy risiko).
    shown["r"] = (255 * shown["risk"].clip(0, 1)).astype(int)
    shown["g"] = (200 * (1 - shown["risk"].clip(0, 1))).astype(int)
    shown["risk_pct"] = (100 * shown["risk"]).round(0)
    layer = pdk.Layer(
        "ScatterplotLayer", data=shown, get_position="[lon, lat]", get_fill_color="[r, g, 60, 200]",
        get_radius=4000, radius_min_pixels=3, pickable=True,
    )
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=65, longitude=13, zoom=3.6),
        tooltip={"text": "{locality_name} ({locality_no})\nRisiko: {risk_pct} %\nVoksne hunnlus nå: {lice}"},
    ))
    alerts = int(shown["alert"].sum())
    st.caption(f"{len(shown)} anlegg vist, {alerts} med varsel (risiko ≥ {bundle['threshold']:.0%}).")
    st.dataframe(
        shown[["locality_name", "locality_no", "area_name", "lice", "limit_future", "risk"]]
        .rename(columns={"locality_name": "Anlegg", "locality_no": "Lokalitetsnr", "area_name": "Område",
                         "lice": "Hunnlus nå", "limit_future": f"Grense om {H} uker", "risk": "Risiko"})
        .head(50),
        hide_index=True,
        column_config={"Risiko": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)},
    )

# --- Enkeltanlegg ---------------------------------------------------------------------
with tab_site:
    options = preds.sort_values("locality_name")
    labels = options["locality_name"] + " (" + options["locality_no"].astype(str) + ")"
    choice = st.selectbox("Velg anlegg", options["locality_no"], format_func=dict(zip(options["locality_no"], labels)).get)
    row = preds[preds["locality_no"] == choice].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Risiko om {H} uker", f"{row['risk']:.0%}", "Varsel" if row["alert"] else "Ikke varsel",
              delta_color="inverse" if row["alert"] else "off")
    c2.metric("Voksne hunnlus per fisk nå", f"{row['lice']:.2f}",
              None if pd.isna(row["lice_lag1"]) else f"{row['lice'] - row['lice_lag1']:+.2f} fra forrige uke",
              delta_color="inverse")
    c3.metric(f"Lusegrense om {H} uker", f"{row['limit_future']:.1f}")

    h = history[history["locality_no"] == choice].copy()
    h["dato"] = pd.to_datetime(h["year"].astype(str) + "-" + h["week"].astype(str).str.zfill(2) + "-1", format="%G-%V-%u")
    h = h.sort_values("dato").tail(104).set_index("dato")
    st.line_chart(h[["adult_female", "lice_limit"]].rename(columns={"adult_female": "Voksne hunnlus",
                                                                       "lice_limit": "Lusegrense"}))
    st.caption("Siste to år med lusetellinger.")

# --- Prøv modellen selv ------------------------------------------------------------------
with tab_try:
    st.write("Legg inn egne tall, så regner modellen ut risikoen direkte.")
    c1, c2, c3 = st.columns(3)
    lice = c1.number_input("Voksne hunnlus nå", 0.0, 10.0, 0.25, 0.01)
    lag1 = c1.number_input("Forrige uke", 0.0, 10.0, 0.18, 0.01)
    lag2 = c1.number_input("For to uker siden", 0.0, 10.0, 0.12, 0.01)
    lag3 = c1.number_input("For tre uker siden", 0.0, 10.0, 0.10, 0.01)
    mobile = c2.number_input("Bevegelige lus nå", 0.0, 30.0, 0.8, 0.05)
    mobile_lag1 = c2.number_input("Bevegelige lus forrige uke", 0.0, 30.0, 0.6, 0.05)
    stationary = c2.number_input("Fastsittende lus nå", 0.0, 30.0, 0.4, 0.05)
    temp = c3.slider("Sjøtemperatur (°C)", 0.0, 20.0, 12.0, 0.5)
    week = c3.slider("Uke", 1, 52, 36)
    area_names = preds.dropna(subset=["area_id"]).drop_duplicates("area_id").sort_values("area_id")
    area_id = c3.selectbox("Produksjonsområde", area_names["area_id"],
                           format_func=dict(zip(area_names["area_id"], area_names["area_name"])).get)
    area_mean = c3.number_input("Snitt hunnlus i området", 0.0, 5.0, 0.2, 0.01)

    future_week = (week + H - 1) % 52 + 1
    limit = float(bundle["limits"].get((area_id, future_week), 0.5))
    lags = [lice, lag1, lag2, lag3]
    x = pd.DataFrame([{
        "lice": lice, "lice_lag1": lag1, "lice_lag2": lag2, "lice_lag3": lag3,
        "lice_diff1": lice - lag1, "lice_roll4_mean": np.mean(lags), "lice_roll4_max": max(lags),
        "mobile": mobile, "stationary": stationary, "mobile_lag1": mobile_lag1,
        "sea_temp": temp, "sea_temp_roll4": temp,
        "limit_future": limit, "ratio_to_limit": lice / limit,
        "area_lice_mean": area_mean, "area_lice_mean_lag1": area_mean,
        "week_sin": np.sin(2 * np.pi * week / 52), "week_cos": np.cos(2 * np.pi * week / 52),
        "lat": float(preds.loc[preds["area_id"] == area_id, "lat"].median()), "area_id": area_id,
    }])[FEATURES]
    p = float(bundle["model"].predict_proba(x)[0, 1])
    st.metric(f"Sannsynlighet for å være over grensen ({limit:.1f}) i uke {future_week}", f"{p:.0%}",
              "Varsel" if p >= bundle["threshold"] else "Ikke varsel",
              delta_color="inverse" if p >= bundle["threshold"] else "off")

# --- Om modellen --------------------------------------------------------------------
with tab_model:
    st.subheader(f"Modell: {metrics['chosen_model']}")
    st.write(
        f"Trent på tellinger {metrics['split']['train']}, validert på {metrics['split']['validation']} og testet "
        f"på {metrics['split']['test']}. Terskelen ({metrics['threshold']:.2f}) er valgt på valideringssettet for å få "
        "høyest mulig recall med høyst 3 falske alarmer per anlegg per år."
    )
    t = pd.DataFrame(metrics["test"]).T[["pr_auc", "recall", "precision", "false_alarms_per_site_year",
                                         "recall_new_exceedances"]]
    st.dataframe(t.rename(columns={"pr_auc": "PR-AUC", "recall": "Recall", "precision": "Precision",
                                   "false_alarms_per_site_year": "Falske alarmer / anlegg / år",
                                   "recall_new_exceedances": "Recall, nye overskridelser"}))
    figs = ROOT / "reports" / "figures"
    c1, c2 = st.columns(2)
    if (figs / "pr_curve.png").exists():
        c1.image(str(figs / "pr_curve.png"))
    if (figs / "feature_importance.png").exists():
        c2.image(str(figs / "feature_importance.png"))
    st.info("Luseradar er et støtteverktøy. Det erstatter ikke lovpålagte tellinger eller Mattilsynets vurderinger.")

st.caption("Data: BarentsWatch / Mattilsynet, Norsk lisens for offentlige data (NLOD).")
