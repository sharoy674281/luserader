"""Luseradar: Streamlit-nettside. Kjør med `streamlit run app.py`."""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from src.features import FEATURES

ROOT = Path(__file__).parent

st.set_page_config(page_title="Luseradar", page_icon="🐟", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 1400px;}
h1 {font-size: 2rem !important; padding: 0 !important;}
[data-testid="stMetricValue"] {font-size: 1.35rem;}
[data-testid="stMetric"] {padding: .5rem .8rem !important;}
[data-testid="stTabs"] {margin-top: -.6rem;}
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stDeckGlJsonChart"] {animation: fade .35s ease-out;}
@keyframes fade {from {opacity: 0; transform: translateY(4px);} to {opacity: 1; transform: none;}}
.legend {display: flex; flex-wrap: wrap; gap: .9rem; font-size: .85rem; color: #4a5a63; margin-top: .3rem;}
.legend span {display: inline-flex; align-items: center; gap: .35rem;}
.legend i {width: .8rem; height: .8rem; border-radius: 50%; display: inline-block;}
.sub {color: #4a5a63; margin-top: -.4rem;}
.big {font-size: 2.6rem; font-weight: 700; line-height: 1;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Laster modellen …")
def load_model():
    return joblib.load(ROOT / "models" / "model.joblib")


@st.cache_data(show_spinner="Laster lusedata …")
def load_data():
    preds = pd.read_parquet(ROOT / "data" / "predictions.parquet")
    history = pd.read_parquet(ROOT / "data" / "lice.parquet", columns=[
        "locality_no", "locality_name", "year", "week", "adult_female", "mobile", "lice_limit",
        "sea_temp", "lat", "lon", "area_name", "municipality"])
    history = history.dropna(subset=["adult_female"])
    history["date"] = pd.to_datetime(
        history["year"].astype(str) + "-" + history["week"].astype(str).str.zfill(2) + "-1", format="%G-%V-%u")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    return preds, history, metrics


bundle = load_model()
preds, history, metrics = load_data()
H, THRESHOLD = bundle["horizon"], bundle["threshold"]

# Fargeskalaer. Risikobåndene følger kalibreringen: under 5 % skjer en overskridelse i under 1 % av tilfellene.
RISK_BINS = [(0.05, "#2a9d8f", "Lav (< 5 %)"), (0.10, "#a7c957", "Moderat (5–10 %)"),
             (0.20, "#f4c430", "Forhøyet (10–20 %)"), (0.40, "#f4842a", "Høy (20–40 %)"),
             (1.01, "#d62828", "Svært høy (> 40 %)")]
NOW_BINS = [(0.5, "#2a9d8f", "Under halve grensen"), (0.8, "#f4c430", "50–80 % av grensen"),
            (1.0, "#f4842a", "80–100 % av grensen"), (np.inf, "#d62828", "Over grensen")]


def colorize(values: pd.Series, bins) -> list:
    out = []
    for v in values.fillna(0):
        hexcol = next(c for upper, c, _ in bins if v < upper)
        out.append([int(hexcol[i:i + 2], 16) for i in (1, 3, 5)] + [210])
    return out


def legend(bins) -> None:
    items = "".join(f'<span><i style="background:{c}"></i>{label}</span>' for _, c, label in bins)
    st.markdown(f'<div class="legend">{items}</div>', unsafe_allow_html=True)


def risk_label(p: float) -> str:
    return next(label for upper, _, label in RISK_BINS if p < upper).split(" (")[0]


if "site" not in st.session_state:
    st.session_state.site = None


def select_from_map():
    objs = st.session_state.map.selection.objects.get("farms", [])
    if objs:
        st.session_state.site = int(objs[0]["locality_no"])


def select_from_list():
    rows = st.session_state.toplist.selection.rows
    if rows:
        st.session_state.site = int(st.session_state.toplist_ids[rows[0]])


# --- Topp: tittel og nøkkeltall ------------------------------------------------------------
head, k1, k2, k3 = st.columns([2, 0.8, 0.8, 1.4], vertical_alignment="center")
with head:
    st.title("🐟 Luseradar")
    st.markdown(f'<p class="sub">Hvor er det fare for lakselus over grensen om {H} uker? '
                f'Prognose fra telleuke {metrics["prediction_week"]}.</p>', unsafe_allow_html=True)
k1.metric("Anlegg vurdert", f"{len(preds)}", border=True,
          help="Anlegg som rapporterte lusetelling i siste fullstendige telleuke.")
k2.metric("Varsler", f"{int(preds['alert'].sum())}", border=True,
          help=f"Anlegg der modellen gir minst {THRESHOLD:.0%} sannsynlighet for overskridelse. Terskelen er valgt "
               "slik at modellen fanger flest mulig overskridelser med høyst rundt 3 falske alarmer per anlegg per år.")
top_area = preds.groupby("area_name")["risk"].mean().idxmax()
k3.metric("Mest utsatte område", top_area, border=True, help="Produksjonsområdet med høyest gjennomsnittlig risiko.")

tab_radar, tab_try, tab_model = st.tabs(["🗺️ Radar", "🧪 Prøv modellen", "📊 Om modellen"])

# --- Radar: kart + detaljpanel ---------------------------------------------------------------
with tab_radar:
    c1, c2, c3 = st.columns([1.3, 1, 1.2], vertical_alignment="bottom")
    view = c1.segmented_control(
        "Vis", [f"Om {H} uker", "Nå", "Historikk"], default=f"Om {H} uker", required=True,
        help=f"**Om {H} uker:** modellens sannsynlighet for å være over grensen.  \n"
             "**Nå:** siste lusetelling sammenlignet med grensen.  \n"
             "**Historikk:** spol gjennom tidligere uker og se hvordan lusa har spredt seg.")
    area = c2.selectbox("Produksjonsområde", ["Hele kysten"] + sorted(preds["area_name"].dropna().unique()))

    if view == "Historikk":
        counts = history.groupby("date").size()
        weeks = list(counts[counts >= 0.8 * counts.tail(10).median()].index[-104:])  # bare fullt rapporterte uker
        chosen = c3.select_slider("Uke", options=weeks, value=weeks[-1],
                                  format_func=lambda d: f"{d.isocalendar().year} uke {d.isocalendar().week}")
        mapdata = history[history["date"] == chosen].rename(columns={"adult_female": "lice"})
        mapdata = mapdata.assign(value=mapdata["lice"] / mapdata["lice_limit"], risk=np.nan)
        bins = NOW_BINS
        tooltip = "{locality_name}\nVoksne hunnlus: {lice}\nGrense: {lice_limit}"
    else:
        names = preds.sort_values("locality_name")
        c3.selectbox("Søk etter anlegg", names["locality_no"], index=None, placeholder="Skriv navn …", key="search",
                     format_func=dict(zip(names["locality_no"], names["locality_name"])).get,
                     on_change=lambda: st.session_state.update(site=st.session_state.search))
        mapdata = preds.copy()
        if view == "Nå":
            mapdata["value"] = mapdata["lice"] / mapdata["limit_future"]
            bins = NOW_BINS
            tooltip = "{locality_name}\nVoksne hunnlus nå: {lice}"
        else:
            mapdata["value"] = mapdata["risk"]
            bins = RISK_BINS
            tooltip = "{locality_name}\nRisiko om " + str(H) + " uker: {risk_pct} %\nVoksne hunnlus nå: {lice}"

    if area != "Hele kysten":
        mapdata = mapdata[mapdata["area_name"] == area]
    mapdata = mapdata.dropna(subset=["lat", "lon"]).sort_values("value").copy()
    mapdata["color"] = colorize(mapdata["value"], bins)
    mapdata["radius"] = 2500 + 6000 * mapdata["value"].clip(0, 1.5)
    mapdata["risk_pct"] = (100 * mapdata["risk"]).round(0)
    mapdata["lice"] = mapdata["lice"].round(2)

    center = (mapdata["lat"].mean(), mapdata["lon"].mean(), 6.2) if area != "Hele kysten" else (65.0, 15.5, 3.35)
    picked = preds[preds["locality_no"] == st.session_state.site]
    if len(picked):
        center = (picked["lat"].iloc[0], picked["lon"].iloc[0], 7.5)
    deck = pdk.Deck(
        layers=[pdk.Layer(
            "ScatterplotLayer", id="selected", data=picked[["lat", "lon"]], get_position="[lon, lat]",
            get_radius=9000, radius_min_pixels=14, filled=False, stroked=True,
            get_line_color=[20, 35, 43, 230], line_width_min_pixels=2.5,
        ), pdk.Layer(
            "ScatterplotLayer", id="farms", data=mapdata[["locality_no", "locality_name", "lat", "lon", "lice",
                                                          "lice_limit" if view == "Historikk" else "limit_future",
                                                          "risk_pct", "color", "radius"]],
            get_position="[lon, lat]", get_fill_color="color", get_radius="radius",
            radius_min_pixels=3, radius_max_pixels=14, stroked=True, get_line_color=[255, 255, 255, 160],
            line_width_min_pixels=0.5, pickable=True, auto_highlight=True,
            transitions={"getFillColor": 600, "getRadius": 600},
        )],
        initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=center[2],
                                         transition_duration=900),
        tooltip={"text": tooltip},
        map_style="light",
    )

    left, right = st.columns([1.75, 1], gap="medium")
    with left:
        st.pydeck_chart(deck, height=450, on_select=select_from_map, selection_mode="single-object", key="map")
        lg, help_col = st.columns([4, 1], vertical_alignment="center")
        with lg:
            legend(bins)
        with help_col.popover("Hva betyr dette?"):
            st.markdown(f"""
**Lusegrensen** er 0,5 voksne hunnlus per fisk i snitt, og 0,2 om våren når villaksen vandrer ut.
Går et anlegg over, må det avluse.

**Risiko om {H} uker** er modellens sannsynlighet for at anlegget er over grensen ved telling om {H} uker.
Når modellen sier 20–40 %, skjedde det i rundt 20 % av tilfellene i testperioden.

**Varsel** gis fra {THRESHOLD:.0%}. Omtrent 1 av 7 varsler slår til, så et varsel betyr *følg ekstra med*,
ikke *avlus nå*.

Klikk på et anlegg i kartet, i lista eller søk for å se detaljer.
""")

    with right, st.container(height=500, border=True):
        site = st.session_state.site
        if site is None or site not in set(preds["locality_no"]):
            st.markdown(f"**Høyest risiko om {H} uker**")
            st.caption("Velg en rad eller klikk i kartet.")
            top = (preds if area == "Hele kysten" else preds[preds["area_name"] == area]).head(25)
            st.session_state.toplist_ids = top["locality_no"].tolist()
            st.dataframe(
                top[["locality_name", "lice", "risk"]].assign(risk=top["risk"] * 100), hide_index=True, height=410,
                on_select=select_from_list, selection_mode="single-row", key="toplist",
                column_config={
                    "locality_name": st.column_config.TextColumn("Anlegg"),
                    "lice": st.column_config.NumberColumn("Hunnlus nå", format="%.2f", width="small"),
                    "risk": st.column_config.ProgressColumn("Risiko", format="%.0f %%", min_value=0, max_value=100,
                                                            width="medium"),
                },
            )
        else:
            row = preds[preds["locality_no"] == site].iloc[0]
            if st.button("← Tilbake til lista", type="tertiary"):
                st.session_state.site = None
                st.rerun()
            rank = int((preds["risk"] > row["risk"]).sum()) + 1
            color = next(c for upper, c, _ in RISK_BINS if row["risk"] < upper)
            st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem">
  <div><div style="font-size:1.35rem; font-weight:700">{row['locality_name']}</div>
       <div style="color:#4a5a63; font-size:.85rem">Lokalitet {site} · {row['area_name']}</div></div>
  <div style="text-align:right"><div class="big" style="color:{color}">{row['risk']:.0%}</div>
       <div style="font-size:.85rem"><b>{risk_label(row['risk'])}</b> · nr. {rank} av {len(preds)}</div></div>
</div>
<p style="color:#4a5a63; font-size:.9rem; margin:.4rem 0 0">Sannsynlighet for å være over lusegrensen om {H} uker.</p>
""", unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric("Hunnlus nå", f"{row['lice']:.2f}",
                      None if pd.isna(row["lice_lag1"]) else f"{row['lice'] - row['lice_lag1']:+.2f}",
                      delta_color="inverse", help="Gjennomsnittlig antall voksne hunnlus per fisk ved siste telling. "
                                                  "Endringen er fra uka før.")
            m2.metric(f"Grense om {H} uker", f"{row['limit_future']:.1f}",
                      help="Lusegrensen er 0,5 voksne hunnlus per fisk, og 0,2 i vårukene da villaksen vandrer ut.")

            h = history[history["locality_no"] == site].tail(52)
            base = alt.Chart(h).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%b")))
            line = base.mark_line(point=alt.OverlayMarkDef(size=18), color="#0f766e").encode(
                y=alt.Y("adult_female:Q", title="Hunnlus per fisk"),
                tooltip=[alt.Tooltip("date:T", title="Uke", format="%d.%m.%Y"),
                         alt.Tooltip("adult_female:Q", title="Hunnlus"), alt.Tooltip("sea_temp:Q", title="Temp °C")])
            limit = base.mark_line(strokeDash=[4, 3], color="#d62828", interpolate="step-after").encode(y="lice_limit:Q")
            st.altair_chart((limit + line).properties(height=190), width="stretch")
            st.caption("Siste år. Stiplet linje er lusegrensen.")

# --- Prøv modellen ----------------------------------------------------------------------------
with tab_try:
    st.markdown("Fyll inn tall for et tenkt anlegg. Modellen regner ut risikoen med en gang du endrer noe.")
    inputs, result = st.columns([2.2, 1], gap="large")
    with inputs:
        a, b, c = st.columns(3)
        with a.container(border=True):
            st.markdown("**Voksne hunnlus per fisk**")
            lice = st.number_input("Denne uka", 0.0, 10.0, 0.25, 0.01)
            lag1 = st.number_input("For 1 uke siden", 0.0, 10.0, 0.18, 0.01)
            lag2 = st.number_input("For 2 uker siden", 0.0, 10.0, 0.12, 0.01)
            lag3 = st.number_input("For 3 uker siden", 0.0, 10.0, 0.10, 0.01)
        with b.container(border=True):
            st.markdown("**Yngre lus per fisk**", help="Bevegelige og fastsittende lus blir voksne hunnlus de neste "
                                                        "ukene, så de varsler hva som kommer.")
            mobile = st.number_input("Bevegelige nå", 0.0, 30.0, 0.8, 0.05)
            mobile_lag1 = st.number_input("Bevegelige forrige uke", 0.0, 30.0, 0.6, 0.05)
            stationary = st.number_input("Fastsittende nå", 0.0, 30.0, 0.4, 0.05)
        with c.container(border=True):
            st.markdown("**Miljø og sted**")
            temp = st.slider("Sjøtemperatur (°C)", 0.0, 20.0, 12.0, 0.5, help="Lusa utvikler seg raskere i varmt vann.")
            week = st.slider("Uke", 1, 52, 36)
            area_names = preds.dropna(subset=["area_id"]).drop_duplicates("area_id").sort_values("area_id")
            area_id = st.selectbox("Produksjonsområde", area_names["area_id"],
                                   format_func=dict(zip(area_names["area_id"], area_names["area_name"])).get)
            area_mean = st.number_input("Snitt hunnlus i området", 0.0, 5.0, 0.2, 0.01,
                                        help="Smittepress fra naboanlegg.")

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
    color = next(c for upper, c, _ in RISK_BINS if p < upper)
    with result, st.container(border=True):
        st.markdown(f"**Risiko for overskridelse i uke {future_week}**")
        st.markdown(f'<div class="big" style="color:{color}">{p:.0%}</div>', unsafe_allow_html=True)
        st.markdown(f"**{risk_label(p)}** · grensen er {limit:.1f} den uka")
        st.progress(min(p, 1.0))
        if p >= THRESHOLD:
            st.warning(f"Over varslingsterskelen ({THRESHOLD:.0%}). Følg ekstra med og vurder å planlegge behandling.")
        else:
            st.success(f"Under varslingsterskelen ({THRESHOLD:.0%}).")
        st.caption("Prøv å øke de bevegelige lusene eller temperaturen og se hvordan risikoen endrer seg.")

# --- Om modellen ----------------------------------------------------------------------------
with tab_model:
    t = metrics["test"]
    best = t[metrics["chosen_model"]]
    persist = t["Persistens (lus nå / grense)"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fanger overskridelser", f"{best['recall']:.0%}", f"{best['recall'] - persist['recall']:+.0%} mot persistens",
              border=True, help="Recall: andelen av alle overskridelser i testperioden som modellen varslet om to uker før.")
    m2.metric("Nye overskridelser fanget", f"{best['recall_new_exceedances']:.0%}", "persistens: 0 %", border=True,
              delta_color="off", help="Overskridelser hos anlegg som er under grensen i dag. De er umulige å se fra dagens tall.")
    m3.metric("Treffsikkerhet", f"{best['precision']:.0%}", border=True,
              help="Precision: hvor mange av varslene som faktisk ble en overskridelse.")
    m4.metric("Falske alarmer", f"{best['false_alarms_per_site_year']:.1f}", border=True,
              help="Gjennomsnittlig antall varsler per anlegg per år som ikke ble en overskridelse.")

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.markdown(f"""
**Slik fungerer det**
1. Ukentlige lusetellinger fra BarentsWatch (2016 til i dag).
2. For hvert anlegg og hver uke lages {len(FEATURES)} features: lusetall og trend, yngre lus, temperatur, smittepress i området, sesong og grense.
3. Modellen ({metrics['chosen_model'].split(' (')[0].lower()}) lærer om anlegget er over grensen {H} uker senere.
4. Trent på {metrics['split']['train'].replace('<= ', 't.o.m. ')}, valgt på {metrics['split']['validation']}, testet på {metrics['split']['test'].replace('>= ', 'f.o.m. ')}.
""")
        table = pd.DataFrame(t).T[["pr_auc", "recall", "precision", "false_alarms_per_site_year"]].astype(float)
        table[["recall", "precision"]] *= 100
        st.dataframe(table, column_config={
            "_index": st.column_config.TextColumn("Modell"),
            "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.2f", help="Hvor godt modellen rangerer "
                                                    "anlegg etter risiko. Tilfeldig gjetting gir 0,03."),
            "recall": st.column_config.NumberColumn("Recall", format="%.0f %%"),
            "precision": st.column_config.NumberColumn("Precision", format="%.0f %%"),
            "false_alarms_per_site_year": st.column_config.NumberColumn("Falske alarmer/år", format="%.1f"),
        })
    with right:
        figs = ROOT / "reports" / "figures"
        f1, f2 = st.tabs(["Precision-recall", "Viktigste features"])
        f1.image(str(figs / "pr_curve.png"))
        f2.image(str(figs / "feature_importance.png"))
    st.info("Luseradar er et støtteverktøy. Det erstatter ikke lovpålagte tellinger eller Mattilsynets vurderinger.")

st.caption("Data: BarentsWatch / Mattilsynet, Norsk lisens for offentlige data (NLOD). DAT158, HVL 2026.")
