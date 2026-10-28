"""Trener og evaluerer Luseradar-modellen.

Leser data/lice.parquet (fra fetch_lice.py), bygger features, splitter på tid,
sammenligner baselines med ML-modeller, velger terskel på valideringssettet og
evaluerer én gang på testsettet. Til slutt trenes den valgte modellen på alle data
og brukes til å predikere risiko for den nyeste uka.

Utdata:
    models/model.joblib         modellen + featureliste + terskel (brukes av app.py)
    models/metrics.json         alle tall som står i rapporten
    data/predictions.parquet    risiko per anlegg for nyeste uke (brukes av app.py)
    reports/figures/*.png       figurer til rapporten

Bruk:
    python scripts/train.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import FEATURES, HORIZON, build_features, limit_table  # noqa: E402

DATA = ROOT / "data" / "lice.parquet"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

# Tidsbasert splitt: modellen ser aldri framtiden under trening.
TRAIN_END, VAL_YEAR = 2022, 2023  # trening <= 2022, validering 2023, test >= 2024
MAX_FALSE_ALARMS = 3  # per anlegg per år (forretningskrav fra rapporten)
SEED = 42


def split(d: pd.DataFrame):
    y = d["date"].dt.isocalendar()["year"]
    # Fjerner rader der label-uka havner i neste periode, så ingen label lekker over grensen.
    target_year = d["target_date"].dt.isocalendar()["year"]
    train = d[(y <= TRAIN_END) & (target_year <= TRAIN_END)]
    val = d[(y == VAL_YEAR) & (target_year == VAL_YEAR)]
    test = d[y > VAL_YEAR]
    return train, val, test


def evaluate(y, score, pred, d) -> dict:
    """Metrikker for ubalanserte klasser + forretningsmetrikkene fra rapporten."""
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    site_years = d.groupby(["locality_no", d["date"].dt.year]).ngroups
    new = (d["lice"] <= d["limit_future"]).to_numpy()  # anlegg som er UNDER grensen nå
    return {
        "pr_auc": round(float(average_precision_score(y, score)), 4),
        "roc_auc": round(float(roc_auc_score(y, score)), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "false_alarms_per_site_year": round(fp / max(site_years, 1), 3),
        # Den vanskelige og nyttige delen: nye overskridelser som ikke er synlige i dag.
        "recall_new_exceedances": round(float(pred[new & (y == 1)].mean()) if (new & (y == 1)).any() else 0.0, 4),
        "n": int(len(y)),
        "positives": int(y.sum()),
    }


def pick_threshold(y, score, d) -> float:
    """Høyest mulig recall, gitt maks MAX_FALSE_ALARMS falske alarmer per anlegg per år."""
    site_years = d.groupby(["locality_no", d["date"].dt.year]).ngroups
    best_t, best_recall = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 97):
        pred = score >= t
        fp = (pred & (y == 0)).sum()
        recall = (pred & (y == 1)).sum() / max(y.sum(), 1)
        if fp / site_years <= MAX_FALSE_ALARMS and recall > best_recall:
            best_t, best_recall = float(t), recall
    return best_t


def main() -> None:
    raw = pd.read_parquet(DATA)
    limits = limit_table(raw)
    d = build_features(raw, limits)
    labeled = d.dropna(subset=["lice", "y"]).reset_index(drop=True)
    train, val, test = split(labeled)
    print(f"Rader med label: {len(labeled):,}  (positiv andel {labeled['y'].mean():.1%})")
    print(f"Trening {len(train):,} | validering {len(val):,} | test {len(test):,}\n")

    Xtr, ytr = train[FEATURES], train["y"].to_numpy()
    Xva, yva = val[FEATURES], val["y"].to_numpy()
    Xte, yte = test[FEATURES], test["y"].to_numpy()

    # --- Baselines (ingen læring) ---
    baselines = {
        # «Alltid nei»: accuracy blir høy, men den fanger ingenting.
        "Alltid nei": lambda X: np.zeros(len(X)),
        # Persistens: om to uker er det som nå.
        "Persistens (lus nå / grense)": lambda X: X["ratio_to_limit"].fillna(0).to_numpy(),
        # Lineær trend: forleng siste ukes endring to uker fram.
        "Lineær trend": lambda X: ((X["lice"] + HORIZON * X["lice_diff1"].fillna(0)) / X["limit_future"]).to_numpy(),
    }

    # --- ML-modeller ---
    candidates = {
        "Logistisk regresjon": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=20, max_samples=0.3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED,
        ),
    }
    # Litt hyperparametersøk for gradient boosting, valgt på validering.
    for lr in (0.05, 0.1):
        for leaves in (15, 31, 63):
            candidates[f"Gradient boosting (lr={lr}, leaves={leaves})"] = HistGradientBoostingClassifier(
                learning_rate=lr, max_leaf_nodes=leaves, max_iter=400, early_stopping=True,
                validation_fraction=0.1, random_state=SEED,
            )

    val_scores = {}
    for name, fn in baselines.items():
        val_scores[name] = average_precision_score(yva, fn(Xva))
        print(f"{name:<45} val PR-AUC {val_scores[name]:.3f}")
    fitted = {}
    for name, model in candidates.items():
        model.fit(Xtr, ytr)
        fitted[name] = model
        val_scores[name] = average_precision_score(yva, model.predict_proba(Xva)[:, 1])
        print(f"{name:<45} val PR-AUC {val_scores[name]:.3f}")

    best_name = max(fitted, key=lambda n: val_scores[n])
    best = fitted[best_name]
    print(f"\nValgt modell: {best_name}")

    # --- Terskel på validering, så én evaluering på test ---
    threshold = pick_threshold(yva, best.predict_proba(Xva)[:, 1], val)
    print(f"Terskel valgt på validering: {threshold:.2f}\n")

    results = {}
    # Baselines bruker den naturlige terskelen 1.0 (= over grensen).
    for name, fn in baselines.items():
        s = fn(Xte)
        results[name] = evaluate(yte, s, (s > 1.0).astype(int), test)
    for name in dict.fromkeys(["Logistisk regresjon", "Random forest", best_name]):
        m = fitted[name]
        t = threshold if name == best_name else pick_threshold(yva, m.predict_proba(Xva)[:, 1], val)
        s = m.predict_proba(Xte)[:, 1]
        results[name] = evaluate(yte, s, (s >= t).astype(int), test)

    table = pd.DataFrame(results).T
    print("Resultater på testsettet:")
    print(table[["pr_auc", "roc_auc", "recall", "precision", "false_alarms_per_site_year",
                 "recall_new_exceedances"]].to_string())

    # --- Figurer ---
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, s in [("Persistens", baselines["Persistens (lus nå / grense)"](Xte)),
                    ("Logistisk regresjon", fitted["Logistisk regresjon"].predict_proba(Xte)[:, 1]),
                    (best_name, best.predict_proba(Xte)[:, 1])]:
        p, r, _ = precision_recall_curve(yte, s)
        ax.plot(r, p, label=f"{name} (AP={average_precision_score(yte, s):.2f})")
    ax.axhline(yte.mean(), ls="--", c="grey", label=f"Tilfeldig ({yte.mean():.2f})")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall, testsett ({HORIZON} uker fram)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr_curve.png", dpi=150)

    sample = test.sample(min(20000, len(test)), random_state=SEED)
    imp = permutation_importance(best, sample[FEATURES], sample["y"], scoring="average_precision",
                                 n_repeats=5, random_state=SEED, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    importance.plot.barh(ax=ax)
    ax.set(xlabel="Fall i PR-AUC når featuren stokkes", title="Permutation importance (testsett)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    print("\nViktigste features:\n" + importance.sort_values(ascending=False).head(8).round(4).to_string())

    # --- Endelig modell: samme oppsett, trent på alle data med label ---
    final = fitted[best_name].__class__(**fitted[best_name].get_params())
    final.fit(labeled[FEATURES], labeled["y"])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES, "threshold": threshold, "horizon": HORIZON,
                 "limits": limits, "name": best_name}, MODEL_DIR / "model.joblib", compress=3)

    # Prediksjoner for nyeste uke der de fleste anleggene har rapportert
    # (inneværende uke er ofte bare delvis rapportert).
    counts = d[d["lice"].notna()].groupby("date").size()
    latest_date = counts[counts >= 0.8 * counts.tail(10).median()].index.max()
    now = d[(d["date"] == latest_date) & d["lice"].notna()].copy()
    now["risk"] = final.predict_proba(now[FEATURES])[:, 1]
    now["alert"] = now["risk"] >= threshold
    cols = ["locality_no", "locality_name", "lat", "lon", "area_id", "area_name", "year", "week",
            "lice", "lice_lag1", "sea_temp", "limit_future", "risk", "alert"]
    now[cols].sort_values("risk", ascending=False).to_parquet(ROOT / "data" / "predictions.parquet", index=False)

    metrics = {
        "horizon_weeks": HORIZON,
        "split": {"train": f"<= {TRAIN_END}", "validation": str(VAL_YEAR), "test": f">= {VAL_YEAR + 1}",
                  "n_train": len(train), "n_val": len(val), "n_test": len(test)},
        "positive_rate": round(float(labeled["y"].mean()), 4),
        "validation_pr_auc": {k: round(float(v), 4) for k, v in val_scores.items()},
        "chosen_model": best_name,
        "threshold": threshold,
        "test": results,
        "permutation_importance": importance.sort_values(ascending=False).round(5).to_dict(),
        "prediction_week": f"{int(now['year'].iloc[0])}-W{int(now['week'].iloc[0]):02d}",
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPrediksjoner for uke {metrics['prediction_week']}: {int(now['alert'].sum())} av {len(now)} anlegg varslet.")
    print("Lagret models/model.joblib, models/metrics.json, data/predictions.parquet, reports/figures/")


if __name__ == "__main__":
    main()
