"""Partie 2 : Prévision de la consommation d'énergie (Energy Forecasting).

Usage : python forecasting.py [clean_data.csv] [dossier_models] [dossier_results]

Ce script entraîne et évalue des modèles Random Forest qui prédisent
Usage_kWh (kWh par intervalle de 15 min). Il compare deux horizons réellement
"prévisionnels" à des modèles naïfs de référence :

- H15  : prévision à 15 min  (utilise les lags 15 min, 1 h, 24 h, la moyenne 1 h)
- H24  : prévision à 24 h    (n'utilise que des informations connues la veille :
                              lags 24 h / 48 h / 168 h + variables calendaires)

Il évalue aussi le modèle initial de la partie 2 (20 variables électriques
mesurées en même temps que la cible). Son R² est quasi parfait car il utilise
le CO2 et la puissance réactive du MEME intervalle : c'est une ESTIMATION,
pas une prévision (fuite d'information). Il est gardé ici pour comparaison.

Séparation temporelle : 80 % d'entraînement (début) / 20 % de test (fin),
sans mélange aléatoire, pour ne pas voir le futur.

Sorties :
- models/forecast_h15.joblib, models/forecast_h24.joblib (+ listes de variables)
- results/forecast_metrics.json
- results/forecast_actual_vs_pred.png
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "Usage_kWh"
TEST_SHARE = 0.20
SEED = 42

# Variables du modèle initial (identiques à models/model_features.joblib)
LEGACY_FEATURES = [
    "Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)", "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor",
    "NSM", "hour", "minute", "dayofweek", "month", "is_weekend",
    "WeekStatus_Weekend", "Day_of_week_Monday", "Day_of_week_Saturday",
    "Day_of_week_Sunday", "Day_of_week_Thursday", "Day_of_week_Tuesday",
    "Day_of_week_Wednesday", "Load_Type_Maximum_Load", "Load_Type_Medium_Load",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les variables de prévision (sans fuite : tout est décalé)."""
    df = df.copy()
    y = df[TARGET]
    # Calendaires (connues à l'avance)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["is_working_hours"] = (
        (df["hour"] >= 8) & (df["hour"] <= 18) & (df["is_weekend"] == 0)
    ).astype(int)
    # Historique (intervalles de 15 min : 4 = 1 h, 96 = 24 h, 672 = 7 j)
    df["lag_15m"] = y.shift(1)
    df["lag_1h"] = y.shift(4)
    df["lag_24h"] = y.shift(96)
    df["lag_48h"] = y.shift(192)
    df["lag_168h"] = y.shift(672)
    df["roll_mean_1h"] = y.shift(1).rolling(4).mean()
    df["roll_std_1h"] = y.shift(1).rolling(4).std()
    return df


CALENDAR = ["NSM", "hour_sin", "hour_cos", "dayofweek", "is_weekend",
            "month_sin", "month_cos", "is_working_hours"]
FEATURES_H15 = CALENDAR + ["lag_15m", "lag_1h", "lag_24h", "roll_mean_1h", "roll_std_1h"]
FEATURES_H24 = CALENDAR + ["lag_24h", "lag_48h", "lag_168h"]


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def make_rf(n_estimators=80):
    # min_samples_leaf élevé : modèle léger (quelques Mo) et moins de sur-apprentissage
    return RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=20, max_features=0.7,
        n_jobs=-1, random_state=SEED)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "clean_data.csv"
    models_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "models")
    results_dir = Path(sys.argv[3] if len(sys.argv) > 3 else "results")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(src, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = add_features(raw)
    results = {}

    # --- Modèles H15 et H24 + références naïves ---------------------------------
    plot_series = {}
    for name, feats, naive_col in [("H15", FEATURES_H15, "lag_15m"),
                                   ("H24", FEATURES_H24, "lag_24h")]:
        d = df.dropna(subset=feats + [TARGET]).reset_index(drop=True)
        cut = int(len(d) * (1 - TEST_SHARE))
        tr, te = d.iloc[:cut], d.iloc[cut:]
        model = make_rf().fit(tr[feats], tr[TARGET])
        pred = model.predict(te[feats])
        results[name] = {
            "random_forest": metrics(te[TARGET], pred),
            "naif": metrics(te[TARGET], te[naive_col]),
            "naif_definition": "même valeur qu'il y a 15 min" if name == "H15"
                               else "même valeur qu'hier à la même heure",
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "test_start": str(te["date"].iloc[0]), "test_end": str(te["date"].iloc[-1]),
            "features": feats,
        }
        joblib.dump(model, models_dir / f"forecast_{name.lower()}.joblib", compress=3)
        plot_series[name] = pd.DataFrame({"date": te["date"], "reel": te[TARGET], "pred": pred})
        imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
        results[name]["importances"] = {k: round(float(v), 4) for k, v in imp.items()}
        print(f"[{name}] RF {results[name]['random_forest']} | naïf {results[name]['naif']}")

    # --- Modèle initial P2 (estimation avec mesures simultanées) ------------------
    leg = pd.get_dummies(raw, columns=["WeekStatus", "Day_of_week", "Load_Type"], drop_first=True)
    missing = [c for c in LEGACY_FEATURES if c not in leg.columns]
    if not missing:
        cut = int(len(leg) * (1 - TEST_SHARE))
        tr, te = leg.iloc[:cut], leg.iloc[cut:]
        m = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=SEED)
        m.fit(tr[LEGACY_FEATURES], tr[TARGET])
        results["modele_initial_P2"] = {
            "random_forest": metrics(te[TARGET], m.predict(te[LEGACY_FEATURES])),
            "remarque": "Estimation avec mesures électriques du même intervalle "
                        "(CO2, réactif, facteur de puissance) : pas une vraie prévision.",
        }
        print("[P2 initial]", results["modele_initial_P2"]["random_forest"])

    with open(results_dir / "forecast_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # --- Graphique : 1 semaine du jeu de test -----------------------------------
    h15 = plot_series["H15"]
    start = h15["date"].iloc[0] + pd.Timedelta(days=7)
    end = start + pd.Timedelta(days=7)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, key, title in [(axes[0], "H15", "Prévision à 15 min"),
                           (axes[1], "H24", "Prévision à 24 h")]:
        s = plot_series[key]
        s = s[(s["date"] >= start) & (s["date"] < end)]
        ax.plot(s["date"], s["reel"], color="#2a78d6", lw=1.4, label="Réel")
        ax.plot(s["date"], s["pred"], color="#eb6834", lw=1.2, label="Prévu (Random Forest)")
        r = results[key]["random_forest"]
        ax.set_title(f"{title} : MAE {r['MAE']} kWh, R² {r['R2']}", loc="left", fontweight="bold")
        ax.set_ylabel("kWh / 15 min")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(results_dir / "forecast_actual_vs_pred.png", dpi=150)
    print("OK : models/forecast_h15.joblib, forecast_h24.joblib, results/forecast_metrics.json, forecast_actual_vs_pred.png")


if __name__ == "__main__":
    main()
