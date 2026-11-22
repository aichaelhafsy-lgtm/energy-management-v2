# Energy Management System

Projet de gestion de l'énergie d'une aciérie (Steel Industry Energy Consumption, relevés toutes les 15 min).

| Partie | Contenu | Fichiers principaux |
|---|---|---|
| 1. Data Analysis & Preprocessing | nettoyage, EDA, feature engineering | `modules/preprocessing.py`, `modules/eda.py`, `modules/feature_engineering.py` |
| 2. Energy Forecasting | prévision à 15 min et à 24 h | `modules/forecasting.py`, `models/forecast_h15.joblib`, `models/forecast_h24.joblib`, `results/forecast_metrics.json` |
| 3. Anomaly Detection & Classification | Isolation Forest + Random Forest | `modules/detection-classification/` |
| 4. Application & Interface | application Streamlit | `app.py` |
| 5. Energy Optimization & CO2 | scénarios d'économie, CO2 | `modules/energy_optimization.py`, `modules/part5_report.py`, `results/` |

## Structure recommandée

```
Energy_Management_System/
├── app.py
├── requirements.txt, requirements-desktop.txt
├── desktop_app.py, Lancer_Fenetre.bat, Lancer_App.bat
├── README.md
├── data/                 clean_data.csv, featured_data.csv, recommendations.csv
├── models/               forecast_h15.joblib, forecast_h24.joblib, rf_model.joblib, model_features.joblib
├── modules/              preprocessing.py, eda.py, feature_engineering.py, forecasting.py,
│   │                     energy_optimization.py, part5_report.py
│   └── detection-classification/   scripts, modèles .pkl, anomaly_results.csv, graphes
├── results/              graphes PNG, forecast_metrics.json, results_summary.json
└── docs/                 rapport, présentation, captures/
```

## Installation et lancement

```bash
pip install -r requirements.txt
streamlit run app.py
```

L'application cherche les fichiers automatiquement dans le dossier du projet.

## Lancement sous Windows

| Fichier | Effet |
|---|---|
| `Lancer_Fenetre.bat` | ouvre l'application dans sa propre fenêtre (sans navigateur), via `desktop_app.py` |
| `Lancer_App.bat` | lance l'application dans le navigateur (`localhost:8501`) |

Prérequis : Python 3.12 installé avec « Add python.exe to PATH ».
`Lancer_Fenetre.bat` installe `pywebview` automatiquement (`requirements-desktop.txt`).
Pour un lancement manuel : `python -m streamlit run app.py`.

## Régénérer les résultats

```bash
# Partie 2 : entraîner et évaluer les modèles de prévision
python modules/forecasting.py data/clean_data.csv models results

# Partie 5 : graphes CO2, facteurs d'influence, optimisation
cd modules
python part5_report.py ../data/clean_data.csv ../results
```

`part5_report.py` fait `import energy_optimization` : le fichier doit donc s'appeler
`energy_optimization.py` (sans `(1)`).

## Résultats de la partie 2 (jeu de test = 20 % les plus récents)

| Horizon | MAE Random Forest (kWh) | MAE naïf (kWh) | R² RF |
|---|---|---|---|
| 15 min | 4,19 | 5,45 | 0,92 |
| 24 h | 14,37 | 14,97 | 0,50 |

Le modèle initial `rf_model.joblib` (R² = 0,998) utilise le CO2 et la puissance réactive
du même intervalle : c'est une estimation, pas une prévision. Les nouveaux modèles
n'utilisent que des informations disponibles à l'avance (lags, calendrier).

## Limites

- Une seule année de données (2018) : la saisonnalité annuelle ne peut pas être apprise.
- Les tarifs (0,06 / 0,09 / 0,13 €/kWh) et les pourcentages de gain sont des hypothèses
  illustratives, modifiables dans l'application (page Optimisation).
