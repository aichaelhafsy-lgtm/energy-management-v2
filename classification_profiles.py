# =====================================================
# CLASSIFICATION DES PROFILS ENERGETIQUES
# RANDOM FOREST OPTIMISE
# =====================================================

# 1. IMPORTATION DES BIBLIOTHEQUES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)


# 2. CHARGEMENT DES DONNEES

df = pd.read_csv("featured_data.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print("===== CLASSIFICATION DES PROFILS ENERGETIQUES =====")

print("\nDimensions du dataset :", df.shape)
print("\nValeurs manquantes :", df.isnull().sum().sum())

mapping = {
    0: "Light_Load",
    1: "Medium_Load",
    2: "Maximum_Load"
}

print("\nDistribution des profils :")
print(df["Load_Type_encoded"].value_counts().sort_index())

print("\nCorrespondance des classes :")
for code, name in mapping.items():
    print(code, "=", name)

print("\nPremiere date :", df["date"].min())
print("Derniere date :", df["date"].max())


# 3. SELECTION DES FEATURES ET TARGET

FEATURES = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "Total_Reactive_Power_kVarh",
    "Apparent_Power_kVAh",
    "Usage_kWh_lag_15m",
    "Usage_kWh_lag_1h",
    "Usage_kWh_lag_24h",
    "Usage_kWh_rolling_mean_1h",
    "Usage_kWh_rolling_std_1h",
    # variables de calendrier (ameliorent la classification)
    "hour",
    "dayofweek",
    "is_weekend",
    "NSM"
]

X = df[FEATURES]
y = df["Load_Type_encoded"]

print("\n===== SELECTION DES VARIABLES =====")
print("Nombre de features :", len(FEATURES))
print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)


# 4. SEPARATION TRAIN / TEST

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n===== SEPARATION TRAIN / TEST =====")
print("Nombre de donnees Train :", len(X_train))
print("Nombre de donnees Test :", len(X_test))

print("\nPeriode Train :")
print(df["date"].iloc[0], "->", df["date"].iloc[split_index - 1])

print("\nPeriode Test :")
print(df["date"].iloc[split_index], "->", df["date"].iloc[-1])


# 5. OPTIMISATION RANDOM FOREST

param_grid = {
    "n_estimators": [150, 200],
    "max_depth": [15, 25, None],
    "min_samples_leaf": [2, 4],
    "max_features": ["sqrt", 0.8],
    "class_weight": ["balanced", "balanced_subsample"]
}

tscv = TimeSeriesSplit(n_splits=3)

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_grid,
    n_iter=6,
    scoring="f1_macro",
    cv=tscv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\n===== OPTIMISATION TERMINEE =====")
print("Meilleurs parametres :")
print(search.best_params_)

print("\nMeilleur F1-score validation :")
print(round(search.best_score_, 4))


# 6. EVALUATION DU MODELE OPTIMISE

y_pred_optimized = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_optimized)
f1_macro = f1_score(y_test, y_pred_optimized, average="macro")

print("\n===== RESULTATS DU MODELE OPTIMISE =====")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("F1-score Macro :", round(f1_macro, 4))

print("\n===== RAPPORT DE CLASSIFICATION =====")

print(classification_report(
    y_test,
    y_pred_optimized,
    labels=[0, 1, 2],
    target_names=["Light_Load", "Medium_Load", "Maximum_Load"]
))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred_optimized))


# 7. SAUVEGARDE DU MODELE

joblib.dump(best_model, "random_forest_optimized.pkl", compress=3)
joblib.dump(FEATURES, "classification_features.pkl")

print("\nModele sauvegarde avec succes !")


# 8. TEST DU MODELE SAUVEGARDE

loaded_model = joblib.load("random_forest_optimized.pkl")
loaded_features = joblib.load("classification_features.pkl")

predictions = loaded_model.predict(X_test[loaded_features])

print("\n===== TEST FINAL =====")
print("Modele recharge avec succes !")
print("Nombre de variables :", len(loaded_features))
print("Exemple de prediction :", predictions[0])


# 9. GRAPHIQUE : MATRICE DE CONFUSION

cm = confusion_matrix(
    y_test,
    y_pred_optimized,
    labels=[0, 1, 2]
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Light_Load", "Medium_Load", "Maximum_Load"],
    yticklabels=["Light_Load", "Medium_Load", "Maximum_Load"]
)

plt.title("Matrice de confusion - Classification des profils")
plt.xlabel("Classe predite")
plt.ylabel("Classe reelle")

plt.tight_layout()
plt.savefig("confusion_matrix_classification.png", dpi=300)
plt.show()


# 10. GRAPHIQUE : DISTRIBUTION DES PROFILS

profile_counts = (
    df["Load_Type_encoded"]
    .value_counts()
    .reindex([0, 1, 2], fill_value=0)
)

profile_names = ["Light_Load", "Medium_Load", "Maximum_Load"]

plt.figure(figsize=(8, 6))

plt.bar(profile_names, profile_counts.values)

plt.title("Distribution des profils energetiques")
plt.xlabel("Type de profil")
plt.ylabel("Nombre d'observations")

plt.tight_layout()
plt.savefig("distribution_profiles.png", dpi=300)
plt.show()


# 11. GRAPHIQUE : ACTUAL VS PREDICTED

actual_counts = y_test.value_counts().reindex([0, 1, 2], fill_value=0)

predicted_counts = (
    pd.Series(y_pred_optimized)
    .value_counts()
    .reindex([0, 1, 2], fill_value=0)
)

labels = ["Light_Load", "Medium_Load", "Maximum_Load"]
x = range(3)
width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    [i - width/2 for i in x],
    actual_counts.values,
    width=width,
    label="Valeurs reelles"
)

plt.bar(
    [i + width/2 for i in x],
    predicted_counts.values,
    width=width,
    label="Valeurs predites"
)

plt.title("Comparaison des profils reels et predits")
plt.xlabel("Type de profil")
plt.ylabel("Nombre d'observations")
plt.xticks(list(x), labels)
plt.legend()

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()


# 12. GRAPHIQUE : PRECISION / RECALL / F1-SCORE

precision, recall, f1, support = precision_recall_fscore_support(
    y_test,
    y_pred_optimized,
    labels=[0, 1, 2],
    zero_division=0
)

labels = ["Light_Load", "Medium_Load", "Maximum_Load"]
x = range(3)
width = 0.25

plt.figure(figsize=(9, 6))

plt.bar([i - width for i in x], precision, width, label="Precision")
plt.bar(x, recall, width, label="Recall")
plt.bar([i + width for i in x], f1, width, label="F1-score")

plt.title("Performance du modele par classe")
plt.xlabel("Type de profil")
plt.ylabel("Score")
plt.xticks(list(x), labels)
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.savefig("classification_metrics.png", dpi=300)
plt.show()


# 13. GRAPHIQUE : FEATURE IMPORTANCE OPTIMISEE

importance = pd.Series(
    best_model.feature_importances_,
    index=FEATURES
).sort_values(ascending=True)

plt.figure(figsize=(9, 6))

importance.plot(kind="barh")

plt.title("Importance des variables - Random Forest optimise")
plt.xlabel("Importance")
plt.ylabel("Variables")

plt.tight_layout()
plt.savefig("feature_importance_optimized.png", dpi=300)
plt.show()


# 14. FIN DU PROGRAMME

print("\n===== CLASSIFICATION TERMINEE AVEC SUCCES =====")

print("\nFichiers generes :")
print("- random_forest_optimized.pkl")
print("- classification_features.pkl")
print("- confusion_matrix_classification.png")
print("- distribution_profiles.png")
print("- actual_vs_predicted.png")
print("- classification_metrics.png")
print("- feature_importance_optimized.png")