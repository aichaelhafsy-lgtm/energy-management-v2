# =====================================================
# DETECTION DES ANOMALIES (Isolation Forest)
# =====================================================
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import joblib
import webbrowser
import os

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1. Charger les donnees
df = pd.read_csv("featured_data.csv", parse_dates=["date"])
if "hour" not in df.columns:
    df["hour"] = df["date"].dt.hour

# 2. Variables utilisees par le modele
FEATURES = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "hour_sin",
    "hour_cos",
    "is_working_hours",
    "Usage_kWh_rolling_std_1h",
]

# 3. Normalisation + entrainement
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])

model = IsolationForest(n_estimators=300, contamination=0.02, random_state=42)
model.fit(X_scaled)

# 4. Detection
df["is_anomaly"] = (model.predict(X_scaled) == -1).astype(int)   # 1 = anomalie
df["anomaly_score"] = -model.score_samples(X_scaled)             # plus haut = plus anormal

print("===== RESULTATS DETECTION =====")
print("Nombre total      :", len(df))
print("Nombre d'anomalies:", df["is_anomaly"].sum())

# 5. Type d'anomalie : Hausse / Baisse
#    Reference UNIQUE = mediane de la consommation pendant les heures de travail
ref = df.loc[df["is_working_hours"] == 1, "Usage_kWh"].median()
print("\nReference (mediane heures de travail) :", round(ref, 2), "kWh")

df["type_anomaly"] = "Normal"
df.loc[(df["is_anomaly"] == 1) & (df["Usage_kWh"] > ref), "type_anomaly"] = "Hausse"
df.loc[(df["is_anomaly"] == 1) & (df["Usage_kWh"] <= ref), "type_anomaly"] = "Baisse"

print("\n===== TYPES D'ANOMALIES =====")
print(df["type_anomaly"].value_counts())

hausse = df[df["type_anomaly"] == "Hausse"]
baisse = df[df["type_anomaly"] == "Baisse"]

# 6. Sauvegarde des resultats et des modeles 
df.to_csv("anomaly_results.csv", index=False)
joblib.dump(model, "isolation_forest.pkl", compress=3)
joblib.dump(scaler, "scaler_anomaly.pkl")
joblib.dump(FEATURES, "anomaly_features.pkl")
joblib.dump(ref, "anomaly_reference.pkl")      # seuil Hausse/Baisse
print("\nFichiers sauvegardes : anomaly_results.csv + 4 fichiers .pkl")

# 7. Graphe interactif (HTML) : sauvegarde + ouverture dans le navigateur
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["date"], y=df["Usage_kWh"], mode="lines",
                         name="Consommation",
                         line=dict(color="lightsteelblue", width=1), hoverinfo="skip"))
for data, nom, couleur in [(hausse, "Anomalies Hausse", "red"),
                           (baisse, "Anomalies Baisse", "orange")]:
    fig.add_trace(go.Scatter(
        x=data["date"], y=data["Usage_kWh"], mode="markers", name=nom,
        marker=dict(color=couleur, size=7), customdata=data["anomaly_score"],
        hovertemplate="<b>%{x|%d %b %Y a %Hh%M}</b><br>"
                      "Consommation : %{y:.2f} kWh<br>"
                      "Score : %{customdata:.3f}<extra></extra>"))
fig.update_layout(title="Anomalies detectees (Isolation Forest)",
                  yaxis_title="Usage_kWh", hovermode="closest")
fig.write_html("anomaly_timeline.html")

webbrowser.open("file://" + os.path.abspath("anomaly_timeline.html"))

# 8. Graphe 1 : timeline
plt.figure(figsize=(15, 6))
plt.plot(df["date"], df["Usage_kWh"], color="lightblue", linewidth=0.7,
         alpha=0.7, label="Consommation energetique")
plt.scatter(hausse["date"], hausse["Usage_kWh"], color="red", s=18,
            label=f"Anomalies Hausse ({len(hausse)})")
plt.scatter(baisse["date"], baisse["Usage_kWh"], color="orange", s=18,
            label=f"Anomalies Baisse ({len(baisse)})")
plt.title("Timeline des anomalies de consommation")
plt.xlabel("Date")
plt.ylabel("Consommation (kWh)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graphe_1_timeline.png", dpi=300)
plt.show()

# 9. Graphe 2 : anomalies de hausse par heure (toutes les heures affichees)
plt.figure(figsize=(10, 5))
hausse["hour"].value_counts().reindex(range(24), fill_value=0).plot(kind="bar", color="red")
plt.title("Anomalies de forte consommation (Hausse) par heure")
plt.xlabel("Heure")
plt.ylabel("Nombre d'anomalies")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graphe_2_hausse_par_heure.png", dpi=300)
plt.show()

# 10. Graphe 3 : anomalies de baisse par heure
plt.figure(figsize=(11, 5))
baisse["hour"].value_counts().reindex(range(24), fill_value=0).plot(
    kind="bar", color="orange", edgecolor="black")
plt.title("Anomalies de faible consommation (Baisse) par heure")
plt.xlabel("Heure de la journee")
plt.ylabel("Nombre d'anomalies")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("graphe_3_baisse_par_heure.png", dpi=300)
plt.show()

# 11. Graphe 4 : distribution normal / hausse / baisse
ordre = ["Normal", "Hausse", "Baisse"]
counts = df["type_anomaly"].value_counts().reindex(ordre, fill_value=0)
pct = counts / counts.sum() * 100

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(ordre, counts.values, color=["#3498DB", "#E74C3C", "#F39C12"], height=0.55)
for bar, c, p in zip(bars, counts.values, pct.values):
    ax.text(bar.get_width() + counts.sum() * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{c:,} ({p:.2f}%)", va="center", fontsize=11)
ax.set_title("Distribution des consommations normales et anormales",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Nombre d'observations")
ax.set_xlim(0, counts.max() * 1.18)
ax.grid(axis="x", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("graphe4_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n===== TERMINE =====")