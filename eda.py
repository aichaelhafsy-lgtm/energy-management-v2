"""Analyse Exploratoire des Données (EDA) - Steel Industry Energy Consumption.

Usage : python eda.py clean_data.csv

Génère les statistiques descriptives dans la console et sauvegarde deux planches
graphiques prêtes pour le rapport :
- eda_dashboard.png (distribution, types de charge, profil horaire, jours)
- eda_correlation.png (matrice de corrélation)
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(path: str) -> None:
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(path, parse_dates=["date"])

    print("=== 1. DIMENSIONS ET VALEURS MANQUANTES ===")
    print(f"Lignes : {df.shape[0]} | Colonnes : {df.shape[1]} | NaN : {df.isna().sum().sum()}")

    print("\n=== 2. STATISTIQUES DESCRIPTIVES (Usage_kWh) ===")
    print(df["Usage_kWh"].describe().round(2))

    print("\n=== 3. REPARTITION DES TYPES DE CHARGE (Load_Type) ===")
    summary_load = pd.DataFrame({
        "Nombre": df["Load_Type"].value_counts(),
        "Pourcentage (%)": (df["Load_Type"].value_counts(normalize=True) * 100).round(2),
    })
    print(summary_load)

    print("\n=== 4. SEMAINE VS WEEK-END ===")
    print(df.groupby("WeekStatus")["Usage_kWh"].agg(["count", "mean", "median", "max"]).round(2))

    # --- Planche 1 : 4 graphiques clés ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Distribution Usage_kWh
    sns.histplot(df["Usage_kWh"], bins=50, kde=True, color="#2b5c8f", ax=axes[0, 0])
    axes[0, 0].axvline(df["Usage_kWh"].mean(), color="red", linestyle="--", label=f"Moyenne: {df['Usage_kWh'].mean():.2f} kWh")
    axes[0, 0].axvline(df["Usage_kWh"].median(), color="green", linestyle="-", label=f"Médiane: {df['Usage_kWh'].median():.2f} kWh")
    axes[0, 0].set_title("1. Distribution de la consommation (Usage_kWh)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Consommation (kWh)")
    axes[0, 0].set_ylabel("Fréquence")
    axes[0, 0].legend()

    # Répartition Load_Type
    order_load = ["Light_Load", "Medium_Load", "Maximum_Load"]
    sns.countplot(data=df, x="Load_Type", order=order_load, palette="viridis", ax=axes[0, 1])
    axes[0, 1].set_title("2. Répartition des types de charge (Load_Type)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Type de charge")
    axes[0, 1].set_ylabel("Nombre de relevés (15 min)")
    for p in axes[0, 1].patches:
        pct = 100 * p.get_height() / len(df)
        axes[0, 1].annotate(
            f"{int(p.get_height())}\n({pct:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Profil horaire Semaine vs Week-end
    hourly_wk = df.groupby(["hour", "WeekStatus"])["Usage_kWh"].mean().reset_index()
    sns.lineplot(data=hourly_wk, x="hour", y="Usage_kWh", hue="WeekStatus", marker="o", linewidth=2.5, ax=axes[1, 0])
    axes[1, 0].set_title("3. Consommation moyenne par heure (Semaine vs Week-end)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Heure de la journée (0h - 23h)")
    axes[1, 0].set_ylabel("Consommation moyenne (kWh)")
    axes[1, 0].set_xticks(range(0, 24, 2))

    # Consommation par jour de la semaine
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_stats = df.groupby("Day_of_week")["Usage_kWh"].mean().reindex(days_order)
    sns.barplot(x=day_stats.index, y=day_stats.values, palette="coolwarm", ax=axes[1, 1])
    axes[1, 1].set_title("4. Consommation moyenne par jour de la semaine", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Jour de la semaine")
    axes[1, 1].set_ylabel("Consommation moyenne (kWh)")
    axes[1, 1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig("eda_dashboard.png", dpi=150)
    plt.close()

    # --- Planche 2 : Matrice de corrélation ---
    num_cols = [
        "Usage_kWh",
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
        "NSM",
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
    ]
    plt.figure(figsize=(11, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, linewidths=0.5)
    plt.title("5. Matrice de corrélation des variables numériques", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("eda_correlation.png", dpi=150)
    plt.close()
    print("\nGraphiques sauvegardés : eda_dashboard.png et eda_correlation.png")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "clean_data.csv"
    run_eda(csv_path)
