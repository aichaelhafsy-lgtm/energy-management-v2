"""Partie 4 : Application & Interface (Streamlit) - Energy Management System.

Lancer :  streamlit run app.py

L'application regroupe les 5 parties du projet :
1. Données & EDA        (clean_data.csv)
2. Prévision            (models/forecast_h15.joblib, forecast_h24.joblib)
3. Anomalies & profils  (anomaly_results.csv, random_forest_optimized.pkl)
4. Optimisation & CO2   (energy_optimization.py, paramètres modifiables)

Les fichiers sont cherchés automatiquement dans le dossier du projet, donc
l'application fonctionne même si l'arborescence n'est pas parfaite.
"""
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
LOAD_ORDER = ["Light_Load", "Medium_Load", "Maximum_Load"]
LOAD_FR = {"Light_Load": "Charge légère", "Medium_Load": "Charge moyenne",
           "Maximum_Load": "Charge max."}
CLASS_NAMES = {0: "Light_Load", 1: "Medium_Load", 2: "Maximum_Load"}

# Palette du projet : bleu acier (industrie) + ambre (énergie) + vert (gains / CO2 évité)
STEEL, AMBER, GREEN, RED = "#1f5f8b", "#f2a30f", "#1baf7a", "#d03b3b"
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [STEEL, AMBER, GREEN, RED, "#6c7a89", "#8e5ea2"]


# --------------------------------------------------------------------------- #
# Utilitaires : retrouver les fichiers et importer les modules du projet
# --------------------------------------------------------------------------- #
def locate(*names):
    """Retourne le premier fichier trouvé (le moins profond) parmi `names`."""
    found = []
    for name in names:
        found += [p for p in ROOT.rglob(name) if p.is_file()]
    return min(found, key=lambda p: len(p.parts)) if found else None


@st.cache_resource(show_spinner=False)
def import_project_module(alias, *filenames):
    path = locate(*filenames)
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def need(path, label):
    """Arrête la page avec un message clair si un fichier manque."""
    if path is None:
        st.error(f"Fichier introuvable : {label}. Vérifie qu'il est dans le dossier du projet.")
        st.stop()
    return path


@st.cache_data(show_spinner="Chargement des données...")
def load_clean(path):
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data(show_spinner="Chargement des anomalies...")
def load_anomalies(path):
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)


def show_image(*names, caption=None):
    p = locate(*names)
    if p:
        st.image(str(p), caption=caption, use_container_width=True)


# --------------------------------------------------------------------------- #
# Page 1 : Données & EDA (Partie 1)
# --------------------------------------------------------------------------- #
def page_data():
    st.header("1. Données & analyse exploratoire")
    df = load_clean(need(locate("clean_data.csv"), "clean_data.csv"))

    d0, d1 = df["date"].min().date(), df["date"].max().date()
    c1, c2 = st.columns([2, 2])
    rng = c1.date_input("Période", value=(d0, d1), min_value=d0, max_value=d1)
    loads = c2.multiselect("Type de charge", LOAD_ORDER, default=LOAD_ORDER,
                           format_func=lambda x: LOAD_FR[x])
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        df = df[(df["date"].dt.date >= rng[0]) & (df["date"].dt.date <= rng[1])]
    df = df[df["Load_Type"].isin(loads)]
    if df.empty:
        st.warning("Aucune donnée pour ce filtre.")
        return

    k = st.columns(4)
    k[0].metric("Consommation totale", f"{df['Usage_kWh'].sum() / 1000:,.0f} MWh")
    k[1].metric("Moyenne / 15 min", f"{df['Usage_kWh'].mean():.1f} kWh")
    k[2].metric("Pic / 15 min", f"{df['Usage_kWh'].max():.1f} kWh")
    k[3].metric("CO₂ enregistré", f"{df['CO2(tCO2)'].sum():,.0f} t")

    st.subheader("Consommation dans le temps")
    freq = st.radio("Agrégation", ["15 min", "Heure", "Jour"], index=2, horizontal=True)
    rule = {"15 min": None, "Heure": "h", "Jour": "D"}[freq]
    s = df.set_index("date")["Usage_kWh"]
    s = s if rule is None else s.resample(rule).sum()
    fig = go.Figure(go.Scattergl(x=s.index, y=s.values, mode="lines", line=dict(color=STEEL)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="kWh")
    st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    with a:
        st.subheader("Profil horaire moyen")
        prof = df.groupby(["hour", "WeekStatus"])["Usage_kWh"].mean().reset_index()
        fig = px.line(prof, x="hour", y="Usage_kWh", color="WeekStatus",
                      labels={"Usage_kWh": "kWh / 15 min", "hour": "Heure"})
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.subheader("Répartition des types de charge")
        rep = df["Load_Type"].value_counts().reindex(LOAD_ORDER).dropna().reset_index()
        rep.columns = ["Load_Type", "intervalles"]
        rep["Load_Type"] = rep["Load_Type"].map(LOAD_FR)
        fig = px.pie(rep, names="Load_Type", values="intervalles", hole=0.4)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Corrélations")
    num = ["Usage_kWh", "Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh",
           "CO2(tCO2)", "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor", "NSM"]
    corr = df[num].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Aperçu des données"):
        st.dataframe(df.head(500), use_container_width=True)


# --------------------------------------------------------------------------- #
# Page 2 : Prévision (Partie 2)
# --------------------------------------------------------------------------- #
def page_forecast():
    st.header("2. Prévision de la consommation")
    fc = import_project_module("forecasting", "forecasting.py")
    metrics_path = locate("forecast_metrics.json")
    if fc is None or metrics_path is None:
        st.error("forecasting.py ou forecast_metrics.json introuvable. "
                 "Lance d'abord : python modules/forecasting.py data/clean_data.csv models results")
        st.stop()
    met = json.loads(metrics_path.read_text(encoding="utf-8"))

    horizon = st.radio("Horizon de prévision", ["15 min", "24 h"], horizontal=True)
    key = "H15" if horizon == "15 min" else "H24"
    m = met[key]

    rows = [{"Modèle": "Random Forest", **m["random_forest"]},
            {"Modèle": f"Naïf ({m['naif_definition']})", **m["naif"]}]
    st.dataframe(pd.DataFrame(rows).set_index("Modèle"), use_container_width=True)
    gain = (1 - m["random_forest"]["MAE"] / m["naif"]["MAE"]) * 100
    st.caption(f"Erreur (MAE) réduite de {gain:.0f} % par rapport au modèle naïf. "
               f"Test : {m['test_start'][:10]} → {m['test_end'][:10]} ({m['n_test']} points).")

    model = load_model(need(locate(f"forecast_{key.lower()}.joblib"), f"forecast_{key.lower()}.joblib"))
    raw = load_clean(need(locate("clean_data.csv"), "clean_data.csv")).sort_values("date")
    feats = m["features"]
    data = fc.add_features(raw).dropna(subset=feats + [fc.TARGET])
    data = data[data["date"] >= pd.Timestamp(m["test_start"])].reset_index(drop=True)

    t0, t1 = data["date"].min().date(), data["date"].max().date()
    day = st.date_input("Choisir un jour à prévoir", value=t0, min_value=t0, max_value=t1)
    nd = st.slider("Nombre de jours affichés", 1, 7, 2)
    start = pd.Timestamp(day)
    sel = data[(data["date"] >= start) & (data["date"] < start + pd.Timedelta(days=nd))].copy()
    if sel.empty:
        st.warning("Pas de données pour cette période.")
        return
    sel["pred"] = model.predict(sel[feats])
    mae = float(np.abs(sel[fc.TARGET] - sel["pred"]).mean())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sel["date"], y=sel[fc.TARGET], name="Réel", line=dict(color=STEEL)))
    fig.add_trace(go.Scatter(x=sel["date"], y=sel["pred"], name="Prévu", line=dict(color=AMBER)))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="kWh / 15 min",
                      title=f"Réel vs prévu : MAE sur la période = {mae:.2f} kWh")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Variables les plus importantes")
    imp = pd.Series(m["importances"]).sort_values()
    fig = px.bar(imp, orientation="h", labels={"value": "Importance", "index": ""})
    fig.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if "modele_initial_P2" in met:
        r = met["modele_initial_P2"]["random_forest"]
        st.info(f"Modèle initial (mesures électriques simultanées) : R² = {r['R2']}. "
                "Score très élevé car il utilise le CO₂ et la puissance réactive du même "
                "intervalle : c'est une estimation, pas une vraie prévision. "
                "Les modèles ci-dessus n'utilisent que des informations disponibles à l'avance.")
    show_image("forecast_actual_vs_pred.png")


# --------------------------------------------------------------------------- #
# Page 3 : Anomalies & classification (Partie 3)
# --------------------------------------------------------------------------- #
def page_anomalies():
    st.header("3. Détection d'anomalies & classification des profils")
    an = load_anomalies(need(locate("anomaly_results.csv"), "anomaly_results.csv"))
    tab1, tab2 = st.tabs(["Anomalies (Isolation Forest)", "Classification des profils"])

    with tab1:
        total = len(an)
        n_h = int((an["type_anomaly"] == "Hausse").sum())
        n_b = int((an["type_anomaly"] == "Baisse").sum())
        k = st.columns(4)
        k[0].metric("Relevés analysés", f"{total:,}")
        k[1].metric("Anomalies", f"{n_h + n_b:,}", f"{(n_h + n_b) / total * 100:.1f} %", delta_color="off")
        k[2].metric("Hausses", f"{n_h:,}")
        k[3].metric("Baisses", f"{n_b:,}")

        types = st.multiselect("Types d'anomalies affichés", ["Hausse", "Baisse"], default=["Hausse", "Baisse"])
        d0, d1 = an["date"].min().date(), an["date"].max().date()
        rng = st.date_input("Période", value=(d0, d1), min_value=d0, max_value=d1, key="anom_rng")
        view = an
        if isinstance(rng, (tuple, list)) and len(rng) == 2:
            view = an[(an["date"].dt.date >= rng[0]) & (an["date"].dt.date <= rng[1])]

        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=view["date"], y=view["Usage_kWh"], mode="lines", name="Consommation",
                                   line=dict(color="#b8b7b2", width=1)))
        colors = {"Hausse": RED, "Baisse": STEEL}
        for t in types:
            sub = view[view["type_anomaly"] == t]
            fig.add_trace(go.Scattergl(x=sub["date"], y=sub["Usage_kWh"], mode="markers", name=t,
                                       marker=dict(color=colors[t], size=6)))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="kWh / 15 min")
        st.plotly_chart(fig, use_container_width=True)

        a, b = st.columns(2)
        with a:
            st.subheader("Anomalies par heure")
            byh = an[an["type_anomaly"] != "Normal"].groupby(["hour", "type_anomaly"]).size().reset_index(name="n")
            fig = px.bar(byh, x="hour", y="n", color="type_anomaly", color_discrete_map=colors,
                         labels={"n": "Nombre", "hour": "Heure", "type_anomaly": ""})
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with b:
            st.subheader("Top 15 des anomalies les plus fortes")
            top = an[an["type_anomaly"] != "Normal"].nlargest(15, "anomaly_score")
            st.dataframe(top[["date", "Usage_kWh", "anomaly_score", "type_anomaly"]].round(3),
                         use_container_width=True, hide_index=True)

    with tab2:
        model_p = locate("random_forest_optimized.pkl")
        feats_p = locate("classification_features.pkl")
        if model_p is None or feats_p is None:
            st.warning("Modèle de classification introuvable.")
        else:
            clf, feats = load_model(model_p), load_model(feats_p)
            st.write("Choisis un relevé : le modèle prédit son profil de charge "
                     "(légère, moyenne ou maximale) et on compare avec la vraie valeur.")
            c1, c2 = st.columns(2)
            day = c1.date_input("Jour", value=an["date"].iloc[500].date(),
                                min_value=an["date"].min().date(), max_value=an["date"].max().date(),
                                key="clf_day")
            day_df = an[an["date"].dt.date == day]
            if day_df.empty:
                st.warning("Pas de données ce jour-là.")
            else:
                pred = clf.predict(day_df[feats])
                out = pd.DataFrame({
                    "Heure": day_df["date"].dt.strftime("%H:%M").values,
                    "Consommation (kWh)": day_df["Usage_kWh"].values,
                    "Profil réel": day_df["Load_Type_encoded"].map(CLASS_NAMES).values,
                    "Profil prédit": [CLASS_NAMES[int(p)] for p in pred],
                })
                acc = float((out["Profil réel"] == out["Profil prédit"]).mean() * 100)
                c2.metric("Précision ce jour-là", f"{acc:.1f} %")
                fig = px.scatter(out, x="Heure", y="Consommation (kWh)", color="Profil prédit",
                                 category_orders={"Profil prédit": LOAD_ORDER})
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(out, use_container_width=True, hide_index=True)
            a, b = st.columns(2)
            with a:
                show_image("confusion_matrix_classification.png")
            with b:
                show_image("classification_metrics.png")


# --------------------------------------------------------------------------- #
# Page 4 : Optimisation & CO2 (Partie 5)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Calcul des scénarios...")
def run_scenarios(path, tariffs, shift_share, standby_thr, standby_red, eff_gain, pf_target):
    eo = import_project_module("energy_optimization", "energy_optimization.py", "energy_optimization (1).py")
    df = eo.load_data(str(path))
    p = eo.Params(price=dict(zip(LOAD_ORDER, tariffs)), shift_share=shift_share,
                  standby_threshold_kwh=standby_thr, standby_reduction=standby_red,
                  efficiency_gain=eff_gain, pf_target=pf_target)
    res = eo.run_all(df, p, with_drivers=False, with_sensitivity=False)
    res.pop("params", None)
    return res


def page_optimization():
    st.header("4. Optimisation énergétique & analyse CO₂")
    path = need(locate("clean_data.csv"), "clean_data.csv")
    if import_project_module("energy_optimization", "energy_optimization.py",
                             "energy_optimization (1).py") is None:
        st.error("energy_optimization.py introuvable.")
        st.stop()

    with st.sidebar:
        st.markdown("### Hypothèses")
        t_l = st.number_input("Prix Light_Load (€/kWh)", 0.01, 1.0, 0.06, 0.01)
        t_m = st.number_input("Prix Medium_Load (€/kWh)", 0.01, 1.0, 0.09, 0.01)
        t_x = st.number_input("Prix Maximum_Load (€/kWh)", 0.01, 1.0, 0.13, 0.01)
        shift = st.slider("Part de charge déplaçable (%)", 0, 30, 10) / 100
        sb_thr = st.slider("Seuil de consommation à vide (kWh/15 min)", 1.0, 12.0, 6.0, 0.5)
        sb_red = st.slider("Réduction de la consommation à vide (%)", 0, 50, 20) / 100
        eff = st.slider("Gain d'efficacité des process (%)", 0, 10, 3) / 100
        pf_t = st.slider("Facteur de puissance cible (%)", 90, 100, 95)
        st.caption("Les tarifs et pourcentages sont des hypothèses illustratives.")

    r = run_scenarios(str(path), (t_l, t_m, t_x), shift, sb_thr, sb_red, eff, float(pf_t))
    scen = {"Décalage de charge": r["shift"], "Efficacité process": r["efficiency"],
            "Facteur de puissance": r["power_factor"], "Consommation à vide": r["standby"]}
    tot_cost = sum(v["cost_saved"] for v in scen.values())
    tot_kwh = sum(v["kwh_saved"] for v in scen.values())
    tot_co2 = sum(v["co2_saved_t"] for v in scen.values())

    k = st.columns(4)
    k[0].metric("Coût énergie actuel", f"{r['baseline_cost']:,.0f} €")
    k[1].metric("Gain total estimé", f"{tot_cost:,.0f} €", f"{tot_cost / r['baseline_cost'] * 100:.1f} %", delta_color="off")
    k[2].metric("Énergie évitée", f"{tot_kwh / 1000:,.1f} MWh")
    k[3].metric("CO₂ évité", f"{tot_co2:,.1f} t")
    st.caption("Total approximatif : les scénarios sont supposés indépendants.")

    st.subheader("Actions recommandées")
    st.dataframe(r["recommendations"], use_container_width=True, hide_index=True)

    gains = pd.DataFrame({"Scénario": list(scen.keys()),
                          "Gain (€/an)": [v["cost_saved"] for v in scen.values()]})
    fig = px.bar(gains.sort_values("Gain (€/an)"), x="Gain (€/an)", y="Scénario", orientation="h")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Analyse du CO₂")
    c = st.columns(3)
    c[0].metric("CO₂ enregistré", f"{r['co2']['co2_recorded_t']:,.0f} t")
    c[1].metric("CO₂ estimé", f"{r['co2']['co2_estimated_t']:,.0f} t")
    c[2].metric("Facteur d'émission", f"{r['co2']['ef_kg_per_kwh']:.3f} kgCO₂/kWh")

    a, b = st.columns(2)
    with a:
        g = r["co2_load"].reset_index()
        g["Load_Type"] = g["Load_Type"].map(LOAD_FR)
        fig = px.bar(g, x="Load_Type", y=["time_share_pct", "co2_share_pct"], barmode="group",
                     labels={"value": "%", "Load_Type": "", "variable": ""},
                     title="Part du temps vs part du CO₂")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        h = r["co2_hour"].reset_index()
        fig = px.line(h, x="hour", y=[c for c in h.columns if c != "hour"],
                      labels={"value": "kgCO₂ / h", "hour": "Heure", "variable": ""},
                      title="CO₂ moyen par heure")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    m = r["co2_month"].reset_index()
    fig = px.bar(m, x="month", y="co2_t", labels={"co2_t": "tCO₂", "month": "Mois"},
                 title="CO₂ par mois")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Graphiques du rapport (Partie 5)"):
        show_image("co2_analysis.png")
        show_image("drivers.png")
        show_image("optimization.png")


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
PAGES = {
    "1. Données & EDA": page_data,
    "2. Prévision": page_forecast,
    "3. Anomalies & profils": page_anomalies,
    "4. Optimisation & CO₂": page_optimization,
}


def main():
    st.set_page_config(page_title="Energy Management System", page_icon="⚡", layout="wide")
    st.sidebar.title("⚡ Energy Management System")
    st.sidebar.caption("Steel Industry : consommation, prévision, anomalies, optimisation")
    choice = st.sidebar.radio("Navigation", list(PAGES.keys()))
    PAGES[choice]()


if __name__ == "__main__":
    main()
