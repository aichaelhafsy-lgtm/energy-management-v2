"""FactoryMind AI - Partie 5 : Optimisation énergétique & analyse CO2.

Module importable (ex. par l'application Streamlit) : aucune dépendance graphique.
Toutes les fonctions prennent le DataFrame `clean_data.csv` (sortie de la partie 1)
et un objet `Params` contenant les hypothèses modifiables (tarifs, % de décalage...).

Fonctions principales
---------------------
process_efficiency_scenario : gain d'efficacité des process (hypothèse)
emission_factor        : facteur d'émission réel (tCO2/kWh) déduit des données
co2_*                  : analyse des émissions (par charge, heure, mois)
driver_analysis        : facteurs qui expliquent la consommation (Random Forest)
standby_scenario       : réduction de la consommation à vide
power_factor_scenario  : compensation de l'énergie réactive (batterie de condensateurs)
load_shift             : décalage de charge optimal (programmation linéaire)
build_recommendations  : tableau final des actions classées par gain
run_all                : exécute tout et renvoie un dictionnaire de résultats

Les tarifs sont des HYPOTHESES illustratives (le dataset n'en contient pas) :
à remplacer par la facture réelle de l'usine.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

LOAD_ORDER = ["Light_Load", "Medium_Load", "Maximum_Load"]
COL_KWH = "Usage_kWh"
COL_CO2 = "CO2(tCO2)"
COL_LAG_Q = "Lagging_Current_Reactive.Power_kVarh"
COL_LEAD_Q = "Leading_Current_Reactive_Power_kVarh"
COL_LAG_PF = "Lagging_Current_Power_Factor"
COL_LEAD_PF = "Leading_Current_Power_Factor"


@dataclass
class Params:
    """Hypothèses des scénarios (toutes modifiables depuis l'interface)."""
    # Prix du kWh selon la période tarifaire = Load_Type (€/kWh, illustratif)
    price: dict = field(default_factory=lambda: {
        "Light_Load": 0.06, "Medium_Load": 0.09, "Maximum_Load": 0.13})
    # Décalage de charge
    shift_share: float = 0.10          # part de la charge (Medium/Max) déplaçable
    receive_cap_kwh: float | None = None  # plafond kWh/15min des créneaux d'accueil
    #                                      (None = moyenne des créneaux Maximum_Load)
    # Consommation à vide
    standby_threshold_kwh: float = 6.0  # <= seuil : l'usine est à l'arrêt/à vide
    standby_reduction: float = 0.20     # réduction visée de cette consommation
    # Efficacité des process en charge
    efficiency_gain: float = 0.03       # hypothèse : -3% sur les relevés en production
    # Facteur de puissance
    pf_threshold: float = 90.0          # seuil de pénalité (%)
    pf_target: float = 95.0             # objectif après compensation (%)
    pf_penalty_per_kvarh: float = 0.01  # €/kVArh réactif excédentaire (illustratif)
    network_loss_share: float = 0.02    # pertes Joule internes (% de l'énergie)
    currency: str = "€"


# --------------------------------------------------------------------------- #
# Chargement et outils
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def price_series(df: pd.DataFrame, p: Params) -> pd.Series:
    return df["Load_Type"].map(p.price)


def baseline_cost(df: pd.DataFrame, p: Params) -> float:
    return float((df[COL_KWH] * price_series(df, p)).sum())


# --------------------------------------------------------------------------- #
# 1. Analyse CO2
# --------------------------------------------------------------------------- #
def emission_factor(df: pd.DataFrame, min_kwh: float = 40.0) -> float:
    """Facteur d'émission (tCO2/kWh).

    La colonne CO2 est arrondie à 0,01 tCO2 : pour les faibles consommations
    (< ~11 kWh) elle vaut 0 alors que l'émission réelle n'est pas nulle.
    On estime donc le facteur sur les relevés >= `min_kwh` où l'arrondi est négligeable.
    """
    big = df[df[COL_KWH] >= min_kwh]
    return float((big[COL_CO2] / big[COL_KWH]).median())


def co2_summary(df: pd.DataFrame, ef: float) -> dict:
    tot_kwh = df[COL_KWH].sum()
    n_top = int(len(df) * 0.10)
    return {
        "kwh_total": float(tot_kwh),
        "co2_recorded_t": float(df[COL_CO2].sum()),
        "co2_estimated_t": float(tot_kwh * ef),
        "ef_kg_per_kwh": ef * 1000,
        "zero_co2_intervals_pct": float((df[COL_CO2] == 0).mean() * 100),
        "top10pct_intervals_share_kwh": float(df[COL_KWH].nlargest(n_top).sum() / tot_kwh * 100),
    }


def co2_by_load_type(df: pd.DataFrame, ef: float) -> pd.DataFrame:
    g = df.groupby("Load_Type")[COL_KWH].agg(intervals="count", kwh="sum", mean_kwh="mean")
    g = g.reindex(LOAD_ORDER)
    g["time_share_pct"] = g["intervals"] / g["intervals"].sum() * 100
    g["co2_t"] = g["kwh"] * ef
    g["co2_share_pct"] = g["co2_t"] / g["co2_t"].sum() * 100
    return g.round(2)


def co2_by_hour(df: pd.DataFrame, ef: float) -> pd.DataFrame:
    """kgCO2 moyen émis par heure (4 relevés de 15 min), semaine vs week-end."""
    d = df.assign(kg=df[COL_KWH] * ef * 1000)
    g = d.groupby(["WeekStatus", "hour"])["kg"].mean() * 4
    return g.unstack(0).rename_axis(None, axis=1)


def co2_by_month(df: pd.DataFrame, ef: float) -> pd.Series:
    return (df.groupby("month")[COL_KWH].sum() * ef).rename("co2_t")


# --------------------------------------------------------------------------- #
# 2. Facteurs d'influence (Random Forest + importance par permutation)
# --------------------------------------------------------------------------- #
TEMPORAL = ["NSM", "dayofweek", "month", "is_weekend", "Load_Type_encoded"]
ELECTRICAL = [COL_LAG_Q, COL_LEAD_Q, COL_LAG_PF, COL_LEAD_PF]


def driver_analysis(df: pd.DataFrame, seed: int = 42) -> dict:
    """Quelles variables expliquent Usage_kWh ?

    CO2 est exclu (calculé à partir de kWh = fuite de cible), ainsi que les lags
    (inertie, pas un levier d'action). Découpage chronologique 80/20.
    """
    d = df.copy()
    d["Load_Type_encoded"] = d["Load_Type"].map({k: i for i, k in enumerate(LOAD_ORDER)})
    cut = int(len(d) * 0.8)
    tr, te = d.iloc[:cut], d.iloc[cut:]

    def fit(cols):
        rf = RandomForestRegressor(n_estimators=120, min_samples_leaf=5,
                                   n_jobs=-1, random_state=seed)
        rf.fit(tr[cols], tr[COL_KWH])
        pred = rf.predict(te[cols])
        return rf, r2_score(te[COL_KWH], pred), mean_absolute_error(te[COL_KWH], pred)

    rf_t, r2_t, mae_t = fit(TEMPORAL)
    _, r2_f, mae_f = fit(TEMPORAL + ELECTRICAL)
    # Importance sur le modèle "leviers d'action" (calendrier + période tarifaire) :
    # les variables électriques reconstruisent kWh par identité physique
    # (P = Q * PF / sqrt(1-PF²), R² ~ 0.998) : ce sont des conséquences, pas des leviers.
    pi = permutation_importance(rf_t, te[TEMPORAL], te[COL_KWH], n_repeats=5,
                                random_state=seed, n_jobs=-1, scoring="r2")
    imp = (pd.DataFrame({"feature": TEMPORAL, "importance": pi.importances_mean,
                         "std": pi.importances_std})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    corr = d[[COL_KWH] + TEMPORAL + ELECTRICAL].corr()[COL_KWH].drop(COL_KWH).round(2)
    return {"importance": imp, "r2_temporal": r2_t, "mae_temporal": mae_t,
            "r2_full": r2_f, "mae_full": mae_f, "corr_with_usage": corr}


# --------------------------------------------------------------------------- #
# 3. Scénarios d'optimisation
# --------------------------------------------------------------------------- #
def _gain(kwh, ef, cost):
    return {"kwh_saved": float(kwh), "co2_saved_t": float(kwh * ef), "cost_saved": float(cost)}


def standby_scenario(df: pd.DataFrame, ef: float, p: Params) -> dict:
    """Réduire la consommation à vide (nuit, week-end, arrêts)."""
    mask = df[COL_KWH] <= p.standby_threshold_kwh
    idle_kwh = df.loc[mask, COL_KWH].sum()
    idle_cost = (df.loc[mask, COL_KWH] * price_series(df, p)[mask]).sum()
    out = _gain(p.standby_reduction * idle_kwh, ef, p.standby_reduction * idle_cost)
    out.update(idle_kwh=float(idle_kwh), idle_share_kwh_pct=float(idle_kwh / df[COL_KWH].sum() * 100),
               idle_share_time_pct=float(mask.mean() * 100))
    return out


def process_efficiency_scenario(df: pd.DataFrame, ef: float, p: Params) -> dict:
    """Efficacité énergétique des process en charge (variateurs, régulation, récupération).

    HYPOTHESE de gain (non déduite des données) appliquée aux relevés > seuil de veille.
    """
    mask = df[COL_KWH] > p.standby_threshold_kwh
    kwh = df.loc[mask, COL_KWH]
    cost = (kwh * price_series(df, p)[mask]).sum()
    out = _gain(p.efficiency_gain * kwh.sum(), ef, p.efficiency_gain * cost)
    out.update(active_kwh=float(kwh.sum()))
    return out


def power_factor_scenario(df: pd.DataFrame, ef: float, p: Params) -> dict:
    """Compensation de l'énergie réactive inductive (batterie de condensateurs).

    Q_nécessaire = Q_lag - P * tan(acos(PF_cible)). Gains : pénalité réactive évitée
    + réduction des pertes Joule (courant ~ S = sqrt(P²+Q²), pertes ~ I²).
    """
    P, Q = df[COL_KWH], df[COL_LAG_Q]
    tan_t = np.tan(np.arccos(p.pf_target / 100))
    tan_th = np.tan(np.arccos(p.pf_threshold / 100))
    low = df[COL_LAG_PF] < p.pf_threshold

    q_needed = (Q - P * tan_t).clip(lower=0).where(low, 0.0)
    excess = (Q - P * tan_th).clip(lower=0).where(low, 0.0)
    penalty_saved = excess.sum() * p.pf_penalty_per_kvarh

    s_before = np.sqrt(P**2 + Q**2)
    s_after = np.sqrt(P**2 + (Q - q_needed) ** 2)
    ratio2 = np.where(s_before > 0, (s_after / s_before.replace(0, np.nan)) ** 2, 1.0)
    ratio2 = np.nan_to_num(ratio2, nan=1.0)
    loss_kwh = p.network_loss_share * float((P * (1 - ratio2))[low].sum())
    loss_cost = float((p.network_loss_share * P * (1 - ratio2) * price_series(df, p))[low].sum())

    pf_after = np.where(s_after > 0, P / s_after.replace(0, np.nan) * 100, 100.0)
    qc_kvar = (q_needed[q_needed > 0] * 4)  # kVArh/15min -> kVAr
    out = _gain(loss_kwh, ef, loss_cost + penalty_saved)
    out.update(
        penalty_saved=float(penalty_saved), loss_cost_saved=loss_cost,
        pct_intervals_below=float(low.mean() * 100),
        pct_kwh_below=float(P[low].sum() / P.sum() * 100),
        pf_weighted_before=float(P.sum() / s_before.sum() * 100),
        pf_weighted_after=float(P.sum() / s_after.sum() * 100),
        bank_kvar_p90=float(qc_kvar.quantile(0.90)), bank_kvar_p95=float(qc_kvar.quantile(0.95)),
        bank_kvar_max=float(qc_kvar.max()),
        pf_after=pd.Series(np.nan_to_num(pf_after, nan=100.0), index=df.index),
    )
    return out


def load_shift(df: pd.DataFrame, ef: float, p: Params) -> dict:
    """Décalage de charge optimal par programmation linéaire, jour par jour.

    Pour chaque jour (96 créneaux de 15 min) :
      variables  s_t >= 0 (charge retirée du créneau t), r_t >= 0 (charge ajoutée)
      minimiser  sum_t prix_t * (r_t - s_t)
      sous       sum s_t = sum r_t                       (énergie conservée)
                 s_t <= shift_share * x_t  si t est Medium/Max  (part flexible)
                 r_t <= max(cap - x_t, 0)                 (pas de nouveau pic > cap)
    L'énergie (kWh) et donc le CO2 ne changent pas : seul le coût baisse
    (le facteur d'émission du dataset est constant).
    """
    d = df[["date", COL_KWH, "Load_Type"]].copy()
    d["day"] = (d["date"] - pd.Timedelta(minutes=15)).dt.normalize()  # jour 00:15 -> 24:00
    price = price_series(d, p).to_numpy()
    x = d[COL_KWH].to_numpy()
    cap = p.receive_cap_kwh or float(d.loc[d["Load_Type"] == "Maximum_Load", COL_KWH].mean())
    flexible = (d["Load_Type"] != "Light_Load").to_numpy()

    new = x.copy()
    moved = 0.0
    for _, idx in d.groupby("day").indices.items():
        xs, ps, n = x[idx], price[idx], len(idx)
        ub_s = np.where(flexible[idx], p.shift_share * xs, 0.0)
        ub_r = np.clip(cap - xs, 0, None)
        # léger départage : on préfère remplir les créneaux déjà actifs (continuité de process)
        tie = 1e-4 * (1 - np.clip(xs / cap, 0, 1))
        c = np.concatenate([-ps, ps + tie])
        A_eq = np.concatenate([np.ones(n), -np.ones(n)])[None, :]
        bounds = np.column_stack([np.zeros(2 * n), np.concatenate([ub_s, ub_r])])
        res = linprog(c, A_eq=A_eq, b_eq=[0.0], bounds=bounds, method="highs")
        if res.status != 0:
            continue
        s, r = res.x[:n], res.x[n:]
        new[idx] = xs - s + r
        moved += float(np.minimum(s, 1e12).sum())

    out_df = df.copy()
    out_df["Usage_kWh_shifted"] = new
    cost_b = float((x * price).sum())
    cost_a = float((new * price).sum())
    month = d["date"].dt.month.to_numpy()
    peak_b = pd.Series(x).groupby(month).max().mean() * 4   # kW (moyenne des pics mensuels)
    peak_a = pd.Series(new).groupby(month).max().mean() * 4
    out = _gain(0.0, ef, cost_b - cost_a)
    out.update(kwh_moved=moved, cost_before=cost_b, cost_after=cost_a,
               peak_kw_before=float(peak_b), peak_kw_after=float(peak_a),
               max_kwh_before=float(x.max()), max_kwh_after=float(new.max()),
               cap_kwh=float(cap), shifted_df=out_df)
    return out


def sensitivity(df: pd.DataFrame, ef: float, p: Params, steps=(0.05, 0.10, 0.15, 0.20, 0.30)) -> pd.DataFrame:
    """Gain (% du coût de base) selon l'intensité de chaque mesure."""
    base = baseline_cost(df, p)
    rows = []
    for s in steps:
        q = Params(**{**p.__dict__, "shift_share": s, "standby_reduction": s})
        shift = load_shift(df, ef, q)["cost_saved"]
        stand = standby_scenario(df, ef, q)["cost_saved"]
        rows.append({"intensity_pct": s * 100, "shift_cost_saved_pct": shift / base * 100,
                     "standby_cost_saved_pct": stand / base * 100,
                     "standby_co2_saved_t": standby_scenario(df, ef, q)["co2_saved_t"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Recommandations
# --------------------------------------------------------------------------- #
def build_recommendations(res: dict, p: Params) -> pd.DataFrame:
    base = res["baseline_cost"]
    kwh_tot = res["co2"]["kwh_total"]
    s, pf, sh, ef_ = res["standby"], res["power_factor"], res["shift"], res["efficiency"]
    cur = p.currency
    rows = [
        {"Action": "Réduire la consommation à vide (nuit, week-end, arrêts)",
         "Levier": "Pilotage des utilitaires : veille, extinction, compresseurs/ventilation à la demande",
         "Hypothèse": f"-{p.standby_reduction:.0%} des relevés <= {p.standby_threshold_kwh:g} kWh/15min "
                      f"({s['idle_share_kwh_pct']:.1f}% de l'énergie, {s['idle_share_time_pct']:.0f}% du temps)",
         "kWh évités": s["kwh_saved"], "tCO2 évitées": s["co2_saved_t"], "Gain": s["cost_saved"]},
        {"Action": "Améliorer l'efficacité des process en production",
         "Levier": "Variateurs de vitesse, régulation des fours/moteurs, récupération de chaleur, maintenance",
         "Hypothèse": f"-{p.efficiency_gain:.0%} sur {ef_['active_kwh']/1000:.0f} MWh de relevés en production (hypothèse, à valider par audit)",
         "kWh évités": ef_["kwh_saved"], "tCO2 évitées": ef_["co2_saved_t"], "Gain": ef_["cost_saved"]},
        {"Action": "Décaler la charge des heures chères vers les heures creuses",
         "Levier": "Planification : process flexibles (fours de maintien, pompage, broyage, lots) hors Maximum_Load",
         "Hypothèse": f"{p.shift_share:.0%} de la charge Medium/Max déplaçable dans la journée, "
                      f"plafond d'accueil {sh['cap_kwh']:.0f} kWh/15min ; "
                      f"pic mensuel moyen {sh['peak_kw_before']:.0f} -> {sh['peak_kw_after']:.0f} kW",
         "kWh évités": 0.0, "tCO2 évitées": 0.0, "Gain": sh["cost_saved"]},
        {"Action": "Corriger le facteur de puissance (batterie de condensateurs automatique)",
         "Levier": "Compensation d'énergie réactive, par gradins régulés (évite la sur-compensation la nuit)",
         "Hypothèse": f"PF {pf['pf_weighted_before']:.1f}% -> {pf['pf_weighted_after']:.1f}% ; "
                      f"{pf['pct_intervals_below']:.0f}% des relevés sous {p.pf_threshold:g}% ; "
                      f"batterie ~{pf['bank_kvar_p90']:.0f} kVAr (P90)",
         "kWh évités": pf["kwh_saved"], "tCO2 évitées": pf["co2_saved_t"], "Gain": pf["cost_saved"]},
    ]
    rec = pd.DataFrame(rows).sort_values("Gain", ascending=False).reset_index(drop=True)
    rec.insert(0, "Priorité", range(1, len(rec) + 1))
    rec[f"Gain ({cur}/an)"] = rec.pop("Gain")
    rec["% coût énergie"] = rec[f"Gain ({cur}/an)"] / base * 100
    rec["% kWh"] = rec["kWh évités"] / kwh_tot * 100
    tot = {"Priorité": "", "Action": "TOTAL (approximatif, mesures supposées indépendantes)",
           "Levier": "", "Hypothèse": "",
           "kWh évités": rec["kWh évités"].sum(), "tCO2 évitées": rec["tCO2 évitées"].sum(),
           f"Gain ({cur}/an)": rec[f"Gain ({cur}/an)"].sum(),
           "% coût énergie": rec["% coût énergie"].sum(), "% kWh": rec["% kWh"].sum()}
    return pd.concat([rec, pd.DataFrame([tot])], ignore_index=True)


# --------------------------------------------------------------------------- #
def run_all(df: pd.DataFrame, p: Params | None = None, with_drivers: bool = True,
            with_sensitivity: bool = True) -> dict:
    p = p or Params()
    ef = emission_factor(df)
    res = {
        "params": p, "ef": ef, "baseline_cost": baseline_cost(df, p),
        "co2": co2_summary(df, ef),
        "co2_load": co2_by_load_type(df, ef),
        "co2_hour": co2_by_hour(df, ef),
        "co2_month": co2_by_month(df, ef),
        "standby": standby_scenario(df, ef, p),
        "efficiency": process_efficiency_scenario(df, ef, p),
        "power_factor": power_factor_scenario(df, ef, p),
        "shift": load_shift(df, ef, p),
    }
    if with_drivers:
        res["drivers"] = driver_analysis(df)
    if with_sensitivity:
        res["sensitivity"] = sensitivity(df, ef, p)
    res["recommendations"] = build_recommendations(res, p)
    return res
