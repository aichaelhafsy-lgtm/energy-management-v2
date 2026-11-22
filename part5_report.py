"""FactoryMind AI - Partie 5 : génération des résultats (graphiques + tableaux).

Usage : python part5_report.py clean_data.csv [dossier_sortie]

Produit dans le dossier de sortie :
- co2_analysis.png        analyse des émissions de CO2
- drivers.png             facteurs d'influence de la consommation
- optimization.png        gains des scénarios d'optimisation
- recommendations.csv     tableau des actions classées par gain
- results_summary.json    chiffres clés (pour le rapport / l'application Streamlit)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import energy_optimization as eo

# Palette validée (slots 1-3 : bleu, orange, aqua), surface claire
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
LOAD_FR = {"Light_Load": "Charge légère", "Medium_Load": "Charge moyenne", "Maximum_Load": "Charge max."}
FEAT_FR = {"Load_Type_encoded": "Période tarifaire\n(Load_Type)", "NSM": "Heure de la journée\n(NSM)",
           "month": "Mois", "dayofweek": "Jour de la semaine", "is_weekend": "Week-end"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "axes.axisbelow": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "legend.frameon": False, "font.size": 10,
})


def style_bars(ax, bars, fmt, pad_frac=0.02):
    top = max(b.get_height() for b in bars) or 1
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * pad_frac, fmt(b.get_height()),
                ha="center", va="bottom", fontsize=9, color=INK)


def fig_co2(df, r):
    ef = r["ef"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    a = ax[0, 0]
    a.scatter(df[eo.COL_KWH], df[eo.COL_CO2], s=5, alpha=0.25, color=BLUE, linewidths=0,
              label="Relevés (CO2 arrondi à 0,01 t)")
    xs = [0, df[eo.COL_KWH].max()]
    a.plot(xs, [x * ef for x in xs], color=ORANGE, lw=2, label=f"Facteur estimé : {ef*1000:.3f} kgCO2/kWh")
    a.set_title("1. CO2 vs consommation : relation proportionnelle")
    a.set_xlabel("Consommation (kWh / 15 min)"); a.set_ylabel("CO2 (tCO2 / 15 min)")
    a.legend(loc="upper left", markerscale=3)

    a = ax[0, 1]
    g = r["co2_load"]; x = range(len(g)); w = 0.38
    b1 = a.bar([i - w / 2 for i in x], g["time_share_pct"], w - 0.04, color=BLUE, label="Part du temps")
    b2 = a.bar([i + w / 2 for i in x], g["co2_share_pct"], w - 0.04, color=ORANGE, label="Part du CO2")
    style_bars(a, list(b1) + list(b2), lambda v: f"{v:.0f}%")
    a.set_xticks(list(x)); a.set_xticklabels([LOAD_FR[i] for i in g.index])
    a.set_ylabel("%"); a.legend(loc="upper right")
    a.set_title("2. La charge max. : 21% du temps, 45% du CO2")

    a = ax[1, 0]
    h = r["co2_hour"]
    a.plot(h.index, h["Weekday"], color=BLUE, lw=2, marker="o", ms=5, label="Semaine")
    a.plot(h.index, h["Weekend"], color=ORANGE, lw=2, marker="o", ms=5, label="Week-end")
    a.set_title("3. Émissions moyennes par heure"); a.set_xlabel("Heure"); a.set_ylabel("kgCO2 / heure")
    a.set_xticks(range(0, 24, 2)); a.legend()

    a = ax[1, 1]
    m = r["co2_month"]
    bars = a.bar(m.index, m.values, color=BLUE, width=0.7)
    style_bars(a, bars, lambda v: f"{v:.0f}")
    a.set_xticks(range(1, 13)); a.set_title("4. Émissions mensuelles estimées (tCO2)")
    a.set_xlabel("Mois"); a.set_ylabel("tCO2")

    fig.suptitle(f"Analyse CO2 : {r['co2']['co2_estimated_t']:.0f} tCO2 estimées sur l'année "
                 f"({r['co2']['co2_recorded_t']:.0f} t enregistrées, arrondi à 0,01 t)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def fig_drivers(df, r):
    d = r["drivers"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    a = ax[0]
    imp = d["importance"].iloc[::-1]
    vals = imp["importance"].clip(lower=0)
    vals = vals / vals.sum() * 100
    bars = a.barh([FEAT_FR[f] for f in imp["feature"]], vals, color=BLUE, height=0.6)
    for b, v in zip(bars, vals):
        a.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", fontsize=9)
    a.set_xlabel("Part de l'importance par permutation (%)")
    a.set_xlim(0, 100)
    a.set_title(f"1. Ce qui explique la consommation (R² = {d['r2_temporal']:.2f})")

    a = ax[1]
    g = df.groupby(["Load_Type", "WeekStatus"])[eo.COL_KWH].mean().unstack().reindex(eo.LOAD_ORDER)
    x = range(len(g)); w = 0.38
    b1 = a.bar([i - w / 2 for i in x], g["Weekday"], w - 0.04, color=BLUE, label="Semaine")
    b2 = a.bar([i + w / 2 for i in x], g["Weekend"], w - 0.04, color=ORANGE, label="Week-end")
    style_bars(a, list(b1) + list(b2), lambda v: f"{v:.0f}")
    a.set_xticks(list(x)); a.set_xticklabels([LOAD_FR[i] for i in g.index])
    a.set_ylabel("kWh moyen / 15 min"); a.legend(loc="upper left")
    a.set_title("2. Consommation moyenne par période et type de jour")
    fig.text(0.01, -0.02, f"Les mesures électriques (réactif, facteur de puissance) reconstruisent kWh quasi exactement "
             f"(R² = {d['r2_full']:.3f}) : relation physique, conséquence et non levier d'action.",
             fontsize=9, color=INK2)
    fig.tight_layout()
    return fig


def fig_optim(df, r):
    p, cur = r["params"], r["params"].currency
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    a = ax[0, 0]
    sh = r["shift"]["shifted_df"]
    wk = sh[sh["WeekStatus"] == "Weekday"]
    before = wk.groupby("hour")[eo.COL_KWH].mean() * 4
    after = wk.groupby("hour")["Usage_kWh_shifted"].mean() * 4
    a.plot(before.index, before.values, color=BLUE, lw=2, label="Avant")
    a.plot(after.index, after.values, color=ORANGE, lw=2, label=f"Après décalage ({p.shift_share:.0%})")
    a.set_title("1. Profil moyen en semaine : lissage des pics")
    a.set_xlabel("Heure"); a.set_ylabel("kWh / heure"); a.set_xticks(range(0, 24, 2)); a.legend()

    rec = r["recommendations"].iloc[:-1].sort_values(f"Gain ({cur}/an)")
    short = {"Décaler": "Décalage de charge", "Améliorer": "Efficacité process",
             "Corriger": "Facteur de puissance", "Réduire": "Consommation à vide"}
    labels = [short[s.split()[0]] for s in rec["Action"]]

    a = ax[0, 1]
    bars = a.barh(labels, rec[f"Gain ({cur}/an)"], color=BLUE, height=0.6)
    for b in bars:
        a.text(b.get_width() + 40, b.get_y() + b.get_height() / 2, f"{b.get_width():,.0f} {cur}".replace(",", " "),
               va="center", fontsize=9)
    a.set_xlim(0, rec[f"Gain ({cur}/an)"].max() * 1.2)
    a.set_title(f"2. Gain financier annuel ({cur}) - tarifs illustratifs"); a.set_xlabel(f"{cur} / an")

    a = ax[1, 0]
    rec2 = rec.sort_values("tCO2 évitées")
    lab2 = [short[s.split()[0]] for s in rec2["Action"]]
    bars = a.barh(lab2, rec2["tCO2 évitées"], color=AQUA, height=0.6)
    for b in bars:
        a.text(b.get_width() + 0.15, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f} t", va="center", fontsize=9)
    a.set_xlim(0, max(rec2["tCO2 évitées"].max() * 1.2, 1))
    a.set_title("3. CO2 évité par an : seul un kWh économisé réduit le CO2")
    a.set_xlabel("tCO2 / an")

    a = ax[1, 1]
    s = r["sensitivity"]
    a.plot(s["intensity_pct"], s["shift_cost_saved_pct"], color=BLUE, lw=2, marker="o", ms=6, label="Décalage de charge")
    a.plot(s["intensity_pct"], s["standby_cost_saved_pct"], color=ORANGE, lw=2, marker="o", ms=6,
           label="Réduction consommation à vide")
    a.set_title("4. Sensibilité : gain selon l'intensité de la mesure")
    a.set_xlabel("Part de charge décalée / part du à-vide supprimée (%)"); a.set_ylabel("Gain (% du coût énergie)")
    a.legend()

    tot = r["recommendations"].iloc[-1]
    fig.suptitle(f"Optimisation : ~{tot[f'Gain ({cur}/an)']:,.0f} {cur}/an ({tot['% coût énergie']:.1f}% du coût) "
                 f"et ~{tot['tCO2 évitées']:.0f} tCO2/an évitées".replace(",", " "),
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main(src, out_dir="results"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = eo.load_data(src)
    r = eo.run_all(df)

    for name, fn in [("co2_analysis", fig_co2), ("drivers", fig_drivers), ("optimization", fig_optim)]:
        fig = fn(df, r)
        fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    r["recommendations"].round(2).to_csv(out / "recommendations.csv", index=False, encoding="utf-8-sig")

    sh, pf = r["shift"], r["power_factor"]
    summary = {
        "emission_factor_kg_per_kwh": round(r["ef"] * 1000, 4),
        "co2": {k: round(v, 2) for k, v in r["co2"].items()},
        "baseline_cost": round(r["baseline_cost"], 0),
        "standby": {k: round(v, 2) for k, v in r["standby"].items()},
        "efficiency": {k: round(v, 2) for k, v in r["efficiency"].items()},
        "power_factor": {k: round(v, 2) for k, v in pf.items() if isinstance(v, float)},
        "shift": {k: round(v, 2) for k, v in sh.items() if isinstance(v, float)},
        "drivers": {"r2_temporal": round(r["drivers"]["r2_temporal"], 3),
                    "r2_with_electrical": round(r["drivers"]["r2_full"], 4),
                    "importance": r["drivers"]["importance"].round(4).to_dict("records")},
        "tariff_assumption": r["params"].price,
    }
    (out / "results_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    print(r["recommendations"][["Priorité", "Action", "kWh évités", "tCO2 évitées",
                                f"Gain ({r['params'].currency}/an)", "% coût énergie"]].round(2).to_string(index=False))
    print(f"\nFichiers écrits dans {out.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage : python part5_report.py clean_data.csv [dossier_sortie]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
