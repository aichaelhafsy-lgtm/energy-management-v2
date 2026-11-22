"""Nettoyage du dataset Steel Industry Energy Consumption (version v1).

Usage : python preprocessing.py Steel_industry_data.csv clean_data.csv

Ce que fait le script :
1. Lit le CSV et convertit la date (format jour/mois/année).
2. Corrige les 365 releves de minuit (NSM == 0) : ils sont etiquetes avec la
   date du jour qui se termine, alors qu'ils appartiennent au lendemain a 00:00.
3. Trie par date et verifie un intervalle regulier de 15 minutes.
4. Ajoute des colonnes de temps calculees a partir de la date corrigee.
Aucune ligne n'est supprimee, aucune valeur aberrante n'est retiree.
"""
import sys
import pandas as pd


def load_and_fix(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df.loc[df["NSM"] == 0, "date"] += pd.Timedelta(days=1)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_time_columns(df):
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    df["dayofweek"] = df["date"].dt.dayofweek  # 0 = lundi ... 6 = dimanche
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def check(df):
    steps = df["date"].diff().dropna().unique()
    assert len(df) == 35040, f"nombre de lignes inattendu : {len(df)}"
    assert df["date"].is_monotonic_increasing, "dates non croissantes"
    assert df["date"].is_unique, "dates dupliquees"
    assert len(steps) == 1 and steps[0] == pd.Timedelta(minutes=15), "intervalle irregulier"
    assert int(df.isna().sum().sum()) == 0, "valeurs manquantes"


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    df = add_time_columns(load_and_fix(src))
    check(df)
    df.to_csv(dst, index=False, date_format="%Y-%m-%d %H:%M")
    print("OK :", df.shape, "|", df["date"].min(), "->", df["date"].max())
