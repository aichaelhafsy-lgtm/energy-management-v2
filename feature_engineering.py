"""Feature Engineering du dataset Steel Industry Energy Consumption.

Usage : python feature_engineering.py clean_data.csv featured_data.csv
"""
import sys
import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Encodage ordinal de la charge
    load_mapping = {"Light_Load": 0, "Medium_Load": 1, "Maximum_Load": 2}
    df["Load_Type_encoded"] = df["Load_Type"].map(load_mapping)

    # 2. Encodage cyclique du temps
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # 3. Variables métier (physique électrique & activité usine)
    df["Total_Reactive_Power_kVarh"] = (
        df["Lagging_Current_Reactive.Power_kVarh"]
        + df["Leading_Current_Reactive_Power_kVarh"]
    )
    df["Apparent_Power_kVAh"] = np.sqrt(
        df["Usage_kWh"] ** 2 + df["Total_Reactive_Power_kVarh"] ** 2
    )
    df["is_working_hours"] = (
        (df["hour"] >= 8) & (df["hour"] <= 18) & (df["is_weekend"] == 0)
    ).astype(int)

    # 4. Variables temporelles décalées (Lags & Rolling sans data leakage)
    df["Usage_kWh_lag_15m"] = df["Usage_kWh"].shift(1)
    df["Usage_kWh_lag_1h"] = df["Usage_kWh"].shift(4)
    df["Usage_kWh_lag_24h"] = df["Usage_kWh"].shift(96)

    df["Usage_kWh_rolling_mean_1h"] = df["Usage_kWh"].shift(1).rolling(window=4).mean()
    df["Usage_kWh_rolling_std_1h"] = df["Usage_kWh"].shift(1).rolling(window=4).std()

    # 5. Nettoyage final pour le Machine Learning
    cols_to_drop = ["WeekStatus", "Day_of_week", "Load_Type"]
    df = df.drop(columns=cols_to_drop).dropna().reset_index(drop=True)
    return df


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "clean_data.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "featured_data.csv"
    df_clean = pd.read_csv(src, parse_dates=["date"])
    df_featured = build_features(df_clean)
    df_featured.to_csv(dst, index=False)
    print("OK :", df_featured.shape, "| Colonnes :", len(df_featured.columns))
