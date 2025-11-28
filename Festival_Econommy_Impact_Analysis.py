# scripts/festival_analysis.py

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

# CONFIG

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_data.csv"  # replace with your file
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True)
FESTIVALS = {
    "Diwali": ["2020-11-14","2021-11-04","2022-10-24","2023-11-12","2024-11-01"],
    "Christmas": ["2020-12-25","2021-12-25","2022-12-25","2023-12-25","2024-12-25"],
    "Eid": ["2020-05-24","2021-05-13","2022-05-03","2023-04-22","2024-04-10"]
}
EVENT_WINDOW_DAYS = 3  # +/- days around festival date to treat as event window

# Utilities

def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["region", "date"]).reset_index(drop=True)
    return df

def tag_festivals(df, festivals=FESTIVALS, window=EVENT_WINDOW_DAYS):
    df = df.copy()
    # initialize columns
    df["festival"] = "None"
    df["is_festival_day"] = 0
    for fest, dates in festivals.items():
        fest_dates = [pd.to_datetime(d) for d in dates]
        for d in fest_dates:
            mask = df["date"].between(d - timedelta(days=window), d + timedelta(days=window))
            # if multiple festivals overlap, we will keep first found (unlikely)
            df.loc[mask & (df["festival"] == "None"), "festival"] = fest
            df.loc[mask, "is_festival_day"] = 1
    df["festival"] = df["festival"].astype("category")
    return df


# Simple event-window uplift estimation

def event_window_uplift(df, value_col="retail_sales"):
    # compute mean on event days vs non-event days per region and festival
    rows = []
    for region in df["region"].unique():
        dfr = df[df["region"] == region]
        for fest in list(dfr["festival"].cat.categories):
            if fest == "None":
                continue
            event_mean = dfr.loc[dfr["festival"] == fest, value_col].mean()
            non_event_mean = dfr.loc[dfr["festival"] == "None", value_col].mean()
            uplift_pct = (event_mean - non_event_mean) / non_event_mean * 100
            rows.append({"region": region, "festival": fest, "event_mean": event_mean,
                         "non_event_mean": non_event_mean, "uplift_pct": uplift_pct})
    return pd.DataFrame(rows)

# Baseline forecasting using SARIMAX

def fit_and_forecast_baseline(series, steps=14, exog=None):
    # Keep a simple SARIMAX: (p,d,q)=(1,1,1) with seasonal (1,0,1,7) weekly seasonality
    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,0,1,7), enforce_stationarity=False, enforce_invertibility=False, exog=exog)
        res = model.fit(disp=False)
        forecast = res.get_forecast(steps=steps, exog=None)
        return res, forecast
    except Exception as e:
        print("SARIMAX failed:", e)
        return None, None

# Feature engineering for prediction model

def build_features_for_prediction(df):
    # For each festival occurrence, compute features: baseline mean prior month, growth last year, region averages
    rows = []
    for fest, dates in FESTIVALS.items():
        for d in dates:
            d = pd.to_datetime(d)
            for region in df["region"].unique():
                dfr = df[df["region"] == region]
                # baseline prior 30 days mean
                prior = dfr[(dfr["date"] >= d - pd.Timedelta(days=60)) & (dfr["date"] < d)]
                prior_mean = prior["retail_sales"].mean() if len(prior)>0 else np.nan
                # same festival last year uplift (if available)
                last_year = d - pd.Timedelta(days=365)
                last_same_fest = dfr[(dfr["date"] >= last_year - pd.Timedelta(days=7)) & (dfr["date"] <= last_year + pd.Timedelta(days=7))]
                last_mean = last_same_fest["retail_sales"].mean() if len(last_same_fest)>0 else np.nan
                # average flight bookings prior month
                prior_flights = prior["flight_bookings"].mean() if len(prior)>0 else np.nan
                rows.append({
                    "festival": fest,
                    "festival_date": d,
                    "region": region,
                    "prior_retail_mean": prior_mean,
                    "prior_flights_mean": prior_flights,
                    "last_year_mean": last_mean
                })
    feat = pd.DataFrame(rows)
    # target: compute actual uplift around the festival this year (if exists in df)
    targets = []
    for idx, r in feat.iterrows():
        region = r["region"]
        fest = r["festival"]
        d = r["festival_date"]
        dfr = df[(df["region"]==region)]
        event_window = dfr[dfr["date"].between(d - pd.Timedelta(days=EVENT_WINDOW_DAYS), d + pd.Timedelta(days=EVENT_WINDOW_DAYS))]
        non_event = dfr[~dfr["date"].between(d - pd.Timedelta(days=60), d + pd.Timedelta(days=60))]
        if len(event_window)>0 and len(non_event)>0:
            target = event_window["retail_sales"].mean() - non_event["retail_sales"].mean()
        else:
            target = np.nan
        targets.append(target)
    feat["target_uplift_abs"] = targets
    feat = feat.dropna()
    # create a label for "will be top festival in region in next year?" - for simplicity, this function returns regression target
    return feat

# MAIN

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print("Tagging festivals...")
    df = tag_festivals(df)

    # Basic EDA summaries
    print("Basic summary:")
    print(df[["retail_sales","flight_bookings","hotel_prices"]].describe().T)

    # Save simple time-series plot (region-level)
    for region in df["region"].unique():
        plt.figure(figsize=(10,4))
        sub = df[df["region"]==region].set_index("date")
        sub["retail_sales"].rolling(7).mean().plot(label="retail_7d")
        plt.title(f"Retail Sales (7-day MA) - {region}")
        plt.ylabel("retail_sales")
        plt.tight_layout()
        plt.savefig(OUT_DIR/f"retail_{region}.png")
        plt.close()

    # Event-window uplift
    print("Estimating event-window uplift by festival...")
    uplift_df = event_window_uplift(df, value_col="retail_sales")
    print(uplift_df)
    uplift_df.to_csv(OUT_DIR/"event_window_uplift.csv", index=False)

    # Baseline forecasting and uplift estimation per region & festival date
    print("Fitting baseline forecasts and computing uplift")
    baseline_results = []
    for region in df["region"].unique():
        dfr = df[df["region"]==region].set_index("date").asfreq("D").fillna(method="ffill")
        series = dfr["retail_sales"]
        # fit model on historical data up to 2024-01-01 to evaluate forecasting; here we just fit on all data
        res, forecast = fit_and_forecast_baseline(series, steps=14)
        if res is None:
            continue
        # compute expected vs actual around festival dates
        for fest, dates in FESTIVALS.items():
            for dstr in dates:
                d = pd.to_datetime(dstr)
                # baseline prediction for the event window: we'll use last fitted model to predict the +/- window
                start = d - pd.Timedelta(days=EVENT_WINDOW_DAYS)
                end = d + pd.Timedelta(days=EVENT_WINDOW_DAYS)
                actual_window = series[start:end]
                # if actual empty, skip
                if actual_window.empty:
                    continue
                # naive baseline: use rolling median of prior 30 days as expected
                prior = series[(series.index >= (start - pd.Timedelta(days=30))) & (series.index < start)]
                if prior.empty:
                    expected_mean = series.mean()
                else:
                    expected_mean = prior.mean()
                actual_mean = actual_window.mean()
                uplift_pct = (actual_mean - expected_mean) / expected_mean * 100
                baseline_results.append({
                    "region": region, "festival": fest, "festival_date": d,
                    "expected_mean": expected_mean, "actual_mean": actual_mean,
                    "uplift_pct": uplift_pct
                })
    baseline_df = pd.DataFrame(baseline_results)
    baseline_df.to_csv(OUT_DIR/"baseline_uplift.csv", index=False)
    print("Baseline uplift saved to outputs/baseline_uplift.csv")

    # Prediction model: which festival will give biggest surge next year (simple regression)
    print("Building features for prediction model...")
    feat = build_features_for_prediction(df)
    if feat.empty:
        print("Not enough data to build features for prediction.")
        return
    X = feat[["prior_retail_mean", "prior_flights_mean", "last_year_mean"]].fillna(0)
    y = feat["target_uplift_abs"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Prediction model RMSE: {rmse:.2f}")
    feat["predicted_uplift"] = rf.predict(X.fillna(0))
    # Now aggregate predictions per festival (sum or mean across regions and dates) for "next year" ranking
    festival_scores = feat.groupby("festival")["predicted_uplift"].mean().sort_values(ascending=False).reset_index()
    print("Predicted festival ranking by average uplift:")
    print(festival_scores)
    festival_scores.to_csv(OUT_DIR/"predicted_festival_ranking.csv", index=False)

    print("All outputs saved in", OUT_DIR)

if __name__ == "__main__":
    main()
