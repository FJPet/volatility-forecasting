import yfinance as yf
import pandas as pd
import numpy as np
from datetime import time
from cachetools import TTLCache, cached
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


MARKET_TZ = "America/New_York"

market_data_cache = TTLCache(
    maxsize=32,
    ttl=1800
)


@cached(market_data_cache)
def download_intraday_data(ticker: str, num_days: int) -> pd.DataFrame:
    ticker = ticker.strip().upper()

    df = yf.download(
        ticker,
        interval="1m",
        period=f"{num_days}d",
        progress=False,
        auto_adjust=False,
        threads=False,
        timeout=30
    )

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for {ticker}. "
            "Yahoo Finance may be temporarily rate-limiting requests."
        )

    return df
def get_recent_full_intraday_days(ticker="AAPL", num_days=7):
    df = yf.download(
        ticker,
        interval="1m",
        period="1mo",
        progress=False,
        auto_adjust=False,
        threads=False,
        timeout=30
    )

    if df.empty:
        raise ValueError(
            f"No data returned for {ticker}. "
            "Yahoo Finance may be rate-limiting requests."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)

    if df.index.tz is not None:
        df = df.tz_convert(MARKET_TZ)
    else:
        df = df.tz_localize("UTC").tz_convert(MARKET_TZ)

    full_days = {}

    for d in sorted(pd.Series(df.index.date).unique()):
        day = df[df.index.date == d].copy()
        day = day.between_time("09:30", "16:00")

        if not day.empty and day.index.max().time() >= time(15, 59):
            full_days[d] = day

    if not full_days:
        raise ValueError(
            f"No complete intraday trading days found for {ticker}"
        )

    full_days = dict(
        list(sorted(full_days.items()))[-num_days:]
    )

    if len(full_days) < num_days:
        raise ValueError(
            f"Only {len(full_days)} complete trading days were returned for {ticker}; "
            f"{num_days} were requested."
        )

    return full_days


def preprocess_yahoo_1min_for_cnn(day, ticker):
    df = day.copy()

    if df.index.tz is None:
        df.index = df.index.tz_localize(MARKET_TZ)
    else:
        df = df.tz_convert(MARKET_TZ)

    df = df.between_time("09:30", "16:00")

    if df.empty:
        raise ValueError(f"No trading-hour data available for {ticker}")

    full_index = pd.date_range(
        start=df.index.min().replace(second=0, microsecond=0),
        end=df.index.max().replace(second=0, microsecond=0),
        freq="1min",
        tz=MARKET_TZ
    )

    df = df.reindex(full_index)

    if "Close" not in df.columns:
        raise ValueError(f"Close column missing for {ticker}")

    df["Close"] = df["Close"].ffill()

    for col in ["Open", "High", "Low"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    df = df[(df.index.time >= time(9, 35)) & (df.index.time <= time(15, 55))]

    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1)) * 100
    df = df.dropna(subset=["Log_Returns"])

    return df


def generate_image(log_returns):
    start = log_returns[:-1]
    end = log_returns[1:]

    heights = end - start
    bases = start
    colors = ["green" if h > 0 else "red" for h in heights]

    fig = plt.figure(figsize=(3.8, 3.8), dpi=100, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])

    ax.bar(
        range(1, len(log_returns)),
        heights,
        bottom=bases,
        color=colors,
        width=1.0
    )

    ax.set_facecolor("black")
    ax.set_xlim([0, 380])
    ax.set_ylim([-0.25, 0.25])
    ax.axis("off")

    canvas = FigureCanvas(fig)
    canvas.draw()

    width, height = canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape((height, width, 4))

    plt.close(fig)

    return img[:, :, :3]


def realized_vol(df):
    return np.sqrt(np.sum(df["Log_Returns"] ** 2)) * 100


def run_volatility_pipeline(ticker, num_days, model):
    days = get_recent_full_intraday_days(ticker, num_days)

    results = []
    images_by_date = {}
    processed_by_date = {}

    dates = sorted(days.keys())

    for i, d in enumerate(dates):
        processed = preprocess_yahoo_1min_for_cnn(days[d], ticker)

        if len(processed) != 380:
            continue

        log_returns = processed["Log_Returns"].values
        img = generate_image(log_returns)

        x = np.expand_dims(img, axis=0).astype(np.float32)
        pred = float(model.predict(x, verbose=0)[0][0])

        images_by_date[d] = img
        processed_by_date[d] = processed

        if i + 1 < len(dates):
            next_day = dates[i + 1]
            next_processed = preprocess_yahoo_1min_for_cnn(days[next_day], ticker)

            if len(next_processed) == 380:
                actual = realized_vol(next_processed)
            else:
                actual = np.nan
        else:
            next_day = "NEXT TRADING DAY"
            actual = np.nan

        results.append({
            "ticker": ticker,
            "input_day": d,
            "predicts_for": next_day,
            "prediction": pred,
            "actual_RV": actual
        })

    if not results:
        raise ValueError(f"No valid 380-row trading days found for {ticker}")

    results_df = pd.DataFrame(results)

    annualization_factor = np.sqrt(252)

    results_df["prediction_daily_pct"] = results_df["prediction"] / 100
    results_df["actual_RV_daily_pct"] = results_df["actual_RV"] / 100

    results_df["prediction_annualized"] = (
        results_df["prediction_daily_pct"] * annualization_factor
    )

    results_df["actual_RV_annualized"] = (
        results_df["actual_RV_daily_pct"] * annualization_factor
    )

    results_df["error_raw"] = results_df["prediction"] - results_df["actual_RV"]

    results_df["error_daily_pct"] = (
        results_df["prediction_daily_pct"] - results_df["actual_RV_daily_pct"]
    )

    results_df["error_annualized"] = (
        results_df["prediction_annualized"] - results_df["actual_RV_annualized"]
    )

    return results_df, images_by_date, processed_by_date


def summarize_forecast(results_df):
    last_row = results_df.iloc[-1]
    previous_actual_rows = results_df.dropna(subset=["actual_RV_annualized"])

    if not previous_actual_rows.empty:
        previous_day_vol = previous_actual_rows.iloc[-1]["actual_RV_annualized"]
        diff_to_previous = last_row["prediction_annualized"] - previous_day_vol

        if diff_to_previous > 0:
            direction = "increase"
        elif diff_to_previous < 0:
            direction = "decrease"
        else:
            direction = "unchanged"
    else:
        previous_day_vol = np.nan
        diff_to_previous = np.nan
        direction = "unknown"

    return {
        "ticker": str(last_row["ticker"]),
        "input_day": str(last_row["input_day"]),
        "forecast_for": str(last_row["predicts_for"]),
        "next_day_prediction_annualized": float(last_row["prediction_annualized"]),
        "previous_actual_vol_annualized": None if pd.isna(previous_day_vol) else float(previous_day_vol),
        "difference_vs_previous_day": None if pd.isna(diff_to_previous) else float(diff_to_previous),
        "direction": direction
    }
