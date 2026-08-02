import time as time_module
from datetime import time as market_time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from cachetools import TTLCache, cached
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


MARKET_TZ = "America/New_York"
SAMPLE_DATA_DIR = Path(__file__).resolve().parent / "sample_data"


market_data_cache = TTLCache(
    maxsize=32,
    ttl=1800
)


def load_sample_intraday_data(ticker: str) -> pd.DataFrame:
    ticker = ticker.strip().upper()
    sample_path = SAMPLE_DATA_DIR / f"{ticker}_1min.csv"

    if not sample_path.exists():
        raise ValueError(
            f"No stored demonstration data is available for {ticker}."
        )

    df = pd.read_csv(
        sample_path,
        index_col="Datetime",
        parse_dates=["Datetime"]
    )

    df.index = pd.to_datetime(df.index)

    return df


@cached(market_data_cache)
def download_intraday_data(
    ticker: str
) -> tuple[pd.DataFrame, str]:
    ticker = ticker.strip().upper()
    last_error = None

    for attempt in range(3):
        try:
            df = yf.download(
                ticker,
                interval="1m",
                period="7d",
                progress=False,
                auto_adjust=False,
                threads=False,
                timeout=30
            )

            if df is not None and not df.empty:
                return df, "live"

        except Exception as error:
            last_error = error

        time_module.sleep(2 ** attempt)

    try:
        fallback_df = load_sample_intraday_data(ticker)
        return fallback_df, "demo"

    except Exception as fallback_error:
        if last_error is not None:
            raise ValueError(
                f"Live Yahoo Finance request failed for {ticker}: "
                f"{last_error}. Stored demonstration data also failed: "
                f"{fallback_error}"
            ) from fallback_error

        raise ValueError(
            f"No live or stored data is available for {ticker}: "
            f"{fallback_error}"
        ) from fallback_error


def get_recent_full_intraday_days(
    ticker: str = "AAPL",
    num_days: int = 7
) -> tuple[dict, str]:
    ticker = ticker.strip().upper()

    df, data_source = download_intraday_data(ticker)
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)

    if df.index.tz is not None:
        df = df.tz_convert(MARKET_TZ)
    else:
        df = df.tz_localize("UTC").tz_convert(MARKET_TZ)

    full_days = {}

    for day_date in sorted(pd.Series(df.index.date).unique()):
        day = df[df.index.date == day_date].copy()
        day = day.between_time("09:30", "16:00")

        if (
            not day.empty
            and day.index.max().time() >= market_time(15, 59)
        ):
            full_days[day_date] = day

    if not full_days:
        raise ValueError(
            f"No complete intraday trading days found for {ticker}"
        )

    full_days = dict(
        list(sorted(full_days.items()))[-num_days:]
    )

    return full_days, data_source


def preprocess_yahoo_1min_for_cnn(
    day: pd.DataFrame,
    ticker: str
) -> pd.DataFrame:
    df = day.copy()

    if df.index.tz is None:
        df.index = df.index.tz_localize(MARKET_TZ)
    else:
        df = df.tz_convert(MARKET_TZ)

    df = df.between_time("09:30", "16:00")

    if df.empty:
        raise ValueError(
            f"No trading-hour data available for {ticker}"
        )

    full_index = pd.date_range(
        start=df.index.min().replace(
            second=0,
            microsecond=0
        ),
        end=df.index.max().replace(
            second=0,
            microsecond=0
        ),
        freq="1min",
        tz=MARKET_TZ
    )

    df = df.reindex(full_index)

    if "Close" not in df.columns:
        raise ValueError(
            f"Close column missing for {ticker}"
        )

    df["Close"] = df["Close"].ffill()

    for column in ["Open", "High", "Low"]:
        if column in df.columns:
            df[column] = df[column].ffill()

    df = df[
        (df.index.time >= market_time(9, 35))
        & (df.index.time <= market_time(15, 55))
    ]

    df["Log_Returns"] = (
        np.log(
            df["Close"] / df["Close"].shift(1)
        )
        * 100
    )

    df = df.dropna(
        subset=["Log_Returns"]
    )

    return df


def generate_image(
    log_returns: np.ndarray
) -> np.ndarray:
    start = log_returns[:-1]
    end = log_returns[1:]

    heights = end - start
    bases = start

    colors = [
        "green" if height > 0 else "red"
        for height in heights
    ]

    fig = plt.figure(
        figsize=(3.8, 3.8),
        dpi=100,
        facecolor="black"
    )

    ax = fig.add_axes(
        [0, 0, 1, 1]
    )

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

    image = np.frombuffer(
        canvas.buffer_rgba(),
        dtype=np.uint8
    )

    image = image.reshape(
        (height, width, 4)
    )

    plt.close(fig)

    return image[:, :, :3]


def realized_vol(
    df: pd.DataFrame
) -> float:
    return float(
        np.sqrt(
            np.sum(
                df["Log_Returns"] ** 2
            )
        )
        * 100
    )


def run_volatility_pipeline(
    ticker: str,
    num_days: int,
    model
):
    ticker = ticker.strip().upper()

    days, data_source = get_recent_full_intraday_days(
        ticker=ticker,
        num_days=num_days
    )

    results = []
    images_by_date = {}
    processed_by_date = {}

    dates = sorted(
        days.keys()
    )

    for index, day_date in enumerate(dates):
        processed = preprocess_yahoo_1min_for_cnn(
            days[day_date],
            ticker
        )

        if len(processed) != 380:
            continue

        log_returns = processed[
            "Log_Returns"
        ].values

        image = generate_image(
            log_returns
        )

        model_input = np.expand_dims(
            image,
            axis=0
        ).astype(np.float32)

        prediction_raw = model.predict(
            model_input,
            verbose=0
        )

        prediction = float(
            np.asarray(
                prediction_raw
            ).reshape(-1)[0]
        )

        images_by_date[day_date] = image
        processed_by_date[day_date] = processed

        if index + 1 < len(dates):
            next_day = dates[index + 1]

            next_processed = (
                preprocess_yahoo_1min_for_cnn(
                    days[next_day],
                    ticker
                )
            )

            if len(next_processed) == 380:
                actual = realized_vol(
                    next_processed
                )
            else:
                actual = np.nan

        else:
            next_day = "NEXT TRADING DAY"
            actual = np.nan

        results.append(
            {
                "ticker": ticker,
                "input_day": day_date,
                "predicts_for": next_day,
                "prediction": prediction,
                "actual_RV": actual
            }
        )

    if not results:
        raise ValueError(
            f"No valid 380-row trading days found for {ticker}"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df.attrs["data_source"] = data_source

    annualization_factor = np.sqrt(252)

    results_df["prediction_daily_pct"] = (
        results_df["prediction"]
        / 100
    )

    results_df["actual_RV_daily_pct"] = (
        results_df["actual_RV"]
        / 100
    )

    results_df["prediction_annualized"] = (
        results_df["prediction_daily_pct"]
        * annualization_factor
    )

    results_df["actual_RV_annualized"] = (
        results_df["actual_RV_daily_pct"]
        * annualization_factor
    )

    results_df["error_raw"] = (
        results_df["prediction"]
        - results_df["actual_RV"]
    )

    results_df["error_daily_pct"] = (
        results_df["prediction_daily_pct"]
        - results_df["actual_RV_daily_pct"]
    )

    results_df["error_annualized"] = (
        results_df["prediction_annualized"]
        - results_df["actual_RV_annualized"]
    )

    return (
        results_df,
        images_by_date,
        processed_by_date
    )


def summarize_forecast(
    results_df: pd.DataFrame
) -> dict:
    last_row = results_df.iloc[-1]

    previous_actual_rows = results_df.dropna(
        subset=["actual_RV_annualized"]
    )

    if not previous_actual_rows.empty:
        previous_day_vol = previous_actual_rows.iloc[-1][
            "actual_RV_annualized"
        ]

        diff_to_previous = (
            last_row["prediction_annualized"]
            - previous_day_vol
        )

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
        "ticker": str(
            last_row["ticker"]
        ),
        "input_day": str(
            last_row["input_day"]
        ),
        "forecast_for": str(
            last_row["predicts_for"]
        ),
        "next_day_prediction_annualized": float(
            last_row["prediction_annualized"]
        ),
        "previous_actual_vol_annualized": (
            None
            if pd.isna(previous_day_vol)
            else float(previous_day_vol)
        ),
        "difference_vs_previous_day": (
            None
            if pd.isna(diff_to_previous)
            else float(diff_to_previous)
        ),
        "direction": direction,
        "data_source": results_df.attrs.get(
            "data_source",
            "unknown"
        )
    }