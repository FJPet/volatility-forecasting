from pathlib import Path

import pandas as pd
import yfinance as yf

OUTPUT_DIR = Path("sample_data")
TICKERS = ["AAPL", "MSFT", "KO"]


def download_and_save(ticker):
    print(f"Downloading {ticker}...")

    df = yf.download(
        ticker,
        interval="1m",
        period="7d",
        progress=False,
        auto_adjust=False,
        threads=False,
        timeout=30
    )

    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Datetime"

    OUTPUT_DIR.mkdir(exist_ok=True)

    df.to_csv(OUTPUT_DIR / f"{ticker}_1min.csv")

    print(f"Saved {ticker}")


for ticker in TICKERS:
    download_and_save(ticker)

print("Finished.")