import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import Iterable, Optional
import logging
from datetime import date

DATA_DIR = Path("data/raw")
DEFAULT_TICKERS = ["AAPL", "TSLA", "MSFT", "BTC-USD"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def fetch_stock_data(
    ticker: str,
    start: str = "2020-01-01",
    end: Optional[str] = "2024-01-01",
    interval: str = "1d",
    retries: int = 2,
) -> pd.DataFrame:
    for attempt in range(retries + 1):
        try:
            logging.info(f"Fetching {ticker} [{start} → {end}] interval={interval}")
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            if df is None or df.empty:
                logging.warning(f"No data returned for {ticker} (attempt {attempt+1})")
            else:
                df = df.reset_index()
                return df
        except Exception as e:
            logging.warning(f"Error fetching {ticker} (attempt {attempt+1}): {e}")
    return pd.DataFrame()

def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "Date" in df.columns:
        df.sort_values("Date", inplace=True)
        df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def save_to_csv(df: pd.DataFrame, ticker: str) -> Optional[Path]:
    if df.empty:
        logging.warning(f"Skip saving empty DataFrame for {ticker}")
        return None
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DATA_DIR / f".{ticker}.csv.tmp"
    out = DATA_DIR / f"{ticker}.csv"
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    logging.info(f"Saved {ticker} → {out}")
    return out

def run_batch(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = "2020-01-01",
    end: Optional[str] = "2024-01-01",
    interval: str = "1d",
):
    for t in tickers:
        df = fetch_stock_data(t, start=start, end=end, interval=interval)
        df = clean_frame(df)
        save_to_csv(df, t)

def main():
    run_batch()

if __name__ == "__main__":
    main()
