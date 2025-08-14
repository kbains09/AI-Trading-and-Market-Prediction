import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

def fetch_stock_data(ticker: str, start: str = "2020-01-01", end: str = "2024-01-01", interval: str = "1d") -> pd.DataFrame:
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=start, end=end, interval=interval)
    data.reset_index(inplace=True)
    return data

def save_to_csv(data: pd.DataFrame, ticker: str):
    filename = DATA_DIR / f"{ticker}.csv"
    data.to_csv(filename, index=False)
    print(f"Saved to {filename}")

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tickers = ["AAPL", "TSLA", "MSFT", "BTC-USD"]  # Add your own
    for ticker in tickers:
        df = fetch_stock_data(ticker)
        save_to_csv(df, ticker)

if __name__ == "__main__":
    main()
