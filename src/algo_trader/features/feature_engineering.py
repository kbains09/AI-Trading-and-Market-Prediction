import pandas as pd
import numpy as np
import talib
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_data(ticker: str) -> pd.DataFrame:
    file_path = RAW_DIR / f"{ticker}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Data for {ticker} not found at {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)

    # Ensure all numeric columns are floats
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Return"] = df["Close"].pct_change()

    # Moving Averages
    df["SMA_20"] = talib.SMA(df["Close"], timeperiod=20)
    df["EMA_20"] = talib.EMA(df["Close"], timeperiod=20)

    # Volatility
    df["Volatility_20"] = df["Return"].rolling(window=20).std()

    # ATR
    df["ATR_14"] = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)

    # RSI
    df["RSI_14"] = talib.RSI(df["Close"], timeperiod=14)

    # MACD
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = talib.MACD(
        df["Close"], fastperiod=12, slowperiod=26, signalperiod=9
    )

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(
        df["Close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    df["BB_upper"] = upper
    df["BB_middle"] = middle
    df["BB_lower"] = lower

    df.dropna(inplace=True)
    return df

def save_features(df: pd.DataFrame, ticker: str):
    out_path = PROCESSED_DIR / f"{ticker}_features.csv"
    df.to_csv(out_path)
    print(f"✅ Features saved to: {out_path}")

def process_ticker(ticker: str):
    print(f"📈 Processing {ticker}...")
    df = load_data(ticker)
    df = add_features(df)
    save_features(df, ticker)

def main():
    tickers = ["AAPL", "TSLA", "MSFT", "BTC-USD"]
    for ticker in tickers:
        try:
            process_ticker(ticker)
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")

if __name__ == "__main__":
    main()
