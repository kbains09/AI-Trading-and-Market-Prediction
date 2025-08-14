import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
LABELED_DIR = Path("data/labeled")
LABELED_DIR.mkdir(parents=True, exist_ok=True)

def generate_labels(df: pd.DataFrame, label_type="binary") -> pd.DataFrame:
    if label_type == "binary":
        # 1 if next day's close is higher, else 0
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    elif label_type == "directional":
        # 1 if up, -1 if down, 0 if flat
        df["Target"] = df["Close"].shift(-1) - df["Close"]
        df["Target"] = df["Target"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    else:
        raise ValueError(f"Unsupported label type: {label_type}")
    
    df.dropna(inplace=True)
    return df

def process_ticker(ticker: str, label_type="binary"):
    print(f"🏷️ Labeling {ticker}...")
    file_path = PROCESSED_DIR / f"{ticker}_features.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Feature file not found for {ticker}")
    
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    df = generate_labels(df, label_type)
    
    out_path = LABELED_DIR / f"{ticker}_labeled.csv"
    df.to_csv(out_path)
    print(f"✅ Labeled file saved: {out_path}")

def main():
    tickers = ["AAPL", "TSLA", "MSFT", "BTC-USD"]
    for ticker in tickers:
        try:
            process_ticker(ticker, label_type="binary")
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")

if __name__ == "__main__":
    main()
