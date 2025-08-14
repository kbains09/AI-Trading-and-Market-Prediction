import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import joblib

LABELED_DIR = Path("data/labeled")
MODEL_DIR = Path("models/xgboost")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_labeled_data(ticker: str) -> pd.DataFrame:
    file_path = LABELED_DIR / f"{ticker}_labeled.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Labeled file not found for {ticker}")
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df

def train_xgboost(ticker: str):
    print(f"📊 Training XGBoost for {ticker}...")

    df = load_labeled_data(ticker)
    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Optional: drop any remaining NaNs
    X = X.dropna()
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy for {ticker}: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model
    model_path = MODEL_DIR / f"{ticker}_xgb_model.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")

def main():
    tickers = ["AAPL", "TSLA", "MSFT", "BTC-USD"]
    for ticker in tickers:
        try:
            train_xgboost(ticker)
        except Exception as e:
            print(f"❌ Error training {ticker}: {e}")

if __name__ == "__main__":
    main()
