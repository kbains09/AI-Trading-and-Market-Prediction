import pathlib as pl
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

TICKERS = ["AAPL", "TSLA", "MSFT", "BTC-USD"]
LABELED_DIR = pl.Path("data/labeled")
MODEL_DIR = pl.Path("models/xgboost")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_and_evaluate(ticker: str):
    print(f"📊 Training XGBoost for {ticker}...")

    file_path = LABELED_DIR / f"{ticker}_labeled.csv"
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)  # Ensure chronological order

    # Drop missing values
    df.dropna(inplace=True)

    # Separate features and target
    X = df.drop(columns=["Date", "Target"])
    y = df["Target"]

    # Time-based train/test split: 80% train, 20% test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train model
    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Test Accuracy for {ticker}: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model
    model_path = MODEL_DIR / f"{ticker}_xgb_model.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")

def main():
    for ticker in TICKERS:
        try:
            train_and_evaluate(ticker)
        except Exception as e:
            print(f"❌ Failed to train for {ticker}: {e}")

if __name__ == "__main__":
    main()
