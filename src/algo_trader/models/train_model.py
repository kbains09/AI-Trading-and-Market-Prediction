import pathlib as pl
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from algo_trader.features.engineering_features import generate_features

TICKERS = ["AAPL", "TSLA", "MSFT", "BTC-USD"]
RAW_DIR = pl.Path("data/raw")
MODEL_DIR = pl.Path("models/xgboost")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_and_evaluate(ticker: str):
    print(f"📊 Training XGBoost for {ticker}...")

    # Generate features from raw CSV
    df = generate_features(ticker, data_dir=str(RAW_DIR))
    df.dropna(inplace=True)

    # Define input features and target
    feature_cols = [
        col for col in df.columns
        if col.startswith(("return", "logret", "SMA", "EMA", "RSI", "MACD", "BB_", "ATR", "HL_", "OC_", "vol"))
        and not col.startswith("next_") and not col.startswith("target")
    ]
    X = df[feature_cols]
    y = df["target_up"]

    # Time-based train/test split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train XGBoost model
    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Test Accuracy for {ticker}: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save trained model
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
