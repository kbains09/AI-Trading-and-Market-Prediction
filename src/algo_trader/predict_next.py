# predict_next.py
import typer
import pathlib as pl
import pandas as pd
import joblib

from algo_trader.features.engineering_features import generate_features

app = typer.Typer()

@app.command()
def predict(
    ticker: str,
    model_path: str = "models/xgboost",
    data_path: str = "data/processed",
):
    model_file = pl.Path(model_path) / f"{ticker}_xgb_model.pkl"
    csv_file = pl.Path(data_path) / f"{ticker}.csv"

    if not model_file.exists():
        raise FileNotFoundError(f"Model not found at: {model_file}")
    if not csv_file.exists():
        raise FileNotFoundError(f"Data not found at: {csv_file}")

    model = joblib.load(model_file)
    df = pd.read_csv(csv_file, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df_features = generate_features(df.copy())
    latest = df_features.dropna().iloc[[-1]]
    
    prediction = model.predict(latest.drop(columns=["Date", "Target"], errors="ignore"))
    print(f"📈 {ticker} prediction: {'BUY' if prediction[0] == 1 else 'SELL'}")

if __name__ == "__main__":
    app()