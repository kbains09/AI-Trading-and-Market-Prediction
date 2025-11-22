from __future__ import annotations
import pathlib as pl
import joblib
import pandas as pd
import typer

app = typer.Typer(add_completion=False)

@app.command()
def predict(
    ticker: str = typer.Option(..., help="Ticker symbol to predict"),
    model_dir: str = typer.Option("models/xgboost", help="Directory containing trained models"),
    feature_dir: str = typer.Option("data/processed", help="Directory containing processed feature CSVs"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show prediction probabilities")
):
    model_dir_path = pl.Path(model_dir)
    feature_path = pl.Path(feature_dir) / f"{ticker}_features.csv"

    # Flexible model file matching
    model_candidates = list(model_dir_path.glob(f"{ticker}_*_model.pkl"))
    if not model_candidates:
        typer.echo(f"[predict_next] ❌ No model file found for {ticker} in {model_dir_path}")
        raise typer.Exit(1)
    model_path = model_candidates[0]  # use the first match

    if not feature_path.exists():
        typer.echo(f"[predict_next] ❌ Feature data not found at {feature_path}")
        raise typer.Exit(1)

    # Load model
    model = joblib.load(model_path)

    # Load features
    df = pd.read_csv(feature_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna()

    # Select last row for prediction
    latest = df.iloc[-1:]
    feature_cols = [
        col for col in df.columns
        if col.startswith(('return', 'logret', 'SMA', 'EMA', 'RSI', 'MACD', 'BB_', 'ATR', 'HL_', 'OC_', 'vol'))
        and not col.startswith('next_') and not col.startswith('target')
    ]
    X = latest[feature_cols]

    # Predict
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    typer.echo(f"[predict_next] ✅ Ticker: {ticker}")
    typer.echo(f"[predict_next] 📈 Prediction: {'UP' if pred == 1 else 'DOWN'}")

    if verbose:
        typer.echo(f"[predict_next] 🧠 Probabilities => DOWN: {proba[0]:.4f}, UP: {proba[1]:.4f}")

if __name__ == "__main__":
    app()
