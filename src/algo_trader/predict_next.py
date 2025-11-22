from __future__ import annotations

import pathlib as pl
from typing import Dict

import joblib
import pandas as pd
import typer
import yaml

app = typer.Typer(add_completion=False)

# Load config for default paths
PROJECT_CONFIG = yaml.safe_load(pl.Path("config/project.yaml").read_text())
FEATURES_CONFIG = yaml.safe_load(pl.Path("config/features.yaml").read_text())

DEFAULT_MODEL_DIR = pl.Path(
    PROJECT_CONFIG["data"].get("models_dir", "models/xgboost")
)
DEFAULT_FEATURE_DIR = pl.Path(PROJECT_CONFIG["data"]["processed_dir"])

# Helper: load latest feature row
def _load_latest_features(
    ticker: str,
    feature_dir: pl.Path,
) -> pd.DataFrame:
    feature_path = feature_dir / f"{ticker}_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature data not found at {feature_path}")

    df = pd.read_csv(feature_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna()
    if df.empty:
        raise ValueError(f"No valid rows after dropna() for {ticker}")

    latest = df.iloc[-1:]
    return latest

# Helper: load model artifacts
def _load_regime_model(
    ticker: str,
    model_dir: pl.Path,
) -> Dict:
    path = model_dir / f"{ticker}_regime_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Regime model not found at {path}")
    artifact = joblib.load(path)
    return artifact


def _load_direction_model(
    ticker: str,
    model_dir: pl.Path,
) -> Dict:
    path = model_dir / f"{ticker}_direction_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Direction model not found at {path}")
    artifact = joblib.load(path)
    return artifact

# CLI command
@app.command()
def predict(
    ticker: str = typer.Option(..., help="Ticker symbol to predict (e.g. AAPL)"),
    model_dir: str = typer.Option(
        str(DEFAULT_MODEL_DIR),
        help="Directory containing trained models",
    ),
    feature_dir: str = typer.Option(
        str(DEFAULT_FEATURE_DIR),
        help="Directory containing processed feature CSVs",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show full probability vectors"
    ),
):
    """
    Predict current regime (trend / mean-reversion / neutral)
    and 5-day direction (UP / DOWN) for the given ticker,
    using the latest engineered feature row.
    """
    model_dir_path = pl.Path(model_dir)
    feature_dir_path = pl.Path(feature_dir)

    # ---- Load latest features ----
    try:
        latest = _load_latest_features(ticker, feature_dir_path)
    except Exception as e:
        typer.echo(f"[predict_next] ❌ {e}")
        raise typer.Exit(1)

    # ---- Load models ----
    try:
        regime_artifact = _load_regime_model(ticker, model_dir_path)
        direction_artifact = _load_direction_model(ticker, model_dir_path)
    except FileNotFoundError as e:
        typer.echo(f"[predict_next] ❌ {e}")
        raise typer.Exit(1)

    regime_model = regime_artifact["model"]
    regime_features = regime_artifact["feature_cols"]
    regime_map = regime_artifact["regime_map"] 
    inv_regime_map = {v: k for k, v in regime_map.items()}

    direction_model = direction_artifact["model"]
    direction_features = direction_artifact["feature_cols"]

    # ---- Build X matrices ----
    missing_regime = [c for c in regime_features if c not in latest.columns]
    missing_dir = [c for c in direction_features if c not in latest.columns]

    if missing_regime:
        typer.echo(
            f"[predict_next] ❌ Missing regime features in latest row: {missing_regime[:5]}..."
        )
        raise typer.Exit(1)
    if missing_dir:
        typer.echo(
            f"[predict_next] ❌ Missing direction features in latest row: {missing_dir[:5]}..."
        )
        raise typer.Exit(1)

    X_regime = latest[regime_features]
    X_dir = latest[direction_features]

    # ---- Regime prediction ----
    regime_probs = regime_model.predict_proba(X_regime)[0]
    regime_class = int(regime_probs.argmax())
    regime_label_int = inv_regime_map.get(regime_class, 0)

    if regime_label_int == 1:
        regime_label_str = "trend"
    elif regime_label_int == -1:
        regime_label_str = "mean_reversion"
    else:
        regime_label_str = "neutral"

    regime_conf = float(regime_probs[regime_class])

    # ---- Direction prediction (5-day) ----
    dir_probs = direction_model.predict_proba(X_dir)[0]
    prob_down = float(dir_probs[0])
    prob_up = float(dir_probs[1])
    dir_pred = "UP" if prob_up >= 0.5 else "DOWN"

    # ---- Output ----
    date_str = latest["Date"].iloc[0].strftime("%Y-%m-%d")

    typer.echo(f"[predict_next] ✅ Ticker: {ticker}")
    typer.echo(f"[predict_next] 📅 As of: {date_str}")
    typer.echo(
        f"[predict_next] 🧭 Regime: {regime_label_str.upper()} "
        f"(class={regime_label_int}, p≈{regime_conf:.3f})"
    )
    typer.echo(
        f"[predict_next] 📈 5D Direction: {dir_pred} "
        f"(p_up={prob_up:.3f}, p_down={prob_down:.3f})"
    )

    if verbose:
        typer.echo("[predict_next] 🔍 Raw regime probabilities (ordered by class 0/1/2):")
        typer.echo(f"  probs = {regime_probs}")
        typer.echo("[predict_next] 🔍 Raw direction probabilities [DOWN, UP]:")
        typer.echo(f"  probs = {dir_probs}")


if __name__ == "__main__":
    app()
