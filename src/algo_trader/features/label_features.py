from __future__ import annotations

import pandas as pd
import numpy as np
import yaml
import pathlib as pl
import typer

app = typer.Typer(add_completion=False)

# Load Config Files
PROJECT_CONFIG = yaml.safe_load(pl.Path("config/project.yaml").read_text())
LABEL_CONFIG = yaml.safe_load(pl.Path("config/labels.yaml").read_text())

DATA_PROCESSED = pl.Path(PROJECT_CONFIG["data"]["processed_dir"])
DATA_LABELED = pl.Path(PROJECT_CONFIG["data"]["labeled_dir"])

# Ensure output directory exists
DATA_LABELED.mkdir(parents=True, exist_ok=True)

# Helper Functions
def compute_future_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """Compute future log returns for horizon days."""
    return np.log(close.shift(-horizon) / close)

def compute_direction(series: pd.Series) -> pd.Series:
    """Convert future return series into direction: -1, 0, +1."""
    return series.apply(lambda x: 1 if x > 0 else -1)

def compute_binary_target(series: pd.Series) -> pd.Series:
    """Simple 1-day up/down label."""
    return (series > 0).astype(int)

def compute_regime_trend_meanrev(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Trend strength = rolling_mean(log returns) / rolling_volatility
    Then classify based on z-thresholds.
    """
    window_ret = params["return_window"]
    window_vol = params["vol_window"]
    upper_z = params["upper_z"]
    lower_z = params["lower_z"]

    # Rolling 20-day log returns
    logret_1 = np.log(df["Close"] / df["Close"].shift(1))

    trend_strength = (
        logret_1.rolling(window_ret).mean()
        /
        logret_1.rolling(window_vol).std()
    )

    # Z-score standardization
    z = (trend_strength - trend_strength.mean()) / trend_strength.std()

    def classify(value):
        if value > upper_z:
            return 1      # Trending
        if value < lower_z:
            return -1     # Mean reversion
        return 0          # Neutral

    return z.apply(classify)

def compute_breakout(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Breakout when Close exceeds rolling high by eps AND volume is elevated.
    """
    lookback = params["lookback_high_window"]
    eps = params["eps"]
    vol_field = params["volume_zscore_field"]
    vol_thresh = params["volume_threshold"]

    rolling_high = df["Close"].shift(1).rolling(lookback).max()
    breakout_price = df["Close"] > rolling_high * (1 + eps)

    breakout_volume = df[vol_field] > vol_thresh

    return (breakout_price & breakout_volume).astype(int)

# Process Single Ticker
def process_ticker(ticker: str):
    input_path = DATA_PROCESSED / f"{ticker}_features.csv"
    if not input_path.exists():
        typer.echo(f"[label_features] ❌ Missing processed features for {ticker}")
        return None

    df = pd.read_csv(input_path, parse_dates=["Date"]).sort_values("Date")
    df.reset_index(drop=True, inplace=True)

    # Targets
    # 1-day simple return
    next_ret_cfg = LABEL_CONFIG["targets"]["future_return_1d"]
    df[next_ret_cfg["name"]] = df["Close"].pct_change().shift(-1)

    # Binary up/down target
    binary_cfg = LABEL_CONFIG["targets"]["binary_1d"]
    df[binary_cfg["name"]] = compute_binary_target(df[next_ret_cfg["name"]])

    # 5-day future log return
    f5_cfg = LABEL_CONFIG["targets"]["future_return_5d"]
    df[f5_cfg["name"]] = compute_future_log_return(df["Close"], f5_cfg["horizon"])

    # 5-day direction (-1/0/+1)
    dir5_cfg = LABEL_CONFIG["targets"]["direction_5d"]
    df[dir5_cfg["name"]] = compute_direction(df[f5_cfg["name"]])

    # Regime Label
    regime_cfg = LABEL_CONFIG["regimes"]["trend_vs_meanrev"]
    df[regime_cfg["name"]] = compute_regime_trend_meanrev(df, regime_cfg["params"])

    # Breakout Label
    breakout_cfg = LABEL_CONFIG["patterns"]["breakout_20d"]
    df[breakout_cfg["name"]] = compute_breakout(df, breakout_cfg["params"])

    # Drop NA rows from rolling calculations
    df = df.dropna().reset_index(drop=True)

    # Save individual labeled file
    out_path = DATA_LABELED / f"{ticker}_labeled.csv"
    df.to_csv(out_path, index=False)
    typer.echo(f"[label_features] ✅ Saved labeled: {out_path}")

    df["ticker"] = ticker
    return df

# Merge All Tickers → market_patterns.parquet
@app.command("build")
def build_labels():
    tickers = PROJECT_CONFIG["universe"]["tickers"]

    frames = []
    for ticker in tickers:
        df = process_ticker(ticker)
        if df is not None:
            frames.append(df)

    if not frames:
        typer.echo("[label_features] ❌ No labeled frames produced")
        return

    merged = pd.concat(frames, ignore_index=True)
    merged_path = DATA_LABELED / "market_patterns.parquet"
    merged.to_parquet(merged_path, index=False)

    typer.echo(f"[label_features] 🎉 All done — merged dataset: {merged_path}")


if __name__ == "__main__":
    app()
