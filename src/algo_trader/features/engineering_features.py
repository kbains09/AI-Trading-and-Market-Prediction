from __future__ import annotations
import os
import math
import pathlib as pl
from typing import Optional, List

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)

# ==== Internal Feature Calculations ====

def _safe_pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)

def _log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(close / close.shift(periods))

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def _sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(window).mean()
    roll_down = pd.Series(down, index=close.index).rolling(window).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    signal_line = _ema(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist

def _bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    ma = _sma(close, window)
    std = close.rolling(window, min_periods=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / ma
    return ma, upper, lower, width

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.rolling(window, min_periods=window).mean()

def _rolling_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window, min_periods=window).std()

def _add_lags(df: pd.DataFrame, cols: List[str], lags: int = 3) -> pd.DataFrame:
    for col in cols:
        for k in range(1, lags + 1):
            df[f"{col}_lag{k}"] = df[col].shift(k)
    return df

def _infer_ticker_from_path(path: pl.Path) -> str:
    return path.stem.split("_")[0].upper()

# ==== Main Feature Engineering Logic ====

def engineer_features(
    df: pd.DataFrame,
    rsi_window: int = 14,
    atr_window: int = 14,
    bb_window: int = 20,
    ema_fast: int = 12,
    ema_slow: int = 26,
    macd_signal: int = 9,
    vol_window: int = 20,
) -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)

    df["return_1d"] = _safe_pct_change(df["Close"])
    df["logret_1d"] = _log_return(df["Close"])

    df[f"SMA_{bb_window}"] = _sma(df["Close"], bb_window)
    df[f"EMA_{ema_fast}"] = _ema(df["Close"], ema_fast)
    df[f"EMA_{ema_slow}"] = _ema(df["Close"], ema_slow)

    df[f"RSI_{rsi_window}"] = _rsi(df["Close"], rsi_window)

    macd, macd_sig, macd_hist = _macd(df["Close"], ema_fast, ema_slow, macd_signal)
    df["MACD"] = macd
    df["MACD_signal"] = macd_sig
    df["MACD_hist"] = macd_hist

    bb_ma, bb_up, bb_lo, bb_w = _bollinger(df["Close"], bb_window)
    df[f"BB_MA_{bb_window}"] = bb_ma
    df[f"BB_UP_{bb_window}"] = bb_up
    df[f"BB_LO_{bb_window}"] = bb_lo
    df[f"BB_WIDTH_{bb_window}"] = bb_w
    df["BB_%B"] = (df["Close"] - bb_lo) / (bb_up - bb_lo)

    df[f"ATR_{atr_window}"] = _atr(df["High"], df["Low"], df["Close"], atr_window)
    df["HL_spread"] = df["High"] - df["Low"]
    df["OC_spread"] = (df["Close"] - df["Open"]).abs()

    df[f"vol_{vol_window}"] = _rolling_vol(df["return_1d"], vol_window)
    
    vol_z_window = 20
    vol_ma = df["Volume"].rolling(vol_z_window, min_periods=vol_z_window).mean()
    vol_std = df["Volume"].rolling(vol_z_window, min_periods=vol_z_window).std()
    df["vol_zscore_20"] = (df["Volume"] - vol_ma) / vol_std

    df["next_return_1d"] = df["return_1d"].shift(-1)
    df["target_up"] = (df["next_return_1d"] > 0).astype(int)

    lag_cols = [
        "return_1d", "logret_1d",
        f"SMA_{bb_window}", f"EMA_{ema_fast}", f"EMA_{ema_slow}",
        f"RSI_{rsi_window}", "MACD", "MACD_signal", "MACD_hist",
        f"BB_MA_{bb_window}", f"BB_UP_{bb_window}", f"BB_LO_{bb_window}", f"BB_WIDTH_{bb_window}", "BB_%B",
        f"ATR_{atr_window}", "HL_spread", "OC_spread", f"vol_{vol_window}",
    ]
    df = _add_lags(df, lag_cols)
    return df

# ==== Public Wrapper for Import ====

def generate_features(ticker: str, data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Loads raw data CSV for given ticker and returns engineered features DataFrame.
    """
    infile = pl.Path(data_dir) / f"{ticker}.csv"
    if not infile.exists():
        raise FileNotFoundError(f"Raw file not found: {infile}")
    
    df = pd.read_csv(infile, parse_dates=["Date"])
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return engineer_features(df)

# ==== CLI Feature Processing ====

def process_file(
    infile: pl.Path,
    outdir: pl.Path,
    date_col: str = "Date"
) -> pl.Path:
    df = pd.read_csv(infile)

    expected = {"Open", "High", "Low", "Close", "Volume"}
    if date_col not in df.columns:
        raise ValueError(f"Missing '{date_col}' in {infile}")
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain {expected} in {infile}")

    for col in expected:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.rename(columns={date_col: "Date"})
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=list(expected))

    features = engineer_features(df)
    ticker = _infer_ticker_from_path(pl.Path(infile))
    outpath = outdir / f"{ticker}_features.csv"
    outdir.mkdir(parents=True, exist_ok=True)
    features.to_csv(outpath, index=False)
    return outpath

@app.command()
def run(
    raw_dir: str = typer.Option("data/raw", help="Directory with raw CSVs"),
    out_dir: str = typer.Option("data/processed", help="Directory for processed feature CSVs"),
) -> None:
    rawp = pl.Path(raw_dir)
    outp = pl.Path(out_dir)
    files = sorted(rawp.glob("*.csv"))
    if not files:
        typer.echo(f"[feature_engineering] No CSVs found in {rawp}")
        raise typer.Exit(1)
    for f in files:
        out = process_file(f, outp)
        typer.echo(f"[feature_engineering] Wrote: {out}")

if __name__ == "__main__":
    app()
