# `engineering_features.py` 

## Imports & CLI app

```python
from __future__ import annotations
import os
import math
import pathlib as pl
from typing import Optional, List

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)
```

- `__future__` import: allows forward annotations (nice for type hints).
    
- `os` is imported but **not used** (you can remove).
    
- `Optional` used in signatures; `List` used for `_add_lags`.
    
- `typer` powers a simple CLI: `python engineering_features.py run --raw-dir data/raw --out-dir data/processed`.
    

---

## Low-level helpers (returns, MAs, indicators)

```python
def _safe_pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)
```

- Wraps `pct_change` and sanitizes divisions by zero to `NaN`.
    
- **Note:** leaves the first `periods` rows as `NaN` (expected).
    

```python
def _log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    # log(Price_t / Price_{t-1})
    return np.log(close / close.shift(periods))
```

- Log returns; equivalent to `np.log(close).diff(periods)`.
    
- **Numerical edge:** if `close` has zeros, division hits `inf`; upstream numeric coercion should avoid zeros for equities (for crypto, 0 close is highly unlikely but still guardable).
    

```python
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()
```

- Exponential moving average with **Wilder‑like warm‑up** (requires `span` values first).
    
- Using `min_periods=span` ensures early values are `NaN` (safer for training).
    

```python
def _sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()
```

- Simple moving average; also delays until full window.
    

```python
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(window).mean()
    roll_down = pd.Series(down, index=close.index).rolling(window).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

- RSI via **SMA** of gains/losses. Original Wilder RSI uses **EMA** of gains/losses; your version is fine but slightly different and a bit noisier.
    
- **Edge cases:**
    
    - When `roll_down==0`, `rs→∞` → RSI→100 (extreme overbought). This is mathematically consistent; if you want to avoid exact 100s, add a small `eps` in denominator.
        

```python
def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    signal_line = _ema(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist
```

- Standard MACD with your `_ema`. Warm‑up is handled by `min_periods`, so early values are `NaN`.
    

```python
def _bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    ma = _sma(close, window)
    std = close.rolling(window, min_periods=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / ma
    return ma, upper, lower, width
```

- Bands and **relative width**.
    
- **Edge:** if `ma` is ~0 (penny stocks / data issues), `width` blows up. Consider `width = (upper - lower) / ma.replace(0, np.nan)`.
    

```python
def _true_range(high: pd.Series, low: pd.Series, close: pd.Series):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr
```

- Classic True Range (max of three ranges).
    

```python
def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.rolling(window, min_periods=window).mean()
```

- ATR via **SMA** of TR (Wilder uses EMA). Either is fine; EMA is smoother.
    

```python
def _rolling_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window, min_periods=window).std()
```

- Rolling **σ** of returns. For annualized later, multiply by `sqrt(252)`.
    

```python
def _add_lags(df: pd.DataFrame, cols: List[str], lags: int = 3) -> pd.DataFrame:
    for col in cols:
        for k in range(1, lags + 1):
            df[f"{col}_lag{k}"] = df[col].shift(k)
    return df
```

- Adds **lagged** versions of features to reduce lookahead and capture short‑term dynamics.
    
- **Note:** Mutates `df` in place; returns same reference (okay given your usage).
    

```python
def _infer_ticker_from_path(path: pl.Path) -> str:
    # e.g., data/raw/AAPL.csv → AAPL
    name = path.stem
    return name.split("_")[0].upper()
```

- Heuristic to extract ticker from filename.
    
- If your raw files include suffixes (e.g., `AAPL_raw`), this still returns `AAPL`.
    

---

## Core transformer: `engineer_features(...)`

```python
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
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)
```

- Copies and ensures chronological order. **Good**: most indicators assume monotonic time.
    

### Returns (basic)

```python
    df["return_1d"] = _safe_pct_change(df["Close"], 1)
    df["logret_1d"] = _log_return(df["Close"], 1)
```

- You compute **both** simple and log returns. That’s fine; you can later drop one to reduce collinearity.
    

### Trend/MA features

```python
    df[f"SMA_{bb_window}"] = _sma(df["Close"], bb_window)
    df[f"EMA_{ema_fast}"] = _ema(df["Close"], ema_fast)
    df[f"EMA_{ema_slow}"] = _ema(df["Close"], ema_slow)
```

- Trend capture across two time scales + a mid SMA.
    

### Oscillator (RSI)

```python
    df[f"RSI_{rsi_window}"] = _rsi(df["Close"], rsi_window)
```

- Values typically in [0, 100]; early windows `NaN`.
    

### MACD trio

```python
    macd, macd_sig, macd_hist = _macd(df["Close"], ema_fast, ema_slow, macd_signal)
    df["MACD"] = macd
    df["MACD_signal"] = macd_sig
    df["MACD_hist"] = macd_hist
```

- Three correlated columns. Keep all or later select via feature importance.
    

### Bollinger bands + %B

```python
    bb_ma, bb_up, bb_lo, bb_w = _bollinger(df["Close"], bb_window, 2.0)
    df[f"BB_MA_{bb_window}"] = bb_ma
    df[f"BB_UP_{bb_window}"] = bb_up
    df[f"BB_LO_{bb_window}"] = bb_lo
    df[f"BB_WIDTH_{bb_window}"] = bb_w
    df["BB_%B"] = (df["Close"] - bb_lo) / (bb_up - bb_lo)
```

- **%B** is standardized location inside the band (0 at lower, 1 at upper).
    
- **Edge:** If `bb_up == bb_lo` (flat price), denominator is 0 → `inf`. Consider adding a small `eps` to denominator.
    

### Volatility & spreads

```python
    df[f"ATR_{atr_window}"] = _atr(df["High"], df["Low"], df["Close"], atr_window)
    df["HL_spread"] = df["High"] - df["Low"]
    df["OC_spread"] = (df["Close"] - df["Open"]).abs()

    df[f"vol_{vol_window}"] = _rolling_vol(df["return_1d"], vol_window)
```

- ATR (range‑based), raw intrabar spreads (HL/OC), and realized σ of returns.
    

### Targets

```python
    df["next_return_1d"] = df["return_1d"].shift(-1)
    df["target_up"] = (df["next_return_1d"] > 0).astype(int)
```

- **Label creation**: binary up/down label (1 if next day’s return > 0).
    
- **Alignment**: `next_return_1d` is forward; rows at time _t_ now contain **t+1** return — this is the correct alignment for backtesting and training when features are at _t_.
    
- ✅ This matches the backtester’s expectation (no post‑hoc shifting needed).
    

> ⚠️ **Pipeline mismatch callout:**  
> Your **`train_model.py`** expects the label column to be named **`Target`** (it drops `["Date", "Target"]` and sets `y = df["Target"]`).  
> Here you create **`target_up`**.  
> To keep everything plug‑and‑play, either:
> 
> - Rename `target_up → Target` **in this function**, or
>     
> - Add a small labeling step later that renames it before saving to `data/labeled/`.
>     

### Lags (leakage guard + dynamics)

```python
    lag_cols = [
        "return_1d", "logret_1d",
        f"SMA_{bb_window}", f"EMA_{ema_fast}", f"EMA_{ema_slow}",
        f"RSI_{rsi_window}", "MACD", "MACD_signal", "MACD_hist",
        f"BB_MA_{bb_window}", f"BB_UP_{bb_window}", f"BB_LO_{bb_window}", f"BB_WIDTH_{bb_window}", "BB_%B",
        f"ATR_{atr_window}", "HL_spread", "OC_spread", f"vol_{vol_window}",
    ]
    df = _add_lags(df, lag_cols, lags=3)
```

- Adds **1–3 bar lags** of each feature; this helps the model avoid accidental peeking at values that depend on the **close at t** when predicting **t+1**, and captures momentum/mean‑reversion dynamics.
    
- You keep the **unlagged** versions too; that’s fine (features at _t_ to predict _t+1_ is legal). The lags often help tree models.
    

```python
    # Drop rows with NaNs introduced by warm-ups if you prefer a clean training frame
    # (You can also leave them and drop later right before modeling.)
    return df
```

- You **don’t** drop NaNs here; that’s okay if you drop later (you do in training/backtest).
    
- If you want to ship a clean dataset, drop with something like:
    
    ```python
    warmup = max(bb_window, ema_slow, rsi_window, atr_window, vol_window) + 3  # +lags
    df = df.iloc[warmup:].reset_index(drop=True)
    ```
    

---

## File processor (CSV in → features CSV out)

```python
def process_file(
    infile: pl.Path,
    outdir: pl.Path,
    date_col: str = "Date"
) -> pl.Path:
    df = pd.read_csv(infile)
    # Normalize schema
    expected = {"Open", "High", "Low", "Close", "Volume"}
    if date_col not in df.columns:
        raise ValueError(f"Missing '{date_col}' in {infile}")
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain {expected} in {infile}")
```

- Enforces a **minimum schema**. You don’t require `Adj Close` (fine).
    
- **Nice to add:** strict dtype casting (numeric) and a quick monotonicity check on dates for quality gating.
    

```python
    df[date_col] = pd.to_datetime(df[date_col], utc=False)
    df = df.rename(columns={date_col: "Date"})
    df = df.sort_values("Date").reset_index(drop=True)
```

- Parses dates (naive timezone; consistent with your other code).
    
- Renames to canonical `"Date"` for downstream consistency.
    

```python
    features = engineer_features(df)

    ticker = _infer_ticker_from_path(pl.Path(infile))
    outpath = outdir / f"{ticker}_features.csv"
    outdir.mkdir(parents=True, exist_ok=True)
    features.to_csv(outpath, index=False)
    return outpath
```

- Calls the transformer and writes to `data/processed/{TICKER}_features.csv`.
    

> ⚠️ **Pipeline mismatch callout (filenames):**  
> Your **`train_model.py`** loads from **`data/labeled/{ticker}_labeled.csv`**, not `data/processed/{ticker}_features.csv`.  
> You’ll need a small **labeling/export step** (or change paths) to produce the exact file the trainer expects:
> 
> - Option A: Change this to write `data/labeled/{ticker}_labeled.csv` **and** rename `target_up → Target`.
>     
> - Option B: Leave this as “processed features” and have a follow‑up step that selects columns and writes the labeled training CSV.
>     

---

## CLI command

```python
@app.command()
def run(
    raw_dir: str = typer.Option("data/raw", help="Directory with raw CSVs"),
    out_dir: str = typer.Option("data/processed", help="Directory for processed feature CSVs"),
) -> None:
    rawp = pl.Path(raw_dir)
    outp = pl.Path(out_dir)
    files = sorted(list(rawp.glob("*.csv")))
    if not files:
        typer.echo(f"[feature_engineering] No CSVs found in {rawp}")
        raise typer.Exit(1)
    for f in files:
        out = process_file(f, outp)
        typer.echo(f"[feature_engineering] Wrote: {out}")
```

- Scans `raw_dir` for all CSVs and runs them through `process_file`.
    
- Emits one processed CSV per input ticker.
    

```python
if __name__ == "__main__":
    app()
```

- Run as:
    
    ```bash
    poetry run python src/algo_trader/engineering_features.py run \
      --raw-dir data/raw --out-dir data/processed
    ```
    

---

## Leakage & alignment audit (important)

- **Good:** All features are built from data up to and including time _t_; the **target** is `next_return_1d` / `target_up` (t+1). That’s leakage‑safe for a next‑bar prediction.
    
- **Good:** You **did not** include `next_return_1d` in the `lag_cols`. It remains a label/PnL column only.
    
- **Downstream:** Ensure training code does **not** include `next_return_1d` in `X`. (Your trainer currently drops only `["Date","Target"]`. If you rename `target_up → Target`, also exclude `next_return_1d` before training.)
    

---

## Quick, high‑impact improvements

1. **Rename label + emit trainer‑ready file**
    
    - If you want this step to produce what `train_model.py` expects:
        
        ```python
        features = engineer_features(df)
        features = features.rename(columns={"target_up": "Target"})
        outpath = outdir / f"{ticker}_labeled.csv"
        ```
        
    - Also **drop** columns you don’t want the model to see (e.g., `next_return_1d`) **before** training (or do it in the trainer).
        
2. **Guard against 0‑denominators**
    
    - `%B` and `BB_WIDTH` denominators can be 0. Add small eps:
        
        ```python
        eps = 1e-12
        df["BB_%B"] = (df["Close"] - bb_lo) / ((bb_up - bb_lo).replace(0, np.nan))
        width = (upper - lower) / ma.replace(0, np.nan)
        ```
        
3. **Indicator variants**
    
    - Consider Wilder RSI / ATR via EMA for smoother signals:
        
        - Replace rolling mean with `ewm(alpha=1/window, adjust=False).mean()`.
            
4. **Warm‑up trim**
    
    - Optionally drop the initial warm‑up where many features are NaN so downstream scripts don’t need to:
        
        ```python
        warmup = max(bb_window, ema_slow, rsi_window, atr_window, vol_window) + 3
        df = df.iloc[warmup:].reset_index(drop=True)
        ```
        
5. **Calendar / session features**
    
    - Add `day_of_week`, `month`, `is_month_end`, `is_options_expiry` (if you maintain a calendar), `overnight_return` (Close→next Open if you add Open of t+1).
        
6. **Regime features (later)**
    
    - Running σ & trend HMM / k‑means cluster label as `regime_id` for the model to condition on.
        
7. **Output format**
    
    - Save to **Parquet** for speed & schema:
        
        ```python
        features.to_parquet(outpath.with_suffix(".parquet"), index=False)
        ```
        

---

## Sanity checks (quick)

- After generating features:
    
    ```python
    df = pd.read_csv("data/processed/AAPL_features.csv", parse_dates=["Date"])
    assert df["Date"].is_monotonic_increasing
    # check leakage: next_return_1d must be a *shifted* return
    (df["next_return_1d"].shift(1).dropna().equals(df["return_1d"].dropna())).any()  # should be roughly True
    # ensure no impossible infinities:
    np.isinf(df.select_dtypes(float)).sum().sum() == 0
    ```
    

---

## How this fits your pipeline (as written)

- **Input:** `data/raw/{TICKER}.csv` (from your `collect_data.py`).
    
- **Output:** `data/processed/{TICKER}_features.csv` with features **and** labels (`next_return_1d`, `target_up`).
    
- **Trainer/backtester:** Expect **`data/labeled/{TICKER}_labeled.csv`** and a label column named **`Target`**. So either:
    
    - Change this script to write `data/labeled/{TICKER}_labeled.csv` (rename label), or
        
    - Add a thin “label export” step that reads processed features, renames, selects columns, and writes labeled files.
        

---

## TL;DR

- You’re computing a solid baseline feature set (trend, momentum, bands, ATR, spreads, rolling vol) and a clean **forward** target.
    
- Alignment is correct and leakage‑safe.
    
- To plug into the rest of your pipeline seamlessly:
    
    - **Rename `target_up` → `Target`** and **save to `data/labeled/{ticker}_labeled.csv`**, or update the trainer to expect `target_up`.
        
    - Ensure the trainer **drops `next_return_1d`** from features.
        
- Add small **denominator guards**, optional **warm‑up trimming**, and consider Wilder variants for smoother RSI/ATR.
    

If you’d like, I can hand you a **drop‑in revision** that:

- Renames `target_up` → `Target`,
    
- Drops `next_return_1d` from the model feature set but keeps it in the CSV for backtesting,
    
- Writes to `data/labeled/{ticker}_labeled.csv`,
    
- Adds the small numerical safety tweaks discussed above.