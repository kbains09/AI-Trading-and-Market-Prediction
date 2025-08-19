# `plot_backtest.py` 

## Imports

```python
import json
import math
import pathlib as pl

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import typer
```

- **json**: read the manifest file created by `backtest_cli.py`.
    
- **math**: square root for Sharpe scaling.
    
- **pathlib as pl**: safe, cross-platform path handling.
    
- **pandas**: load CSV, handle time series indexing.
    
- **matplotlib.pyplot**: plotting engine for equity/drawdown/Sharpe figures.
    
- **numpy**: vector math (rolling Sharpe, NaN handling).
    
- **typer**: command-line parsing for options like `--ticker`, `--logy`, etc.
    

---

## Helper: load manifest

```python
def _load_manifest(manifest_path: pl.Path) -> dict:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text())
        except Exception:
            return {}
    return {}
```

- Tries to read JSON from `{ticker}_backtest.manifest.json`.
    
- Returns `{}` on failure (safe fallback).
    
- **Purpose:** enriches plot titles with Sharpe/CAGR if available.
    

---

## Helper: rolling Sharpe

```python
def _rolling_sharpe(ret: pd.Series, window: int = 63, trading_days: int = 252) -> pd.Series:
    mu = ret.rolling(window).mean()
    sd = ret.rolling(window).std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = (mu / sd) * math.sqrt(trading_days)
    return rs.replace([np.inf, -np.inf], np.nan)
```

- Computes **rolling Sharpe**: mean/σ × √252.
    
- Uses a lookback `window` (default 63 trading days ≈ 3 months).
    
- **NaN handling:** replaces infinities with NaN.
    
- **Interpretation:** shows if Sharpe is stable across regimes.
    

---

## Helper: column hardening

```python
def _ensure_two_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["cumulative_market_return", "cumulative_strategy_return"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Backtest CSV is missing required columns: {missing}")
    out = df[cols].dropna(how="any").copy()
    return out
```

- Validates that both cumulative return columns exist.
    
- Drops rows with NaN in either.
    
- Returns a clean DataFrame for plotting equity curves.
    
- **Fails fast** with a clear error if columns are missing.
    

---

## CLI definition

```python
@app.callback()
def main(
    ticker: str = typer.Option(..., help="Ticker symbol, e.g., AAPL"),
    backtest_dir: str = typer.Option("data/backtests", help="Directory with backtest results"),
    logy: bool = typer.Option(False, help="Use log scale for equity plot"),
    window: int = typer.Option(63, help="Rolling window (days) for Sharpe (≈ 3 months)"),
    show: bool = typer.Option(True, help="Show plots interactively"),
    save_prefix: str = typer.Option(
        "", help="If provided, save PNGs using this prefix (e.g., reports/plots/AAPL)"
    ),
):
```

- **`ticker`**: required; name of asset to plot.
    
- **`backtest_dir`**: folder containing CSV + manifest.
    
- **`logy`**: switch to log-scale y-axis for equity curves.
    
- **`window`**: rolling Sharpe window length.
    
- **`show`**: show interactively (`plt.show()`) or just save.
    
- **`save_prefix`**: base path prefix for saved PNGs (`*_equity.png`, `*_drawdown.png`, `*_rollsharpe.png`).
    
- Example:
    
    ```bash
    poetry run python src/algo_trader/plot_backtest.py \
      --ticker AAPL --logy --no-show --save-prefix reports/plots/AAPL
    ```
    

---

## File resolution + manifest read

```python
    backtest_path = pl.Path(backtest_dir) / f"{ticker}_backtest.csv"
    if not backtest_path.exists():
        typer.echo(f"❌ Backtest file not found: {backtest_path}")
        raise typer.Exit(code=1)

    manifest_path = pl.Path(backtest_dir) / f"{ticker}_backtest.manifest.json"
    manifest = _load_manifest(manifest_path)
    m = manifest.get("metrics", {})
    sharpe_txt = f" | Sharpe {m.get('sharpe'):.2f}" if isinstance(m.get("sharpe"), (int, float)) else ""
    cagr = m.get("cagr")
    cagr_txt = f" | CAGR {cagr:.2%}" if isinstance(cagr, (int, float)) else ""
```

- Validates CSV exists.
    
- Loads optional manifest, extracts Sharpe/CAGR for title.
    
- **String enrichment:** appends `" | Sharpe x.xx | CAGR yy%"` if metrics are present.
    

---

## Load + preprocess data

```python
    df = pd.read_csv(backtest_path)
    if "Date" not in df.columns:
        typer.echo("❌ Backtest CSV must include a 'Date' column.")
        raise typer.Exit(code=1)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    core = _ensure_two_cols(df)
```

- Reads CSV.
    
- Ensures `Date` exists, parses to datetime, drops bad dates.
    
- Sets `Date` as index, sorts ascending.
    
- Calls `_ensure_two_cols` to pull just the cumulative return columns and drop NaNs.
    
- **Guarantees:** clean chronological time series for plotting.
    

---

## Plot 1: Equity curves

```python
    fig1 = plt.figure(figsize=(12, 6))
    if logy:
        plt.yscale("log")
    plt.plot(core.index, core["cumulative_market_return"], label="Market Return", linestyle="--")
    plt.plot(core.index, core["cumulative_strategy_return"], label="Strategy Return", linewidth=2)
    plt.title(f"{ticker} - Strategy vs Market Return{sharpe_txt}{cagr_txt}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return" + (" (log)" if logy else ""))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
```

- Strategy vs market, bold vs dashed, optional log scale.
    
- Title enriched with Sharpe/CAGR if manifest present.
    
- X axis = time, Y axis = cumulative multiple.
    
- **Log scale** useful if strategy grows >3× while market lags.
    

---

## Plot 2: Underwater (drawdown)

```python
    eq = core["cumulative_strategy_return"].clip(lower=1e-12)
    dd = eq / eq.cummax() - 1.0
    fig2 = plt.figure(figsize=(12, 2.8))
    plt.plot(dd, label="Drawdown")
    plt.title(f"{ticker} - Underwater (Drawdown)")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
```

- Drawdown = equity / rolling peak − 1.0.
    
- Shows max pain points; important for risk evaluation.
    
- Clipping avoids divide-by-zero.
    
- Shorter figure height (2.8") makes it dashboard-friendly.
    

---

## Plot 3: Rolling Sharpe

```python
    if "strategy_return" in df.columns:
        ret = pd.to_numeric(df["strategy_return"], errors="coerce").fillna(0.0)
    else:
        ret = core["cumulative_strategy_return"].pct_change().fillna(0.0)

    rs = _rolling_sharpe(ret, window=window, trading_days=252)
    fig3 = plt.figure(figsize=(12, 2.8))
    plt.plot(rs)
    plt.title(f"{ticker} - Rolling Sharpe ({window}d)")
    plt.grid(True)
    plt.tight_layout()
```

- If the CSV has `strategy_return` (per-bar returns), uses that.
    
- Otherwise falls back to differences of cumulative strategy curve.
    
- Computes rolling Sharpe with `_rolling_sharpe`.
    
- Plots ~3-month rolling Sharpe by default.
    
- **Diagnostic:** shows regime dependence (is the strategy only strong in certain years?).
    

---

## Save to disk if requested

```python
    if save_prefix:
        base = pl.Path(save_prefix)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(base.with_suffix("").as_posix() + "_equity.png", dpi=180, bbox_inches="tight")
        fig2.savefig(base.with_suffix("").as_posix() + "_drawdown.png", dpi=180, bbox_inches="tight")
        fig3.savefig(base.with_suffix("").as_posix() + "_rollsharpe.png", dpi=180, bbox_inches="tight")
        typer.echo(f"💾 Saved plots to prefix: {base}")
```

- If `--save-prefix` is provided, writes three PNGs:
    
    - `<prefix>_equity.png`
        
    - `<prefix>_drawdown.png`
        
    - `<prefix>_rollsharpe.png`
        
- Creates parent directories if needed.
    
- Uses high DPI (180) and tight bounding box.
    

---

## Show or close

```python
    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
```

- If `--no-show`, closes figures (useful in CI/headless).
    
- If `--show` (default), pops up GUI windows.
    

---

## Entrypoint

```python
if __name__ == "__main__":
    app()
```

- Runs as a Typer CLI.
    
- Typical usage:
    
    ```bash
    poetry run python src/algo_trader/plot_backtest.py --ticker AAPL --save-prefix reports/plots/AAPL --no-show
    ```
    

---

# 🔑 Key Features of This Version

- **Hardened input**: sorts dates, drops NaNs, checks required columns.
    
- **Rich plots**: equity (with log option), drawdown, rolling Sharpe.
    
- **Manifest integration**: pulls Sharpe/CAGR for title.
    
- **Headless-safe**: can skip GUI and save plots directly.
    
- **Configurable**: log scale, rolling window, save path.
    

---

# Potential Sharp Edges

- **Manifest metrics mismatch**: Title may show Sharpe/CAGR from a previous run if CSV and manifest are out of sync.
    
- **Rolling Sharpe on sparse data**: For short series (< window), Sharpe = NaN at start. Expected behavior.
    
- **Performance**: For large datasets (intraday, >1M rows), matplotlib may be slow. Downsample before plotting if needed.
    

---

# TL;DR

- This upgraded `plot_backtest.py` produces **three diagnostic plots** (equity, drawdown, rolling Sharpe) with **optional log scale** and **file saving**.
    
- It integrates with your backtest manifest for enriched titles, works in headless environments, and is hardened against missing columns/NaNs.
    

---

👉 Do you want me to do the same deep-notes + upgrade pass next for **`collect_data.py`**? That would complete the four core scripts.