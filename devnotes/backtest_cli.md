# `backtest_cli.py`

## Imports (standard + ML + CLI)

```python
import json
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import pathlib as pl
import typer
import joblib
from sklearn.metrics import accuracy_score
```

- **json** / **hashlib** / **datetime**: create a reproducibility **manifest** per run (JSON), with SHA‑256 hashes of the model and features, and a UTC timestamp.
    
- **numpy**, **pandas**: numerical ops and DataFrame manipulation.
    
- **pathlib**: safe, cross‑platform file paths.
    
- **typer**: CLI framework (nice help, typed options).
    
- **joblib**: load scikit/XGBoost models (`.pkl`).
    
- **accuracy_score**: quick sanity metric for classification; not trading‑aware, but useful to catch obvious model regressions.
    

---

## Typer App

```python
app = typer.Typer(add_completion=False)
```

Creates the CLI app. Tab‑completion is off to keep things minimal.

---

## 📈 Metrics & Helper Functions

These functions compute risk/performance stats and utilities used later.

```python
def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())
```

- **Input:** `equity` is a cumulative curve starting near **1.0** (e.g., `(1+ret).cumprod()`).
    
- **Logic:** classic **underwater** computation; most negative drawdown is returned.
    
- **Edge cases:** Requires non‑empty series; handles flat series (dd=0).
    

```python
def _cagr(cumret: pd.Series, trading_days: int = 252) -> float:
    n = len(cumret)
    if n == 0 or cumret.iloc[-1] <= 0:
        return float("nan")
    years = n / trading_days
    return float(cumret.iloc[-1] ** (1.0 / years) - 1.0)
```

- **Assumes** daily bars; `trading_days` scales to years.
    
- Uses **ending multiple** to compute CAGR. Returns NaN for empty/invalid series.
    

```python
def _ann_vol(ret: pd.Series, trading_days: int = 252) -> float:
    return float(ret.std(ddof=0) * np.sqrt(trading_days))
```

- **Population** std (`ddof=0`) then scales by √252.
    

```python
def _sharpe(ret: pd.Series, rf_annual: float = 0.0, trading_days: int = 252) -> float:
    if ret.std(ddof=0) == 0 or np.isnan(ret.std(ddof=0)):
        return float("nan")
    rf_daily = rf_annual / trading_days
    return float(((ret.mean() - rf_daily) / ret.std(ddof=0)) * np.sqrt(trading_days))
```

- **Annualized Sharpe** with optional risk‑free rate.
    
- Guards against zero/NaN std.
    

```python
def _turnover(position: pd.Series) -> float:
    return float(position.diff().abs().fillna(0.0).sum())
```

- Sums absolute **position changes** period‑to‑period. Higher = more trading (costs/slippage).
    

```python
def _hit_rate(trade_pnl: pd.Series) -> float:
    wins = (trade_pnl > 0).sum()
    total = (trade_pnl != 0).sum()
    return float(wins / total) if total > 0 else float("nan")
```

- Fraction of non‑zero trades with positive P&L.
    
- **Note:** If you include flats, they’re excluded from denominator.
    

```python
def _sha256(path: pl.Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""
```

- Returns a file’s SHA‑256 for **provenance**. Empty string on error (file deleted, perm issues, etc.).
    

---

## 🔁 Core PnL Simulation

```python
def simulate_trades(
    df: pd.DataFrame,
    *,
    cost_bps: float = 0.0,
    vol_target_annual: float = 0.0,
    roll_window: int = 20,
    max_leverage: float = 3.0,
    trading_days: int = 252,
) -> pd.DataFrame:
    ...
```

### Inputs & Assumptions

- `df` contains **at least**:
    
    - `prediction` — model signal per bar, numeric. Can be {‑1,0,1} or {0,1}. Float allowed for confidence/vol‑scaled sizing.
        
    - `next_return_1d` — **forward** one‑period return aligned to _this_ row (created during labeling). **No shifts inside backtest.**
        
- `cost_bps` — round‑trip **transaction cost** in basis points applied on **position changes** (not per bar).
    
- `vol_target_annual` — if >0, we **scale** positions to target annualized volatility.
    
- `roll_window` — realized vol lookback (days) for the scaling.
    
- `max_leverage` — cap to avoid exploding leverage in low‑vol regimes.
    

### Body (step‑by‑step)

```python
out = df.copy()
pos = pd.to_numeric(out["prediction"], errors="coerce").fillna(0.0).astype(float)
```

- Clone input; coerce `prediction` to float numeric — any non‑numeric becomes 0 (flat).
    

```python
if vol_target_annual and vol_target_annual > 0.0:
    daily_target = vol_target_annual / np.sqrt(trading_days)
    roll_std = out["next_return_1d"].rolling(roll_window).std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lev = (daily_target / roll_std).clip(upper=max_leverage)
        lev = lev.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    pos = pos * lev
out["position"] = pos
```

- **Vol targeting:** scales the raw signal by **`daily_target / realized_daily_vol`**.
    
- **Guards:** replaces inf/NaN with 0 (flat) and caps leverage.
    
- **Result:** `position` can be fractional (e.g., 0.5, 1.7) and will respect sign of `prediction`.
    

```python
out["strategy_return_gross"] = out["position"] * out["next_return_1d"]
```

- **Gross** per‑bar return before costs: classic “position × next return”.
    

```python
if cost_bps and cost_bps > 0.0:
    delta_pos = out["position"].diff().abs().fillna(0.0)
    cost_rate = (cost_bps / 1e4)
    out["tc"] = -cost_rate * delta_pos
else:
    out["tc"] = 0.0
```

- Simple **turnover‑based cost model**:
    
    - Moving from +1 to −1 costs **2 × cost_rate**.
        
    - `cost_bps` is **basis points**, so 5 → 0.0005.
        

```python
out["strategy_return"] = out["strategy_return_gross"] + out["tc"]
out["cumulative_strategy_return"] = (1.0 + out["strategy_return"]).cumprod()
out["cumulative_market_return"] = (1.0 + out["next_return_1d"]).cumprod()
```

- Net strategy return includes costs.
    
- Build **equity curves** for strategy and buy‑and‑hold (market).
    

**Key design decision:**  
No shifting/mutation of `next_return_1d`. Alignment is **fixed** and consistent between the saved CSV and the equity curves.

---

## 🧰 CLI Command

```python
@app.command("run")
def run_backtest(
    model_dir: str = typer.Option("models/xgboost", ...),
    feature_dir: str = typer.Option("data/labeled", ...),
    ticker: str = typer.Option("AAPL", ...),
    cost_bps: float = typer.Option(0.0, ...),
    vol_target: float = typer.Option(0.0, ...),
    roll_window: int = typer.Option(20, ...),
    max_leverage: float = typer.Option(3.0, ...),
    rf_annual: float = typer.Option(0.0, ...),
    trading_days: int = typer.Option(252, ...),
):
    ...
```

- Adds configurable **costs**, **vol targeting**, **risk‑free rate**, and **market frequency**.
    
- **Usage example:**
    
    ```bash
    poetry run python src/algo_trader/backtest_cli.py run \
      --ticker AAPL --cost-bps 5 --vol-target 0.10 --roll-window 20 --max-leverage 3
    ```
    

### Path setup + existence checks

```python
model_path = pl.Path(model_dir) / f"{ticker}_xgb_model.pkl"
feature_path = pl.Path(feature_dir) / f"{ticker}_labeled.csv"
if not model_path.exists(): ...
if not feature_path.exists(): ...
```

- Fails fast with clear messages if anything’s missing.
    

### Load model/data & hygiene

```python
model = joblib.load(model_path)
df = pd.read_csv(feature_path)
df = df.dropna(how="any").reset_index(drop=True)
```

- Loads artifacts.
    
- **Simplest cleanup**: drop any row with NaNs and reset row index.  
    (If you prefer deterministic handling, consider imputing specific features instead.)
    

### Validate required columns + feature split

```python
if "Target" not in df.columns: ...
if "next_return_1d" not in df.columns: ...

drop_cols = [c for c in ["Date", "Target"] if c in df.columns]
X = df.drop(columns=drop_cols, errors="ignore")
y = df["Target"]
```

- Ensures the labeled frame has the necessary **label** and **forward return** fields.
    
- `X` uses every column except `Date` and `Target`.  
    ⚠️ _You should validate your feature engineering ensured **no leakage** (i.e., all features use only past info)._
    

### Predict + quick metric

```python
preds = model.predict(X)
df["prediction"] = preds
acc = accuracy_score(y, preds)
```

- Stores predictions; prints **accuracy** as a rough, _non trading‑aware_ check.
    
- If your model is probabilistic, you can later extend with `predict_proba` and confidence thresholds.
    

### Run simulation

```python
results = simulate_trades(
    df,
    cost_bps=cost_bps,
    vol_target_annual=vol_target,
    roll_window=roll_window,
    max_leverage=max_leverage,
    trading_days=trading_days,
)
```

- Passes CLI parameters into the PnL engine.
    
- **No alignment mutations** inside.
    

### Compute performance summary

```python
strat_ret = results["strategy_return"]
mkt_ret = results["next_return_1d"]
eq_strat = results["cumulative_strategy_return"]
eq_mkt = results["cumulative_market_return"]

final_strat_return = float(eq_strat.iloc[-1]) if len(eq_strat) else float("nan")
final_market_return = float(eq_mkt.iloc[-1]) if len(eq_mkt) else float("nan")

strat_cagr = _cagr(eq_strat, trading_days)
strat_vol = _ann_vol(strat_ret, trading_days)
strat_sharpe = _sharpe(strat_ret, rf_annual=rf_annual, trading_days=trading_days)
strat_mdd = _max_drawdown(eq_strat)
strat_turnover = _turnover(results["position"])
trade_pnl = (results["position"] * results["next_return_1d"]).where(results["position"] != 0.0, 0.0)
strat_hit_rate = _hit_rate(trade_pnl)
```

- **final_*_return** are **multiples** (e.g., 1.35x), directly printable.
    
- **CAGR/vol/Sharpe/MDD**: core risk stats.
    
- **Turnover**: sum of |Δposition| across the series.
    
- **Hit rate**: % of non‑zero trades that made money.
    

### Console output

```python
typer.echo(f"📈 Strategy Return: {final_strat_return:.4f}x")
typer.echo(f"📉 Market Return:   {final_market_return:.4f}x")
typer.echo(f"📊 CAGR: {strat_cagr:.4%} | Vol: {strat_vol:.4%} | Sharpe: {strat_sharpe:.2f} | MaxDD: {strat_mdd:.2%}")
typer.echo(f"🔁 Turnover: {strat_turnover:.4f} | 🎯 Hit rate: {strat_hit_rate:.2%}")
```

- Gives a **compact snapshot** of strategy quality.
    

### Save outputs (CSV + manifest)

```python
out_dir = pl.Path("data/backtests")
out_dir.mkdir(parents=True, exist_ok=True)

outpath_csv = out_dir / f"{ticker}_backtest.csv"
results.to_csv(outpath_csv, index=False)
```

- Saves full per‑bar results (including `position`, `strategy_return`, cumulative curves).
    

```python
manifest = {
  "ticker": ticker,
  "timestamp_utc": ...,
  "model_path": ..., "model_sha256": ...,
  "feature_path": ..., "feature_sha256": ...,
  "cost_bps": cost_bps, "vol_target_annual": vol_target,
  "roll_window": roll_window, "max_leverage": max_leverage,
  "rf_annual": rf_annual, "trading_days": trading_days,
  "metrics": {...}, "output_csv": str(outpath_csv),
}
outpath_manifest = out_dir / f"{ticker}_backtest.manifest.json"
outpath_manifest.write_text(json.dumps(manifest, indent=2))
```

- **Manifest** captures everything needed to reproduce/trace a run:
    
    - **Parameters** (costs, vol target, leverage, etc.)
        
    - **Data/Model hashes**
        
    - **Key metrics**
        
    - **Timestamp**
        
    - **Output file path**
        
- This gives you _MLflow‑lite_ without extra infra.
    

### Entrypoint

```python
if __name__ == "__main__":
    app(prog_name="backtest_cli.py")
```

- Required to run as a script.
    
- `prog_name` sets the CLI help header.
    

---

## Validation Checklist (quick tests you can run)

1. **Basic run (no costs/vol targeting):**
    
    ```bash
    poetry run python src/algo_trader/backtest_cli.py run --ticker AAPL
    ```
    
    - Expect CSV + manifest under `data/backtests/`.
        
    - `next_return_1d` column should **match** `cumulative_market_return`’s construction.
        
2. **With costs:**
    
    ```bash
    --cost-bps 5
    ```
    
    - `strategy_return` decreases on bars where `position` changes.
        
    - `turnover` > 0; `tc` column non‑zero around trades.
        
3. **With vol targeting:**
    
    ```bash
    --vol-target 0.10 --roll-window 20
    ```
    
    - `position` becomes fractional and scales inversely to rolling volatility.
        
    - Leverage should not exceed `--max-leverage`.
        
4. **Edge cases:**
    
    - Very short series (< roll window): `lev` becomes 0 until enough data; results should still save.
        
    - All‑zero predictions: flat `position`, zero `strategy_return`, equity stays ~1.0.
        

---

## Common Pitfalls + How This Version Avoids Them

- **Label alignment leak:** No in‑function shifting. Assumes labeling already aligned `next_return_1d` forward.
    
- **Misleading metrics:** Adds Sharpe/MDD/CAGR/turnover/hit‑rate beyond accuracy.
    
- **Reproducibility gaps:** Manifest with hashes + params.
    
- **Unbounded leverage:** `max_leverage` cap during vol targeting.
    
- **Invisible costs:** Turnover‑based cost model wired in.
    

---

## Nice Extensions You Can Add Later

- **Proba‑based sizing:** use `predict_proba` to map confidence → position size.
    
- **Side‑by‑side benchmarks:** SPY/QQQ cumulative curves (if multi‑asset context).
    
- **Per‑trade ledger:** detect entry/exit events, compute per‑trade P&L/hold time.
    
- **Config file:** support a YAML/JSON run config in addition to CLI flags.
    
- **Multiple tickers batch:** loop through a list and emit one manifest per ticker.
    

---
