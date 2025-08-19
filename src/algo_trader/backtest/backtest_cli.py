import json
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import pathlib as pl
import typer
import joblib
from sklearn.metrics import accuracy_score

app = typer.Typer(add_completion=False)

# -----------------------------
# Metrics & Helpers
# -----------------------------
def _max_drawdown(equity: pd.Series) -> float:
    """Max drawdown given an equity curve (cumprod-style, starting at 1.0)."""
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())

def _cagr(cumret: pd.Series, trading_days: int = 252) -> float:
    """Compound annual growth rate from cumulative returns (starts at 1.0)."""
    n = len(cumret)
    if n == 0 or cumret.iloc[-1] <= 0:
        return float("nan")
    years = n / trading_days
    return float(cumret.iloc[-1] ** (1.0 / years) - 1.0)

def _ann_vol(ret: pd.Series, trading_days: int = 252) -> float:
    return float(ret.std(ddof=0) * np.sqrt(trading_days))

def _sharpe(ret: pd.Series, rf_annual: float = 0.0, trading_days: int = 252) -> float:
    if ret.std(ddof=0) == 0 or np.isnan(ret.std(ddof=0)):
        return float("nan")
    rf_daily = rf_annual / trading_days
    return float(((ret.mean() - rf_daily) / ret.std(ddof=0)) * np.sqrt(trading_days))

def _turnover(position: pd.Series) -> float:
    """Sum of absolute position changes (per period)."""
    return float(position.diff().abs().fillna(0.0).sum())

def _hit_rate(trade_pnl: pd.Series) -> float:
    wins = (trade_pnl > 0).sum()
    total = (trade_pnl != 0).sum()
    return float(wins / total) if total > 0 else float("nan")

def _sha256(path: pl.Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""


# -----------------------------
# Core PnL Simulation
# -----------------------------
def simulate_trades(
    df: pd.DataFrame,
    *,
    cost_bps: float = 0.0,
    vol_target_annual: float = 0.0,
    roll_window: int = 20,
    max_leverage: float = 3.0,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Assumptions:
    - df has columns: 'next_return_1d' (forward return on *this* row), 'prediction' (∈ {-1,0,1} or {0,1})
    - No in-function shifting: alignment must be handled upstream during labeling.
    - Transaction cost is applied when position changes (simple bps of notional).
    - Vol targeting scales today's position based on rolling realized vol of next_return_1d.
    """
    out = df.copy()

    # 1) Base position from model prediction
    pos = pd.to_numeric(out["prediction"], errors="coerce").fillna(0.0).astype(float)

    # 2) Volatility targeting (optional)
    #    Target daily vol = vol_target_annual / sqrt(trading_days)
    if vol_target_annual and vol_target_annual > 0.0:
        daily_target = vol_target_annual / np.sqrt(trading_days)
        roll_std = out["next_return_1d"].rolling(roll_window).std(ddof=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lev = (daily_target / roll_std).clip(upper=max_leverage)
            lev = lev.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        pos = pos * lev
    out["position"] = pos

    # 3) Strategy return BEFORE costs
    out["strategy_return_gross"] = out["position"] * out["next_return_1d"]

    # 4) Transaction costs (bps) on position changes
    #    If you move from +1 to -1 with leverage, you "trade" 2 units, so pay 2 * cost.
    if cost_bps and cost_bps > 0.0:
        delta_pos = out["position"].diff().abs().fillna(0.0)
        cost_rate = (cost_bps / 1e4)
        out["tc"] = -cost_rate * delta_pos
    else:
        out["tc"] = 0.0

    out["strategy_return"] = out["strategy_return_gross"] + out["tc"]

    # 5) Equity curves
    out["cumulative_strategy_return"] = (1.0 + out["strategy_return"]).cumprod()
    out["cumulative_market_return"] = (1.0 + out["next_return_1d"]).cumprod()

    return out


# -----------------------------
# CLI Command
# -----------------------------
@app.command("run")
def run_backtest(
    model_dir: str = typer.Option("models/xgboost", help="Directory with trained models"),
    feature_dir: str = typer.Option("data/labeled", help="Directory with labeled feature files"),
    ticker: str = typer.Option("AAPL", help="Ticker to backtest"),
    cost_bps: float = typer.Option(0.0, help="Round-trip transaction cost in basis points per unit turnover"),
    vol_target: float = typer.Option(0.0, help="Annualized volatility target (e.g., 0.10 for 10%%). 0 disables"),
    roll_window: int = typer.Option(20, help="Rolling window for vol targeting (days)"),
    max_leverage: float = typer.Option(3.0, help="Max leverage cap when vol targeting"),
    rf_annual: float = typer.Option(0.0, help="Annual risk-free rate for Sharpe calc"),
    trading_days: int = typer.Option(252, help="Periods per year (252 for daily)"),
):
    """
    Run backtest for a specific ticker with optional transaction costs and volatility targeting.
    Expects the labeled CSV to include:
      - Date
      - Target
      - next_return_1d (forward 1-period return aligned to THIS row)
      - feature columns used by the model
    """
    # Paths
    model_path = pl.Path(model_dir) / f"{ticker}_xgb_model.pkl"
    feature_path = pl.Path(feature_dir) / f"{ticker}_labeled.csv"

    if not model_path.exists():
        typer.echo(f"❌ Missing model for {ticker}: {model_path}")
        raise typer.Exit(1)
    if not feature_path.exists():
        typer.echo(f"❌ Missing labeled data for {ticker}: {feature_path}")
        raise typer.Exit(1)

    # Load model & data
    model = joblib.load(model_path)
    df = pd.read_csv(feature_path)

    # Basic hygiene
    # Keep Date as-is (string) for CSV roundtrip; parse later when plotting.
    df = df.dropna(how="any").reset_index(drop=True)

    # Split features/labels
    if "Target" not in df.columns:
        typer.echo("❌ Labeled file must include 'Target' column.")
        raise typer.Exit(1)
    if "next_return_1d" not in df.columns:
        typer.echo("❌ Labeled file must include 'next_return_1d' column.")
        raise typer.Exit(1)

    # Ensure we don't accidentally feed label/date columns into the model
    drop_cols = [c for c in ["Date", "Target"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["Target"]

    # Predict signals
    try:
        preds = model.predict(X)
    except Exception as e:
        typer.echo(f"❌ Model prediction failed: {e}")
        raise typer.Exit(1)

    df["prediction"] = preds

    # Sanity metric (not trading-aware)
    try:
        acc = accuracy_score(y, preds)
        typer.echo(f"✅ Model Accuracy on {ticker}: {acc:.4f}")
    except Exception:
        typer.echo("⚠️ Could not compute accuracy_score (non-classification or invalid labels).")

    # Simulate PnL (clean alignment; no shifting here)
    results = simulate_trades(
        df,
        cost_bps=cost_bps,
        vol_target_annual=vol_target,
        roll_window=roll_window,
        max_leverage=max_leverage,
        trading_days=trading_days,
    )

    # Compute metrics
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
    # Per-trade PnL proxy: position * next_return_1d when position != 0
    trade_pnl = (results["position"] * results["next_return_1d"]).where(results["position"] != 0.0, 0.0)
    strat_hit_rate = _hit_rate(trade_pnl)

    # Console summary
    typer.echo(f"📈 Strategy Return: {final_strat_return:.4f}x")
    typer.echo(f"📉 Market Return:   {final_market_return:.4f}x")
    typer.echo(f"📊 CAGR: {strat_cagr:.4%} | Vol: {strat_vol:.4%} | Sharpe: {strat_sharpe:.2f} | MaxDD: {strat_mdd:.2%}")
    typer.echo(f"🔁 Turnover: {strat_turnover:.4f} | 🎯 Hit rate: {strat_hit_rate:.2%}")

    # Save outputs
    out_dir = pl.Path("data/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)

    outpath_csv = out_dir / f"{ticker}_backtest.csv"
    results.to_csv(outpath_csv, index=False)

    # Save a small run manifest (JSON)
    manifest = {
        "ticker": ticker,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "feature_path": str(feature_path),
        "feature_sha256": _sha256(feature_path),
        "cost_bps": cost_bps,
        "vol_target_annual": vol_target,
        "roll_window": roll_window,
        "max_leverage": max_leverage,
        "rf_annual": rf_annual,
        "trading_days": trading_days,
        "metrics": {
            "final_strategy_multiple": final_strat_return,
            "final_market_multiple": final_market_return,
            "cagr": strat_cagr,
            "ann_vol": strat_vol,
            "sharpe": strat_sharpe,
            "max_drawdown": strat_mdd,
            "turnover": strat_turnover,
            "hit_rate": strat_hit_rate,
        },
        "output_csv": str(outpath_csv),
    }
    outpath_manifest = out_dir / f"{ticker}_backtest.manifest.json"
    outpath_manifest.write_text(json.dumps(manifest, indent=2))

    typer.echo(f"💾 Backtest results saved to {outpath_csv}")
    typer.echo(f"🧾 Manifest saved to {outpath_manifest}")


if __name__ == "__main__":
    app(prog_name="backtest_cli.py")
