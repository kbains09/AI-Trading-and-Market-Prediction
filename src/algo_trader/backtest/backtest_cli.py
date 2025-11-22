import json
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import pathlib as pl
import typer
import joblib
import yaml

app = typer.Typer(add_completion=False)

# Config & paths
PROJECT_CONFIG = yaml.safe_load(pl.Path("config/project.yaml").read_text())
DATA_LABELED = pl.Path(PROJECT_CONFIG["data"]["labeled_dir"])
MODELS_DIR = pl.Path(PROJECT_CONFIG["data"].get("models_dir", "models/xgboost"))

# Metrics & Helpers
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

# Core PnL Simulation
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
    - df has columns:
        'next_return_1d' (forward 1-day return on this row),
        'prediction' (position ∈ {-1, 0, 1} or scaled float).
    - No shifting inside; labeling already aligned.
    """
    out = df.copy()

    # 1) Base position from model prediction
    pos = pd.to_numeric(out["prediction"], errors="coerce").fillna(0.0).astype(float)

    # 2) Volatility targeting (optional)
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

    # 4) Transaction costs on position changes
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

# Strategy: use regime + direction
def _build_positions_from_models(
    df: pd.DataFrame,
    regime_artifact: dict,
    direction_artifact: dict,
    up_thresh: float = 0.55,
    down_thresh: float = 0.45,
) -> pd.DataFrame:
    """
    For each row:
      - predict regime (trend / neutral / mean-reversion)
      - predict 5D direction (UP/DOWN probabilities)
      - derive trading position:
          * if regime == trend and p_up > up_thresh:  +1
          * if regime == mean-reversion and p_up < down_thresh: -1
          * else: 0
    """
    model_reg = regime_artifact["model"]
    reg_feats = regime_artifact["feature_cols"]
    regime_map = regime_artifact["regime_map"]
    inv_regime_map = {v: k for k, v in regime_map.items()}

    model_dir = direction_artifact["model"]
    dir_feats = direction_artifact["feature_cols"]

    needed_cols = set(reg_feats) | set(dir_feats) | {"next_return_1d"}
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in labeled data: {missing[:8]}")

    # Drop rows with NaNs in any used feature or next_return_1d
    clean = df.dropna(subset=list(needed_cols)).copy().reset_index(drop=True)

    X_reg = clean[reg_feats]
    X_dir = clean[dir_feats]

    # Regime predictions
    reg_probs = model_reg.predict_proba(X_reg)  # shape (n, 3)
    reg_class = reg_probs.argmax(axis=1)        # 0/1/2
    reg_label_int = np.array([inv_regime_map.get(int(c), 0) for c in reg_class])

    # Map to strings for inspection
    reg_label_str = np.where(
        reg_label_int == 1, "trend",
        np.where(reg_label_int == -1, "mean_reversion", "neutral")
    )
    reg_conf = reg_probs.max(axis=1)

    # Direction predictions
    dir_probs = model_dir.predict_proba(X_dir)  # shape (n, 2) -> [DOWN, UP]
    prob_down = dir_probs[:, 0]
    prob_up = dir_probs[:, 1]
    dir_label_bin = (prob_up >= 0.5).astype(int)  # 1 = UP, 0 = DOWN
    dir_label_str = np.where(dir_label_bin == 1, "UP", "DOWN")

    # Trading rule
    position = np.zeros(len(clean), dtype=float)
    is_trend = reg_label_int == 1
    is_meanrev = reg_label_int == -1

    position[(is_trend) & (prob_up > up_thresh)] = 1.0
    position[(is_meanrev) & (prob_up < down_thresh)] = -1.0

    # Attach to frame
    clean["regime_label_int"] = reg_label_int
    clean["regime_label_str"] = reg_label_str
    clean["regime_conf"] = reg_conf

    clean["direction_label"] = dir_label_str
    clean["direction_prob_up"] = prob_up
    clean["direction_prob_down"] = prob_down

    clean["prediction"] = position

    return clean

# CLI Command
@app.command("run")
def run_backtest(
    ticker: str = typer.Option("AAPL", help="Ticker to backtest"),
    model_dir: str = typer.Option(str(MODELS_DIR), help="Directory with trained models"),
    feature_dir: str = typer.Option(str(DATA_LABELED), help="Directory with labeled feature files"),
    cost_bps: float = typer.Option(0.0, help="Round-trip transaction cost in basis points per unit turnover"),
    vol_target: float = typer.Option(0.0, help="Annualized volatility target (e.g., 0.10 for 10%%). 0 disables"),
    roll_window: int = typer.Option(20, help="Rolling window for vol targeting (days)"),
    max_leverage: float = typer.Option(3.0, help="Max leverage cap when vol targeting"),
    rf_annual: float = typer.Option(0.0, help="Annual risk-free rate for Sharpe calc"),
    trading_days: int = typer.Option(252, help="Periods per year (252 for daily)"),
):
    """
    Backtest a regime + direction strategy for a specific ticker using:
      - regime_trend_meanrev model (multi-class, trend/neutral/meanrev)
      - direction_5 model (binary UP/DOWN)

    Uses labeled CSV:
      data/labeled/{ticker}_labeled.csv
    which must contain:
      - Date
      - next_return_1d
      - direction_5 (optional, for accuracy metrics)
      - regime_trend_meanrev (optional, for accuracy metrics)
      - feature columns used by the models
    """
    model_dir_path = pl.Path(model_dir)
    feature_dir_path = pl.Path(feature_dir)

    regime_path = model_dir_path / f"{ticker}_regime_model.pkl"
    direction_path = model_dir_path / f"{ticker}_direction_model.pkl"
    feature_path = feature_dir_path / f"{ticker}_labeled.csv"

    if not regime_path.exists():
        typer.echo(f"❌ Missing regime model for {ticker}: {regime_path}")
        raise typer.Exit(1)
    if not direction_path.exists():
        typer.echo(f"❌ Missing direction model for {ticker}: {direction_path}")
        raise typer.Exit(1)
    if not feature_path.exists():
        typer.echo(f"❌ Missing labeled data for {ticker}: {feature_path}")
        raise typer.Exit(1)

    # Load models & data
    regime_artifact = joblib.load(regime_path)
    direction_artifact = joblib.load(direction_path)
    df = pd.read_csv(feature_path, parse_dates=["Date"])

    # Build positions from models
    try:
        df_signals = _build_positions_from_models(df, regime_artifact, direction_artifact)
    except Exception as e:
        typer.echo(f"❌ Failed to build positions: {e}")
        raise typer.Exit(1)

    # Optional: basic label accuracies
    if "direction_5" in df_signals.columns:
        # Convert direction_5 (-1/+1) → 0/1
        y_dir = (df_signals["direction_5"] == 1).astype(int)
        y_hat_dir = (df_signals["direction_prob_up"] >= 0.5).astype(int)
        dir_acc = float((y_dir == y_hat_dir).mean())
        typer.echo(f"✅ Direction (5D) accuracy for {ticker}: {dir_acc:.4f}")
    else:
        dir_acc = float("nan")

    if "regime_trend_meanrev" in df_signals.columns:
        y_reg = df_signals["regime_trend_meanrev"].astype(int)
        y_hat_reg = df_signals["regime_label_int"].astype(int)
        regime_acc = float((y_reg == y_hat_reg).mean())
        typer.echo(f"✅ Regime accuracy for {ticker}: {regime_acc:.4f}")
    else:
        regime_acc = float("nan")

    # Run PnL simulation
    results = simulate_trades(
        df_signals,
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
    trade_pnl = (results["position"] * results["next_return_1d"]).where(
        results["position"] != 0.0, 0.0
    )
    strat_hit_rate = _hit_rate(trade_pnl)

    # Console summary
    typer.echo(f"📈 Strategy Return: {final_strat_return:.4f}x")
    typer.echo(f"📉 Market Return:   {final_market_return:.4f}x")
    typer.echo(
        f"📊 CAGR: {strat_cagr:.4%} | Vol: {strat_vol:.4%} | "
        f"Sharpe: {strat_sharpe:.2f} | MaxDD: {strat_mdd:.2%}"
    )
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
        "regime_model_path": str(regime_path),
        "regime_model_sha256": _sha256(regime_path),
        "direction_model_path": str(direction_path),
        "direction_model_sha256": _sha256(direction_path),
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
            "direction_accuracy": dir_acc,
            "regime_accuracy": regime_acc,
        },
        "output_csv": str(outpath_csv),
    }
    outpath_manifest = out_dir / f"{ticker}_backtest.manifest.json"
    outpath_manifest.write_text(json.dumps(manifest, indent=2))

    typer.echo(f"💾 Backtest results saved to {outpath_csv}")
    typer.echo(f"🧾 Manifest saved to {outpath_manifest}")


if __name__ == "__main__":
    app(prog_name="backtest_cli.py")
