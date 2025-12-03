from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import typer

from ..backtest import backtest_cli as backtest_mod
from ..features.feature_set import LIVE_FEATURE_COLUMNS
from ..plots.base import load_backtest_csv
from ..plots.drawdown import plot_drawdown
from ..plots.equity_curve import plot_equity_curve
from ..plots.feature_importance import plot_feature_importance
from ..plots.rolling_sharpe import plot_rolling_sharpe
from ..training import train_direction_model as dir_mod
from ..training import train_regime_model as reg_mod

app = typer.Typer(add_completion=False)


# ---------------------------
# Defaults for this pipeline
# ---------------------------

DEFAULT_COST_BPS = 10.0
DEFAULT_VOL_TARGET = 0.15
DEFAULT_ROLL_WINDOW = 20
DEFAULT_MAX_LEVERAGE = 3.0
DEFAULT_RF_ANNUAL = 0.0
DEFAULT_TRADING_DAYS = 252
DEFAULT_UP_THRESHOLD = 0.55
DEFAULT_DOWN_THRESHOLD = 0.45
DEFAULT_SHARPE_WINDOW = 252


def _project_paths() -> Tuple[Path, Path, Path, Path]:
    """
    Resolve project-root-relative paths so this works no matter
    where you run the command from.

    Layout assumed:
      repo_root/
        src/algo_trader/cli/full_report.py   <-- this file
        models/xgboost/
        data/backtests/
        plots/
    """
    # __file__ = .../src/algo_trader/cli/full_report.py
    # parents[0] = cli
    # parents[1] = algo_trader
    # parents[2] = src
    # parents[3] = repo root
    root = Path(__file__).resolve().parents[3]

    models_dir = root / "models" / "xgboost"
    backtest_dir = root / "data" / "backtests"
    plots_dir = root / "plots"

    return root, models_dir, backtest_dir, plots_dir


@app.command()
def main(
    ticker: str = typer.Option(
        ...,
        "--ticker",
        "-t",
        help="Ticker symbol, e.g. AAPL",
    ),
    cost_bps: float = typer.Option(
        DEFAULT_COST_BPS,
        "--cost-bps",
        help="Round-trip transaction cost (bps) per unit turnover for backtest.",
    ),
    vol_target: float = typer.Option(
        DEFAULT_VOL_TARGET,
        "--vol-target",
        help="Annualized volatility target (e.g. 0.15 for 15%%). 0 disables targeting.",
    ),
    roll_window: int = typer.Option(
        DEFAULT_ROLL_WINDOW,
        "--roll-window",
        help="Rolling window (days) for realized volatility in targeting.",
    ),
    max_leverage: float = typer.Option(
        DEFAULT_MAX_LEVERAGE,
        "--max-leverage",
        help="Leverage cap when applying vol targeting.",
    ),
    rf_annual: float = typer.Option(
        DEFAULT_RF_ANNUAL,
        "--rf-annual",
        help="Annual risk-free rate for Sharpe calculation.",
    ),
    trading_days: int = typer.Option(
        DEFAULT_TRADING_DAYS,
        "--trading-days",
        help="Trading days per year (252 for daily).",
    ),
    up_threshold: float = typer.Option(
        DEFAULT_UP_THRESHOLD,
        "--up-threshold",
        help="Min P(UP) to go long in TREND regime.",
    ),
    down_threshold: float = typer.Option(
        DEFAULT_DOWN_THRESHOLD,
        "--down-threshold",
        help="Max P(UP) to go short in MEAN-REVERSION regime.",
    ),
    sharpe_window: int = typer.Option(
        DEFAULT_SHARPE_WINDOW,
        "--sharpe-window",
        help="Rolling window for Sharpe plot.",
    ),
) -> None:
    """
    End-to-end pipeline for a single ticker:

      1. Train direction model
      2. Train regime model
      3. Run regime+direction backtest
      4. Generate plots (equity, drawdown, rolling Sharpe)
      5. Generate feature-importance plots

    All outputs are written under:
      - models/xgboost/
      - data/backtests/
      - plots/{TICKER}/
    """
    ticker = ticker.upper()
    root, models_dir, backtest_dir, plots_root = _project_paths()

    labeled_dir = root / "data" / "labeled"
    plots_dir = plots_root / ticker
    plots_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"🚀 Full report pipeline for {ticker}")
    typer.echo(f"📁 Project root: {root}")
    typer.echo(f"📁 Labeled data dir: {labeled_dir}")
    typer.echo(f"📁 Models dir:       {models_dir}")
    typer.echo(f"📁 Backtest dir:     {backtest_dir}")
    typer.echo(f"📁 Plots dir:        {plots_dir}")
    typer.echo("")

    # -------------------------
    # 1) Train direction model
    # -------------------------
    typer.echo(f"🧠 [1/5] Training DIRECTION model for {ticker}...")
    dir_mod.main(
        ticker=ticker,
        labeled_dir=str(labeled_dir),
        model_dir=str(models_dir),
    )
    typer.echo("")

    # -------------------------
    # 2) Train regime model
    # -------------------------
    typer.echo(f"🧠 [2/5] Training REGIME model for {ticker}...")
    reg_mod.main(
        ticker=ticker,
        labeled_dir=str(labeled_dir),
        model_dir=str(models_dir),
    )
    typer.echo("")

    # -------------------------
    # 3) Run backtest
    # -------------------------
    typer.echo(f"📉 [3/5] Running backtest for {ticker}...")

    # IMPORTANT: pass plain Python floats into backtest_mod.main so we don't
    # accidentally use Typer's OptionInfo defaults there.
    backtest_mod.main(
        ticker=ticker,
        model_dir=str(models_dir),
        feature_dir=str(labeled_dir),
        cost_bps=cost_bps,
        vol_target=vol_target,
        roll_window=roll_window,
        max_leverage=max_leverage,
        rf_annual=rf_annual,
        trading_days=trading_days,
        up_threshold=up_threshold,
        down_threshold=down_threshold,
    )
    typer.echo("")

    # Path to backtest CSV saved by backtest_cli
    backtest_csv = backtest_dir / f"{ticker}_backtest.csv"
    if not backtest_csv.exists():
        raise SystemExit(f"❌ Backtest CSV not found: {backtest_csv}")

    # -------------------------
    # 4) Plots: equity / DD / rolling Sharpe
    # -------------------------
    typer.echo("📊 [4/5] Generating equity / drawdown / rolling Sharpe plots...")

    results = load_backtest_csv(
        backtest_csv,
        date_col="Date",
        strategy_col="cumulative_strategy_return",
        benchmark_col="cumulative_market_return",
        signal_col="position",
    )
    # Override inferred ticker name to be safe
    results.ticker = ticker

    ec_path = plot_equity_curve(results, plots_dir)
    dd_path = plot_drawdown(results, plots_dir)
    rs_path = plot_rolling_sharpe(results, plots_dir, window=sharpe_window)

    typer.echo(f"   ✅ Equity curve:   {ec_path}")
    typer.echo(f"   ✅ Drawdown:       {dd_path}")
    typer.echo(f"   ✅ Rolling Sharpe: {rs_path}")
    typer.echo("")

    # -------------------------
    # 5) Feature importance plots
    # -------------------------
    typer.echo("🧬 [5/5] Generating feature-importance plots...")

    dir_model_path = models_dir / f"{ticker}_direction_model.pkl"
    reg_model_path = models_dir / f"{ticker}_regime_model.pkl"

    if dir_model_path.exists():
        dir_imp = plot_feature_importance(
            dir_model_path,
            out_dir=plots_dir,
            title=f"{ticker} Direction Model – Feature Importance",
            feature_names=LIVE_FEATURE_COLUMNS,
        )
        typer.echo(f"   ✅ Direction feature importance: {dir_imp}")
    else:
        typer.echo(f"   ⚠️ Direction model not found: {dir_model_path}")

    if reg_model_path.exists():
        reg_imp = plot_feature_importance(
            reg_model_path,
            out_dir=plots_dir,
            title=f"{ticker} Regime Model – Feature Importance",
            feature_names=LIVE_FEATURE_COLUMNS,
        )
        typer.echo(f"   ✅ Regime feature importance:    {reg_imp}")
    else:
        typer.echo(f"   ⚠️ Regime model not found: {reg_model_path}")

    typer.echo("")
    typer.echo("✅ Full report completed.")
    typer.echo(f"   Ticker:      {ticker}")
    typer.echo(
        f"   Timestamp:   "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    typer.echo(f"   Backtest CSV: {backtest_csv}")
    typer.echo(f"   Plots dir:    {plots_dir}")


if __name__ == "__main__":
    app()
