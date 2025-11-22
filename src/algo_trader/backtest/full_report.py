from __future__ import annotations

import json
import pathlib as pl
from datetime import datetime

import typer
import yaml

from . import backtest_cli
from . import plot_backtest

app = typer.Typer(add_completion=False)

# Shared config for paths
PROJECT_CONFIG_PATH = pl.Path("config/project.yaml")

if PROJECT_CONFIG_PATH.exists():
    PROJECT_CONFIG = yaml.safe_load(PROJECT_CONFIG_PATH.read_text())
    DATA_BACKTESTS = pl.Path(
        PROJECT_CONFIG.get("data", {}).get("backtests_dir", "data/backtests")
    )
else:
    DATA_BACKTESTS = pl.Path("data/backtests")

PLOTS_DIR = DATA_BACKTESTS / "plots"

# CLI command
@app.command("run")
def full_report(
    ticker: str = typer.Option("AAPL", help="Ticker to run full report for"),
):
    """
    Run a full analytical report for a single ticker:

    1) Run backtest (saves CSV + manifest)
    2) Generate plots (equity, underwater, rolling Sharpe)
    3) Print out artifact paths
    """
    typer.echo(f"[full_report] 🚀 Starting full report for {ticker}")

    # 1) Run backtest (explicit args so we avoid Typer's OptionInfo defaults)
    typer.echo(f"[full_report] ▶️ Running backtest for {ticker} ...")
    backtest_cli.run_backtest(
        model_dir="models/xgboost",
        feature_dir="data/labeled",
        ticker=ticker,
        cost_bps=0.0,
        vol_target=0.0,
        roll_window=20,
        max_leverage=3.0,
        rf_annual=0.0,
        trading_days=252,
        up_threshold=0.55,
        down_threshold=0.45,
    )

    # 2) Generate plots from the generated backtest CSV
    typer.echo(f"[full_report] 🖼 Generating plots for {ticker} ...")
    plot_backtest.plot_backtest(
        ticker=ticker,
        # let plot_backtest handle defaults, but we pass explicit dir so paths align
        backtest_dir=str(DATA_BACKTESTS),
        window=63,
        trading_days=252,
        rf_annual=0.0,
        show=False,
    )

    # 3) Resolve paths to artifacts
    manifest_path = DATA_BACKTESTS / f"{ticker}_backtest.manifest.json"
    csv_path = DATA_BACKTESTS / f"{ticker}_backtest.csv"

    # If manifest exists, try to read output CSV path from it (more robust)
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            csv_from_manifest = manifest.get("output_csv")
            if csv_from_manifest:
                csv_path = pl.Path(csv_from_manifest)
        except Exception:
            # fallback to default csv_path
            pass

    equity_png = PLOTS_DIR / f"{ticker}_equity.png"
    underwater_png = PLOTS_DIR / f"{ticker}_underwater.png"
    rolling_sharpe_png = PLOTS_DIR / f"{ticker}_rolling_sharpe_63d.png"

    typer.echo("")
    typer.echo(f"[full_report] ✅ Done for {ticker}")
    typer.echo("Artifacts:")
    typer.echo(f"  🧾 Manifest: {manifest_path}")
    typer.echo(f"  📄 Backtest CSV: {csv_path}")
    typer.echo(f"  🖼 Equity curve: {equity_png}")
    typer.echo(f"  🖼 Underwater:   {underwater_png}")
    typer.echo(f"  🖼 Rolling Sharpe: {rolling_sharpe_png}")
    typer.echo("")


if __name__ == "__main__":
    app()
