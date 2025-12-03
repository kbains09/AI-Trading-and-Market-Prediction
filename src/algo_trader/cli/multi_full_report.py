# src/algo_trader/cli/multi_full_report.py

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    tickers: List[str] = typer.Argument(
        ...,
        help="List of ticker symbols, e.g. AAPL TSLA MSFT BTC-USD",
    ),
    cost_bps: float = typer.Option(
        10.0,
        "--cost-bps",
        help="Round-trip transaction cost (bps) per unit turnover for backtest.",
    ),
    vol_target: float = typer.Option(
        0.15,
        "--vol-target",
        help="Annualized volatility target (e.g. 0.15 for 15%%). 0 disables targeting.",
    ),
    roll_window: int = typer.Option(
        20,
        "--roll-window",
        help="Rolling window (days) for realized volatility in targeting.",
    ),
    max_leverage: float = typer.Option(
        3.0,
        "--max-leverage",
        help="Leverage cap when applying vol targeting.",
    ),
    rf_annual: float = typer.Option(
        0.0,
        "--rf-annual",
        help="Annual risk-free rate for Sharpe calculation.",
    ),
    trading_days: int = typer.Option(
        252,
        "--trading-days",
        help="Trading days per year (252 for daily).",
    ),
    up_threshold: float = typer.Option(
        0.55,
        "--up-threshold",
        help="Min P(UP) to go long in TREND regime.",
    ),
    down_threshold: float = typer.Option(
        0.45,
        "--down-threshold",
        help="Max P(UP) to go short in MEAN-REVERSION regime.",
    ),
    sharpe_window: int = typer.Option(
        252,
        "--sharpe-window",
        help="Rolling window for Sharpe plot.",
    ),
) -> None:
    """
    Run the full_report pipeline for multiple tickers in sequence.

    This is just an orchestrator:
      - For each ticker, it calls:
        `python -m algo_trader.cli.full_report --ticker TICKER ...`
    """

    root = Path(__file__).resolve().parents[3]

    tickers = [t.upper() for t in tickers]

    typer.echo("🚀 Multi-ticker full report pipeline")
    typer.echo(f"🧾 Tickers: {', '.join(tickers)}")
    typer.echo("")

    for idx, ticker in enumerate(tickers, start=1):
        typer.echo(f"========== [{idx}/{len(tickers)}] {ticker} ==========")

        cmd = [
            sys.executable,
            "-m",
            "algo_trader.cli.full_report",
            "--ticker",
            ticker,
            "--cost-bps",
            str(cost_bps),
            "--vol-target",
            str(vol_target),
            "--roll-window",
            str(roll_window),
            "--max-leverage",
            str(max_leverage),
            "--rf-annual",
            str(rf_annual),
            "--trading-days",
            str(trading_days),
            "--up-threshold",
            str(up_threshold),
            "--down-threshold",
            str(down_threshold),
            "--sharpe-window",
            str(sharpe_window),
        ]

        # Run under the repo root so paths resolve exactly like when you run full_report directly
        result = subprocess.run(cmd, cwd=root)

        if result.returncode != 0:
            raise SystemExit(
                f"❌ full_report failed for {ticker} with exit code {result.returncode}"
            )

    typer.echo("")
    typer.echo("✅ Multi-ticker pipeline completed.")


if __name__ == "__main__":
    app()
