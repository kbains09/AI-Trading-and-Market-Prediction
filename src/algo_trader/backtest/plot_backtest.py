from __future__ import annotations

import pathlib as pl
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import yaml

app = typer.Typer(add_completion=False)

# Paths / Config
PROJECT_CONFIG_PATH = pl.Path("config/project.yaml")

if PROJECT_CONFIG_PATH.exists():
    PROJECT_CONFIG = yaml.safe_load(PROJECT_CONFIG_PATH.read_text())
    DATA_BACKTESTS = pl.Path(
        PROJECT_CONFIG.get("data", {}).get("backtests_dir", "data/backtests")
    )
else:
    DATA_BACKTESTS = pl.Path("data/backtests")


PLOTS_DIR = DATA_BACKTESTS / "plots"

# Helpers
def _ensure_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame is indexed by Date (if present).
    """
    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date")
        df = df.set_index("Date")
    return df


def _max_drawdown_curve(equity: pd.Series) -> pd.Series:
    """
    Compute drawdown series: equity / equity.cummax() - 1
    """
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return dd


def _rolling_sharpe(
    ret: pd.Series,
    window: int = 63,
    trading_days: int = 252,
    rf_annual: float = 0.0,
) -> pd.Series:
    """
    Rolling Sharpe ratio over 'window' periods.
    """
    rf_daily = rf_annual / trading_days

    def _sharpe_window(x: pd.Series) -> float:
        if x.std(ddof=0) == 0 or np.isnan(x.std(ddof=0)):
            return np.nan
        return (x.mean() - rf_daily) / x.std(ddof=0) * np.sqrt(trading_days)

    return ret.rolling(window).apply(_sharpe_window, raw=False)

# Core plotting logic
def _plot_equity_curves(
    df: pd.DataFrame,
    ticker: str,
    outdir: pl.Path,
    show: bool = False,
) -> pl.Path:
    """
    Plot strategy vs market equity curves.
    """
    eq_strat = df["cumulative_strategy_return"]
    eq_mkt = df["cumulative_market_return"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq_strat.index, eq_strat.values, label="Strategy", linewidth=1.5)
    ax.plot(eq_mkt.index, eq_mkt.values, label="Market (buy & hold)", linestyle="--", linewidth=1.0)

    ax.set_title(f"Equity Curve — {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth (x)")
    ax.legend()
    ax.grid(alpha=0.3)

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{ticker}_equity.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return outpath


def _plot_underwater(
    df: pd.DataFrame,
    ticker: str,
    outdir: pl.Path,
    show: bool = False,
) -> pl.Path:
    """
    Plot underwater (drawdown) curve for strategy.
    """
    eq_strat = df["cumulative_strategy_return"]
    dd = _max_drawdown_curve(eq_strat)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(dd.index, dd.values, 0, step="pre", alpha=0.5)
    ax.set_title(f"Underwater Curve — {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{ticker}_underwater.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return outpath


def _plot_rolling_sharpe(
    df: pd.DataFrame,
    ticker: str,
    outdir: pl.Path,
    window: int = 63,
    trading_days: int = 252,
    rf_annual: float = 0.0,
    show: bool = False,
) -> pl.Path:
    """
    Plot rolling Sharpe ratio of strategy returns.
    """
    ret = df["strategy_return"]
    rs = _rolling_sharpe(
        ret,
        window=window,
        trading_days=trading_days,
        rf_annual=rf_annual,
    )

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(rs.index, rs.values, linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Rolling Sharpe ({window}-day) — {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe")
    ax.grid(alpha=0.3)

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{ticker}_rolling_sharpe_{window}d.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return outpath

# CLI command
@app.command("run")
def plot_backtest(
    ticker: str = typer.Option("AAPL", help="Ticker to plot"),
    backtest_dir: str = typer.Option(str(DATA_BACKTESTS), help="Directory for backtest CSVs"),
    window: int = typer.Option(63, help="Window size (days) for rolling Sharpe"),
    trading_days: int = typer.Option(252, help="Trading days per year"),
    rf_annual: float = typer.Option(0.0, help="Annual risk-free rate for Sharpe"),
    show: bool = typer.Option(False, "--show", help="Display plots interactively as well as save"),
):
    """
    Generate plots for a completed backtest:
      - Equity curve (strategy vs market)
      - Underwater (drawdown)
      - Rolling Sharpe (windowed)
    """
    backtest_dir_path = pl.Path(backtest_dir)
    csv_path = backtest_dir_path / f"{ticker}_backtest.csv"

    if not csv_path.exists():
        typer.echo(f"❌ Backtest CSV not found: {csv_path}")
        raise typer.Exit(1)

    df = pd.read_csv(csv_path)
    df = _ensure_date_index(df)

    if "cumulative_strategy_return" not in df.columns or "cumulative_market_return" not in df.columns:
        typer.echo("❌ Backtest CSV is missing required equity columns.")
        raise typer.Exit(1)
    if "strategy_return" not in df.columns:
        typer.echo("❌ Backtest CSV is missing 'strategy_return' column.")
        raise typer.Exit(1)

    outdir = PLOTS_DIR
    eq_path = _plot_equity_curves(df, ticker, outdir, show=show)
    uw_path = _plot_underwater(df, ticker, outdir, show=show)
    rs_path = _plot_rolling_sharpe(
        df,
        ticker,
        outdir,
        window=window,
        trading_days=trading_days,
        rf_annual=rf_annual,
        show=show,
    )

    typer.echo(f"🖼 Saved equity curve → {eq_path}")
    typer.echo(f"🖼 Saved underwater plot → {uw_path}")
    typer.echo(f"🖼 Saved rolling Sharpe plot → {rs_path}")


if __name__ == "__main__":
    app()
