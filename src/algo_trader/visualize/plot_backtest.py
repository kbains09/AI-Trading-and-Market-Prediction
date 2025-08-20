import json
import math
import pathlib as pl

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(add_completion=False)


def _load_manifest(manifest_path: pl.Path) -> dict:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text())
        except Exception:
            return {}
    return {}


def _rolling_sharpe(ret: pd.Series, window: int = 63, trading_days: int = 252) -> pd.Series:
    mu = ret.rolling(window).mean()
    sd = ret.rolling(window).std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = (mu / sd) * math.sqrt(trading_days)
    return rs.replace([np.inf, -np.inf], np.nan)


def _ensure_two_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["cumulative_market_return", "cumulative_strategy_return"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Backtest CSV is missing required columns: {missing}")
    return df[cols].dropna(how="any").copy()


@app.command()
def plot_backtest(
    ticker: str = typer.Option(..., help="Ticker symbol, e.g., AAPL"),
    backtest_dir: str = typer.Option("data/backtests", help="Directory with backtest results"),
    logy: bool = typer.Option(False, help="Use log scale for equity plot"),
    window: int = typer.Option(63, help="Rolling window (days) for Sharpe (≈ 3 months)"),
    show: bool = typer.Option(True, help="Show plots interactively"),
    save_prefix: str = typer.Option(
        "", help="If provided, save PNGs using this prefix (e.g., reports/plots/AAPL)"
    ),
):
    """
    Plot strategy vs market cumulative returns, Underwater (drawdown), and Rolling Sharpe.

    Usage:
      poetry run python -m src.algo_trader.visualize.plot_backtest plot-backtest --ticker AAPL --logy --save-prefix reports/plots/AAPL
      poetry run python -m src.algo_trader.visualize.plot_backtest plot-backtest --ticker AAPL --no-show --save-prefix reports/plots/AAPL
    """
    backtest_path = pl.Path(backtest_dir) / f"{ticker}_backtest.csv"
    if not backtest_path.exists():
        typer.echo(f"❌ Backtest file not found: {backtest_path}")
        raise typer.Exit(code=1)

    # Optional manifest
    manifest_path = pl.Path(backtest_dir) / f"{ticker}_backtest.manifest.json"
    manifest = _load_manifest(manifest_path)
    m = manifest.get("metrics", {})
    sharpe_txt = f" | Sharpe {m.get('sharpe'):.2f}" if isinstance(m.get("sharpe"), (int, float)) else ""
    cagr_txt = f" | CAGR {m.get('cagr'):.2%}" if isinstance(m.get("cagr"), (int, float)) else ""

    # Load CSV
    df = pd.read_csv(backtest_path)
    if "Date" not in df.columns:
        typer.echo("❌ Backtest CSV must include a 'Date' column.")
        raise typer.Exit(code=1)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    # Filter and clean
    core = _ensure_two_cols(df)

    # ---- Plot 1: Equity curves ----
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

    # ---- Plot 2: Underwater (drawdown) ----
    eq = core["cumulative_strategy_return"].clip(lower=1e-12)
    dd = eq / eq.cummax() - 1.0
    fig2 = plt.figure(figsize=(12, 2.8))
    plt.plot(dd, label="Drawdown")
    plt.title(f"{ticker} - Underwater (Drawdown)")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()

    # ---- Plot 3: Rolling Sharpe ----
    if "strategy_return" in df.columns:
        ret = pd.to_numeric(df["strategy_return"], errors="coerce").fillna(0.0)
    else:
        ret = core["cumulative_strategy_return"].pct_change().fillna(0.0)

    rs = _rolling_sharpe(ret, window=window)
    fig3 = plt.figure(figsize=(12, 2.8))
    plt.plot(rs)
    plt.title(f"{ticker} - Rolling Sharpe ({window}d)")
    plt.grid(True)
    plt.tight_layout()

    # ---- Save if needed ----
    if save_prefix:
        base = pl.Path(save_prefix)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(base.with_suffix("").as_posix() + "_equity.png", dpi=180, bbox_inches="tight")
        fig2.savefig(base.with_suffix("").as_posix() + "_drawdown.png", dpi=180, bbox_inches="tight")
        fig3.savefig(base.with_suffix("").as_posix() + "_rollsharpe.png", dpi=180, bbox_inches="tight")
        typer.echo(f"💾 Saved plots to prefix: {base}")

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)


if __name__ == "__main__":
    app()
