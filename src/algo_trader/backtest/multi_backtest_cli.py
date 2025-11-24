from __future__ import annotations

import json
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import pathlib as pl
import typer

from .backtest_cli import (
    _cagr,
    _ann_vol,
    _sharpe,
    _max_drawdown,
    _turnover,
    _hit_rate,
)

app = typer.Typer(add_completion=False)


def _load_single_backtest(path: pl.Path, ticker: str) -> pd.DataFrame:
    """
    Load a single-ticker backtest CSV and rename key columns so we can join
    multiple assets without collisions.
    Expects columns:
      - Date
      - strategy_return
      - next_return_1d
    plus anything else (ignored for portfolio calc).
    """
    if not path.exists():
        raise FileNotFoundError(f"Backtest CSV not found for {ticker}: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])

    # rename per-ticker return columns
    df = df.rename(
        columns={
            "strategy_return": f"{ticker}_strategy_return",
            "next_return_1d": f"{ticker}_next_return_1d",
        }
    )
    return df[["Date", f"{ticker}_strategy_return", f"{ticker}_next_return_1d"]]


@app.command()
def main(
    tickers: List[str] = typer.Argument(..., help="List of tickers to combine, e.g. AAPL TSLA MSFT BTC-USD"),
    backtest_dir: str = typer.Option("data/backtests", help="Directory containing {TICKER}_backtest.csv files"),
    trading_days: int = typer.Option(252, help="Periods per year (252 for daily data)"),
    portfolio_name: str = typer.Option("EQW_PORT", help="Name tag for the portfolio"),
):
    """
    Combine multiple single-ticker backtests into an equal-weight portfolio.

    Assumes you've already run:
      backtest_cli --ticker <TICKER> ...
    for each ticker, so that:
      data/backtests/{TICKER}_backtest.csv
    exists and contains:
      - Date
      - strategy_return
      - next_return_1d
    """
    backtest_dir_path = pl.Path(backtest_dir)
    if not tickers:
        typer.echo("❌ No tickers provided.")
        raise typer.Exit(1)

    # 1) Load and merge all assets on Date
    merged: pd.DataFrame | None = None
    for t in tickers:
        path = backtest_dir_path / f"{t}_backtest.csv"
        df_t = _load_single_backtest(path, t)

        if merged is None:
            merged = df_t
        else:
            merged = pd.merge(
                merged,
                df_t,
                on="Date",
                how="outer",
                sort=True,
            )

    if merged is None or merged.empty:
        typer.echo("❌ No data loaded for given tickers.")
        raise typer.Exit(1)

    merged = merged.sort_values("Date").reset_index(drop=True)

    # 2) Build portfolio returns
    strat_cols = [f"{t}_strategy_return" for t in tickers]
    mkt_cols = [f"{t}_next_return_1d" for t in tickers]

    # If some days are missing for a ticker, treat missing returns as 0 (no position).
    strat_ret = merged[strat_cols].fillna(0.0).mean(axis=1)
    mkt_ret = merged[mkt_cols].fillna(0.0).mean(axis=1)

    merged["portfolio_strategy_return"] = strat_ret
    merged["portfolio_market_return"] = mkt_ret

    # 3) Equity curves
    merged["portfolio_cum_strat"] = (1.0 + merged["portfolio_strategy_return"]).cumprod()
    merged["portfolio_cum_mkt"] = (1.0 + merged["portfolio_market_return"]).cumprod()

    # 4) Metrics on portfolio
    eq_strat = merged["portfolio_cum_strat"]
    final_strat_multiple = float(eq_strat.iloc[-1]) if len(eq_strat) else float("nan")

    eq_mkt = merged["portfolio_cum_mkt"]
    final_mkt_multiple = float(eq_mkt.iloc[-1]) if len(eq_mkt) else float("nan")

    cagr = _cagr(eq_strat, trading_days=trading_days)
    vol = _ann_vol(merged["portfolio_strategy_return"], trading_days=trading_days)
    sharpe = _sharpe(merged["portfolio_strategy_return"], trading_days=trading_days)
    mdd = _max_drawdown(eq_strat)

    # For turnover: sum of absolute daily changes in equal-weight allocation
    # Approximation: average the per-asset position turnover.
    # We don't have per-asset positions here, so we skip or load them in a future version.
    # For now, we'll leave turnover/hit_rate as NaN for the portfolio.
    portfolio_turnover = float("nan")
    portfolio_hit_rate = float("nan")

    # 5) Console summary
    typer.echo(f"📦 Portfolio: {portfolio_name}")
    typer.echo(f"🧾 Tickers: {', '.join(tickers)}")
    typer.echo(f"📈 Strategy Return (portfolio): {final_strat_multiple:.4f}x")
    typer.echo(f"📉 Market Return   (avg):      {final_mkt_multiple:.4f}x")
    typer.echo(
        f"📊 CAGR: {cagr:.4%} | Vol: {vol:.4%} | Sharpe: {sharpe:.2f} | MaxDD: {mdd:.2%}"
    )

    # 6) Save portfolio backtest
    out_dir = backtest_dir_path
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers_tag = "_".join(t.replace("-", "_") for t in tickers)
    out_csv = out_dir / f"PORT_{portfolio_name}_{tickers_tag}_backtest.csv"
    merged.to_csv(out_csv, index=False)
    typer.echo(f"💾 Portfolio backtest saved to {out_csv}")

    # 7) Save manifest
    manifest = {
        "portfolio_name": portfolio_name,
        "tickers": tickers,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "metrics": {
            "final_strategy_multiple": final_strat_multiple,
            "final_market_multiple": final_mkt_multiple,
            "cagr": cagr,
            "ann_vol": vol,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "turnover": portfolio_turnover,
            "hit_rate": portfolio_hit_rate,
        },
        "backtest_csv": str(out_csv),
    }

    out_manifest = out_dir / f"PORT_{portfolio_name}_{tickers_tag}_backtest.manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2))
    typer.echo(f"🧾 Portfolio manifest saved to {out_manifest}")


if __name__ == "__main__":
    app()
