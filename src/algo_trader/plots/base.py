from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

@dataclass
class BacktestResults:
    df: pd.DataFrame
    ticker: Optional[str] = None
    portfolio_name: Optional[str] = None

    @property
    def name(self) -> str:
        if self.portfolio_name:
            return self.portfolio_name
        if self.ticker:
            return self.ticker
        return "strategy"


def load_backtest_csv(
    path: str | Path,
    date_col: str = "date",
    strategy_col: str = "equity_curve",
    benchmark_col: str = "benchmark_equity",
    signal_col: str = "signal",
    extra_index_cols: Optional[Sequence[str]] = None,
) -> BacktestResults:
    """
    Load a backtest CSV file and return a BacktestResults object.

    Expected columns (can be overridden):
    - date_col: time index
    - strategy_col: cumulative equity of the strategy
    - benchmark_col: cumulative equity of the reference / market
    - signal_col: trading signal or position
    """
    path = Path(path)
    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise ValueError(f"Missing required date column '{date_col}' in {path}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col)

    col_map = {}
    if strategy_col in df.columns:
        col_map[strategy_col] = "equity_curve"
    if benchmark_col in df.columns:
        col_map[benchmark_col] = "benchmark_equity"
    if signal_col in df.columns:
        col_map[signal_col] = "signal"

    if col_map:
        df = df.rename(columns=col_map)

    ticker = None
    portfolio_name = None

    stem = path.stem
    if "PORT_" in stem:
        portfolio_name = stem
    else:
        ticker = stem.split("_")[0]

    return BacktestResults(df=df, ticker=ticker, portfolio_name=portfolio_name)

