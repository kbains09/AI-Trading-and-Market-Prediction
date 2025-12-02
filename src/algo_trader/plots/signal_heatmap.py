from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


files = [
    ("AAPL", Path("data/backtests/AAPL_backtest.csv")),
    ("TSLA", Path("data/backtests/TSLA_backtest.csv")),
    ("MSFT", Path("data/backtests/MSFT_backtest.csv")),
    ("BTC-USD", Path("data/backtests/BTC-USD_backtest.csv")),
]

rows = []
for ticker, path in files:
    df = pd.read_csv(path)
    df["ticker"] = ticker
    # Normalize date col name if needed
    df["date"] = pd.to_datetime(df["Date"])
    df["signal"] = df["position"]
    rows.append(df[["date", "ticker", "signal"]])

all_signals = pd.concat(rows, ignore_index=True)


def build_signal_matrix(
    df: pd.DataFrame,
    tickers: Iterable[str] | None = None,
    signal_col: str = "signal",
) -> pd.DataFrame:
    """
    Build a pivoted matrix of signals over time: index=date, columns=ticker, values=signal.

    Expects columns: ['date', 'ticker', signal_col].
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column")
    if "ticker" not in df.columns:
        raise ValueError("DataFrame must contain a 'ticker' column")
    if signal_col not in df.columns:
        raise ValueError(f"DataFrame must contain a '{signal_col}' column")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if tickers is not None:
        df = df[df["ticker"].isin(list(tickers))]

    pivot = df.pivot_table(
        index="date",
        columns="ticker",
        values=signal_col,
    )

    pivot = pivot.sort_index()
    return pivot


def plot_signal_heatmap(
    df: pd.DataFrame,
    out_dir: str | Path,
    tickers: Iterable[str] | None = None,
    signal_col: str = "signal",
) -> Path:
    """
    Plot a heatmap of signals over time across tickers.

    Good for seeing when multiple assets fire at the same time.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = build_signal_matrix(df, tickers=tickers, signal_col=signal_col)

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        mat.T,
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Signal / Position"},
    )

    plt.title("Signal Heatmap Across Tickers")
    plt.xlabel("Date")
    plt.ylabel("Ticker")
    plt.tight_layout()

    out_path = out_dir / "signal_heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path
