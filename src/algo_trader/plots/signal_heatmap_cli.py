from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .signal_heatmap import plot_signal_heatmap


def main() -> None:
    parser = argparse.ArgumentParser(description="Build signal heatmap across tickers.")
    parser.add_argument(
        "--backtest-dir",
        type=str,
        default="data/backtests",
        help="Directory with per-ticker backtest CSVs.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "TSLA", "MSFT", "BTC-USD"],
        help="Tickers to include.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="plots",
        help="Where to save the heatmap.",
    )

    args = parser.parse_args()

    rows = []
    for ticker in args.tickers:
        path = Path(args.backtest_dir) / f"{ticker}_backtest.csv"
        df = pd.read_csv(path)
        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["Date"])
        df["signal"] = df["position"]
        rows.append(df[["date", "ticker", "signal"]])

    all_signals = pd.concat(rows, ignore_index=True)

    out_path = plot_signal_heatmap(
        all_signals,
        out_dir=args.out_dir,
        tickers=args.tickers,
        signal_col="signal",
    )

    print(f"Signal heatmap saved to {out_path}")


if __name__ == "__main__":
    main()
