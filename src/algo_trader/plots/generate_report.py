from __future__ import annotations

import argparse
from pathlib import Path

from .base import load_backtest_csv
from .drawdown import plot_drawdown
from .equity_curve import plot_equity_curve
from .rolling_sharpe import plot_rolling_sharpe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots for a backtest CSV."
    )
    parser.add_argument(
        "backtest_csv",
        type=str,
        help="Path to backtest CSV file.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="plots",
        help="Directory to write plots into.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=252,
        help="Rolling window for Sharpe ratio.",
    )

    args = parser.parse_args()

    results = load_backtest_csv(
        args.backtest_csv,
        date_col="Date",
        strategy_col="cumulative_strategy_return",
        benchmark_col="cumulative_market_return",
        signal_col="position",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating plots for {results.name} into {out_dir} ...")

    ec_path = plot_equity_curve(results, out_dir)
    print(f"  - Equity curve: {ec_path}")

    dd_path = plot_drawdown(results, out_dir)
    print(f"  - Drawdown: {dd_path}")

    rs_path = plot_rolling_sharpe(results, out_dir, window=args.window)
    print(f"  - Rolling Sharpe: {rs_path}")

    print("Done.")


if __name__ == "__main__":
    main()
