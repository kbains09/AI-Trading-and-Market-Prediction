from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .base import BacktestResults


def compute_drawdown(equity: pd.Series) -> pd.Series:
    """
    Compute drawdown series from an equity curve.

    Drawdown(t) = equity(t) / peak_to_date(t) - 1
    """
    rolling_max = equity.cummax()
    dd = equity / rolling_max - 1.0
    return dd


def plot_drawdown(
    results: BacktestResults,
    out_dir: str | Path,
) -> Path:
    """
    Plot drawdown over time for the strategy equity curve.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = results.df
    if "equity_curve" not in df.columns:
        raise ValueError("DataFrame must contain 'equity_curve' column")

    dd = compute_drawdown(df["equity_curve"])

    plt.figure(figsize=(10, 4))
    plt.plot(dd.index, dd, label="Drawdown")
    plt.fill_between(dd.index, dd, 0, alpha=0.3)

    plt.title(f"Drawdown – {results.name}")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / f"{results.name}_drawdown.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path
