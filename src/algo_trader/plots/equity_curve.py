from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .base import BacktestResults


def plot_equity_curve(
    results: BacktestResults,
    out_dir: str | Path,
    show_benchmark: bool = True,
) -> Path:
    """
    Plot strategy equity curve (and optional benchmark) over time.

    Saves a PNG and returns the path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = results.df
    if "equity_curve" not in df.columns:
        raise ValueError("DataFrame must contain 'equity_curve' column")

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["equity_curve"], label=f"{results.name} strategy")

    if show_benchmark and "benchmark_equity" in df.columns:
        plt.plot(df.index, df["benchmark_equity"], label="benchmark")

    plt.title(f"Equity Curve – {results.name}")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / f"{results.name}_equity_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path
