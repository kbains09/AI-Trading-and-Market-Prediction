from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import BacktestResults


def compute_log_returns(equity: pd.Series) -> pd.Series:
    """
    Compute log returns from an equity curve.
    """
    return (equity / equity.shift(1)).apply(lambda x: pd.NA if pd.isna(x) else float(x)).pipe(
        lambda s: (s).map(lambda x: None if x is None else (0.0 if x <= 0 else pd.np.log(x))) 
    )


def rolling_sharpe(equity: pd.Series, window: int = 252, trading_days: int = 252) -> pd.Series:
    """
    Compute rolling Sharpe ratio from an equity curve.
    
    Converts equity curve to simple returns, then computes rolling Sharpe ratio
    as (mean return / std return) * sqrt(trading_days).
    """
    # Convert equity curve to simple returns
    returns = equity.pct_change().dropna()
    
    # Compute rolling mean and std
    mu = returns.rolling(window).mean()
    sd = returns.rolling(window).std(ddof=0)
    
    # Compute Sharpe ratio with error handling
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = (mu / sd) * np.sqrt(trading_days)
    
    # Replace infinities with NaN
    return rs.replace([np.inf, -np.inf], np.nan)


def plot_rolling_sharpe(
    results: BacktestResults,
    out_dir: str | Path,
    window: int = 252,
) -> Path:
    """
    Plot rolling Sharpe ratio for the strategy equity curve.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = results.df
    if "equity_curve" not in df.columns:
        raise ValueError("DataFrame must contain 'equity_curve' column")

    rs = rolling_sharpe(df["equity_curve"], window=window)

    plt.figure(figsize=(10, 4))
    plt.plot(rs.index, rs, label=f"{window}-day rolling Sharpe")

    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Rolling Sharpe ({window} days) – {results.name}")
    plt.xlabel("Date")
    plt.ylabel("Sharpe")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / f"{results.name}_rolling_sharpe_{window}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path
