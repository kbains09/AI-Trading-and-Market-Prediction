from __future__ import annotations

import pathlib as pl
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import typer
from xgboost import XGBClassifier

from ..features.feature_set import LIVE_FEATURE_COLUMNS

app = typer.Typer(add_completion=False)


def _chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simple time-ordered split: first train, then val, then test.
    Assumes df is already sorted chronologically.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


@app.command()
def main(
    ticker: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL"),
    labeled_dir: str = typer.Option("data/labeled", help="Directory with *_labeled.csv"),
    model_dir: str = typer.Option("models/xgboost", help="Directory to save regime models"),
):
    """
    Train a regime classification model on the 'regime_trend_meanrev' label.

    Original label space:
        -1 = mean-reversion
         0 = neutral / sideway
         1 = trend

    We encode to {0, 1, 2} for XGBoost and save a mapping dict.

    Saves an artifact dict to:
      models/xgboost/{TICKER}_regime_model.pkl

    Artifact schema:
      {
        "model": XGBClassifier,
        "feature_cols": [ ... ],
        "regime_map": { original_label_int -> encoded_class_int }
      }
    """
    labeled_path = pl.Path(labeled_dir) / f"{ticker}_labeled.csv"
    model_dir_path = pl.Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    if not labeled_path.exists():
        typer.echo(f"[train_regime] ❌ Labeled file not found: {labeled_path}")
        raise typer.Exit(1)

    df = pd.read_csv(labeled_path)

    if "regime_trend_meanrev" not in df.columns:
        typer.echo("[train_regime] ❌ Column 'regime_trend_meanrev' not found in labeled data.")
        raise typer.Exit(1)

    # Drop rows with missing labels
    df = df.dropna(subset=["regime_trend_meanrev"]).reset_index(drop=True)

    # Original integer labels
    y_orig = df["regime_trend_meanrev"].astype(int)
    unique_orig = sorted(y_orig.unique())
    typer.echo(f"[train_regime] Raw regime label classes: {unique_orig}")

    # Build a stable mapping orig_label -> encoded_label (0..K-1)
    regime_map: Dict[int, int] = {orig: idx for idx, orig in enumerate(unique_orig)}
    y_enc = y_orig.map(regime_map).astype(int)

    typer.echo(f"[train_regime] Encoded regime label classes: {sorted(y_enc.unique().tolist())}")
    typer.echo(f"[train_regime] Regime map (orig -> enc): {regime_map}")

    # 🔑 Use SAME feature set as live / backtest
    available = [c for c in LIVE_FEATURE_COLUMNS if c in df.columns]
    missing = sorted(set(LIVE_FEATURE_COLUMNS) - set(available))
    if missing:
        typer.echo(f"[train_regime] ⚠️ Missing features in labeled data (ignored): {missing}")

    if not available:
        typer.echo("[train_regime] ❌ No usable features found from LIVE_FEATURE_COLUMNS.")
        raise typer.Exit(1)

    X = df[available].copy()

    X = X.dropna(axis=1, how="all") 
    X = X.fillna(0.0)              

    # Merge X and encoded y for a clean chronological split
    combined = pd.concat([X, y_enc.rename("regime_enc")], axis=1)

    train_df, val_df, test_df = _chronological_split(combined)
    X_train, y_train = train_df[available], train_df["regime_enc"]
    X_val, y_val = val_df[available], val_df["regime_enc"]
    X_test, y_test = test_df[available], test_df["regime_enc"]

    typer.echo(f"[train_regime] 📊 Ticker: {ticker}")
    typer.echo(
        f"[train_regime] Train size: {len(X_train)}, "
        f"Val size: {len(X_val)}, Test size: {len(X_test)}"
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate on test set (encoded space)
    y_pred_enc = model.predict(X_test)
    acc_enc = float((y_pred_enc == y_test.values).mean())

    typer.echo(f"[train_regime] ✅ Test accuracy (encoded labels 0/1/2): {acc_enc:.4f}")

    # Also compute accuracy in original label space (mainly for sanity)
    inv_regime_map = {v: k for k, v in regime_map.items()}
    y_test_orig = y_test.map(inv_regime_map)
    y_pred_orig = pd.Series(y_pred_enc).map(inv_regime_map)
    acc_orig = float((y_test_orig.values == y_pred_orig.values).mean())
    typer.echo(f"[train_regime] ℹ️  Test accuracy (original label space): {acc_orig:.4f}")

    # Save artifact (model + feature metadata + mapping)
    artifact = {
        "model": model,
        "feature_cols": list(X.columns),
        "regime_map": regime_map,
    }

    out_path = model_dir_path / f"{ticker}_regime_model.pkl"
    joblib.dump(artifact, out_path)

    typer.echo(f"[train_regime] 💾 Saved regime model → {out_path}")


if __name__ == "__main__":
    app()
