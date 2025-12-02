from __future__ import annotations

import pathlib as pl
from typing import Tuple

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
    model_dir: str = typer.Option("models/xgboost", help="Directory to save direction models"),
):
    """
    Train a binary direction model on the 'direction_5' label.

    - direction_5 is originally in {-1, 1}
        * -1 = DOWN (mapped → 0)
        *  1 = UP   (mapped → 1)

    Saves an artifact dict to:
      models/xgboost/{TICKER}_direction_model.pkl

    Artifact schema:
      {
        "model": XGBClassifier,
        "feature_cols": [ ... ]  # columns used for training / prediction
      }
    """
    labeled_path = pl.Path(labeled_dir) / f"{ticker}_labeled.csv"
    model_dir_path = pl.Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    if not labeled_path.exists():
        typer.echo(f"[train_direction] ❌ Labeled file not found: {labeled_path}")
        raise typer.Exit(1)

    df = pd.read_csv(labeled_path)

    if "direction_5" not in df.columns:
        typer.echo("[train_direction] ❌ Column 'direction_5' not found in labeled data.")
        raise typer.Exit(1)

    # Drop rows with missing labels
    df = df.dropna(subset=["direction_5"]).reset_index(drop=True)

    # 🎯 Label encoding: map {-1, 1} → {0, 1}
    y_raw = df["direction_5"].astype(int)
    unique_raw = sorted(y_raw.unique())
    typer.echo(f"[train_direction] Raw label classes: {unique_raw}")

    label_map = {-1: 0, 1: 1}
    if not set(unique_raw).issubset(label_map.keys()):
        typer.echo(f"[train_direction] ❌ Unexpected direction_5 classes: {unique_raw}")
        raise typer.Exit(1)

    y = y_raw.map(label_map).astype(int)
    typer.echo(f"[train_direction] Encoded label classes: {sorted(y.unique().tolist())}")

    # Use SAME feature set as live / backtest
    available = [c for c in LIVE_FEATURE_COLUMNS if c in df.columns]
    missing = sorted(set(LIVE_FEATURE_COLUMNS) - set(available))
    if missing:
        typer.echo(f"[train_direction] ⚠️ Missing features in labeled data (ignored): {missing}")

    if not available:
        typer.echo("[train_direction] ❌ No usable features found from LIVE_FEATURE_COLUMNS.")
        raise typer.Exit(1)

    X = df[available].copy()

    # Clean features
    X = X.dropna(axis=1, how="all")  # drop columns that are entirely NaN
    X = X.fillna(0.0)                # simple imputation for any remaining NaNs

    # Merge X and y for a clean chronological split
    combined = pd.concat([X, y.rename("direction_bin")], axis=1)

    train_df, val_df, test_df = _chronological_split(combined)
    X_train, y_train = train_df[available], train_df["direction_bin"]
    X_val, y_val = val_df[available], val_df["direction_bin"]
    X_test, y_test = test_df[available], test_df["direction_bin"]

    typer.echo(f"[train_direction] 📊 Ticker: {ticker}")
    typer.echo(
        f"[train_direction] Train size: {len(X_train)}, "
        f"Val size: {len(X_val)}, Test size: {len(X_test)}"
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic", 
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate on test set
    prob_up = model.predict_proba(X_test)[:, 1]
    y_pred = (prob_up >= 0.5).astype(int)
    acc = float((y_pred == y_test.values).mean())

    typer.echo(f"[train_direction] ✅ Test accuracy (encoded labels 0/1): {acc:.4f}")

    # Save artifact (model + feature metadata)
    artifact = {
        "model": model,
        "feature_cols": list(X.columns),
    }

    out_path = model_dir_path / f"{ticker}_direction_model.pkl"
    joblib.dump(artifact, out_path)
    json_path = model_dir_path / f"{ticker}_direction_model.json"
    model.save_model(json_path) 

    typer.echo(f"[train_direction] 💾 Saved direction model → {out_path}")
    typer.echo(f"[train_direction] 💾 Saved direction model → {json_path}")


if __name__ == "__main__":
    app()
