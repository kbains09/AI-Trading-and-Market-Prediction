from __future__ import annotations

import pathlib as pl
from typing import List

import joblib
import numpy as np
import pandas as pd
import typer
import yaml
from xgboost import XGBClassifier

app = typer.Typer(add_completion=False)

# Load config
PROJECT_CONFIG = yaml.safe_load(pl.Path("config/project.yaml").read_text())
FEATURES_CONFIG = yaml.safe_load(pl.Path("config/features.yaml").read_text())

DATA_LABELED = pl.Path(PROJECT_CONFIG["data"]["labeled_dir"])
MODELS_DIR = pl.Path(PROJECT_CONFIG["data"]["models_dir"]) if "models_dir" in PROJECT_CONFIG["data"] else pl.Path("models/xgboost")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Pick model features using prefixes from features.yaml."""
    prefixes = FEATURES_CONFIG["training"]["feature_prefixes"]
    exclude_prefixes = FEATURES_CONFIG["training"]["exclude_prefixes"]

    cols: List[str] = []
    for col in df.columns:
        if any(col.startswith(p) for p in prefixes) and not any(
            col.startswith(e) for e in exclude_prefixes
        ):
            cols.append(col)
    return cols


def _split_by_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into train/val/test using project.yaml date ranges."""
    splits = PROJECT_CONFIG["splits"]
    train_start = pd.to_datetime(splits["train"]["start"])
    train_end = pd.to_datetime(splits["train"]["end"])
    val_start = pd.to_datetime(splits["validation"]["start"])
    val_end = pd.to_datetime(splits["validation"]["end"])
    test_start = pd.to_datetime(splits["test"]["start"])
    test_end = pd.to_datetime(splits["test"]["end"])

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    train = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)]
    val = df[(df["Date"] >= val_start) & (df["Date"] <= val_end)]
    test = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)]

    return train, val, test


@app.command()
def main(
    ticker: str = typer.Argument(..., help="Ticker to train direction model for"),
    n_estimators: int = typer.Option(400, help="Number of trees"),
    max_depth: int = typer.Option(4, help="Max tree depth"),
    learning_rate: float = typer.Option(0.05, help="Learning rate"),
    subsample: float = typer.Option(0.8, help="Subsample ratio"),
):
    """
    Train a direction model for 5-day horizon (UP/DOWN),
    using the 'direction_5' label.
    """
    parquet_path = DATA_LABELED / "market_patterns.parquet"
    if not parquet_path.exists():
        typer.echo(f"[train_direction] ❌ Missing parquet: {parquet_path}")
        raise typer.Exit(1)

    df_all = pd.read_parquet(parquet_path)

    if "ticker" not in df_all.columns:
        typer.echo("[train_direction] ❌ 'ticker' column not found in merged dataset")
        raise typer.Exit(1)

    df = df_all[df_all["ticker"] == ticker].copy()
    if df.empty:
        typer.echo(f"[train_direction] ❌ No rows found for ticker {ticker}")
        raise typer.Exit(1)

    # Select features & label
    feature_cols = _select_feature_columns(df)
    label_col = "direction_5"

    if label_col not in df.columns:
        typer.echo(f"[train_direction] ❌ Label column '{label_col}' not found")
        raise typer.Exit(1)

    # Map direction -1 / +1 → 0 / 1
    df[label_col] = (df[label_col] == 1).astype(int)

    # Drop NaNs
    df = df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)

    train_df, val_df, test_df = _split_by_date(df)

    def _xy(frame: pd.DataFrame):
        return frame[feature_cols].values, frame[label_col].values.astype(int)

    X_train, y_train = _xy(train_df)
    X_val, y_val = _xy(val_df)
    X_test, y_test = _xy(test_df)

    typer.echo(f"[train_direction] 📊 Ticker: {ticker}")
    typer.echo(f"[train_direction] Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
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

    test_preds = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int) if len(y_test) > 0 else np.array([])
    test_acc = (test_preds == y_test).mean() if len(y_test) > 0 else float("nan")
    typer.echo(f"[train_direction] ✅ Test accuracy: {test_acc:.4f}")

    out_path = MODELS_DIR / f"{ticker}_direction_model.pkl"
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
        },
        out_path,
    )
    typer.echo(f"[train_direction] 💾 Saved direction model → {out_path}")


if __name__ == "__main__":
    app()
