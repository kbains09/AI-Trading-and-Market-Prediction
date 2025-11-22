from __future__ import annotations

import datetime
import json
import pathlib as pl

import joblib
import pandas as pd
import typer

app = typer.Typer(add_completion=False)

# Where to log simulated trades
TRADE_LOG = pl.Path("data/live_trades.csv")


def _load_features(
    ticker: str,
    feature_dir: str = "data/processed",
) -> pd.DataFrame:
    """Load processed feature CSV and return latest row + full DF."""
    feature_path = pl.Path(feature_dir) / f"{ticker}_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature data not found at {feature_path}")

    df = pd.read_csv(feature_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna()

    if df.empty:
        raise ValueError(f"No usable rows in {feature_path}")

    latest = df.iloc[-1:]
    return df, latest


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Match the same feature selection logic used in training/predict_next:
    keep engineered features, drop target/label-style columns.
    """
    feature_cols = [
        col
        for col in df.columns
        if col.startswith(
            (
                "return",
                "logret",
                "SMA",
                "EMA",
                "RSI",
                "MACD",
                "BB_",
                "ATR",
                "HL_",
                "OC_",
                "vol",
            )
        )
        and not col.startswith("next_")
        and not col.startswith("target")
        and col
        not in (
            "Target",
            "future_logret_5",
            "direction_5",
            "regime_trend_meanrev",
        )
    ]
    if not feature_cols:
        raise ValueError("No feature columns selected for model input.")
    return feature_cols


def _load_models(
    ticker: str,
    model_dir: str = "models/xgboost",
):
    """
    Load regime + direction XGBoost models for given ticker.
    """
    model_dir_path = pl.Path(model_dir)
    regime_path = model_dir_path / f"{ticker}_regime_model.pkl"
    direction_path = model_dir_path / f"{ticker}_direction_model.pkl"

    if not regime_path.exists():
        raise FileNotFoundError(f"Regime model file not found: {regime_path}")
    if not direction_path.exists():
        raise FileNotFoundError(
            f"Direction model file not found: {direction_path}"
        )

    regime_model = joblib.load(regime_path)
    direction_model = joblib.load(direction_path)
    return regime_model, direction_model


def _regime_label(idx: int) -> str:
    mapping = {
        0: "NEUTRAL",
        1: "TRENDING",
        2: "MEAN_REVERTING",
    }
    return mapping.get(int(idx), f"UNKNOWN_{idx}")


def _log_trade(
    ticker: str,
    action: str,
    price: float,
    quantity: float,
    regime_class: int,
    regime_proba: list[float],
    dir_proba: list[float],
):
    """
    Append a simulated trade to TRADE_LOG as CSV.
    """
    TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp_utc": datetime.datetime.utcnow()
        .replace(microsecond=0)
        .isoformat()
        + "Z",
        "ticker": ticker,
        "action": action,
        "price": float(price),
        "quantity": float(quantity),
        "regime_class": int(regime_class),
        "regime_proba_json": json.dumps(list(map(float, regime_proba))),
        "dir_proba_json": json.dumps(list(map(float, dir_proba))),
    }

    if TRADE_LOG.exists():
        df_log = pd.read_csv(TRADE_LOG)
        df_log = pd.concat(
            [df_log, pd.DataFrame([row])], ignore_index=True
        )
    else:
        df_log = pd.DataFrame([row])

    df_log.to_csv(TRADE_LOG, index=False)


@app.command("trade")
def paper_trade(
    ticker: str = typer.Option(..., help="Ticker symbol to trade"),
    model_dir: str = typer.Option(
        "models/xgboost", help="Directory with trained models"
    ),
    feature_dir: str = typer.Option(
        "data/processed", help="Directory with processed feature CSVs"
    ),
    up_threshold: float = typer.Option(
        0.55,
        help="If P(UP) >= this, go long (BUY).",
    ),
    down_threshold: float = typer.Option(
        0.45,
        help="If P(UP) <= this, go short (SELL).",
    ),
    quantity: float = typer.Option(
        1.0, help="Simulated position size for paper trades."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Print full probability info."
    ),
):
    """
    Single-shot paper trade:
    - Load latest bar's features
    - Predict regime + 5D direction
    - Decide BUY/SELL/HOLD based on P(UP)
    - Log a simulated trade to CSV
    """
    typer.echo(f"[paper_trader] 🚀 Paper trading for {ticker}")

    # 1) Load models
    try:
        regime_model, direction_model = _load_models(
            ticker=ticker, model_dir=model_dir
        )
    except Exception as e:
        typer.echo(f"[paper_trader] ❌ Failed to load models: {e}")
        raise typer.Exit(1)

    # 2) Load features
    try:
        df, latest = _load_features(
            ticker=ticker, feature_dir=feature_dir
        )
    except Exception as e:
        typer.echo(f"[paper_trader] ❌ Failed to load features: {e}")
        raise typer.Exit(1)

    feature_cols = _select_feature_columns(df)
    X = latest[feature_cols]

    last_date = latest["Date"].iloc[0]
    last_close = (
        float(latest["Close"].iloc[0])
        if "Close" in latest.columns
        else float("nan")
    )

    typer.echo(f"[paper_trader] 📅 As of: {last_date.date()}")
    if not pd.isna(last_close):
        typer.echo(f"[paper_trader] 💰 Last close: {last_close:.2f}")

    # 3) Predict regime + direction
    try:
        regime_proba = regime_model.predict_proba(X)[0]
        regime_class = int(regime_proba.argmax())
        dir_proba = direction_model.predict_proba(X)[0]
    except Exception as e:
        typer.echo(f"[paper_trader] ❌ Model prediction failed: {e}")
        raise typer.Exit(1)

    p_up = float(dir_proba[1])
    p_down = float(dir_proba[0])

    typer.echo(
        f"[paper_trader] 🧭 Regime: {_regime_label(regime_class)} "
        f"(class={regime_class}, p≈{regime_proba[regime_class]:.3f})"
    )
    typer.echo(
        f"[paper_trader] 📈 5D direction ⇒ P(UP)={p_up:.3f}, P(DOWN)={p_down:.3f}"
    )

    if verbose:
        typer.echo(
            f"[paper_trader] 🔍 Raw regime probs: "
            f"{[round(float(x), 4) for x in regime_proba]}"
        )
        typer.echo(
            f"[paper_trader] 🔍 Raw direction probs [DOWN, UP]: "
            f"{[round(float(x), 4) for x in dir_proba]}"
        )

    # 4) Decide signal
    if p_up >= up_threshold:
        action = "BUY"
        pos_qty = +quantity
    elif p_up <= down_threshold:
        action = "SELL"
        pos_qty = -quantity
    else:
        action = "HOLD"
        pos_qty = 0.0

    typer.echo(
        f"[paper_trader] 🎯 Signal for {ticker}: {action} "
        f"(P(UP)={p_up:.3f}, thresholds: up≥{up_threshold}, down≤{down_threshold})"
    )

    # 5) Log simulated trade (only if not HOLD)
    if action != "HOLD":
        _log_trade(
            ticker=ticker,
            action=action,
            price=last_close,
            quantity=pos_qty,
            regime_class=regime_class,
            regime_proba=list(regime_proba),
            dir_proba=list(dir_proba),
        )
        typer.echo(
            f"[paper_trader] 📝 Logged simulated trade → {TRADE_LOG}"
        )
    else:
        typer.echo("[paper_trader] 🧊 No trade logged (HOLD signal)")


if __name__ == "__main__":
    app()
