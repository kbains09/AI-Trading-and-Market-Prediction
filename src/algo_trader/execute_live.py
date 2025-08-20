# execute_live.py
import typer
import random
import datetime
import pandas as pd
from pathlib import Path

app = typer.Typer()

TRADE_LOG = Path("data/live_trades.csv")

@app.command()
def execute(
    ticker: str,
    signal: str,
    price: float = typer.Option(..., help="Current price of the asset"),
):
    now = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
    trade = {
        "timestamp": now,
        "ticker": ticker,
        "action": signal.upper(),
        "price": price,
        "quantity": 1,
        "simulated": True,
    }

    if TRADE_LOG.exists():
        df = pd.read_csv(TRADE_LOG)
        df = pd.concat([df, pd.DataFrame([trade])], ignore_index=True)
    else:
        df = pd.DataFrame([trade])

    df.to_csv(TRADE_LOG, index=False)
    print(f"✅ Simulated trade executed: {trade['action']} {ticker} @ ${price:.2f}")

if __name__ == "__main__":
    app()
