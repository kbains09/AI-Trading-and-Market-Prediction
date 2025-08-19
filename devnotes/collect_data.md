# `collect_data.py`

## Imports & globals

```python
import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import Iterable, Optional
import logging
from datetime import date
```

- **yfinance**: pulls OHLCV and dividends/splits via Yahoo endpoints. Returns a DataFrame indexed by date (until you reset).
    
- **pandas**: CSV‑friendly tabular ops and cleaning.
    
- **Pathlib.Path**: OS‑agnostic path building and atomic file moves.
    
- **typing**: `Iterable`, `Optional` for function signatures.
    
- **logging**: standard library logging; you configure root logger below.
    
- **datetime.date**: imported but not currently used (could be used to default `end` to today).
    

```python
DATA_DIR = Path("data/raw")
DEFAULT_TICKERS = ["AAPL", "TSLA", "MSFT", "BTC-USD"]
```

- Output directory for raw files (per‑ticker CSV).
    
- Default batch tickers: 3 equities + 1 crypto (mixed trading calendars).
    

```python
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
```

- Sets root logging config: time‑stamped INFO‑level messages.
    
- **Note:** If another module already configured logging, `basicConfig` may be ignored. Fine for a top‑level script.
    

---

## `fetch_stock_data(...)`

```python
def fetch_stock_data(
    ticker: str,
    start: str = "2020-01-01",
    end: Optional[str] = "2024-01-01",
    interval: str = "1d",
    retries: int = 2,
) -> pd.DataFrame:
```

- Pulls OHLCV for one `ticker` across a date range and interval with simple retry logic.
    
- **Types:** `start`/`end` are ISO date strings (yfinance accepts strings). `end=None` means “until latest”.
    

```python
    for attempt in range(retries + 1):
        try:
            logging.info(f"Fetching {ticker} [{start} → {end}] interval={interval}")
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            if df is None or df.empty:
                logging.warning(f"No data returned for {ticker} (attempt {attempt+1})")
            else:
                df = df.reset_index()
                return df
        except Exception as e:
            logging.warning(f"Error fetching {ticker} (attempt {attempt+1}): {e}")
    return pd.DataFrame()
```

- **Retry loop:** tries `retries + 1` times (e.g., 3 total if `retries=2`).
    
- **Download:** `yf.download` returns a DataFrame indexed by `Date` with columns `Open, High, Low, Close, Adj Close, Volume` (for daily). For intraday intervals (`1m`, `5m`, …) Yahoo restricts history length.
    
- **Success path:** `reset_index()` moves index `Date` to a column `Date` (helpful downstream).
    
- **Failure modes handled:** empty/None results (network, symbol delisting, calendar gaps) and thrown exceptions.
    
- Returns **empty DataFrame** on full failure.
    

**Assumptions & edges**

- `yf.download` can be rate‑limited or partial during live market hours (missing last bar). Retries help transient issues but not caps.
    
- BTC trading hours differ from equities; `start/end` are still calendar dates; Yahoo returns 24/7 series for crypto.
    
- If you pass a list of tickers to `yf.download`, it returns a **column MultiIndex**; you’re using single‑ticker calls, so no MultiIndex here.
    

---

## `clean_frame(df)`

```python
def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "Date" in df.columns:
        df.sort_values("Date", inplace=True)
        df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
```

- **No‑op on empty** to avoid exceptions.
    
- **Sorts** by `Date` (ascending) and **deduplicates** duplicate bars, keeping the **last** duplicate (sensible if later fetch overwrote earlier partial bars).
    
- **Coerces** standard numeric columns to numeric, putting invalid entries to `NaN` (e.g., “null” strings).
    
- **Note:** Does **not** enforce a specific trading calendar or fill gaps; it simply orders and cleans.
    

**Sharp edges**

- No timezone normalization (Yahoo returns naive timestamps; for daily, that’s fine).
    
- No gap handling: if holidays/weekends appear, they remain absent (which is usually desired for daily bars).
    
- If intraday, dedup by `Date` alone may be insufficient—intraday needs `Datetime` granularity to minute. Here the column is still called `Date`; for intraday, it’s actually a datetime. Works ok but name is slightly misleading.
    

---

## `save_to_csv(df, ticker)`

```python
def save_to_csv(df: pd.DataFrame, ticker: str) -> Optional[Path]:
    if df.empty:
        logging.warning(f"Skip saving empty DataFrame for {ticker}")
        return None
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DATA_DIR / f".{ticker}.csv.tmp"
    out = DATA_DIR / f"{ticker}.csv"
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    logging.info(f"Saved {ticker} → {out}")
    return out
```

- **Guard:** skips saving empties (avoids creating misleading zero‑byte files).
    
- Ensures the output dir exists.
    
- **Atomic-ish write pattern:** write to a **temp** path, then `replace` to final target. On POSIX, `replace` is atomic within the same filesystem—prevents partially written files on crash.
    
- Returns final path on success.
    

**Notes**

- `index=False` is correct since you have a `Date` column after `reset_index()`.
    
- If a previous process is reading `out` while you swap, it’ll see either old or new file (good). On Windows, `replace` semantics differ but are acceptable.
    

---

## `run_batch(...)`

```python
def run_batch(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = "2020-01-01",
    end: Optional[str] = "2024-01-01",
    interval: str = "1d",
):
    for t in tickers:
        df = fetch_stock_data(t, start=start, end=end, interval=interval)
        df = clean_frame(df)
        save_to_csv(df, t)
```

- Iterates over `tickers`, fetches, cleans, and saves per‑ticker CSVs.
    
- **No concurrency** (keeps it simple, respects rate limits better). You can parallelize later with care.
    
- **Edge cases:** Mixed assets with different calendars are fine here; each ticker is handled independently.
    

---

## Script entrypoint

```python
def main():
    run_batch()

if __name__ == "__main__":
    main()
```

- Runs a default batch using `DEFAULT_TICKERS` and hardcoded date range.
    
- **Note:** There’s no CLI yet; parameters are fixed unless you edit code or call functions from elsewhere.
    

---

## What the output CSV looks like

- Columns (typical daily): `Date, Open, High, Low, Close, Adj Close, Volume`
    
- Sorted by `Date`, unique per row (post‑dedupe).
    
- One file per ticker at `data/raw/{TICKER}.csv`.
    

---

## Strengths

- **Robust basics**: retry logic, dedupe, numeric coercion, and atomic file writes.
    
- **Clear logs**: helpful INFO/WARN statements for observability.
    
- **Simple contract**: one CSV per ticker, ready for your feature pipeline.
    

---

## Sharp edges & assumptions to watch

- **Calendar alignment**: Not enforced. Downstream multi‑asset joins need careful `inner`/`outer` logic and forward fills if required.
    
- **Corporate actions**: You store both `Close` and `Adj Close`. If you do ML on price levels, prefer **returns** on `Adj Close` (split/dividend adjusted). Keep raw `Close` for plotting/trader sanity.
    
- **Intraday**: If you switch `interval` to intraday, consider column naming (`Datetime`), rate limits, and much larger file sizes.
    
- **End date default**: Hardcoded `"2024-01-01"`. You may want `end=None` (up to latest) or `str(date.today())`.
    
- **Logging config**: If used within a larger app that already configures logging, `basicConfig` may be ignored. Not an issue for a stand‑alone script.
    

---

## High‑impact upgrades (low effort)

1. **CLI with Typer**
    
    ```python
    import typer
    app = typer.Typer(add_completion=False)
    
    @app.command()
    def run(
        tickers: str = typer.Option(",".join(DEFAULT_TICKERS), help="Comma-separated tickers"),
        start: str = typer.Option("2020-01-01"),
        end: Optional[str] = typer.Option(None, help="Default: latest"),
        interval: str = typer.Option("1d"),
    ):
        run_batch(tickers=[t.strip() for t in tickers.split(",")], start=start, end=end, interval=interval)
    
    if __name__ == "__main__":
        app()
    ```
    
    - Lets you run:  
        `poetry run python src/algo_trader/collect_data.py run --tickers AAPL,MSFT --start 2018-01-01 --end 2025-08-01`
        
2. **Metadata sidecar**
    
    - Save a `{ticker}.meta.json` next to the CSV with source, fetch time, interval, start/end, row count, min/max date. Helps reproducibility/debugging.
        
3. **Automatic end date**
    
    ```python
    if end is None:
        end = date.today().isoformat()
    ```
    
4. **Data quality gate**
    
    - Before saving, compute and log:
        
        - duplicate rate, missing numeric rate, strictly increasing dates check.
            
        - For daily equities, warn if unexpected weekend bars appear.
            
5. **Corporate actions snapshot**
    
    - `yf.Ticker(ticker).actions` returns dividends/splits. Save them to `data/raw/{ticker}_actions.csv` for later adjustment checks.
        
6. **Backfill append mode**
    
    - If `data/raw/{ticker}.csv` exists, load last date and only fetch the missing tail (saves time and bandwidth). Then append + dedupe + save.
        
7. **Retry backoff**
    
    - Add exponential backoff between retries to be gentle with the API and avoid rate‑limit flaps.
        
8. **Validation for intraday**
    
    - If `interval` endswith `"m"`, ensure `Date` is truly datetime with timezone (or document it). Consider renaming to `Datetime` when intraday.
        

---

## Quick sanity checks you can run

- **Row count & dates**
    
    ```python
    df = pd.read_csv("data/raw/AAPL.csv", parse_dates=["Date"])
    df["Date"].is_monotonic_increasing  # should be True
    df.duplicated("Date").sum()         # should be 0
    df[["Open","High","Low","Close","Adj Close","Volume"]].isna().mean()  # near 0
    ```
    
- **Latest date reach**
    
    - If `end=None`, confirm the last `Date` is up to today for markets that were open.
        

---

## How this fits your pipeline

- This script **feeds** your feature engineering step with clean per‑ticker OHLCV CSVs.
    
- The cleaned output goes to `data/raw/`, then your labeling/feature scripts read from there and write to `data/labeled/`.
    

---

## TL;DR

- Fetches per‑ticker OHLCV with retries, cleans (sort/dedup/coerce), and saves atomically to `data/raw/{ticker}.csv`.
    
- Good foundations; add a CLI, metadata logs, corporate actions snapshot, append‑mode backfills, and quality checks to make it production‑grade.