# `label_features.py` 

## Imports & directories

```python
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
LABELED_DIR = Path("data/labeled")
LABELED_DIR.mkdir(parents=True, exist_ok=True)
```

- Uses **`data/processed`** as the source for feature CSVs (produced by your feature engineering step).
    
- Ensures **`data/labeled/`** exists before writing out label files (idempotent).
    

---

## Label generator

```python
def generate_labels(df: pd.DataFrame, label_type="binary") -> pd.DataFrame:
    df["next_return_1d"] = df["Close"].pct_change().shift(-1)  # Add this line
```

- Creates a **forward one‑bar return** aligned to the current row:
    
    - `pct_change()` at t is (Ct/Ct−1)−1(C_t/C_{t-1})-1.
        
    - `shift(-1)` moves future return **up one row**, so at time _t_ you now have (Ct+1/Ct)−1(C_{t+1}/C_{t})-1.
        
- ✅ This is exactly what your backtester expects (forward return known at _t+1_ but **stored on row t**).
    
- ⚠️ **Duplicate risk:** your feature generator already creates `next_return_1d`. If the processed file already contains it, this line **overwrites** it (same definition, so fine). If you want to be explicit, you can `pop` the old one first.
    

```python
    if label_type == "binary":
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
```

- **Binary label**: 1 if next close is higher than current close, else 0.
    
- Equivalent to `Target = (next_return_1d > 0)`. Nice and leakage‑safe.
    

```python
    elif label_type == "directional":
        df["Target"] = df["Close"].shift(-1) - df["Close"]
        df["Target"] = df["Target"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
```

- **Ternary label**: −1, 0, +1. Same forward‑looking idea, but discretized 3‑way.
    
- Consider adding a **no‑trade band** (e.g., mark small moves around 0 as 0) to combat noise:  
    `elif abs(next_return_1d) < eps: 0 else sign`.
    

```python
    else:
        raise ValueError(f"Unsupported label type: {label_type}")
    
    df.dropna(inplace=True)
    return df
```

- Drops all rows with NaNs (warm‑ups, last row with `shift(-1)`, indicator windows).
    
- ⚠️ **Global `dropna`** can be aggressive; it drops rows where **any** column is NaN (including features). That’s often fine, but if you’ve got a few features that are sparsely missing, consider targeting:
    
    ```python
    df = df.dropna(subset=["Close", "next_return_1d", "Target"])
    ```
    
    and/or impute non‑critical features elsewhere.
    

---

## Per‑ticker processing

```python
def process_ticker(ticker: str, label_type="binary"):
    print(f"🏷️ Labeling {ticker}...")
    file_path = PROCESSED_DIR / f"{ticker}_features.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Feature file not found for {ticker}")
```

- Expecting your **feature engineering** step to have written `data/processed/{ticker}_features.csv`.
    

```python
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
```

- Parses `Date` and sets it as the **index** (good for time‑series semantics).
    
- Note: other scripts (e.g., `train_model.py`) read labeled files with `parse_dates=["Date"]` but **do not set index**. That’s fine because `to_csv` below will write the index as a `Date` column header.
    

```python
    df = generate_labels(df, label_type)
    
    out_path = LABELED_DIR / f"{ticker}_labeled.csv"
    df.to_csv(out_path)
    print(f"✅ Labeled file saved: {out_path}")
```

- Adds `next_return_1d` and `Target`, drops NaNs, and writes to `data/labeled/{ticker}_labeled.csv`.
    
- Because the index is named `Date`, `to_csv` will emit it as a **`Date` column** in the CSV, which matches how `train_model.py` expects to read it.
    

---

## Batch runner

```python
def main():
    tickers = ["AAPL", "TSLA", "MSFT", "BTC-USD"]
    for ticker in tickers:
        try:
            process_ticker(ticker, label_type="binary")
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")

if __name__ == "__main__":
    main()
```

- Simple loop; keeps going on per‑ticker errors.
    
- You can later promote this to a Typer CLI (optional).
    

---

## Alignment & leakage audit (important)

- **Good alignment:** Labels and `next_return_1d` are **forward‑shifted** to row _t_. This is compatible with your updated backtester (which no longer shifts anything internally).
    
- **Leakage hazard in training/backtest features:** Your current `train_model.py` and the earlier `backtest_cli.py` build `X` as `df.drop(columns=["Date","Target"])`. That **still includes `next_return_1d`**, which is a **future** value.  
    ➜ You must **exclude** `next_return_1d` from `X` in both training and backtest inference:
    
    ```python
    X = df.drop(columns=["Date", "Target", "next_return_1d"])
    ```
    
    (We flagged this earlier; mentioning again here because this file introduces/overwrites that column.)
    

---

## Small correctness & robustness tweaks

1. **Use adjusted prices** (optional but recommended)
    
    - If your processed features are based on raw `Close`, corporate actions can bias labels. Prefer labels from `Adj Close` if available:
        
        ```python
        price = df.get("Adj Close", df["Close"])
        df["next_return_1d"] = price.pct_change().shift(-1)
        df["Target"] = (price.shift(-1) > price).astype(int)
        ```
        
2. **No‑trade band for directional labels**
    
    ```python
    eps = 0.0005  # 5 bps band
    nxt = df["next_return_1d"]
    df["Target"] = np.where(nxt > eps, 1, np.where(nxt < -eps, -1, 0))
    ```
    
3. **Avoid global dropna**
    
    - Replace with targeted drops to avoid losing rows due to irrelevant feature NaNs:
        
        ```python
        df = df.dropna(subset=["next_return_1d", "Target"])
        ```
        
4. **Keep index & schema consistent**
    
    - You’re writing with index; downstream `pd.read_csv(..., parse_dates=["Date"])` will recreate the `Date` column (not as index) which your training script expects. That’s consistent. If you ever switch to `index=False`, update the reader.
        
5. **Manifest / metadata (optional)**
    
    - Save a small JSON next to the labeled file with: label_type, thresholds (eps), source file, row counts, min/max dates.
        

---

## End‑to‑end compatibility check

- **engineering_features.py** → writes `data/processed/{ticker}_features.csv`.
    
    - It may already include `next_return_1d` (same definition). Your label step will overwrite; OK.
        
- **label_features.py** → writes `data/labeled/{ticker}_labeled.csv` with `Target` and `next_return_1d`.
    
- **train_model.py** → reads `{ticker}_labeled.csv`.
    
    - 🔧 Update `X = df.drop(columns=["Date","Target","next_return_1d"])` to avoid leakage.
        
- **backtest_cli.py (updated)** → also should **not** use `next_return_1d` as a feature; only as P&L input. Ensure the same drop there (your updated version still drops only `Date` and `Target`—adjust it as well).
    

---

## Quick sanity tests you can run

- **Label direction correctness**
    
    ```python
    df = pd.read_csv("data/labeled/AAPL_labeled.csv", parse_dates=["Date"])
    chk = ((df["Close"].shift(-1) > df["Close"]).astype(int) == df["Target"]).mean()
    print("Binary label match:", chk)  # should be 1.0
    ```
    
- **Forward return shape**
    
    ```python
    ret_forward = df["next_return_1d"]
    ret_now = df["Close"].pct_change()
    # ret_forward at t should equal ret_now at t+1
    print((ret_forward.dropna().values == ret_now.shift(-1).dropna().values).mean())
    ```
    
- **No leakage in X**
    
    ```python
    leak_cols = set(["next_return_1d"]) & set(df.columns)
    print("Leakage columns present:", leak_cols)  # ensure you drop them before training
    ```
    

---

## TL;DR

- This script reads processed feature files, **adds forward returns** and a **Target** label, and writes a trainer‑ready CSV per ticker.
    
- Alignment is correct and matches your backtester.
    
- The main pitfall to fix in the rest of your pipeline: **exclude `next_return_1d` from features** when training and predicting.
    
- Optional upgrades: adjusted prices for labels, no‑trade band, targeted `dropna`, and saving simple manifest metadata.
    

If you want, I can give you a tiny patch diff for `train_model.py` and your updated `backtest_cli.py` to **drop `next_return_1d`** from `X` so we fully eliminate leakage.