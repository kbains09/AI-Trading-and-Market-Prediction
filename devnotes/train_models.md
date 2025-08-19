# `train_model.py` 

## Imports

```python
import pathlib as pl
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
```

- `pathlib as pl`: Path objects for robust, cross‑platform filesystem ops.
    
- `pandas`: CSV I/O + tabular transforms.
    
- `joblib`: serialize/deserialize Python objects (here, the trained XGBoost model).
    
- `XGBClassifier`: gradient‑boosted trees optimized for tabular classification.
    
- `classification_report`, `accuracy_score`: basic classification metrics (precision/recall/F1 + overall accuracy).
    
- `train_test_split`: imported but **not used** (you do a manual time‑based split below). Safe to remove.
    

---

## Globals & Directories

```python
TICKERS = ["AAPL", "TSLA", "MSFT", "BTC-USD"]
LABELED_DIR = pl.Path("data/labeled")
MODEL_DIR = pl.Path("models/xgboost")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
```

- `TICKERS`: batch training list; mix of equities and crypto.
    
- `LABELED_DIR`: expected location of labeled feature CSVs named `{TICKER}_labeled.csv`.
    
- `MODEL_DIR`: destination for saved models; created if missing.
    
- **Assumption:** each `{TICKER}_labeled.csv` contains at least `Date`, `Target`, and feature columns. Also assumes labeling already handled leakage by forward‑shifting targets.
    

---

## Train/Evaluate per Ticker

```python
def train_and_evaluate(ticker: str):
    print(f"📊 Training XGBoost for {ticker}...")
```

- Entry point for a single dataset. Simple console feedback.
    

### Load + Order Data

```python
    file_path = LABELED_DIR / f"{ticker}_labeled.csv"
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)  # Ensure chronological order
```

- Reads the labeled feature frame; parses `Date` to `datetime64`.
    
- Sort ensures **chronological order** (critical for time‑aware splits).
    
- **Pitfall:** If there are duplicate timestamps or timezone inconsistencies, they persist here. Consider de‑duping and normalizing timezones upstream.
    

### Missing Data Handling

```python
    # Drop missing values
    df.dropna(inplace=True)
```

- Drops **any** row with NaNs across **all** columns.
    
- **Pros:** keeps model input clean.
    
- **Cons:** can be overly aggressive (e.g., you might prefer imputing specific indicators or trimming only the warm‑up period for rolling features). Also, if class imbalance is present, this can distort the label distribution.
    

### Feature/Target Split

```python
    # Separate features and target
    X = df.drop(columns=["Date", "Target"])
    y = df["Target"]
```

- `X` contains all feature columns (anything not `Date`/`Target`).
    
- `y` is the label used for classification.
    
- **Leakage watch:** Ensure no **future‑looking** columns remain in `X` (e.g., `next_return_1d` or anything derived from future bars). Those should be used for backtesting PnL, not training features.
    

### Time‑based Split (Manual)

```python
    # Time-based train/test split: 80% train, 20% test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
```

- Simple chronological split: first 80% for training, last 20% for testing.
    
- **Good:** avoids shuffling/leakage across time.
    
- **Limitations:** single hold‑out; no cross‑validation. For time series, prefer **purged/embargoed K‑Fold** or **walk‑forward** to reduce overfitting risk and leakage via overlapping windows.
    

### Model Definition & Fit

```python
    # Train model
    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
```

- `n_estimators=100`: small forest; fast to train, may underfit on complex signals.
    
- `use_label_encoder=False`: suppresses deprecated label encoder behavior.
    
- `eval_metric="logloss"`: standard for probabilistic classification.
    
- **Defaults:** learning_rate=0.3, max_depth=6, subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0. Defaults can overfit; typically tune or at least reduce learning rate & raise `n_estimators` + use early stopping.
    
- **Upgrade path:** provide `eval_set=[(X_test,y_test)]` and `early_stopping_rounds` to auto‑stop; set a `random_state` for reproducibility; use `tree_method="hist"` by default (XGBoost may auto‑select).
    

### Evaluation

```python
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Test Accuracy for {ticker}: {acc:.4f}")
    print(classification_report(y_test, y_pred))
```

- Predicts class labels on hold‑out.
    
- Reports **accuracy** and **precision/recall/F1** per class.
    
- **Caveat (trading relevance):** These metrics ignore costs/asymmetry and do not guarantee profitability. Useful for sanity checks only. You’re already covering trading metrics in `backtest_cli.py`.
    

### Persist the Model

```python
    # Save model
    model_path = MODEL_DIR / f"{ticker}_xgb_model.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")
```

- Saves the estimator to disk for later inference.
    
- **Note:** Persisting with `joblib` is fine; XGBoost also supports `model.save_model()` (JSON/binary) which can be language‑agnostic. `joblib` keeps scikit wrapping (easier if you’ll reload in Python).
    

---

## Orchestration

```python
def main():
    for ticker in TICKERS:
        try:
            train_and_evaluate(ticker)
        except Exception as e:
            print(f"❌ Failed to train for {ticker}: {e}")

if __name__ == "__main__":
    main()
```

- Loops over tickers, trains/evaluates independently; catches exceptions to keep the batch running.
    
- **Idea:** accumulate per‑ticker metrics into a table/log for quick comparison.
    

---

## Key Assumptions You’re Relying On

1. **Labels are clean and leakage‑free.**  
    `Target` must be created with proper **forward shift** (e.g., triple‑barrier or sign of future return). All features must be based only on info available **up to t**.
    
2. **Class balance is reasonable.**  
    If positive/negative classes are imbalanced, accuracy becomes misleading. Consider class weights.
    
3. **Feature set is stable across time.**  
    No column drift (e.g., column added/removed mid‑history). Dropping NaNs would otherwise lop off chunks unevenly.
    

---

## Sharp Edges / Pitfalls

- **Over‑aggressive `dropna`**: you might unintentionally bias or shrink your test set; consider targeted imputations or drop only rows with NaNs in **required** features.
    
- **No CV / early stopping**: a single split can overfit or under‑represent regime diversity. Use **walk‑forward** or **purged K‑Fold** with **embargo**; add `early_stopping_rounds` with a validation set.
    
- **No `random_state`**: results won’t be reproducible across runs.
    
- **Class imbalance**: if `Target` is skewed (e.g., many 0/no‑trade), add `scale_pos_weight` or class weights; consider thresholding probabilities for trade vs. no‑trade.
    
- **Train/test leakage via rolling features**: ensure rolling windows don’t peek into the future; they shouldn’t, if computed correctly upstream.
    
- **Heterogeneous assets**: BTC vs equities have different trading days/vol regimes; a single model per ticker is fine, but keep this in mind if you ever move to multi‑asset models.
    

---

## Quick, High‑Impact Upgrades (drop‑in changes)

1. **Reproducibility + basic regularization**
    
    ```python
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
        early_stopping_rounds=50,
    )
    ```
    
    - Lower LR + more trees + early stopping typically improves generalization.
        
2. **Class imbalance handling**
    
    - Compute `scale_pos_weight = (neg/pos)` for binary labels, or use `class_weight`‑like logic (XGB doesn’t take sklearn’s `class_weight`, but you can supply sample weights).
        
    - Alternatively, **probability thresholding** at inference: trade only if `p(class=1) > τ`.
        
3. **Feature importance dump**  
    After training:
    
    ```python
    import numpy as np
    import json
    imp = model.get_booster().get_score(importance_type="gain")
    (MODEL_DIR / f"{ticker}_feature_importance.json").write_text(json.dumps(imp, indent=2))
    ```
    
    Helps you audit which features the model leans on.
    
4. **Persist training manifest** (like you did for backtests)  
    Save a small JSON with: ticker, time, rows, feature list, params, best_iteration, eval scores.
    
5. **Move to a `Pipeline`** (if you add scaling/selection)  
    Keep transforms + model bundled so inference uses the identical preprocessing steps.
    
6. **Multiple splits (walk‑forward)**  
    Instead of one 80/20 split, iterate sliding windows, average metrics, and keep the final model trained on the **most recent** window.
    

---

## Sanity Checks Before/After Training

- **Label balance:** `y.value_counts(normalize=True)`—check skew.
    
- **Leakage probe:** ensure no columns like `next_return_*` are in `X`; ensure all rolling features are `shift(1)`‑safe if needed.
    
- **Temporal coverage:** print `df["Date"].min(), df["Date"].max()` for train vs. test.
    
- **Feature drift:** confirm `X_train.columns.equals(X_test.columns)`.
    

---

## How This File Fits the Pipeline

- Consumes **labeled** data from `data/labeled/`.
    
- Produces a **model artifact** `{ticker}_xgb_model.pkl` that your `backtest_cli.py` loads to generate predictions and simulate P&L.
    
- Complements your research loop: data → features/labels → **train** → backtest → visualize.
    

---

## TL;DR

- The script trains an XGBoost classifier per ticker with a simple **time‑aware 80/20 split**, evaluates accuracy/F1, and saves the model.
    
- Watch for **leakage**, **class imbalance**, and **over‑aggressive NaN drops**.
    
- Add **early stopping**, **walk‑forward/purged CV**, **random_state**, and optionally **class weights**. Persist a **training manifest** and **feature importances** to make results reproducible and auditable.
    

Want me to apply those upgrades and hand you an **updated `train_model.py`** with early stopping, reproducibility, simple manifest, and optional class weighting?