# AI Algorithmic Trading & Market Prediction

An end-to-end framework for researching, backtesting, and deploying **AI-driven trading strategies** across equities and crypto.  
The pipeline takes you from **raw OHLCV data → engineered features → labeled datasets → trained models → backtests → diagnostic plots**, with reproducibility guardrails built-in.

---

## Quickstart Example (Apple AAPL)

This runs the full pipeline for Apple. Adjust tickers as you like.

```bash
# 0) Install dependencies (once)
poetry install

# 1) Collect raw data → data/raw/AAPL.csv
poetry run python src/algo_trader/collect_data.py

# 2) Engineer features → data/processed/AAPL_features.csv
poetry run python src/algo_trader/engineering_features.py run \
  --raw-dir data/raw \
  --out-dir data/processed

# 3) Label features → data/labeled/AAPL_labeled.csv
poetry run python src/algo_trader/label_features.py

# 4) Train a model → models/xgboost/AAPL_xgb_model.pkl
# (Trainer must drop `next_return_1d` from features to avoid leakage)
poetry run python src/algo_trader/train_model.py

# 5) Backtest → data/backtests/AAPL_backtest.csv + manifest JSON
poetry run python src/algo_trader/backtest_cli.py run \
  --ticker AAPL \
  --cost-bps 5 \
  --vol-target 0.10

# 6) Visualize results (saves plots headless if --no-show)
poetry run python src/algo_trader/plot_backtest.py \
  --ticker AAPL \
  --no-show \
  --save-prefix reports/plots/AAPL
````

### Generated Artifacts

- `data/raw/AAPL.csv` — raw OHLCV
    
- `data/processed/AAPL_features.csv` — engineered features
    
- `data/labeled/AAPL_labeled.csv` — features + `Target`
    
- `models/xgboost/AAPL_xgb_model.pkl` — trained model
    
- `data/backtests/AAPL_backtest.csv` — per-bar results
    
- `data/backtests/AAPL_backtest.manifest.json` — metadata + metrics
    
- `reports/plots/AAPL_equity.png`, `_drawdown.png`, `_rollsharpe.png`
    

---

## Project Structure

```
.
├── data/
│   ├── raw/                 # Raw vendor OHLCV dumps
│   ├── processed/           # Feature-engineered datasets
│   └── labeled/             # Training-ready labeled data
├── models/
│   └── xgboost/             # Trained model artifacts
├── reports/
│   └── plots/               # Visualization outputs
├── src/algo_trader/
│   ├── collect_data.py          # Fetch + clean raw OHLCV
│   ├── engineering_features.py  # Build indicators/features
│   ├── label_features.py        # Add forward-looking targets
│   ├── train_model.py           # Train ML model (XGBoost baseline)
│   ├── backtest_cli.py          # Backtest strategy with costs
│   └── plot_backtest.py         # Equity, drawdown, Sharpe plots
├── pyproject.toml
└── README.md
```

---

## Pipeline Overview

1. **Data Collection**
    
    - Pulls OHLCV from APIs (`yfinance` etc).
        
    - Cleans, sorts, deduplicates, coercing numeric types.
        
    - Saves **atomically** to `data/raw/{TICKER}.csv`.
        
2. **Feature Engineering**
    
    - Computes indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, volatility).
        
    - Adds lagged returns/features for leakage safety.
        
    - Saves to `data/processed/{TICKER}_features.csv`.
        
3. **Labeling**
    
    - Adds `Target` column (binary ↑/↓ or directional -1/0/1).
        
    - Also includes `next_return_1d` (forward return, used for evaluation not training).
        
    - Outputs to `data/labeled/{TICKER}_labeled.csv`.
        
4. **Model Training**
    
    - Time-aware train/validation split (chronological).
        
    - Baseline: XGBoost classifier/regressor.
        
    - Drops `next_return_1d` to prevent look-ahead bias.
        
    - Saves model to `models/xgboost/{TICKER}_xgb_model.pkl`.
        
5. **Backtesting**
    
    - Generates signals, applies **transaction costs** + **vol targeting**.
        
    - Computes metrics: CAGR, Sharpe, Max Drawdown, Turnover, Hit Rate.
        
    - Saves per-bar results (`_backtest.csv`) and manifest JSON (metadata, hashes, params).
        
6. **Visualization**
    
    - Equity curve vs benchmark.
        
    - Underwater (drawdown) chart.
        
    - Rolling Sharpe ratio.
        
    - Saved as PNGs under `reports/plots/`.
        

---

## Setup

- **Python**: 3.10+
    
- **Poetry**: dependency management
    

```bash
# Install dependencies
poetry install

# Confirm
poetry run python -V
```

---

## Guardrails & Gotchas

- **Feature/label alignment**: predict _t+1_ using features at _t_.
    
- **Avoid leakage**: never include `next_return_1d` in training features.
    
- **Costs matter**: use `--cost-bps` in backtest (realized P&L depends on turnover).
    
- **Vol targeting**: normalize position sizing to achieve stable risk.
    
- **Class imbalance**: binary targets often skewed — consider weights or thresholds.
    
- **Reproducibility**: backtest manifest JSON stores params, hashes, metrics.
    

---

## Future Extensions

- Walk-forward CV + early stopping.
    
- Advanced labeling: triple-barrier, meta-labeling.
    
- Multi-asset / portfolio backtests.
    
- MLflow or Weights & Biases for experiment tracking.
    
- Paper/live trading via broker APIs (Alpaca, Binance).
    

---

## References

- López de Prado — _Advances in Financial Machine Learning_
    
- Ernest Chan — _Machine Trading_
    
- Bacidore — _Algorithmic Trading with Python_
    
