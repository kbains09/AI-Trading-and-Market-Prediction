
# AI Algorithmic Trading & Market Prediction

An end-to-end framework for researching, backtesting, and deploying **AI-driven trading strategies** across equities and crypto.

The pipeline takes you from **raw OHLCV data → engineered features → labeled datasets → trained models → backtests → diagnostic plots**, with reproducibility guardrails built in.

---

## Highlights

- End-to-end **quant research pipeline** (data → features → labels → models → backtests → plots)
- **XGBoost** baseline models trained on engineered features
- Full **backtesting engine** with transaction costs + volatility targeting
- Auto-generated performance metrics (CAGR, Sharpe, Max Drawdown, Turnover, Hit Rate)
- Visualization module for **equity curve**, **drawdown**, and **rolling Sharpe**
- Reproducible runs via backtest **manifest JSON** (params, hashes, metrics)

---

## Quickstart Example (AAPL)

This runs the full pipeline for Apple (AAPL). Adjust tickers as needed.

```bash
# 0) Install dependencies (once)
poetry install

# 1) Collect raw data → data/raw/AAPL.csv
poetry run python src/algo_trader/collect_data.py --ticker AAPL

# 2) Engineer features → data/processed/AAPL_features.csv
poetry run python src/algo_trader/engineering_features.py run \
  --raw-dir data/raw \
  --out-dir data/processed

# 3) Label features → data/labeled/AAPL_labeled.csv
poetry run python src/algo_trader/label_features.py --ticker AAPL

# 4) Train a model → models/xgboost/AAPL_xgb_model.pkl
# (Trainer must drop `next_return_1d` from features to avoid leakage)
poetry run python src/algo_trader/train_model.py --ticker AAPL

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
```

---

## Results & Visuals (AAPL Example)

These charts are generated automatically from backtest results.

### **Equity Curve**
![AAPL Equity Curve](data/backtests/plots/AAPL_equity.png)

---

### **Drawdown (Underwater Chart)**
![AAPL Drawdown](data/backtests/plots/AAPL_underwater.png)

---

### **Rolling Sharpe Ratio**
![AAPL Rolling Sharpe](data/backtests/plots/AAPL_rolling_sharpe.png)

---

##  Sample Backtest Metrics (AAPL)

Example from the manifest:

```
File: data/backtests/AAPL_backtest.manifest.json
```

| Metric        | Value (Example) |
|---------------|------------------|
| CAGR          | 12.4%            |
| Sharpe        | 1.31             |
| Max Drawdown  | -9.8%            |
| Hit Rate      | 54.2%            |
| Turnover      | 135%             |

---

## Reproducibility: Backtest Manifest

Every backtest generates a manifest JSON recording:

- Strategy + backtest parameters  
- Train/validation splits  
- Dataset hashes  
- Model artifact paths  
- Key performance metrics  
- Timestamps + run ID  

Example:

```
data/backtests/AAPL_backtest.manifest.json
```

---

## Project Structure

```
.
├── data/
│   ├── raw/
│   ├── processed/
│   ├── labeled/
│   └── backtests/
├── models/
│   └── xgboost/
├── reports/
│   └── plots/
├── src/
│   └── algo_trader/
│       ├── collect_data.py
│       ├── engineering_features.py
│       ├── label_features.py
│       ├── train_model.py
│       ├── backtest_cli.py
│       └── plot_backtest.py
└── pyproject.toml
```

---

## 🔄 Pipeline Overview

### **1. Data Collection**

- Pull OHLCV from APIs (e.g., yfinance)
- Clean + normalize
- Save to:

```
data/raw/{TICKER}.csv
```

---

### **2. Feature Engineering**

- SMA, EMA  
- RSI, MACD  
- Bollinger Bands  
- ATR, realized volatility  
- Lagged features  
- Saves to:

```
data/processed/{TICKER}_features.csv
```

---

### **3. Labeling**

- Binary / directional label (↑/↓ or -1/0/1)
- Adds `next_return_1d` (not used for training)
- Saves to:

```
data/labeled/{TICKER}_labeled.csv
```

---

### **4. Model Training**

- Chronological train/validation split  
- XGBoost baseline model  
- Drops leakage columns (`next_return_1d`)
- Outputs:

```
models/xgboost/{TICKER}_xgb_model.pkl
```

---

### **5. Backtesting**

- Predict signals  
- Apply transaction costs + vol targeting  
- Compute Sharpe, CAGR, MaxDD, Turnover  
- Outputs:

```
data/backtests/{TICKER}_backtest.csv
data/backtests/{TICKER}_backtest.manifest.json
```

---

### **6. Visualization**

- Equity curve  
- Drawdown  
- Rolling Sharpe  
- Plots saved to:

```
reports/plots/{TICKER}_equity.png
reports/plots/{TICKER}_drawdown.png
reports/plots/{TICKER}_rollsharpe.png
```

---

## 🛠 Setup

```bash
poetry install
poetry run python -V
```

---

## ⚠️ Guardrails & Gotchas

- Predict **t+1** using **t** features  
- Never train on `next_return_1d`  
- Costs can destroy edge  
- Vol-targeting stabilizes risk  
- Watch out for class imbalance  
- Manifest JSONs track exactly what was run  

---

## 🛣️ Roadmap

- Walk‑forward CV  
- Triple-barrier labeling  
- Multi-asset portfolio  
- MLflow / W&B tracking  
- Real-time FastAPI inference API  
- Live trading via Alpaca/Binance  
- Transformers/LSTMs  

---

## 📚 References

- López de Prado — *Advances in Financial Machine Learning*  
- Ernest Chan — *Machine Trading*  
- Stefan Jansen — *Machine Learning for Algorithmic Trading*  
- Bacidore — *Algorithmic Trading with Python*
