# AI Algorithmic Trading & Market Prediction

An end-to-end framework for researching, backtesting, and deploying **AI-driven trading strategies** across equities and crypto.

This pipeline takes you from **raw OHLCV data → engineered features → labeled datasets → trained ML models → backtests → visual performance diagnostics**, with reproducibility guardrails built in.

---

## Highlights

- Full **quant research** workflow  
- Feature engineering: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, volatility, lags  
- **XGBoost model training** with time-aware splits  
- **Backtesting engine** with:
  - Transaction costs (bps)
  - Volatility targeting
  - Daily P&L breakdowns
- Automatic generation of:
  - Equity curve  
  - Drawdown curve  
  - Rolling Sharpe ratio  
- Backtest **manifest JSON** for full reproducibility  
- Clean project layout designed for scaling to multiple tickers & models  

---

## Quickstart (Example: AAPL)

Run the entire pipeline for Apple (AAPL). Adjust tickers as needed.

```bash
# 0) Install dependencies
poetry install

# 1) Collect raw data
poetry run python src/algo_trader/collect_data.py --ticker AAPL

# 2) Engineer features
poetry run python src/algo_trader/engineering_features.py run \
    --raw-dir data/raw \
    --out-dir data/processed

# 3) Label feature dataset
poetry run python src/algo_trader/label_features.py --ticker AAPL

# 4) Train ML model (XGBoost)
poetry run python src/algo_trader/train_model.py --ticker AAPL

# 5) Run backtest (with costs + vol targeting)
poetry run python src/algo_trader/backtest_cli.py run \
    --ticker AAPL \
    --cost-bps 5 \
    --vol-target 0.10

# 6) Generate performance plots
poetry run python src/algo_trader/plot_backtest.py \
    --ticker AAPL \
    --no-show \
    --save-prefix reports/plots/AAPL
```

Results & Visuals (AAPL Example)

These plots are auto-generated from your backtest results.

Equity Curve


Drawdown (Underwater Chart)

Rolling Sharpe Ratio

Sample Backtest Metrics (AAPL)
From:
data/backtests/AAPL_backtest.manifest.json

Metric	Example Value
CAGR	12.4%
Sharpe	1.31
Max Drawdown	-9.8%
Hit Rate	54.2%
Turnover	135%

Values vary depending on data range, parameters, and model output.

Reproducibility via Backtest Manifest
Every backtest produces a JSON manifest file containing:

Parameters used

Cost assumptions

Vol targeting settings

Dataset hashes

Model artifact paths

Sharpe, CAGR, MaxDD, Hit Rate

Timestamps + unique run identifiers

Example:

bash
Copy code
data/backtests/AAPL_backtest.manifest.json
This ensures results are fully reproducible, auditable, and traceable.

Project Structure
Your real repo layout:

graphql
Copy code
.
├── config/                      # Optional settings
├── data/
│   ├── raw/                     # Raw vendor OHLCV dumps
│   ├── processed/               # Feature-engineered datasets
│   ├── labeled/                 # Training-ready labeled data
│   └── backtests/               # Backtest results + manifest files
├── devnotes/                    # Notes / drafts / experiments
├── models/
│   └── xgboost/                 # Saved XGBoost models
├── reports/
│   └── plots/                   # Auto-generated PNG performance charts
├── scripts/                     # Utility shell scripts
├── src/
│   └── algo_trader/
│       ├── collect_data.py
│       ├── engineering_features.py
│       ├── label_features.py
│       ├── train_model.py
│       ├── backtest_cli.py
│       └── plot_backtest.py
├── tests/                       # Test suite
├── pyproject.toml
└── README.md

Pipeline Overview
Data Collection
Uses yfinance

Cleans/sorts OHLCV

Saves to:
data/raw/{TICKER}.csv

Feature Engineering
Computes:

SMA / EMA

RSI / MACD

Bollinger Bands

ATR

Realized volatility

Lagged features

Saves to:
data/processed/{TICKER}_features.csv

Labeling
Adds:

Direction label (Target)

next_return_1d (forward return – NOT used for training)

Saves to:
data/labeled/{TICKER}_labeled.csv

Model Training
Chronological split

XGBoost baseline

Removes leakage features (next_return_1d)

Saves model to:
models/xgboost/{TICKER}_xgb_model.pkl

Backtesting
Generates signals

Applies:

Transaction costs

Volatility targeting

Outputs:

backtests/{TICKER}_backtest.csv

backtests/{TICKER}_backtest.manifest.json

Visualization
Produces:

Equity Curve

Drawdown Chart

Rolling Sharpe Ratio

Saved to:
reports/plots/{TICKER}_*.png

Setup
bash
Copy code
poetry install
poetry run python -V
Run any script via:

bash
Copy code
poetry run python <script>.py

Guardrails & Best Practices
Avoid Leakage
Never include next_return_1d as a training feature

Ensure rolling windows do not peek ahead

Time-Aware Splits
No random splits

Always chronological

Costs Matter
Use --cost-bps

High turnover can eliminate edge

Volatility Targeting
Stabilizes risk

Essential for fair model comparison

Class Imbalance
Use:

class weights

threshold tuning

probability calibration

Manifest = Truth
Treat *_manifest.json as your official experiment record

Roadmap (Future Enhancements)
Walk-forward cross-validation

Triple-barrier & meta-labeling

Multi-asset / portfolio optimization

MLflow or W&B experiment tracking

Real-time inference API (FastAPI)

Broker integration (Alpaca, Binance)

LSTM / Transformer models for sequence prediction

References
Marcos López de Prado — Advances in Financial Machine Learning

Ernest P. Chan — Machine Trading

Stefan Jansen — Machine Learning for Algorithmic Trading

Bacidore — Algorithmic Trading with Python

