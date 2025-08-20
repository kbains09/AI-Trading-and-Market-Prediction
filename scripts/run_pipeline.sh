#!/bin/bash

set -e  # Exit if any command fails

TICKER=${1:-AAPL}

echo "🔄 Running full pipeline for ticker: $TICKER"

# Step 1: Collect data
poetry run python src/algo_trader/data/collect_data.py --ticker "$TICKER"

# Step 2: Feature engineering
poetry run python src/algo_trader/features/engineer_features.py --ticker "$TICKER"

# Step 3: Train model
poetry run python src/algo_trader/models/train_model.py --ticker "$TICKER"

# Step 4: Backtest
poetry run python -m algo_trader.backtest.backtest_cli --ticker "$TICKER"

# Step 5: Plot results
poetry run python src/algo_trader/visualize/plot_backtest.py --ticker "$TICKER"

echo "✅ Pipeline completed for $TICKER"
