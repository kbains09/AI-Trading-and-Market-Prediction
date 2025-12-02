# src/algo_trader/plots/test.py

from pathlib import Path

from algo_trader.plots.feature_importance import plot_feature_importance
from algo_trader.features.feature_set import LIVE_FEATURE_COLUMNS

MODELS_DIR = Path("models/xgboost")
OUT_AAPL = Path("plots/AAPL")

if __name__ == "__main__":
    OUT_AAPL.mkdir(parents=True, exist_ok=True)

    # AAPL direction:
    plot_feature_importance(
        MODELS_DIR / "AAPL_direction_model.pkl",
        out_dir=OUT_AAPL,
        title="AAPL Direction Model – Feature Importance",
        feature_names=LIVE_FEATURE_COLUMNS,  # aligns f0,f1,... to features
    )

    # AAPL regime:
    plot_feature_importance(
        MODELS_DIR / "AAPL_regime_model.pkl",
        out_dir=OUT_AAPL,
        title="AAPL Regime Model – Feature Importance",
        feature_names=LIVE_FEATURE_COLUMNS,
    )
