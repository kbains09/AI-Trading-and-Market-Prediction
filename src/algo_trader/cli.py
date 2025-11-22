import typer

from .predict_next import app as predict_app
from .features import engineering_features as features_app
from .backtest import backtest_cli as backtest_app
from .backtest import plot_backtest as plot_backtest_app
from .backtest import full_report as full_report_app
from .live import paper_trader as paper_trading_app

app = typer.Typer(add_completion=False)

# Feature engineering
app.add_typer(features_app.app, name="features")

# Prediction (regime + 5D direction on latest bar)
app.add_typer(predict_app, name="predict")

# Backtesting (PnL, manifests, stats)
app.add_typer(backtest_app.app, name="backtest")

# Visualization / reporting (plots from backtest CSV)
app.add_typer(plot_backtest_app.app, name="plot-backtest")

# Full report (backtest + plots + summary paths)
app.add_typer(full_report_app.app, name="full-report")

# Paper trading (simulate trades based on predicted probabilities)
app.add_typer(paper_trading_app.app, name="paper")


if __name__ == "__main__":
    app()
