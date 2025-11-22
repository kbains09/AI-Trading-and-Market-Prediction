import typer
from .features import engineering_features as features_app
from .predict_next import predict as predict_command
from .backtest.backtest_cli import run_backtest as backtest_command

app = typer.Typer(add_completion=False)

# Sub-app for feature engineering
app.add_typer(features_app.app, name="features")

# Top-level commands
app.command("predict")(predict_command)
app.command("backtest")(backtest_command)

if __name__ == "__main__":
    app()
