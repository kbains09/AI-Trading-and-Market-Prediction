import typer
from .features import engineering_features

app = typer.Typer()
app.add_typer(engineering_features.app, name="features")

if __name__ == "__main__":
    app()
