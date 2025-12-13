from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

import typer

app = typer.Typer(add_completion=False)


@dataclass
class BacktestRow:
    ticker: str
    strategy_return: float | None
    market_return: float | None
    cagr: float | None
    sharpe: float | None
    max_drawdown: float | None
    hit_rate: float | None
    turnover: float | None
    cost_bps: float | None
    vol_target: float | None

    @classmethod
    def from_manifest(cls, ticker: str, manifest: Dict[str, Any]) -> "BacktestRow":
        """
        Map manifest JSON → BacktestRow.

        If your manifest uses slightly different keys, just tweak
        the get(...) calls below.
        """
        return cls(
            ticker=ticker,
            strategy_return=manifest.get("strategy_return"),
            market_return=manifest.get("market_return"),
            cagr=manifest.get("cagr"),
            sharpe=manifest.get("sharpe"),
            max_drawdown=manifest.get("max_drawdown"),
            hit_rate=manifest.get("hit_rate"),
            turnover=manifest.get("turnover"),
            cost_bps=manifest.get("cost_bps"),
            vol_target=manifest.get("vol_target"),
        )


def _project_paths() -> Tuple[Path, Path, Path]:
    """
    Reuse the same idea as full_report: resolve project root
    relative to this file.

    .../src/algo_trader/reports/backtest_summary.py
    parents[0] = reports
    parents[1] = algo_trader
    parents[2] = src
    parents[3] = repo root
    """
    root = Path(__file__).resolve().parents[3]
    backtest_dir = root / "data" / "backtests"
    reports_dir = root / "reports"
    return root, backtest_dir, reports_dir


def _load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_manifests(backtest_dir: Path) -> List[Path]:
    """
    Find all backtest manifest JSONs:
      e.g. AAPL_backtest.manifest.json
    """
    return sorted(backtest_dir.glob("*_backtest.manifest.json"))


def _ticker_from_filename(path: Path) -> str:
    """
    Extract ticker from filename like:
      'AAPL_backtest.manifest.json' -> 'AAPL'
      'BTC-USD_backtest.manifest.json' -> 'BTC-USD'
    """
    name = path.name
    return name.split("_backtest", 1)[0]


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


@app.command()
def main(
    include_header: bool = typer.Option(
        True,
        "--include-header/--no-header",
        help="Include header row when printing table to stdout.",
    ),
) -> None:
    """
    Aggregate all backtest manifests into:

      - reports/backtest_summary.csv
      - reports/backtest_summary.md

    and print a compact table to stdout.
    """
    root, backtest_dir, reports_dir = _project_paths()
    reports_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"📁 Project root:      {root}")
    typer.echo(f"📁 Backtests dir:     {backtest_dir}")
    typer.echo(f"📁 Reports output dir:{reports_dir}")
    typer.echo("")

    manifest_files = _find_manifests(backtest_dir)
    if not manifest_files:
        raise SystemExit(
            f"❌ No backtest manifests found in {backtest_dir} "
            "(expected '*_backtest.manifest.json')."
        )

    rows: List[BacktestRow] = []
    for mf in manifest_files:
        ticker = _ticker_from_filename(mf)
        manifest = _load_manifest(mf)
        row = BacktestRow.from_manifest(ticker, manifest)
        rows.append(row)

    # --------------------------
    # 1) Write CSV summary
    # --------------------------
    csv_path = reports_dir / "backtest_summary.csv"
    fieldnames = list(asdict(rows[0]).keys())

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    # --------------------------
    # 2) Write Markdown summary
    # --------------------------
    md_path = reports_dir / "backtest_summary.md"
    md_lines: List[str] = []

    md_lines.append("# Backtest Summary\n")
    md_lines.append(
        "| Ticker | Strategy Return (x) | Market Return (x) | CAGR | Sharpe | MaxDD | Hit rate | Turnover | Cost (bps) | Vol target |"
    )
    md_lines.append(
        "|--------|---------------------|-------------------|------|--------|-------|----------|----------|-----------|------------|"
    )

    for r in rows:
        md_lines.append(
            "| {ticker} | {sr} | {mr} | {cagr} | {sharpe} | {maxdd} | {hit} | {turn} | {cost} | {vol} |".format(
                ticker=r.ticker,
                sr=_format_float(r.strategy_return, 4),
                mr=_format_float(r.market_return, 4),
                cagr=_format_float(r.cagr, 4),
                sharpe=_format_float(r.sharpe, 2),
                maxdd=_format_float(r.max_drawdown, 2),
                hit=_format_float(r.hit_rate, 2),
                turn=_format_float(r.turnover, 2),
                cost=_format_float(r.cost_bps, 1),
                vol=_format_float(r.vol_target, 2),
            )
        )

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # --------------------------
    # 3) Print compact table to stdout
    # --------------------------
    typer.echo("Backtest summary:\n")

    if include_header:
        typer.echo(
            f"{'Ticker':<8} {'StratRet':>9} {'MktRet':>9} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} {'Hit%':>7}"
        )
        typer.echo("-" * 60)

    for r in rows:
        typer.echo(
            f"{r.ticker:<8} "
            f"{_format_float(r.strategy_return, 4):>9} "
            f"{_format_float(r.market_return, 4):>9} "
            f"{_format_float(r.cagr, 4):>7} "
            f"{_format_float(r.sharpe, 2):>7} "
            f"{_format_float(r.max_drawdown, 2):>7} "
            f"{_format_float(r.hit_rate, 2):>7}"
        )

    typer.echo("")
    typer.echo(f"✅ CSV summary written to: {csv_path}")
    typer.echo(f"✅ Markdown summary written to: {md_path}")


if __name__ == "__main__":
    app()
