"""Compare a signal at different capital levels and buying caps.

Larger orders incur more modelled impact. The buying cap can also leave
cash uninvested, so the sweep reports deployed capital alongside returns
and costs. Rankings and all other configuration stay the same.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import pandas as pd

from .engine import BacktestConfig, run_backtest

log = logging.getLogger("capacity")

DEFAULT_CAPITALS = [100_000, 300_000, 1_000_000, 3_000_000, 10_000_000, 30_000_000]
DEFAULT_PARTICIPATIONS = [0.01, 0.02, 0.05, 0.10]


@dataclass
class CapacityPoint:
    capital: float
    participation: float
    cagr: float
    sharpe: float
    max_dd: float
    turnover: float
    cost_drag: float
    avg_positions: float
    deployed_frac: float   # how much of the capital actually got invested


def sweep(signal: pd.DataFrame, bars: pd.DataFrame,
          capitals: list[float] | None = None,
          participations: list[float] | None = None,
          base: BacktestConfig | None = None) -> pd.DataFrame:
    capitals = capitals or DEFAULT_CAPITALS
    participations = participations or DEFAULT_PARTICIPATIONS
    base = base or BacktestConfig()
    rows: list[CapacityPoint] = []
    for part in participations:
        for cap in capitals:
            # Preserve all base settings except capital and participation.
            cfg = replace(base, capital=cap, max_participation=part)
            res = run_backtest(signal, bars, cfg)
            m = res.metrics
            # Report invested capital as well as returns.
            deployed = 1.0 - (res.daily["cash"].mean() / max(res.equity.mean(), 1))
            rows.append(CapacityPoint(
                capital=cap, participation=part, cagr=m["cagr"], sharpe=m["sharpe"],
                max_dd=m["max_dd"], turnover=m["turnover"], cost_drag=m["cost_drag"],
                avg_positions=m["avg_positions"], deployed_frac=deployed,
            ))
            log.info("capital=%-12.0f part=%.0f%% CAGR=%+.1f%% Sharpe=%.2f deployed=%.0f%%",
                     cap, part * 100, m["cagr"] * 100, m["sharpe"], deployed * 100)
    return pd.DataFrame([r.__dict__ for r in rows])


def render(df: pd.DataFrame) -> str:
    """Format sweep results and show where Sharpe drops below half its starting value."""
    if df.empty:
        return "  (no capacity points computed)"
    out = ["=" * 84, "CAPACITY / ALPHA DECAY", "=" * 84,
           f"  {'participation':>13} {'capital':>12} {'CAGR':>8} {'Sharpe':>7} "
           f"{'maxDD':>7} {'turn':>6} {'cost/yr':>8} {'deployed':>9}"]
    for part, g in df.groupby("participation"):
        out.append("")
        for r in g.sort_values("capital").itertuples():
            out.append(f"  {r.participation:>12.0%} {r.capital:>12,.0f} {r.cagr:>7.1%} "
                       f"{r.sharpe:>7.2f} {r.max_dd:>7.1%} {r.turnover:>5.1f}x "
                       f"{r.cost_drag:>7.2%} {r.deployed_frac:>8.0%}")
    out.append("")
    out.append("  Capital at which Sharpe falls below half its small-size value:")
    for part, g in df.groupby("participation"):
        g = g.sort_values("capital")
        best = g["sharpe"].iloc[0]
        if best <= 0:
            out.append(f"    participation {part:.0%}: no positive Sharpe at any size")
            continue
        breach = g[g["sharpe"] < best / 2]
        where = f"{breach['capital'].iloc[0]:,.0f}" if len(breach) else "above the tested range"
        out.append(f"    participation {part:.0%}: ¥{where}  (small-size Sharpe {best:.2f})")
    out.append("=" * 84)
    return "\n".join(out)
