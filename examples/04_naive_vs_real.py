"""Same signal, three backtests: how much of a naive backtest's return is fiction.

    python examples/04_naive_vs_real.py bars.parquet [out.png]

`bars.parquet` is your own daily data, one row per stock per day, with columns
date, code, open, close, preclose, ret, amount, tradable and optionally
limit_width (default 0.10). `ret` is the total return (baostock's pctChg / 100),
`preclose` the adjusted previous close. Without a file it runs on synthetic bars.

The signal is the textbook 5-day reversal: every 5 trading days buy the 50
stocks that fell the most over the past 5 days, equal weight.

  naive          fills at the signal day's CLOSE, no costs, no limits, no T+1
  naive + costs  the same, minus the default round-trip cost on each turnover
  ashare-bt      next-open fills, limit-up/down gates, T+1, 100-share lots,
                 the CNY 5 minimum commission, participation cap, delisting
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ashare_bt import BacktestConfig, CostModel, run_backtest
from ashare_bt.sizing import limit_prices
from ashare_bt.synthetic import make_market

N, HOLD, CAPITAL = 50, 5, 1_000_000


def load(path: str | None) -> pd.DataFrame:
    if path is None:
        return make_market(n_stocks=300, n_days=750, seed=5)
    b = pd.read_parquet(path)
    b["date"] = pd.to_datetime(b["date"])
    width = b["limit_width"].astype(float) if "limit_width" in b else 0.10
    up, dn = limit_prices(b["preclose"].astype(float), width)
    b["open_limit_up"] = (b["open"] >= up) | b["preclose"].isna()
    b["open_limit_down"] = (b["open"] <= dn) | b["preclose"].isna()
    b["ret"] = b["ret"].fillna(0.0)
    return b


def reversal_signal(bars: pd.DataFrame) -> pd.DataFrame:
    b = bars.sort_values(["code", "date"])
    lr = np.log1p(b["ret"].astype(float))
    ret5 = np.expm1(lr.groupby(b["code"]).transform(lambda s: s.rolling(5, min_periods=5).sum()))
    sig = b[["date", "code"]].assign(score=-ret5).dropna()
    return sig.sort_values(["date", "code"]).reset_index(drop=True)


def naive(signal: pd.DataFrame, bars: pd.DataFrame, round_trip: float) -> pd.Series:
    """Close-to-close book: buy at the signal close, hold HOLD days, pay `round_trip` per turnover."""
    idx = (1 + bars.pivot(index="date", columns="code", values="ret")).cumprod().ffill()
    days = [d for d in sorted(signal["date"].unique()) if d in idx.index][::HOLD]
    by_day = dict(tuple(signal.groupby("date")))
    eq, prev, out = 1.0, set(), {days[0]: 1.0}
    for t, t_next in zip(days[:-1], days[1:]):
        picks = by_day[t].nlargest(N, "score")["code"].tolist()
        r = float((idx.loc[t_next, picks] / idx.loc[t, picks]).mean() - 1)
        turnover = 1 - len(prev & set(picks)) / N
        eq *= (1 + r) * (1 - turnover * round_trip)
        prev = set(picks)
        out[t_next] = eq
    return pd.Series(out)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else None
    bars = load(path)
    sig = reversal_signal(bars)
    rt = CostModel().round_trip_bps(1e6) / 1e4
    curves = {
        "naive": naive(sig, bars, 0.0),
        "naive + costs": naive(sig, bars, rt),
        "ashare-bt": run_backtest(sig, bars, BacktestConfig(capital=CAPITAL, n_positions=N,
                                                            rebalance_days=HOLD)).equity / CAPITAL,
    }
    for name, c in curves.items():
        yrs = (c.index[-1] - c.index[0]).days / 365.25
        print(f"{name:>14}: x{c.iloc[-1]:8.2f}   CAGR {c.iloc[-1] ** (1 / yrs) - 1:+7.1%}")
    if out:
        plot(curves, out)


def plot(curves: dict[str, pd.Series], out: str) -> None:
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for mode, surface, ink, muted, colors in (
        ("light", "#fcfcfb", "#1f1f1e", "#6b6a63", ("#eb6834", "#eda100", "#2a78d6")),
        ("dark", "#1a1a19", "#ffffff", "#c3c2b7", ("#d95926", "#c98500", "#3987e5")),
    ):
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
        fig.patch.set_facecolor(surface)
        ax.set_facecolor(surface)
        for (name, c), color, style in zip(curves.items(), colors, ("--", ":", "-")):
            ax.plot(c.index, c.values, color=color, lw=2, ls=style, solid_capstyle="round")
            ax.annotate(f"{name}  x{c.iloc[-1]:.1f}", (c.index[-1], c.iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=ink, fontsize=10)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel("growth of 1 (log scale)", color=muted)
        ax.tick_params(colors=muted)
        ax.grid(axis="y", color=muted, alpha=0.15)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("Same 5-day reversal signal, three backtests", loc="left", color=ink, fontsize=13)
        fig.tight_layout()
        target = out if mode == "light" else out.replace(".png", "-dark.png")
        fig.savefig(target, facecolor=surface, bbox_inches="tight")
        plt.close(fig)
        print("wrote", target)


if __name__ == "__main__":
    main()
