"""用涨停和反转信号，比较收盘成交与下一开盘成交。

    python examples/04_naive_vs_real.py [bars.parquet] [--out chart.png] [--signal limit_up|reversal]

不传文件时使用合成行情。parquet 文件需要 date、code、open、close、
preclose、ret、amount、tradable；可选 limit_width 设置涨跌停幅度，默认 0.10。
ret 是日总收益，preclose 是当日除权口径的昨收。
读取 parquet 需要 pyarrow 等引擎，保存图表需要 matplotlib。

limit_up：选收盘涨停的股票，按成交额排名。
reversal：从成交额排名前 800 的股票中，选过去 5 日跌幅最大的。
两者都会排除 load() 根据涨跌停幅度标记的 ST 股票。
每只入选股票的目标仓位为 1/50，剩余资金留现金。

比较三种口径：当天收盘成交、收盘成交加固定费用、ashare-bt 下一开盘成交。
成交时点和多项约束同时变化，曲线差距不能归因于某一条规则。
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ashare_bt import BacktestConfig, CostModel, run_backtest
from ashare_bt.sizing import limit_prices
from ashare_bt.synthetic import make_market

N, CAPITAL, LIQUID = 50, 1_000_000, 800


def load(path: str | None) -> pd.DataFrame:
    b = make_market(n_stocks=300, n_days=750, seed=5) if path is None else pd.read_parquet(path)
    b["date"] = pd.to_datetime(b["date"])
    width = b["limit_width"].astype(float) if "limit_width" in b else 0.10
    up, dn = limit_prices(b["preclose"].astype(float), width)
    b["open_limit_up"] = (b["open"] >= up) | b["preclose"].isna()
    b["open_limit_down"] = (b["open"] <= dn) | b["preclose"].isna()
    b["close_limit_up"] = b["close"] >= up
    b["st"] = width < 0.06 if "limit_width" in b else False
    b["ret"] = b["ret"].fillna(0.0)
    return b


def limit_up_signal(bars: pd.DataFrame) -> pd.DataFrame:
    """选可交易、非 ST、收盘涨停的股票，按成交额排名。"""
    hit = bars[bars["close_limit_up"] & bars["tradable"] & ~bars["st"]]
    return hit[["date", "code"]].assign(score=hit["amount"].astype(float)).reset_index(drop=True)


def reversal_signal(bars: pd.DataFrame) -> pd.DataFrame:
    """在 LIQUID 只高成交额非 ST 股票中，用负的 5 日收益打分。"""
    b = bars.sort_values(["code", "date"])
    lr = np.log1p(b["ret"].astype(float))
    ret5 = np.expm1(lr.groupby(b["code"]).transform(lambda s: s.rolling(5, min_periods=5).sum()))
    amt20 = b.groupby("code")["amount"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    sig = b[["date", "code"]].assign(score=-ret5, amt20=amt20.where(~b["st"])).dropna()
    sig = sig[sig.groupby("date")["amt20"].rank(ascending=False) <= LIQUID]
    return sig.drop(columns="amt20").sort_values(["date", "code"]).reset_index(drop=True)


SIGNALS = {"limit_up": (limit_up_signal, 1), "reversal": (reversal_signal, 5)}


def naive(signal: pd.DataFrame, bars: pd.DataFrame, hold: int, round_trip: float) -> pd.Series:
    """信号日收盘买入，持有 hold 天；每只占 1/N，其余留现金。"""
    idx = (1 + bars.pivot(index="date", columns="code", values="ret")).cumprod().ffill()
    days = list(idx.index[idx.index >= signal["date"].min()])[::hold]
    by_day = dict(tuple(signal.groupby("date")))
    eq, prev, out = 1.0, set(), {days[0]: 1.0}
    for t, t_next in zip(days[:-1], days[1:]):
        picks = by_day[t].nlargest(N, "score")["code"].tolist() if t in by_day else []
        r = float((idx.loc[t_next, picks] / idx.loc[t, picks] - 1).sum() / N) if picks else 0.0
        bought = len(set(picks) - prev) / N
        eq *= (1 + r) * (1 - bought * round_trip)
        prev = set(picks)
        out[t_next] = eq
    return pd.Series(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bars", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--signal", choices=sorted(SIGNALS), default="limit_up")
    a = ap.parse_args()
    bars = load(a.bars)
    make, hold = SIGNALS[a.signal]
    sig = make(bars)
    rt = CostModel().round_trip_bps(1e6) / 1e4
    curves = {
        "naive": naive(sig, bars, hold, 0.0),
        "naive + costs": naive(sig, bars, hold, rt),
        "ashare-bt": run_backtest(sig, bars, BacktestConfig(capital=CAPITAL, n_positions=N,
                                                            rebalance_days=hold)).equity / CAPITAL,
    }
    for name, c in curves.items():
        yrs = (c.index[-1] - c.index[0]).days / 365.25
        print(f"{name:>14}: x{c.iloc[-1]:8.2f}   CAGR {c.iloc[-1] ** (1 / yrs) - 1:+7.1%}")
    if a.out:
        plot(curves, a.out, TITLES[a.signal])


TITLES = {"limit_up": "Buy every stock that closes limit-up: three backtests",
          "reversal": "Same 5-day reversal signal, three backtests"}


def plot(curves: dict[str, pd.Series], out: str, title: str) -> None:
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
            yrs = (c.index[-1] - c.index[0]).days / 365.25
            ax.annotate(f"{name}  {c.iloc[-1] ** (1 / yrs) - 1:+.0%} a year", (c.index[-1], c.iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=ink, fontsize=10)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel("growth of 1 (log scale)", color=muted)
        ax.tick_params(colors=muted)
        ax.grid(axis="y", color=muted, alpha=0.15)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc="left", color=ink, fontsize=13)
        fig.tight_layout()
        target = out if mode == "light" else out.replace(".png", "-dark.png")
        fig.savefig(target, facecolor=surface, bbox_inches="tight")
        plt.close(fig)
        print("wrote", target)


if __name__ == "__main__":
    main()
