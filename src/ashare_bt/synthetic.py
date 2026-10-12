"""Synthetic A-share-like bars and signals, for examples and tests.

Nothing here is market data. The generator only has to produce panels on which
every venue rule in the engine fires: opens at the limit, halts, stocks whose
data ends (delisting), and prices that respect the daily band.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sizing import limit_prices

BARS_COLUMNS = ["date", "code", "open", "close", "preclose", "ret", "amount",
                "tradable", "open_limit_up", "open_limit_down"]


def make_market(n_stocks: int = 50, n_days: int = 500, start: str = "2020-01-02",
                seed: int = 0, limit_width: float = 0.10, daily_vol: float = 0.025,
                limit_open_prob: float = 0.004, halt_prob: float = 0.002,
                delist_frac: float = 0.1) -> pd.DataFrame:
    """Random-walk bars for `n_stocks` over `n_days` business days.

    Prices are rounded to the fen and clipped to +-`limit_width` of the prior
    close. With probability `limit_open_prob` a day opens at the limit (half
    up, half down). A halted day has no turnover and its price does not move.
    The first `delist_frac` of the stocks stop trading part-way through.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    n_delist = int(round(n_stocks * delist_frac))
    frames = []
    for i in range(n_stocks):
        n = int(rng.integers(n_days // 4, n_days * 3 // 4)) if i < n_delist else n_days
        price = float(rng.uniform(5.0, 50.0))
        o = np.empty(n)
        c = np.empty(n)
        pc = np.empty(n)
        amt = np.empty(n)
        trad = np.ones(n, dtype=bool)
        for t in range(n):
            up, dn = (float(x) for x in limit_prices(price, limit_width))
            pc[t] = price
            if rng.random() < halt_prob:
                o[t] = c[t] = price
                amt[t] = 0.0
                trad[t] = False
                continue
            u = rng.random()
            if u < limit_open_prob / 2:
                o[t] = up
            elif u < limit_open_prob:
                o[t] = dn
            else:
                o[t] = round(min(max(price * (1 + rng.normal(0.0, daily_vol / 3)), dn), up), 2)
            c[t] = round(min(max(o[t] * (1 + rng.normal(0.0, daily_vol)), dn), up), 2)
            amt[t] = float(rng.lognormal(np.log(1e8), 1.0))
            price = c[t]
        frames.append(pd.DataFrame({
            "date": dates[:n], "code": f"sh.{600000 + i:06d}", "open": o, "close": c,
            "preclose": pc, "ret": c / pc - 1.0, "amount": amt, "tradable": trad,
        }))
    bars = pd.concat(frames, ignore_index=True)
    up, dn = limit_prices(bars["preclose"], limit_width)
    bars["open_limit_up"] = bars["open"] >= up
    bars["open_limit_down"] = bars["open"] <= dn
    return bars.sort_values(["date", "code"]).reset_index(drop=True)[BARS_COLUMNS]


def make_signal(bars: pd.DataFrame, kind: str = "momentum", lookback: int = 20,
                seed: int = 0) -> pd.DataFrame:
    """A [date, code, score] frame built only from data up to each date.

    `momentum`: trailing `lookback`-day total return. `random`: uniform noise,
    the null a real signal has to beat.
    """
    b = bars.sort_values(["code", "date"])
    if kind == "momentum":
        lr = np.log1p(b["ret"].astype(float))
        score = np.expm1(lr.groupby(b["code"]).transform(
            lambda s: s.rolling(lookback, min_periods=lookback).sum()))
    elif kind == "random":
        score = pd.Series(np.random.default_rng(seed).random(len(b)), index=b.index)
    else:
        raise ValueError(f"unknown signal kind {kind!r}; use 'momentum' or 'random'")
    out = b[["date", "code"]].assign(score=score.astype(float)).dropna(subset=["score"])
    return out.sort_values(["date", "code"]).reset_index(drop=True)
