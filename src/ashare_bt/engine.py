"""Daily portfolio backtest with next-open execution.

Signals supply stock rankings; this module handles selection, sizing, costs,
and valuation. Holdings are tracked in yuan and updated using open/preclose
and close/open ratios. Raw prices are used for lot rounding.

Defaults use 100-share lots and T+1. Opening limit flags block buys at
limit-up and sells at limit-down. These are daily-data approximations of
execution, not an auction or order-book simulation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import sizing
from .costs import LOT_SIZE, CostModel

log = logging.getLogger("backtest")

# preclose supplies the reference for overnight returns, including ex-rights days.
REQUIRED_BARS = ("date", "code", "open", "close", "preclose", "amount",
                 "tradable", "open_limit_up", "open_limit_down")


@dataclass
class BacktestConfig:
    capital: float = 1_000_000.0
    n_positions: int = 50            # breadth; smaller = more concentrated
    rebalance_days: int = 5          # 5 = weekly
    max_participation: float = 0.05  # cap position at 5% of 20d median ¥ turnover
    long_only: bool = True           # only long_only=True is supported
    # Optional score percentile range, e.g. (0.80, 0.95).
    score_band: tuple[float, float] | None = None
    # Trim and top up existing holdings on rebalance days.
    rebalance_weights: bool = True
    # Override settlement and lot size for other instruments.
    settlement_t0: bool = False      # allow same-day sales
    lot_size: int | None = None      # None = the A-share default of 100
    costs: CostModel = None          # type: ignore[assignment]
    # By default, exit proceeds wait for the next rebalance.
    refill: bool = False             # buy replacements at the exit open
    cash_annual_rate: float = 0.0    # interest on idle cash
    # Explicit signal dates replace the fixed rebalance interval. Exits still run daily.
    rebalance_dates: frozenset | None = None

    def __post_init__(self) -> None:
        if self.costs is None:
            self.costs = CostModel()
        if self.lot_size is None:
            self.lot_size = LOT_SIZE
        if self.lot_size < 1:
            raise ValueError(f"lot_size must be >= 1, got {self.lot_size}")


def _size_order(px: float, adv, want_value: float, cash: float,
                cfg: "BacktestConfig", costs: CostModel) -> tuple[int, float, float]:
    """Size a buy using the shared lot, turnover, and cash constraints."""
    return sizing.fit_buy(px, want_value, adv, cfg.max_participation,
                          cash, cfg.lot_size, costs)


@dataclass
class BacktestResult:
    equity: pd.Series
    daily: pd.DataFrame
    trades: pd.DataFrame
    config: BacktestConfig
    metrics: dict

    def summary(self) -> str:
        m = self.metrics
        rows = [
            ("period", f"{m['start']} -> {m['end']} ({m['years']:.1f}y)"),
            ("total return", f"{m['total_return']:>8.1%}"),
            ("CAGR", f"{m['cagr']:>8.1%}"),
            ("volatility", f"{m['vol']:>8.1%}"),
            ("Sharpe (rf=0)", f"{m['sharpe']:>8.2f}"),
            ("max drawdown", f"{m['max_dd']:>8.1%}"),
            ("hit rate (days)", f"{m['hit_rate']:>8.1%}"),
            ("annual turnover", f"{m['turnover']:>8.1f}x"),
            ("cost drag / yr", f"{m['cost_drag']:>8.2%}"),
            ("avg positions", f"{m['avg_positions']:>8.1f}"),
            ("capital", f"{m['capital']:>12,.0f}"),
        ]
        w = max(len(k) for k, _ in rows)
        return "\n".join(f"  {k:.<{w + 2}} {v}" for k, v in rows)


def _metrics(equity: pd.Series, daily: pd.DataFrame, cfg: BacktestConfig) -> dict:
    rets = equity.pct_change().dropna()
    # Use calendar time for CAGR, with a floor for very short samples.
    span_days = (equity.index[-1] - equity.index[0]).days
    years = max(span_days / 365.25, len(equity) / 252.0 * 0.5, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0
    vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0.0
    dd = (equity / equity.cummax() - 1.0).min()
    cost_total = daily["cost"].sum()
    # Daily arithmetic Sharpe, annualised at 252 sessions with zero risk-free rate.
    sharpe = (rets.mean() * 252 / vol) if vol > 0 and len(rets) > 1 else 0.0
    return {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "years": years,
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": dd,
        "hit_rate": (rets > 0).mean() if len(rets) else 0.0,
        "turnover": daily["traded_notional"].sum() / max(equity.mean(), 1) / years,
        "cost_drag": cost_total / max(equity.mean(), 1) / years,
        "avg_positions": daily["n_positions"].mean(),
        "capital": cfg.capital,
        "total_cost": cost_total,
    }


def run_backtest(signal: pd.DataFrame, bars: pd.DataFrame, cfg: BacktestConfig | None = None,
                 exposure: pd.Series | None = None,
                 exit_flags: pd.DataFrame | None = None) -> BacktestResult:
    """Run a long-only portfolio from stock scores and daily bars.

    `signal` requires date, code, score, with an optional relative weight `w`.
    `bars` requires the columns in REQUIRED_BARS. Limit flags describe the open;
    preclose must use the ex-rights reference price for that day.
    Supply signal rows for every trading day to advance valuation daily.

    `exposure` is a Series indexed by signal date, clipped to [0, 1]. Missing
    entries use zero exposure. Reducing existing positions requires
    cfg.rebalance_weights=True.

    `exit_flags` contains date and code. Each signal day, flagged holdings are
    sold at the next open if tradable and settled. Repeat a flag on later dates
    to retry a blocked exit. Cash waits for the next rebalance unless cfg.refill
    is enabled, which buys replacements at the same open as the exit.
    To prevent re-entry on later days, remove the stock from the signal or set
    its score to NaN.
    """
    cfg = cfg or BacktestConfig()
    if not cfg.long_only:
        raise NotImplementedError("short side omitted: individual A-share shorting is impractical")
    for side in ("up", "down"):
        old, new = f"at_limit_{side}", f"open_limit_{side}"
        if new not in bars.columns and old in bars.columns:
            raise ValueError(
                f"bars has `{old}` but the engine reads `{new}`: rename the "
                "columns, and make sure they describe the OPEN, not the close. A "
                "close-based flag is a look-ahead for an order that fills at the open."
            )
    missing = [c for c in REQUIRED_BARS if c not in bars.columns]
    if missing:
        raise ValueError(f"bars is missing columns {missing}; see run_backtest's docstring")

    bars = bars.sort_values(["date", "code"]).copy()
    bars["date"] = pd.to_datetime(bars["date"])
    # Missing flags block trading in the affected direction.
    for col, default in (("tradable", False), ("open_limit_up", True), ("open_limit_down", True)):
        if col in bars.columns:
            # np.where avoids the pandas 2.x fillna downcasting warning.
            bars[col] = np.where(bars[col].isna(), default, bars[col]).astype(bool)
    signal = signal.copy()
    signal["date"] = pd.to_datetime(signal["date"])
    if exposure is not None:
        exposure = exposure.copy()
        exposure.index = pd.to_datetime(exposure.index)
        # Keep exposure within the long-only, unlevered range.
        exposure = exposure.clip(lower=0.0, upper=1.0)

    # Shift the turnover reference so execution-day amounts cannot affect sizing.
    bars["adv20"] = (
        bars.groupby("code")["amount"].transform(lambda s: s.rolling(20, min_periods=5).median().shift(1))
    )

    # With an ex-rights preclose, these ratios carry position value through
    # splits and dividends while orders still use raw open prices:
    #     open / preclose: prior close to this open
    #     close / open:   this open to this close
    _pc = pd.to_numeric(bars["preclose"], errors="coerce")
    # Missing ratios leave position value unchanged.
    bars["overnight"] = (bars["open"] / _pc).where(_pc > 0).fillna(1.0)
    bars["intraday"] = (bars["close"] / bars["open"]).where(bars["open"] > 0).fillna(1.0)

    # Map each date to the next trading date in the bars.
    days = sorted(bars["date"].unique())
    nxt = {d: days[i + 1] for i, d in enumerate(days[:-1])}

    bars_by_date = {d: g.set_index("code") for d, g in bars.groupby("date")}
    # A missing bar before the final date is carried; after it, the position is liquidated.
    last_bar_date = bars.groupby("code")["date"].max().to_dict()
    sig_by_date = {d: g for d, g in signal.groupby("date")}
    exit_by_date: dict[pd.Timestamp, set[str]] = {}
    if exit_flags is not None and len(exit_flags):
        ef = exit_flags.copy()
        ef["date"] = pd.to_datetime(ef["date"])
        exit_by_date = {d: set(g["code"]) for d, g in ef.groupby("date")}
    sig_days = [d for d in sorted(sig_by_date) if d in nxt]
    if not sig_days:
        raise ValueError("no signal dates align with a following trading day")
    if cfg.rebalance_dates is not None:
        want = {pd.Timestamp(x) for x in cfg.rebalance_dates}
        rebal_days = {d for d in sig_days if d in want}
    else:
        rebal_days = set(sig_days[:: cfg.rebalance_days])

    holdings: dict[str, dict] = {}   # code -> {value, bought_on, last_seen}
    cash = cfg.capital
    rows, trade_rows = [], []

    for d in sig_days:
        ex_day = nxt[d]
        ex = bars_by_date.get(ex_day)
        if ex is None:
            continue
        day_costs = cfg.costs.on(ex_day)

        # Mark existing holdings to the execution open.
        for code, h in list(holdings.items()):
            if code in ex.index:
                h["value"] *= float(ex.at[code, "overnight"])

        cost_today = traded_notional = 0.0

        # Apply the configured cash rate using a 243.6-session year.
        if cfg.cash_annual_rate and cash > 0:
            cash *= 1.0 + cfg.cash_annual_rate / 243.6

        # Check exits on every signal day, including non-rebalance days.
        # Blocked exits need another flag on a later signal day to be retried.
        exited_today: set[str] = set()
        for code in list(holdings):
            if code not in exit_by_date.get(d, ()):
                continue
            if code not in ex.index:
                continue
            row = ex.loc[code]
            if bool(row["open_limit_down"]) or not bool(row["tradable"]):
                continue
            if not cfg.settlement_t0 and holdings[code]["bought_on"] >= ex_day:
                continue
            notional = holdings[code]["value"]
            if notional <= 0:
                del holdings[code]
                continue
            adv = row["adv20"]
            part = notional / adv if (pd.notna(adv) and adv > 0) else 0.0
            c = day_costs.trade_cost(notional, True, part)
            cash += notional - c
            cost_today += c
            traded_notional += notional
            trade_rows.append({"date": ex_day, "code": code, "side": "exit",
                               "notional": notional, "cost": c})
            del holdings[code]
            exited_today.add(code)

        # Optionally buy replacements at the same open as the exit.
        # Exclude stocks just sold so refill cannot immediately buy them back.
        if (cfg.refill and exited_today and d not in rebal_days
                and len(holdings) < cfg.n_positions):
            sc = sig_by_date.get(d)
            if sc is not None:
                cand = sc.dropna(subset=["score"])
                cand = cand[~cand["code"].isin(holdings) & ~cand["code"].isin(exited_today)]
                cand = cand[cand["code"].isin(ex.index)]
                if len(cand):
                    m = ex.loc[cand["code"]]
                    cand = cand[m["tradable"].to_numpy() & ~m["open_limit_up"].to_numpy()]
                picks = cand.nlargest(cfg.n_positions - len(holdings), "score")
                if len(picks):
                    equity_now = cash + sum(h["value"] for h in holdings.values())
                    ex_pt = 1.0 if exposure is None else float(exposure.get(d, 0.0))
                    per_name = equity_now * ex_pt / max(cfg.n_positions, 1)
                    wcol = picks.set_index("code")["w"] if "w" in picks.columns else None
                    for code in picks["code"]:
                        row = ex.loc[code]
                        mult = 1.0
                        if wcol is not None:
                            v = pd.to_numeric(pd.Series([wcol[code]]), errors="coerce").iloc[0]
                            if pd.notna(v) and v > 0:
                                mult = float(v)
                        shares, notional, c = _size_order(
                            float(row["open"]), row["adv20"], per_name * mult, cash, cfg, day_costs)
                        if shares <= 0:
                            continue
                        cash -= notional + c
                        cost_today += c
                        traded_notional += notional
                        holdings[code] = {"value": notional, "bought_on": ex_day,
                                          "last_seen": ex_day}
                        trade_rows.append({"date": ex_day, "code": code, "side": "buy",
                                           "notional": notional, "cost": c})

        if d in rebal_days:
            scored = sig_by_date[d].dropna(subset=["score"])
            elig = [c for c in scored["code"] if c in ex.index]
            avail = ex.loc[elig]
            # Keep ranked holdings at limit-up in the target; new buys and top-ups are blocked.
            held = avail.index.isin(list(holdings))
            ok = avail[avail["tradable"].astype(bool)
                       & (~avail["open_limit_up"].astype(bool) | held)]
            pool = scored[scored["code"].isin(ok.index)]
            if cfg.score_band is not None and len(pool) > 1:
                lo, hi = cfg.score_band
                # Rank percentiles within this date's eligible pool.
                pct = pool["score"].rank(pct=True)
                banded = pool[(pct >= lo) & (pct <= hi)]
                if len(banded) >= cfg.n_positions:
                    pool = banded
            if exited_today:
                pool = pool[~pool["code"].isin(exited_today)]
            target = pool.nlargest(cfg.n_positions, "score")["code"].tolist()

            # Normalise relative weights to a mean of 1. Unfilled slots stay in cash,
            # matching the equal-weight path.
            wmap: dict[str, float] = {}
            if "w" in pool.columns and target:
                raw = pool.set_index("code")["w"].reindex(target)
                raw = pd.to_numeric(raw, errors="coerce").clip(lower=0.0)
                tot = float(raw.sum())
                if tot > 0 and raw.notna().all():
                    wmap = (raw / tot * len(target)).to_dict()   # mean 1.0

            # Sell holdings outside the target where trading rules allow.
            for code in list(holdings):
                if code in target:
                    continue
                if code not in ex.index:
                    continue                       # delisted/suspended: carry, cannot trade
                row = ex.loc[code]
                if bool(row["open_limit_down"]) or not bool(row["tradable"]):
                    continue                       # cannot sell into a limit-down or a halt
                if not cfg.settlement_t0 and holdings[code]["bought_on"] >= ex_day:
                    continue                       # T+1: bought today, not sellable
                notional = holdings[code]["value"]   # already marked to today's open
                if notional <= 0:
                    del holdings[code]
                    continue
                adv = row["adv20"]
                part = notional / adv if (pd.notna(adv) and adv > 0) else 0.0
                c = day_costs.trade_cost(notional, True, part)
                cash += notional - c
                cost_today += c
                traded_notional += notional
                trade_rows.append({"date": ex_day, "code": code, "side": "sell",
                                   "notional": notional, "cost": c})
                del holdings[code]

            # Set target values, then trim and buy.
            keep = [c for c in target if c in holdings]
            need = [c for c in target if c not in holdings]
            equity_now = cash + sum(h["value"] for h in holdings.values())
            # When exposure is supplied, missing dates use zero.
            expo = 1.0 if exposure is None else float(exposure.get(d, 0.0))
            # Divide by the requested position count so unfilled slots stay in cash.
            per_name = equity_now * expo / max(cfg.n_positions, 1)
            # Relative weights redistribute the target value across selected stocks.
            tgt = lambda c: per_name * wmap.get(c, 1.0)

            # Trim first to release cash for buys.
            if cfg.rebalance_weights:
                for code in keep:
                    if code not in ex.index:
                        continue                   # halted/delisted: cannot trade
                    row = ex.loc[code]
                    if not bool(row["tradable"]):
                        continue
                    px = float(row["open"])
                    if px <= 0:
                        continue
                    # Use raw prices for lot rounding; holdings are tracked in yuan.
                    excess = holdings[code]["value"] - tgt(code)
                    if excess < px * cfg.lot_size:
                        continue                   # under- or on-weight: handled below
                    if bool(row["open_limit_down"]):
                        continue                   # cannot sell into a limit-down
                    if not cfg.settlement_t0 and holdings[code]["bought_on"] >= ex_day:
                        continue                   # T+1: bought today
                    shares = sizing.fit_sell(px, excess, cfg.lot_size)
                    notional = min(px * shares, holdings[code]["value"])
                    adv = row["adv20"]
                    part = notional / adv if (pd.notna(adv) and adv > 0) else 0.0
                    c = day_costs.trade_cost(notional, True, part)
                    cash += notional - c
                    cost_today += c
                    traded_notional += notional
                    holdings[code]["value"] -= notional
                    trade_rows.append({"date": ex_day, "code": code, "side": "sell",
                                       "notional": notional, "cost": c})

            # Buy underweight holdings alongside new positions.
            top_up = []
            if cfg.rebalance_weights:
                for code in keep:
                    if code not in ex.index:
                        continue
                    row = ex.loc[code]
                    px = float(row["open"])
                    if px <= 0 or not bool(row["tradable"]) or bool(row["open_limit_up"]):
                        continue
                    if tgt(code) - holdings[code]["value"] >= px * cfg.lot_size:
                        top_up.append(code)

            if need or top_up:
                for code in need + top_up:
                    row = ex.loc[code]
                    px = float(row["open"])
                    if px <= 0:
                        continue
                    adv = row["adv20"]
                    # For a top-up, buy only the gap to target value.
                    want_value = tgt(code)
                    if code in holdings:
                        want_value = tgt(code) - holdings[code]["value"]
                        if want_value <= 0:
                            continue
                    shares, notional, c = _size_order(px, adv, want_value, cash, cfg, day_costs)
                    if shares <= 0:
                        continue
                    cash -= notional + c
                    cost_today += c
                    traded_notional += notional
                    if code in holdings:
                        holdings[code]["value"] += notional
                        holdings[code]["bought_on"] = ex_day
                        holdings[code]["last_seen"] = ex_day
                    else:
                        holdings[code] = {"value": notional, "bought_on": ex_day,
                                          "last_seen": ex_day}
                    trade_rows.append({"date": ex_day, "code": code, "side": "buy",
                                       "notional": notional, "cost": c})

        # Mark holdings to the close. If a stock's history has ended, release
        # its last marked value as cash. This assumes a data endpoint is a delisting;
        # it does not model an actual recovery payment.
        pos_val = 0.0
        for code, h in list(holdings.items()):
            if code in ex.index:
                h["value"] *= float(ex.at[code, "intraday"])
                h["last_seen"] = ex_day
            elif h.get("last_seen") is not None and code in last_bar_date:
                if ex_day > last_bar_date[code]:
                    cash += h["value"]                          # liquidate at last mark
                    trade_rows.append({"date": ex_day, "code": code, "side": "delist",
                                       "notional": h["value"], "cost": 0.0})
                    del holdings[code]
                    continue
            pos_val += h["value"]

        rows.append({"date": ex_day, "equity": cash + pos_val, "cash": cash,
                     "n_positions": len(holdings), "cost": cost_today,
                     "traded_notional": traded_notional})

    daily = pd.DataFrame(rows).set_index("date").sort_index()
    if daily.empty:
        raise ValueError("backtest produced no days")
    # Include starting capital so the first execution day contributes to all metrics.
    equity = pd.concat([pd.Series([float(cfg.capital)], index=[sig_days[0]]), daily["equity"]])
    equity.name = "equity"
    return BacktestResult(equity, daily, pd.DataFrame(trade_rows), cfg, _metrics(equity, daily, cfg))
