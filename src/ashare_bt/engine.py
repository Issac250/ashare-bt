"""Strategy-agnostic cross-sectional backtest engine.

The engine knows NOTHING about any particular strategy. It takes a signal
panel and returns performance. Strategies are therefore just functions that
produce signals, and can be swapped or tuned without touching this file.

    signal:  DataFrame[date, code, score]   higher score = more attractive
    output:  BacktestResult (equity curve, per-date detail, metrics)

Execution model. The DEFAULTS match how A-share equity actually trades; the two
venue-specific rules are config, because convertible bonds trade on the same
exchanges under different ones (T+0 settlement, 10-bond lots, no stamp duty).
  * Signal computed from data up to and including date t.
  * Trades execute at t+1's OPEN. Nothing uses t+1 close to decide t+1 trades.
  * T+1 settlement: a position bought on day t cannot be sold before t+1.
    Set `settlement_t0=True` for instruments that settle same-day.
  * Lot size defaults to 100 (A-share). Set `lot_size=10` for convertible bonds.
    Stamp duty is already a CostModel field: pass CostModel(stamp_duty=0.0).
  * Cannot buy a stock that opens at its limit-up, nor sell one at limit-down.
  * Position size is capped by a share of the stock's recent ¥ turnover, so
    the same signal can be re-run at several capital levels to expose the
    capacity/alpha trade-off rather than assuming it away.
  * Positions are carried as ¥ VALUE and advanced by the total return, so a
    split or a dividend is not a loss. Share counts are derived at the raw
    price only where the real share count matters: lot rounding.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import sizing
from .costs import LOT_SIZE, CostModel

log = logging.getLogger("backtest")

# `preclose` is required rather than defaulted: it is what makes a split or a
# dividend not look like a loss, and defaulting the overnight ratio to 1.0
# would silently drop every overnight move.
REQUIRED_BARS = ("date", "code", "open", "close", "preclose", "amount",
                 "tradable", "open_limit_up", "open_limit_down")


@dataclass
class BacktestConfig:
    capital: float = 1_000_000.0
    n_positions: int = 50            # breadth; smaller = more concentrated
    rebalance_days: int = 5          # 5 = weekly
    max_participation: float = 0.05  # cap position at 5% of 20d median ¥ turnover
    long_only: bool = True           # A-share retail cannot practically short
    # Select from a percentile band of the score instead of the extreme top.
    # Several A-share factors invert in the extreme tail, so the very top names
    # can be falling knives. (0.80, 0.95) means "well-ranked but not extreme".
    score_band: tuple[float, float] | None = None
    # Trim held names back toward their target weight on each rebalance.
    # False lets winners drift and concentrate the book.
    rebalance_weights: bool = True
    # Venue rules. Defaults are A-share equity; convertible bonds on the same
    # exchanges settle T+0 and trade in lots of 10 (¥1,000 face).
    settlement_t0: bool = False      # True = a position bought today CAN be sold today
    lot_size: int | None = None      # None = the A-share default of 100
    costs: CostModel = None          # type: ignore[assignment]
    # What happens to cash freed by an exit: `refill` redeploys the slot at the
    # next open; otherwise it waits for the next rebalance, earning
    # `cash_annual_rate` (243.6 trading days a year).
    refill: bool = False             # redeploy a freed slot at the next open
    cash_annual_rate: float = 0.0    # interest on idle cash
    # Rebalance when the INFORMATION arrives instead of every N days: a fixed
    # N-day calendar re-picks a ranking that has not moved. Signal dates NOT in this
    # set are still checked for exits; they just do not re-pick the book.
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
    """Largest whole-lot order that fits the cap, the want and the cash.

    Delegates to `sizing.fit_buy` so order-placing code can round identically.
    """
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
    # CAGR is a calendar-time quantity, so `years` is calendar days / 365.25.
    # Counting trading days / 252 overstates CAGR on A-shares, which trade
    # about 243 days a year.
    span_days = (equity.index[-1] - equity.index[0]).days
    years = max(span_days / 365.25, len(equity) / 252.0 * 0.5, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0
    vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0.0
    dd = (equity / equity.cummax() - 1.0).min()
    cost_total = daily["cost"].sum()
    # `sharpe` is the arithmetic daily Sharpe, annualised by sqrt(252), with a
    # zero risk-free rate. CAGR / vol is a different, geometric quantity and is
    # deliberately not reported under this name.
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
    """Run a cross-sectional long book.

    `signal` needs [date, code, score]; `bars` needs [date, code, open, close,
    ret, amount, tradable, open_limit_up, open_limit_down]. Both are plain frames
    so this is testable without any database.

    `exposure` is an optional Series indexed by signal date giving the fraction
    of equity to hold in stocks, in [0, 1]; the remainder sits in cash. It is
    how a risk control that SCALES exposure (volatility targeting, drawdown
    de-risking) is expressed, as opposed to a binary in/out overlay.

    Scaling DOWN only works with `cfg.rebalance_weights=True`: the trim step is
    what sells an existing book down to the smaller target. With trimming off,
    a falling exposure would only slow new buying.

    `exit_flags` is an optional frame of [date, code] meaning "sell this holding
    at the next open", checked on EVERY signal day rather than only on rebalance
    days. It is how an exit rule is expressed, separately from what to buy.

    The freed capital sits in CASH until the next rebalance. That idle drag is
    the honest cost of exiting early and is exactly what a cross-tab of
    "how much each position gave back" cannot show.

    The engine only SELLS here. Suppressing re-entry is the caller's job (drop
    the name from `signal`, or NaN its score); keeping those two apart means the
    engine never has to hold an opinion about how long a flag lasts.

    Re-entry is deliberately not suppressed by default: an exit that sells a
    spike can earn its keep by buying the SAME name back cheaper at the next
    rebalance, when its rank has not changed.
    """
    cfg = cfg or BacktestConfig()
    if not cfg.long_only:
        raise NotImplementedError("short side omitted: individual A-share shorting is impractical")
    if "open_limit_up" not in bars.columns and "at_limit_up" in bars.columns:
        raise ValueError(
            "bars has `at_limit_up` but the engine reads `open_limit_up`: rename the "
            "columns, and make sure they describe the OPEN, not the close. A "
            "close-based flag is a look-ahead for an order that fills at the open."
        )
    missing = [c for c in REQUIRED_BARS if c not in bars.columns]
    if missing:
        raise ValueError(f"bars is missing columns {missing}; see run_backtest's docstring")

    bars = bars.sort_values(["date", "code"]).copy()
    bars["date"] = pd.to_datetime(bars["date"])
    # Defensive: a NULL flag from any source must not crash a boolean cast, and
    # "unknown" must never be read as "tradable" or "not at a limit".
    for col, default in (("tradable", False), ("open_limit_up", True), ("open_limit_down", True)):
        if col in bars.columns:
            bars[col] = bars[col].fillna(default).astype(bool)
    signal = signal.copy()
    signal["date"] = pd.to_datetime(signal["date"])
    if exposure is not None:
        exposure = exposure.copy()
        exposure.index = pd.to_datetime(exposure.index)
        # Clipped, not trusted: an exposure above 1 would be leverage, which a
        # cash A-share account does not have, and a negative one would be a
        # short. Both fail closed rather than silently backtesting something
        # the account cannot do.
        exposure = exposure.clip(lower=0.0, upper=1.0)

    # Liquidity reference: trailing 20d median ¥ turnover, shifted so the cap
    # on day t+1 uses only information available by t.
    bars["adv20"] = (
        bars.groupby("code")["amount"].transform(lambda s: s.rolling(20, min_periods=5).median().shift(1))
    )

    # CORPORATE ACTIONS -- the second most important thing in this file.
    #
    # baostock serves RAW prices (adjustflag=3), so a split or a bonus issue
    # moves the raw price with no economic event behind it. Valuing a position
    # as shares x raw price therefore booked sh.603636's 2017-04-07 4-for-1
    # split as a -72.5% day, and every dividend as a small loss.
    #
    # The adjustment is already in `preclose`, which baostock adjusts:
    #     close_t / preclose_t == 1 + ret_t          (a TOTAL return)
    # Splitting the day at the open gives two ratios that multiply back to it,
    # while still letting a trade fill at the raw open price:
    #     overnight   open_t / preclose_t     prior close -> today's open
    #     intraday    close_t / open_t        inside one day; no action fits here
    _pc = pd.to_numeric(bars["preclose"], errors="coerce")
    # A missing ratio means "no information about the move", which is 1.0
    # (carry the value) -- never NaN, which would poison the curve from that
    # day on, and never 0.
    bars["overnight"] = (bars["open"] / _pc).where(_pc > 0).fillna(1.0)
    bars["intraday"] = (bars["close"] / bars["open"]).where(bars["open"] > 0).fillna(1.0)

    # Execution happens on the day AFTER the signal date.
    days = sorted(bars["date"].unique())
    nxt = {d: days[i + 1] for i, d in enumerate(days[:-1])}

    bars_by_date = {d: g.set_index("code") for d, g in bars.groupby("date")}
    # Final observed bar per code: used to distinguish "delisted" from
    # "merely suspended today", since a suspended stock reappears later.
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

        # Advance the existing book from last night's close to this morning's
        # open, through any corporate action that happened in between.
        for code, h in list(holdings.items()):
            if code in ex.index:
                h["value"] *= float(ex.at[code, "overnight"])

        cost_today = traded_notional = 0.0

        # Idle cash earns its rate. 243.6 is the average A-share trading year;
        # using 252 here would quietly pay
        # eight days of interest a year that the calendar does not contain.
        if cfg.cash_annual_rate and cash > 0:
            cash *= 1.0 + cfg.cash_annual_rate / 243.6

        # ---- exit rule: runs EVERY day, not only on rebalance days ----
        # Same venue rules as any other sell: no selling into a limit-down or a
        # halt, and T+1 still binds. A blocked exit is simply retried tomorrow
        # if the caller still flags it -- it is never silently treated as done.
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

        # ---- refill: put an exited slot back to work at the next open ----
        #
        # The alternative is that the freed cash waits for the next calendar
        # rebalance. On a book whose score gradient is steep that wait is
        # expensive; on one where the exit earns its keep by buying the SAME
        # name back cheaper it is the point. Which dominates is measurable, so
        # it is a flag rather than an assumption.
        #
        # The name just exited is never re-bought here: it was flagged for sale
        # this morning, and buying it back the same morning would only pay the
        # spread twice.
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
            # An open at limit-up blocks BUYING it, not HOLDING it: a held name
            # stays eligible for the target, or it would be sold at that open
            # while still ranked. Top-ups of it are blocked further down.
            held = avail.index.isin(list(holdings))
            ok = avail[avail["tradable"].astype(bool)
                       & (~avail["open_limit_up"].astype(bool) | held)]
            pool = scored[scored["code"].isin(ok.index)]
            if cfg.score_band is not None and len(pool) > 1:
                lo, hi = cfg.score_band
                # Percentile within THIS date's tradable cross-section, so the
                # band means the same thing regardless of how many names
                # happened to be tradable.
                pct = pool["score"].rank(pct=True)
                banded = pool[(pct >= lo) & (pct <= hi)]
                if len(banded) >= cfg.n_positions:
                    pool = banded
            if exited_today:
                pool = pool[~pool["code"].isin(exited_today)]
            target = pool.nlargest(cfg.n_positions, "score")["code"].tolist()

            # ---- target weights ----
            # An optional `w` column on the signal frame sets RELATIVE position
            # sizes; absent (or constant) it reproduces equal weight exactly.
            #
            # The normalisation keeps the "size against the BREADTH CAP" rule
            # that the equal-weight path uses: k names out of n_positions stay
            # k/n invested with the remainder in cash, rather than being
            # renormalised up onto the survivors (which would silently lever an
            # event book on days when few names qualify).
            wmap: dict[str, float] = {}
            if "w" in pool.columns and target:
                raw = pool.set_index("code")["w"].reindex(target)
                raw = pd.to_numeric(raw, errors="coerce").clip(lower=0.0)
                tot = float(raw.sum())
                if tot > 0 and raw.notna().all():
                    wmap = (raw / tot * len(target)).to_dict()   # mean 1.0

            # ---- sells: exit anything not in target, unless blocked ----
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

            # ---- rebalance + buys: equal-weight the target ----
            keep = [c for c in target if c in holdings]
            need = [c for c in target if c not in holdings]
            equity_now = cash + sum(h["value"] for h in holdings.values())
            # A missing exposure reading means no basis for a position, so it
            # is 0, not 1.
            expo = 1.0 if exposure is None else float(exposure.get(d, 0.0))
            # Size against the BREADTH CAP, not against how many names happened
            # to qualify, and leave the rest in cash.
            #
            # For a factor that ranks the whole market the two are identical.
            # For an EVENT strategy, where most days only a handful of names
            # qualify, dividing by the count found would put half the account
            # into each of two names, and the backtest would measure that bet
            # rather than the signal.
            per_name = equity_now * expo / max(cfg.n_positions, 1)
            # Per-name target value. `wmap` has mean 1.0 over the target names,
            # so the book's total target is unchanged and only its DISTRIBUTION
            # moves; an empty wmap leaves every name at `per_name`.
            tgt = lambda c: per_name * wmap.get(c, 1.0)

            # Trim the overweight names FIRST: it raises the cash that the
            # underweight ones and the new names are then funded from.
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
                    # Lot rounding is done at the RAW price because the lot is
                    # 100 REAL shares; the position itself is tracked in ¥.
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

            # Underweight held names are topped up alongside the new ones, so
            # both directions of the drift are corrected in the same pass.
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
                    # A TOP-UP must fund only the DEFICIT. Sizing it to
                    # `per_name` would buy a second full position on top of the
                    # one already held and double the weight it was meant to
                    # correct.
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
                        # A top-up ADDS to the position. Overwriting it would
                        # silently discard the value already held.
                        holdings[code]["value"] += notional
                        holdings[code]["bought_on"] = ex_day
                        holdings[code]["last_seen"] = ex_day
                    else:
                        holdings[code] = {"value": notional, "bought_on": ex_day,
                                          "last_seen": ex_day}
                    trade_rows.append({"date": ex_day, "code": code, "side": "buy",
                                       "notional": notional, "cost": c})

        # Hold through the execution day: mark to its close.
        #
        # DELISTING: when a held stock stops appearing in the data its history
        # has ended, and carrying it at a frozen price forever would quietly
        # understate delisting risk -- A-share delisting usually runs through a
        # 退市整理期 in which the price collapses (sh.600193 fell to CNY 0.19).
        # The bars DO contain that collapse, so a position held to the final bar
        # already absorbs most of it; here we force a liquidation at the last
        # observed price so the capital is released rather than stranded.
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
    equity = daily["equity"]
    return BacktestResult(equity, daily, pd.DataFrame(trade_rows), cfg, _metrics(equity, daily, cfg))
