"""How big an order is, and whether it can be placed at all.

These rules live outside `engine.py` so that any code that places real orders
can import the SAME rounding and the same participation cap the backtest used.
If the two rounded lots differently, a live book would drift from the tested
one with nothing to make the drift visible.

Nothing here knows about a strategy or a date. Given a price, a budget and the
venue's rules, it answers how many shares.
"""
from __future__ import annotations

import pandas as pd

from .costs import CostModel


def whole_lots(budget: float, px: float, lot_size: int) -> int:
    """Shares affordable at `px` within `budget`, rounded DOWN to whole lots.

    The lot is the binding constraint at small capital: CNY 200,000 spread over
    100 names is CNY 2,000 each, which cannot buy even one lot of any stock
    priced above CNY 20.
    """
    if px <= 0 or budget <= 0 or lot_size < 1:
        return 0
    return int(budget // (px * lot_size)) * lot_size


def participation_cap(adv20: float, max_participation: float) -> float:
    """Largest position the stock's own turnover supports, or 0 if unknown.

    Zero means "no basis to size against", and callers read it as "fall back to
    the target weight" rather than "place no order" -- which is what the engine
    did inline and is preserved here deliberately. It matters only for names
    with no 20-day turnover history, i.e. recent listings.
    """
    if pd.isna(adv20) or adv20 <= 0:
        return 0.0
    return float(adv20) * max_participation


def fit_buy(px: float, want_value: float, adv20: float, max_participation: float,
            cash: float, lot_size: int, costs: CostModel) -> tuple[int, float, float]:
    """(shares, notional, cost) for a buy that actually fits the cash on hand.

    Steps DOWN a lot at a time until the order plus its costs fits. Sizing to
    the full budget and then rejecting the order would silently skip every buy
    whenever the budget equalled the available cash -- which, at a rebalance
    that is deploying the whole book, is most of them.

    `shares == 0` means no order.
    """
    if px <= 0 or want_value <= 0 or cash <= 0:
        return 0, 0.0, 0.0
    cap = participation_cap(adv20, max_participation)
    has_adv = cap > 0
    budget = min(want_value, cap if has_adv else want_value, cash)
    shares = whole_lots(budget, px, lot_size)
    notional = cost = 0.0
    while shares > 0:
        notional = px * shares
        part = notional / adv20 if has_adv else 0.0
        cost = costs.trade_cost(notional, False, part)
        if notional + cost <= cash:
            break
        shares -= lot_size
    if shares <= 0:
        return 0, 0.0, 0.0
    return shares, notional, cost


def fit_sell(px: float, value: float, lot_size: int) -> int:
    """Whole lots to sell to raise about `value` at `px`.

    Used for a TRIM, where the position is being reduced toward equal weight
    rather than closed. A full exit does not come through here: it sells the
    share count held, which is already an integer and may legitimately not be a
    whole number of lots after a bonus issue -- an A-share account can sell an
    odd lot but not buy one.
    """
    return whole_lots(value, px, lot_size)


def can_buy(row: pd.Series) -> bool:
    """A buy fills at the open, so the gate is the OPEN's limit state.

    Gating on the close is a look-ahead: most stocks that close at limit-up
    did not open there, so a close-based gate refuses exactly the names that
    went on to rise. The flags passed in must describe the open.
    """
    return (float(row["open"]) > 0
            and bool(row["tradable"])
            and not bool(row["open_limit_up"]))


def can_sell(row: pd.Series) -> bool:
    """Cannot sell into a limit-down or a halt. Same open-price reasoning."""
    return bool(row["tradable"]) and not bool(row["open_limit_down"])


def settled(bought_on, execution_day, settlement_t0: bool = False) -> bool:
    """T+1: shares bought on day t cannot be sold before t+1.

    Comparing dates rather than counting days, because the engine advances by
    trading session and a weekend must not count as settlement time.
    """
    if settlement_t0:
        return True
    return bought_on < execution_day
