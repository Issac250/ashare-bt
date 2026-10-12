"""Order sizing and trading checks shared by the engine and other callers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import CostModel


def limit_prices(preclose, width: float):
    """Return upper and lower limit prices, rounded half-up to cents.

    For example, 27.95 * 1.1 gives 30.75, where float rounding gives 30.74.
    Accepts scalars, arrays, and Series; a Series keeps its index.
    """
    def half_up(x):
        return np.floor(x * 100 + 0.5 + 1e-6) / 100
    return half_up(preclose * (1 + width)), half_up(preclose * (1 - width))


def whole_lots(budget: float, px: float, lot_size: int) -> int:
    """Return the affordable share count, rounded down to whole lots."""
    if px <= 0 or budget <= 0 or lot_size < 1:
        return 0
    return int(budget // (px * lot_size)) * lot_size


def participation_cap(adv20: float, max_participation: float) -> float:
    """Return adv20 * max_participation, or zero for missing/nonpositive turnover.

    fit_buy treats a zero cap as uncapped and sizes from the target and cash.
    """
    if pd.isna(adv20) or adv20 <= 0:
        return 0.0
    return float(adv20) * max_participation


def fit_buy(px: float, want_value: float, adv20: float, max_participation: float,
            cash: float, lot_size: int, costs: CostModel) -> tuple[int, float, float]:
    """Return (shares, notional, cost) within the buying cap and available cash.

    Reduce by one lot at a time until notional plus fees fits. A zero share
    count means no order.
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
    """Round a partial sale down to whole lots. Full exits are handled by the engine."""
    return whole_lots(value, px, lot_size)


def can_buy(row: pd.Series) -> bool:
    """Allow buys at a positive open price when tradable and off limit-up."""
    return (float(row["open"]) > 0
            and bool(row["tradable"])
            and not bool(row["open_limit_up"]))


def can_sell(row: pd.Series) -> bool:
    """Allow sells when tradable and off limit-down at the open."""
    return bool(row["tradable"]) and not bool(row["open_limit_down"])


def settled(bought_on, execution_day, settlement_t0: bool = False) -> bool:
    """Check whether a holding can be sold on execution_day.

    The engine passes trading dates: T+1 requires a later date; T+0 allows
    same-day sales.
    """
    if settlement_t0:
        return True
    return bought_on < execution_day
