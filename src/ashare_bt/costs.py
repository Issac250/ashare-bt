"""Commission, stamp duty, transfer fees, and estimated execution costs.

Defaults charge 0.05% stamp duty on sells. For earlier periods, enable
historical_stamp_duty to use the trade-date schedule. Other fees remain
fixed unless overridden in CostModel.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

# (effective from, buy-side rate, sell-side rate). Each row holds until the next.
STAMP_DUTY_SCHEDULE: tuple[tuple[str, float, float], ...] = (
    ("2001-11-16", 0.002, 0.002),
    ("2005-01-24", 0.001, 0.001),
    ("2007-05-30", 0.003, 0.003),
    ("2008-04-24", 0.001, 0.001),
    ("2008-09-19", 0.0, 0.001),     # sell side only from here on
    ("2023-08-28", 0.0, 0.0005),
)


def stamp_duty_on(date) -> tuple[float, float]:
    """(buy, sell) stamp-duty rates in force on `date`."""
    ts = pd.Timestamp(date)
    rate = None
    for start, buy, sell in STAMP_DUTY_SCHEDULE:
        if ts >= pd.Timestamp(start):
            rate = (buy, sell)
    if rate is None:
        raise ValueError(f"no stamp-duty rate on file before {STAMP_DUTY_SCHEDULE[0][0]}: {ts.date()}")
    return rate


@dataclass(frozen=True)
class CostModel:
    commission_rate: float = 0.00025   # broker, both sides
    commission_min: float = 5.0        # minimum per order, in yuan
    stamp_duty: float = 0.0005         # sell side only (0.001 before 2023-08-28)
    transfer_fee: float = 0.00001      # both sides, both exchanges since 2022
    slippage_bps: float = 5.0          # baseline half-spread on top of impact
    impact_coef: float = 0.10          # impact = coef * participation_rate
    stamp_duty_buy: float = 0.0        # buy side; non-zero only before 2008-09-19
    historical_stamp_duty: bool = False  # take both rates from STAMP_DUTY_SCHEDULE by trade date

    def __post_init__(self) -> None:
        # Historical mode replaces both duty rates; reject conflicting overrides.
        if self.historical_stamp_duty and (self.stamp_duty != 0.0005 or self.stamp_duty_buy != 0.0):
            raise ValueError("historical_stamp_duty=True takes stamp_duty and stamp_duty_buy from "
                             "the schedule; leave them at their defaults")

    def trade_cost(self, notional: float, is_sell: bool, participation: float = 0.0) -> float:
        """Return the cost of one order in yuan.

        `participation` is notional / reference turnover. The engine uses its
        lagged 20-day median turnover as the reference.
        """
        if self.historical_stamp_duty:
            raise ValueError("a historical_stamp_duty model has no rate until it is dated: "
                             "call .on(date) first")
        if notional <= 0:
            return 0.0
        commission = max(notional * self.commission_rate, self.commission_min)
        duty = notional * (self.stamp_duty if is_sell else self.stamp_duty_buy)
        transfer = notional * self.transfer_fee
        slip = notional * self.slippage_bps / 1e4
        impact = notional * self.impact_coef * max(participation, 0.0)
        return commission + duty + transfer + slip + impact

    def on(self, date) -> "CostModel":
        """Return a model with trade-date stamp duty, or self when historical mode is off."""
        if not self.historical_stamp_duty:
            return self
        buy, sell = stamp_duty_on(date)
        return replace(self, stamp_duty=sell, stamp_duty_buy=buy, historical_stamp_duty=False)

    def round_trip_bps(self, notional: float, participation: float = 0.0) -> float:
        """Return combined buy and sell costs in basis points of notional."""
        if notional <= 0:
            return 0.0
        c = self.trade_cost(notional, False, participation) + self.trade_cost(notional, True, participation)
        return c / notional * 1e4


LOT_SIZE = 100  # default buy lot size
