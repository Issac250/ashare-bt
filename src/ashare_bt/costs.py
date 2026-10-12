"""A-share transaction costs and tradability rules.

Rates are the retail-typical 2026 schedule; override via CostModel for your
own broker. Stamp duty is sell-side only and was halved to 0.05% on
2023-08-28.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_rate: float = 0.00025   # broker, both sides
    commission_min: float = 5.0        # ¥5 minimum per order -- brutal on small tickets
    stamp_duty: float = 0.0005         # sell side only (0.001 before 2023-08-28)
    transfer_fee: float = 0.00001      # both sides, both exchanges since 2022
    slippage_bps: float = 5.0          # baseline half-spread on top of impact
    impact_coef: float = 0.10          # impact = coef * participation_rate

    def trade_cost(self, notional: float, is_sell: bool, participation: float = 0.0) -> float:
        """Cost in ¥ for one order.

        `participation` is the order's share of that day's ¥ turnover; impact
        grows with it, which is what makes capacity bite as size increases.
        """
        if notional <= 0:
            return 0.0
        commission = max(notional * self.commission_rate, self.commission_min)
        duty = notional * self.stamp_duty if is_sell else 0.0
        transfer = notional * self.transfer_fee
        slip = notional * self.slippage_bps / 1e4
        impact = notional * self.impact_coef * max(participation, 0.0)
        return commission + duty + transfer + slip + impact

    def round_trip_bps(self, notional: float, participation: float = 0.0) -> float:
        """Convenience: full in-and-out cost in basis points of notional."""
        if notional <= 0:
            return 0.0
        c = self.trade_cost(notional, False, participation) + self.trade_cost(notional, True, participation)
        return c / notional * 1e4


LOT_SIZE = 100  # A-share minimum tradable unit; binds hard at small capital
