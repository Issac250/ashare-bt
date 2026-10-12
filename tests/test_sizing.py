"""Lot rounding, turnover caps, cash limits, and trading checks."""
from __future__ import annotations

import pandas as pd

from ashare_bt import sizing
from ashare_bt.costs import CostModel

COSTS = CostModel(slippage_bps=10.0)



def test_whole_lots_rounds_down_never_up():
    # CNY 1,050 at CNY 10 is 105 shares, which is one lot plus a fraction.
    assert sizing.whole_lots(1_050, 10.0, 100) == 100
    assert sizing.whole_lots(999, 10.0, 100) == 0
    assert sizing.whole_lots(1_000, 10.0, 100) == 100


def test_whole_lots_handles_the_lot_floor_that_binds_at_small_capital():
    """One lot at CNY 24.79 requires CNY 2,479 before fees."""
    assert sizing.whole_lots(2_000, 24.79, 100) == 0
    assert sizing.whole_lots(2_479, 24.79, 100) == 100


def test_whole_lots_refuses_nonsense_rather_than_dividing_by_zero():
    assert sizing.whole_lots(1_000, 0.0, 100) == 0
    assert sizing.whole_lots(1_000, -1.0, 100) == 0
    assert sizing.whole_lots(-1_000, 10.0, 100) == 0
    assert sizing.whole_lots(1_000, 10.0, 0) == 0


def test_whole_lots_respects_a_non_default_lot_size():
    # Convertible bonds trade in tens on the same exchanges.
    assert sizing.whole_lots(1_050, 10.0, 10) == 100
    assert sizing.whole_lots(105, 10.0, 10) == 10



def test_participation_cap_is_a_share_of_the_stocks_own_turnover():
    assert sizing.participation_cap(1_000_000, 0.02) == 20_000


def test_unknown_turnover_caps_at_zero_which_callers_read_as_no_cap():
    """Missing or nonpositive turnover returns zero; fit_buy treats this as uncapped."""
    assert sizing.participation_cap(float("nan"), 0.02) == 0.0
    assert sizing.participation_cap(0.0, 0.02) == 0.0
    assert sizing.participation_cap(-5.0, 0.02) == 0.0



def test_fit_buy_sizes_to_the_target_weight_when_nothing_else_binds():
    shares, notional, cost = sizing.fit_buy(
        px=10.0, want_value=10_000, adv20=1e9, max_participation=0.02,
        cash=1_000_000, lot_size=100, costs=COSTS)
    assert shares == 1_000
    assert notional == 10_000
    assert cost > 0


def test_fit_buy_is_capped_by_participation_not_by_the_target():
    # 2% of CNY 100,000 turnover is CNY 2,000 -- 200 shares at CNY 10.
    shares, notional, _ = sizing.fit_buy(
        px=10.0, want_value=10_000, adv20=100_000, max_participation=0.02,
        cash=1_000_000, lot_size=100, costs=COSTS)
    assert shares == 200
    assert notional == 2_000


def test_fit_buy_steps_down_until_the_order_plus_costs_fits_the_cash():
    """Leave room for fees by reducing the share count instead of skipping the buy."""
    shares, notional, cost = sizing.fit_buy(
        px=10.0, want_value=10_000, adv20=1e9, max_participation=0.02,
        cash=10_000, lot_size=100, costs=COSTS)
    assert 0 < shares < 1_000
    assert notional + cost <= 10_000


def test_fit_buy_returns_no_order_when_even_one_lot_cannot_be_afforded():
    shares, notional, cost = sizing.fit_buy(
        px=24.79, want_value=10_000, adv20=1e9, max_participation=0.02,
        cash=2_000, lot_size=100, costs=COSTS)
    assert (shares, notional, cost) == (0, 0.0, 0.0)


def test_fit_buy_with_unknown_turnover_falls_back_to_the_target_weight():
    shares, _, _ = sizing.fit_buy(
        px=10.0, want_value=10_000, adv20=float("nan"), max_participation=0.02,
        cash=1_000_000, lot_size=100, costs=COSTS)
    assert shares == 1_000       # not zero: no cap, not "no order"


def test_fit_buy_refuses_a_non_positive_deficit():
    for want in (0.0, -500.0):
        assert sizing.fit_buy(10.0, want, 1e9, 0.02, 1e6, 100, COSTS)[0] == 0


def test_fit_buy_refuses_when_there_is_no_cash():
    assert sizing.fit_buy(10.0, 10_000, 1e9, 0.02, 0.0, 100, COSTS)[0] == 0


def test_fit_buy_impact_cost_grows_with_participation():
    """The same order costs more against lower reference turnover."""
    _, n1, c1 = sizing.fit_buy(10.0, 2_000, 1e9, 1.0, 1e6, 100, COSTS)
    _, n2, c2 = sizing.fit_buy(10.0, 2_000, 4_000, 1.0, 1e6, 100, COSTS)
    assert n1 == n2 == 2_000
    assert c2 > c1



def test_fit_sell_trims_in_whole_lots():
    assert sizing.fit_sell(10.0, 2_050, 100) == 200
    assert sizing.fit_sell(10.0, 999, 100) == 0



def _row(open_=10.0, tradable=True, lu=False, ld=False) -> pd.Series:
    return pd.Series({"open": open_, "tradable": tradable,
                      "open_limit_up": lu, "open_limit_down": ld})


def test_can_buy_requires_a_price_a_tradable_bar_and_an_open_off_limit_up():
    assert sizing.can_buy(_row())
    assert not sizing.can_buy(_row(open_=0.0))
    assert not sizing.can_buy(_row(tradable=False))
    assert not sizing.can_buy(_row(lu=True))


def test_can_buy_is_judged_at_the_open_not_the_close():
    """Buying eligibility depends only on the open flags, not the later close."""
    assert sizing.can_buy(_row(open_=10.0, lu=False))
    assert list(_row().index) == ["open", "tradable", "open_limit_up", "open_limit_down"]


def test_can_sell_blocks_a_limit_down_and_a_halt():
    assert sizing.can_sell(_row())
    assert not sizing.can_sell(_row(ld=True))
    assert not sizing.can_sell(_row(tradable=False))


def test_settled_enforces_t_plus_one_by_date_not_by_elapsed_days():
    """A Friday purchase is sellable on Monday, but not on Friday."""
    fri, mon = pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")
    assert not sizing.settled(fri, fri)
    assert sizing.settled(fri, mon)


def test_settlement_t0_lets_a_position_be_sold_the_day_it_was_bought():
    # Convertible bonds on the same exchanges settle same-day.
    d = pd.Timestamp("2024-01-05")
    assert sizing.settled(d, d, settlement_t0=True)


def test_limit_prices_round_half_up_like_the_exchange():
    """27.95 * 1.1 = 30.745 -> 30.75; float rounding gives 30.74 and flags a normal open."""
    up, dn = sizing.limit_prices(pd.Series([27.95, 10.0]), 0.10)
    assert list(up) == [30.75, 11.0]
    assert list(dn) == [25.16, 9.0]   # 27.95 * 0.9 = 25.155 -> 25.16
