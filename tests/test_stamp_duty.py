from __future__ import annotations

import pandas as pd
import pytest
from test_engine import make_bars

from ashare_bt.costs import CostModel, stamp_duty_on
from ashare_bt.engine import BacktestConfig, run_backtest


@pytest.mark.parametrize("date,buy,sell", [
    ("2001-11-16", 0.002, 0.002),
    ("2005-01-24", 0.001, 0.001),
    ("2007-05-29", 0.001, 0.001),
    ("2007-05-30", 0.003, 0.003),
    ("2008-04-24", 0.001, 0.001),
    ("2008-09-18", 0.001, 0.001),
    ("2008-09-19", 0.0, 0.001),
    ("2023-08-27", 0.0, 0.001),
    ("2023-08-28", 0.0, 0.0005),
    ("2026-01-05", 0.0, 0.0005),
])
def test_schedule_boundaries(date, buy, sell):
    assert stamp_duty_on(date) == (buy, sell)


def test_dates_before_the_schedule_are_refused():
    with pytest.raises(ValueError, match="2001-11-16"):
        stamp_duty_on("2001-11-15")


def test_default_model_is_unchanged_and_ignores_the_date():
    m = CostModel()
    assert m.on("2007-06-01") is m
    assert m.trade_cost(100_000, is_sell=False) == pytest.approx(
        CostModel(stamp_duty=0.0).trade_cost(100_000, is_sell=False))


ZERO = dict(commission_rate=0.0, commission_min=0.0, transfer_fee=0.0,
            slippage_bps=0.0, impact_coef=0.0)


def _first_buy(start: str, costs: CostModel):
    dates = pd.bdate_range(start, periods=10).strftime("%Y-%m-%d").tolist()
    bars = make_bars({"A": [{"date": d, "open": 10.0, "close": 10.0} for d in dates]})
    signal = pd.DataFrame({"date": [dates[0]], "code": ["A"], "score": [1.0]})
    cfg = BacktestConfig(capital=1_000_000, n_positions=1, rebalance_days=99, costs=costs)
    return run_backtest(signal, bars, cfg).trades.query("side == 'buy'").iloc[0]


def test_engine_charges_buy_side_duty_in_2007():
    buy = _first_buy("2007-06-01", CostModel(historical_stamp_duty=True, **ZERO))
    assert buy["cost"] == pytest.approx(buy["notional"] * 0.003)


def test_engine_charges_no_buy_side_duty_in_2024():
    buy = _first_buy("2024-03-01", CostModel(historical_stamp_duty=True, **ZERO))
    assert buy["cost"] == pytest.approx(0.0)
