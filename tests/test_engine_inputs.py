"""What the engine does with inputs a user is likely to get slightly wrong."""
from __future__ import annotations

import pandas as pd
import pytest
from test_engine import DATES, flat_series, make_bars

from ashare_bt.engine import BacktestConfig, run_backtest


def _signal(codes_scores):
    return pd.DataFrame([{"date": d, "code": c, "score": s}
                         for d in DATES for c, s in codes_scores])


def test_old_close_based_flag_names_are_refused_with_a_hint():
    bars = make_bars({"A": flat_series("A", 10.0)}).rename(
        columns={"open_limit_up": "at_limit_up", "open_limit_down": "at_limit_down"})
    with pytest.raises(ValueError, match="open_limit_up.*OPEN"):
        run_backtest(_signal([("A", 1.0)]), bars, BacktestConfig(n_positions=1))


def test_missing_columns_are_named():
    bars = make_bars({"A": flat_series("A", 10.0)}).drop(columns=["amount"])
    with pytest.raises(ValueError, match="amount"):
        run_backtest(_signal([("A", 1.0)]), bars, BacktestConfig(n_positions=1))


def test_unknown_limit_state_is_treated_as_at_the_limit():
    bars = make_bars({
        "A": flat_series("A", 10.0),
        "B": [{"date": d, "open": 10.0, "close": 10.0, "open_limit_up": None} for d in DATES],
    })
    res = run_backtest(_signal([("A", 1.0), ("B", 2.0)]), bars,
                       BacktestConfig(n_positions=1, rebalance_days=5))
    assert set(res.trades["code"]) == {"A"}, "a name with an unknown limit state was bought"


def test_string_and_timestamp_signal_dates_give_the_same_curve():
    bars = make_bars({"A": flat_series("A", 10.0), "B": flat_series("B", 20.0)})
    sig = _signal([("A", 1.0), ("B", 2.0)])
    cfg = BacktestConfig(n_positions=1, rebalance_days=5)
    a = run_backtest(sig, bars, cfg).equity
    b = run_backtest(sig.assign(date=pd.to_datetime(sig["date"])), bars, cfg).equity
    pd.testing.assert_series_equal(a, b)
