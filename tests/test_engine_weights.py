"""Relative position weights, equal-weight defaults, and cash for unfilled slots."""
from __future__ import annotations

import pandas as pd
import pytest
from test_engine import DATES, flat_series, make_bars

from ashare_bt.costs import CostModel
from ashare_bt.engine import BacktestConfig, run_backtest

FREE = CostModel(commission_rate=0.0, commission_min=0.0, stamp_duty=0.0,
                 transfer_fee=0.0, slippage_bps=0.0, impact_coef=0.0)


def _signal(codes_scores: dict[str, float], weights: dict[str, float] | None = None) -> pd.DataFrame:
    rows = []
    for d in DATES:
        for code, sc in codes_scores.items():
            r = {"date": d, "code": code, "score": sc}
            if weights is not None:
                r["w"] = weights[code]
            rows.append(r)
    return pd.DataFrame(rows)


def _cfg(n=2):
    return BacktestConfig(capital=1_000_000.0, n_positions=n, rebalance_days=5,
                          max_participation=1.0, costs=FREE)


def test_constant_weights_reproduce_equal_weight_exactly():
    bars = make_bars({"a": flat_series("a", 10.0), "b": flat_series("b", 25.0)})
    plain = run_backtest(_signal({"a": 2.0, "b": 1.0}), bars, _cfg())
    with_w = run_backtest(_signal({"a": 2.0, "b": 1.0}, {"a": 0.5, "b": 0.5}), bars, _cfg())
    pd.testing.assert_series_equal(plain.equity, with_w.equity)


def test_weights_are_honoured_in_position_values():
    """3:1 weights apply to position values, regardless of stock price."""
    bars = make_bars({"a": flat_series("a", 10.0), "b": flat_series("b", 25.0)})
    res = run_backtest(_signal({"a": 2.0, "b": 1.0}, {"a": 0.75, "b": 0.25}), bars, _cfg())
    buys = res.trades[res.trades["side"] == "buy"]
    first = buys[buys["date"] == buys["date"].min()].set_index("code")["notional"]
    assert first["a"] / first["b"] == pytest.approx(3.0, rel=0.02)


def test_weights_do_not_lever_when_a_name_is_missing():
    """Two qualifying stocks out of three target slots leave about one-third in cash."""
    bars = make_bars({"a": flat_series("a", 10.0), "b": flat_series("b", 25.0)})
    res = run_backtest(_signal({"a": 2.0, "b": 1.0}, {"a": 0.5, "b": 0.5}), bars, _cfg(n=3))
    invested = res.daily["n_positions"].iloc[-1]
    assert invested == 2
    held = res.daily["equity"].iloc[-1] - res.daily["cash"].iloc[-1]
    assert 0.60 <= held / res.daily["equity"].iloc[-1] <= 0.70
