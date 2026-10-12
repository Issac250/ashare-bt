"""Three things the engine could not express, all on the SELL side's aftermath.

When an exit rule sells a position, the default is that the cash waits until
the next calendar rebalance, up to `rebalance_days` away. On a book whose score
gradient is steep that is a real decision, so it is configurable.

  refill            redeploy the freed slot the next morning instead of waiting
  cash_annual_rate  pay interest on idle cash instead of assuming 0%
  rebalance_dates   rebalance when the INFORMATION arrives rather than every N
                    days, so a fixed calendar does not re-pick a ranking
                    that has not moved
"""
from __future__ import annotations

import pandas as pd
from test_engine import DATES, flat_series, make_bars

from ashare_bt.costs import CostModel
from ashare_bt.engine import BacktestConfig, run_backtest

FREE = CostModel(commission_rate=0.0, commission_min=0.0, stamp_duty=0.0,
                 transfer_fee=0.0, slippage_bps=0.0, impact_coef=0.0)


def _sig(scores: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{"date": d, "code": c, "score": s}
                         for d in DATES for c, s in scores.items()])


def _cfg(**kw):
    base = dict(capital=1_000_000.0, n_positions=2, rebalance_days=20,
                max_participation=1.0, costs=FREE)
    base.update(kw)
    return BacktestConfig(**base)


def test_cash_earns_nothing_by_default():
    """An all-cash book must stay exactly flat unless a rate is asked for."""
    bars = make_bars({"a": flat_series("a", 10.0)})
    res = run_backtest(_sig({"a": 1.0}), bars, _cfg(n_positions=2, capital=1_000_000.0))
    # half the book is cash (1 name, n_positions=2); with no rate it must not grow
    plain = res.equity
    with_rate = run_backtest(_sig({"a": 1.0}), bars,
                             _cfg(n_positions=2, cash_annual_rate=0.0)).equity
    pd.testing.assert_series_equal(plain, with_rate)


def test_cash_rate_compounds_on_the_idle_half():
    """2% a year on a book that is half cash must show up, and only on the cash."""
    bars = make_bars({"a": flat_series("a", 10.0)})
    flat = run_backtest(_sig({"a": 1.0}), bars, _cfg()).equity
    paid = run_backtest(_sig({"a": 1.0}), bars, _cfg(cash_annual_rate=0.02)).equity
    assert paid.iloc[-1] > flat.iloc[-1]
    # ~39 trading days of 2%/yr on ~half the book
    gain = paid.iloc[-1] / flat.iloc[-1] - 1
    assert 0.0005 < gain < 0.003, gain


def test_refill_redeploys_an_exited_slot_before_the_next_rebalance():
    """Exit 'a' on day 2. With refill the freed slot buys 'c' the next morning;
    without it the money waits for the rebalance 20 days out."""
    bars = make_bars({"a": flat_series("a", 10.0), "b": flat_series("b", 10.0),
                      "c": flat_series("c", 10.0)})
    sig = _sig({"a": 3.0, "b": 2.0, "c": 1.0})
    exits = pd.DataFrame([{"date": DATES[2], "code": "a"}])
    off = run_backtest(sig, bars, _cfg(), exit_flags=exits)
    on = run_backtest(sig, bars, _cfg(refill=True), exit_flags=exits)
    day = pd.Timestamp(DATES[6])
    assert off.daily.loc[day, "n_positions"] == 1
    assert on.daily.loc[day, "n_positions"] == 2
    bought = on.trades[(on.trades["side"] == "buy") & (on.trades["code"] == "c")]
    assert len(bought) == 1
    assert not len(off.trades[(off.trades["side"] == "buy") & (off.trades["code"] == "c")])


def test_refill_never_rebuys_the_name_it_just_exited():
    bars = make_bars({"a": flat_series("a", 10.0), "b": flat_series("b", 10.0)})
    exits = pd.DataFrame([{"date": DATES[2], "code": "a"}])
    res = run_backtest(_sig({"a": 3.0, "b": 2.0}), bars, _cfg(refill=True), exit_flags=exits)
    same_day = res.trades[(res.trades["side"] == "buy") & (res.trades["code"] == "a")]
    assert same_day["date"].min() != pd.Timestamp(DATES[3])


def test_explicit_rebalance_dates_replace_the_calendar():
    """With rebalance_dates given, no trade may happen on any other signal day."""
    bars = make_bars({"a": flat_series("a", 10.0), "b": flat_series("b", 10.0)})
    picked = {pd.Timestamp(DATES[0]), pd.Timestamp(DATES[17])}
    res = run_backtest(_sig({"a": 3.0, "b": 2.0}), bars,
                       _cfg(rebalance_days=5, rebalance_dates=picked))
    # trades execute the day AFTER a signal date
    ok = {pd.Timestamp(DATES[1]), pd.Timestamp(DATES[18])}
    assert set(res.trades["date"].unique()) <= ok
