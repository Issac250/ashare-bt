from __future__ import annotations

import pandas as pd
import pytest

from ashare_bt import BacktestConfig, run_backtest
from ashare_bt.sizing import limit_prices
from ashare_bt.synthetic import BARS_COLUMNS, make_market, make_signal


@pytest.fixture(scope="module")
def bars():
    return make_market(seed=1)


def test_columns_and_determinism(bars):
    assert list(bars.columns) == BARS_COLUMNS
    pd.testing.assert_frame_equal(bars, make_market(seed=1))
    assert not bars.equals(make_market(seed=2))


def test_prices_stay_inside_the_daily_band(bars):
    up, dn = limit_prices(bars["preclose"], 0.10)
    assert (bars["close"] <= up).all() and (bars["close"] >= dn).all()
    assert (bars["open"] <= up).all() and (bars["open"] >= dn).all()


def test_limit_flags_describe_the_open(bars):
    up, dn = limit_prices(bars["preclose"], 0.10)
    assert (bars["open_limit_up"] == (bars["open"] >= up)).all()
    assert (bars["open_limit_down"] == (bars["open"] <= dn)).all()


def test_every_venue_rule_has_something_to_act_on(bars):
    assert bars["open_limit_up"].sum() > 0
    assert bars["open_limit_down"].sum() > 0
    assert (~bars["tradable"]).sum() > 0
    last = bars.groupby("code")["date"].max()
    assert (last < bars["date"].max()).sum() > 0, "no stock's data ends early (no delisting)"


def test_halted_days_have_no_turnover(bars):
    h = bars[~bars["tradable"]]
    assert (h["amount"] == 0).all() and (h["open"] == h["preclose"]).all()


def test_momentum_signal_uses_only_the_past(bars):
    sig = make_signal(bars, kind="momentum", lookback=5)
    one = bars[bars["code"] == bars["code"].iloc[0]].sort_values("date").reset_index(drop=True)
    d = one["date"].iloc[10]
    want = (1 + one["ret"].iloc[6:11]).prod() - 1
    got = sig[(sig["code"] == one["code"].iloc[0]) & (sig["date"] == d)]["score"].iloc[0]
    assert got == pytest.approx(want)


def test_runs_through_the_engine(bars):
    res = run_backtest(make_signal(bars), bars, BacktestConfig(capital=1_000_000, n_positions=10))
    assert len(res.equity) > 400
    assert {"buy", "sell"} <= set(res.trades["side"])


def test_unknown_signal_kind_is_refused(bars):
    with pytest.raises(ValueError, match="kind"):
        make_signal(bars, kind="oracle")


def test_limit_flags_match_half_up_limit_prices():
    """Seed 3 contains opens one fen below a half-up limit (e.g. preclose 27.95)."""
    from decimal import ROUND_HALF_UP, Decimal
    bars = make_market(seed=3)
    up = bars["preclose"].map(lambda p: float((Decimal(str(p)) * Decimal("1.1")).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP)))
    assert (bars["open_limit_up"] == (bars["open"] >= up)).all()
