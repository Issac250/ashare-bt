"""Execution-rule tests for the backtest engine.

These encode the A-share rules that silently inflate returns when they are
missing. Each test is built on a tiny synthetic panel so a failure points at
one rule rather than at the data.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ashare_bt.costs import CostModel
from ashare_bt.engine import BacktestConfig, run_backtest


def make_bars(spec: dict[str, list[dict]]) -> pd.DataFrame:
    """spec: {code: [ {date, open, close, preclose, ret, ...}, ... ]}

    `preclose` defaults to the PREVIOUS bar's raw close, which is the
    no-corporate-action case. Pass it explicitly to model a split or a
    dividend: baostock's preclose is already adjusted for both, so
    `preclose != prev_close` is exactly how a corporate action shows up.
    """
    rows = []
    for code, recs in spec.items():
        prev_close = None
        for r in recs:
            rows.append({
                "code": code,
                "date": r["date"],
                "open": r["open"],
                "close": r["close"],
                "preclose": r.get("preclose", prev_close if prev_close is not None else r["open"]),
                "ret": r.get("ret", 0.0),
                "amount": r.get("amount", 1e9),
                "tradable": r.get("tradable", True),
                "open_limit_up": r.get("open_limit_up", False),
                "open_limit_down": r.get("open_limit_down", False),
            })
            prev_close = r["close"]
    return pd.DataFrame(rows)


DATES = pd.bdate_range("2024-01-01", periods=40).strftime("%Y-%m-%d").tolist()


def flat_series(code: str, price: float, n: int = 40, **kw) -> list[dict]:
    return [{"date": DATES[i], "open": price, "close": price, **kw} for i in range(n)]


def test_trades_execute_on_the_day_after_the_signal():
    """A signal dated t must transact at t+1's open, never at t's price."""
    # 'winner' is cheap on day 0 and expensive from day 1 onward. If the engine
    # executed on the signal date it would buy at 10; executing at t+1 pays 20.
    bars = make_bars({
        "A": [{"date": d, "open": 10.0 if i == 0 else 20.0,
               "close": 10.0 if i == 0 else 20.0} for i, d in enumerate(DATES)],
    })
    signal = pd.DataFrame({"date": [DATES[0]], "code": ["A"], "score": [1.0]})
    cfg = BacktestConfig(capital=1_000_000, n_positions=1, rebalance_days=1,
                         costs=CostModel(commission_min=0.0, slippage_bps=0.0, impact_coef=0.0))
    res = run_backtest(signal, bars, cfg)
    buys = res.trades[res.trades.side == "buy"]
    assert len(buys) == 1
    # notional/shares recovers the fill price; it must be 20, not 10.
    fill = buys.iloc[0]["notional"]
    assert fill > 0
    assert res.trades.iloc[0]["date"] == pd.Timestamp(DATES[1]), "must fill on t+1"


def test_cannot_buy_a_stock_at_limit_up():
    bars = make_bars({
        "A": [{"date": d, "open": 10.0, "close": 10.0, "open_limit_up": True} for d in DATES],
        "B": [{"date": d, "open": 10.0, "close": 10.0} for d in DATES],
    })
    # A scores higher, but is unbuyable at limit-up, so the book must hold B.
    signal = pd.DataFrame({"date": [DATES[0], DATES[0]], "code": ["A", "B"], "score": [2.0, 1.0]})
    cfg = BacktestConfig(capital=1_000_000, n_positions=1, rebalance_days=1)
    res = run_backtest(signal, bars, cfg)
    bought = set(res.trades[res.trades.side == "buy"]["code"])
    assert "A" not in bought, "limit-up stock must not be purchased"
    assert "B" in bought


def test_cannot_sell_into_limit_down():
    """Once held, a limit-down stock cannot be exited that day."""
    bars = make_bars({
        "A": [{"date": d, "open": 10.0, "close": 10.0,
               "open_limit_down": i >= 2} for i, d in enumerate(DATES)],
        "B": [{"date": d, "open": 10.0, "close": 10.0} for d in DATES],
    })
    # Buy A first, then switch the signal to B; A is stuck at limit-down.
    signal = pd.concat([
        pd.DataFrame({"date": [DATES[0]], "code": ["A"], "score": [2.0]}),
        pd.DataFrame({"date": DATES[1:6], "code": ["B"] * 5, "score": [2.0] * 5}),
    ])
    cfg = BacktestConfig(capital=1_000_000, n_positions=1, rebalance_days=1)
    res = run_backtest(signal, bars, cfg)
    sells = res.trades[res.trades.side == "sell"]
    assert "A" not in set(sells["code"]), "must not sell a limit-down stock"


def test_t_plus_one_blocks_same_day_exit():
    """A-share settles T+1: a stock bought today cannot be sold today."""
    bars = make_bars({
        "A": [{"date": d, "open": 10.0, "close": 10.0} for d in DATES],
        "B": [{"date": d, "open": 10.0, "close": 10.0} for d in DATES],
    })
    # Signal flips every day, so a naive engine would buy and sell on one day.
    sig = []
    for i, d in enumerate(DATES[:6]):
        sig.append({"date": d, "code": "A" if i % 2 == 0 else "B", "score": 2.0})
    signal = pd.DataFrame(sig)
    cfg = BacktestConfig(capital=1_000_000, n_positions=1, rebalance_days=1)
    res = run_backtest(signal, bars, cfg)
    t = res.trades
    for day in t["date"].unique():
        day_t = t[t["date"] == day]
        for code in day_t["code"].unique():
            sides = set(day_t[day_t["code"] == code]["side"])
            assert sides != {"buy", "sell"}, f"{code} bought and sold on {day} (violates T+1)"


def test_costs_reduce_returns():
    bars = make_bars({"A": flat_series("A", 10.0)})
    signal = pd.DataFrame({"date": DATES[:10], "code": ["A"] * 10, "score": [1.0] * 10})
    free = BacktestConfig(capital=1_000_000, n_positions=1, rebalance_days=1,
                          costs=CostModel(commission_rate=0, commission_min=0, stamp_duty=0,
                                          transfer_fee=0, slippage_bps=0, impact_coef=0))
    dear = BacktestConfig(capital=1_000_000, n_positions=1, rebalance_days=1,
                          costs=CostModel(slippage_bps=50.0))
    r_free = run_backtest(signal, bars, free)
    r_dear = run_backtest(signal, bars, dear)
    assert r_dear.equity.iloc[-1] < r_free.equity.iloc[-1]
    assert r_dear.metrics["total_cost"] > 0


def test_participation_cap_limits_position_size():
    """A thin stock must not absorb a full equal-weight slice."""
    thin = make_bars({"A": [{"date": d, "open": 10.0, "close": 10.0, "amount": 1_000_000.0}
                            for d in DATES]})
    signal = pd.DataFrame({"date": [DATES[25]], "code": ["A"], "score": [1.0]})
    cfg = BacktestConfig(capital=100_000_000, n_positions=1, rebalance_days=1,
                         max_participation=0.01)
    res = run_backtest(signal, thin, cfg)
    buys = res.trades[res.trades.side == "buy"]
    assert len(buys) == 1
    # 1% of ¥1,000,000 median turnover = ¥10,000 cap, not the ¥100m of capital.
    assert buys.iloc[0]["notional"] <= 10_000 * 1.001


def test_whole_lots_only():
    """A-share trades in 100-share lots."""
    bars = make_bars({"A": flat_series("A", 33.0)})
    signal = pd.DataFrame({"date": [DATES[25]], "code": ["A"], "score": [1.0]})
    cfg = BacktestConfig(capital=100_000, n_positions=1, rebalance_days=1)
    res = run_backtest(signal, bars, cfg)
    buys = res.trades[res.trades.side == "buy"]
    if len(buys):
        shares = buys.iloc[0]["notional"] / 33.0
        assert abs(shares % 100) < 1e-6, f"{shares} is not a whole lot"


def test_t_plus_one_does_not_bind_in_this_engine_so_t0_changes_nothing():
    """Convertibles settle T+0. One might expect the engine's T+1 rule to
    understate them. It does not.

    The engine transacts once per execution day and runs SELLS BEFORE BUYS, so a name
    bought on an execution day is never a sell candidate on that same day. The
    `bought_on >= ex_day` guard is unreachable under a normal calendar, which means
    settlement_t0 is inert here. It is kept because it documents the venue rule and
    would bind if execution ever became intraday -- but it must not silently change
    any existing result, which is what this test pins down.
    """
    bars = make_bars({
        "A": [{"date": d, "open": 10.0, "close": 10.0 + i * 0.1} for i, d in enumerate(DATES)],
        "B": [{"date": d, "open": 10.0, "close": 10.0 - i * 0.05} for i, d in enumerate(DATES)],
    })
    sig = [{"date": d, "code": "A" if i % 2 == 0 else "B", "score": 2.0}
           for i, d in enumerate(DATES[:10])]
    signal = pd.DataFrame(sig)
    base = dict(capital=1_000_000, n_positions=1, rebalance_days=1)
    r_t1 = run_backtest(signal, bars, BacktestConfig(**base))
    r_t0 = run_backtest(signal, bars, BacktestConfig(**base, settlement_t0=True))
    pd.testing.assert_series_equal(r_t1.equity, r_t0.equity)
    assert len(r_t1.trades) == len(r_t0.trades)

    # and neither one ever round-trips a name within one day
    for res in (r_t1, r_t0):
        t = res.trades
        for day in t["date"].unique():
            day_t = t[t["date"] == day]
            for code in day_t["code"].unique():
                assert set(day_t[day_t["code"] == code]["side"]) != {"buy", "sell"}


def test_lot_size_is_configurable_for_convertible_bonds():
    """Convertibles trade in lots of 10 (¥1,000 face), not 100 shares. With capital
    that cannot fill a 100-lot, the bond lot must still allow a position."""
    bars = make_bars({"A": flat_series("A", 130.0)})
    signal = pd.DataFrame({"date": [DATES[25]], "code": ["A"], "score": [1.0]})
    base = dict(capital=50_000, n_positions=10, rebalance_days=1)

    def bought(res):
        t = res.trades
        if not len(t):
            return 0.0
        return t[t["side"] == "buy"]["notional"].sum()

    res100 = run_backtest(signal, bars, BacktestConfig(**base))
    res10 = run_backtest(signal, bars, BacktestConfig(**base, lot_size=10))
    n100 = bought(res100)
    n10 = bought(res10)
    # ¥5,000 per name buys zero 100-lots at ¥130 (¥13,000) but three 10-lots.
    assert n100 == 0
    assert n10 > 0
    shares = n10 / 130.0
    assert abs(shares % 10) < 1e-6, f"{shares} is not a whole 10-lot"


def test_lot_size_must_be_positive():
    with pytest.raises(ValueError):
        BacktestConfig(lot_size=0)


def test_zero_stamp_duty_is_cheaper_to_sell():
    """Convertibles pay no stamp duty. It is already a CostModel field, so bond mode
    needs no code change -- this test exists so that stays true."""
    bars = make_bars({
        "A": [{"date": d, "open": 10.0, "close": 10.0} for d in DATES],
        "B": [{"date": d, "open": 10.0, "close": 10.0} for d in DATES],
    })
    sig = [{"date": d, "code": "A" if i % 2 == 0 else "B", "score": 2.0}
           for i, d in enumerate(DATES[:10])]
    signal = pd.DataFrame(sig)
    base = dict(capital=1_000_000, n_positions=1, rebalance_days=1)
    equity = dict(costs=CostModel())
    bond = dict(costs=CostModel(stamp_duty=0.0))
    r_eq = run_backtest(signal, bars, BacktestConfig(**base, **equity))
    r_bd = run_backtest(signal, bars, BacktestConfig(**base, **bond))
    assert r_bd.metrics["total_cost"] < r_eq.metrics["total_cost"]


def test_short_side_is_refused_rather_than_faked():
    bars = make_bars({"A": flat_series("A", 10.0)})
    signal = pd.DataFrame({"date": [DATES[0]], "code": ["A"], "score": [1.0]})
    with pytest.raises(NotImplementedError):
        run_backtest(signal, bars, BacktestConfig(long_only=False))


def test_delisted_holding_is_liquidated_not_stranded():
    """A stock whose data ends must release its capital, not freeze it.

    Carrying a delisted name at a frozen price forever would let a backtest
    park capital in a stock that no longer exists.
    """
    # A trades for 10 days then disappears; B trades throughout so the loop
    # keeps advancing past A's final bar.
    bars = make_bars({
        "A": [{"date": DATES[i], "open": 10.0, "close": 10.0} for i in range(10)],
        "B": [{"date": d, "open": 5.0, "close": 5.0} for d in DATES],
    })
    # Rebalance ONCE (buying A), then hold while the clock runs on. Signal rows
    # must exist on later dates or the engine has no days left to simulate.
    signal = pd.concat([
        pd.DataFrame({"date": [DATES[0]], "code": ["A"], "score": [2.0]}),
        pd.DataFrame({"date": DATES[1:20], "code": ["A"] * 19, "score": [2.0] * 19}),
    ])
    cfg = BacktestConfig(capital=100_000, n_positions=1, rebalance_days=99)
    res = run_backtest(signal, bars, cfg)
    sides = set(res.trades["side"])
    assert "delist" in sides, f"delisted holding never liquidated (sides={sides})"
    assert res.daily["n_positions"].iloc[-1] == 0
    # Capital comes back as cash rather than vanishing.
    assert res.daily["cash"].iloc[-1] > 0.9 * 100_000


def test_score_band_avoids_the_extreme_tail():
    """A band must skip the highest-scoring names, which can be falling knives.

    For a factor that inverts in the tail, selecting a band rather than the
    extreme is a correctness requirement, not a preference.
    """
    codes = [f"C{i:03d}" for i in range(100)]
    bars = make_bars({c: [{"date": d, "open": 10.0, "close": 10.0} for d in DATES]
                      for c in codes})
    # Score ascending with the index: C099 is the most extreme.
    signal = pd.DataFrame({"date": [DATES[25]] * 100, "code": codes,
                           "score": [float(i) for i in range(100)]})
    top = run_backtest(signal, bars, BacktestConfig(
        capital=5_000_000, n_positions=5, rebalance_days=1))
    banded = run_backtest(signal, bars, BacktestConfig(
        capital=5_000_000, n_positions=5, rebalance_days=1, score_band=(0.50, 0.80)))

    top_picks = set(top.trades[top.trades.side == "buy"]["code"])
    band_picks = set(banded.trades[banded.trades.side == "buy"]["code"])
    assert "C099" in top_picks, "unbanded run should hold the most extreme name"
    assert "C099" not in band_picks, "banded run must exclude the extreme tail"
    # A 50-80th percentile band cannot contain the very top names.
    assert max(int(c[1:]) for c in band_picks) <= 82


# --------------------------------------------------------------------------
# Weight rebalancing. Held positions are trimmed back toward target weight;
# without this a stock that tripled would keep 3x weight forever and the book
# would silently concentrate into past winners.

def _two_stock_bars(a_path, b_path):
    return make_bars({
        "A": [{"date": d, "open": a_path[i], "close": a_path[i]} for i, d in enumerate(DATES)],
        "B": [{"date": d, "open": b_path[i], "close": b_path[i]} for i, d in enumerate(DATES)],
    })


def _weights(res, bars, day_idx):
    """Value weights of the book after the rebalance on DATES[day_idx].

    Each trade's share count must be recovered at ITS OWN price, not at the
    measurement day's -- a first version of this divided every notional by the
    final price and reported an equal-weighted book as 0.62/0.38.
    """
    at = {(r["date"], r["code"]): r["open"] for _, r in bars.iterrows()}
    end_px = {r["code"]: r["open"] for _, r in
              bars[bars["date"] == DATES[day_idx]].iterrows()}
    net = {}
    for _, tr in res.trades[res.trades["date"] <= pd.Timestamp(DATES[day_idx])].iterrows():
        key = (tr["date"].strftime("%Y-%m-%d"), tr["code"])
        px_then = at[key]
        sign = 1 if tr["side"] == "buy" else -1
        net[tr["code"]] = net.get(tr["code"], 0.0) + sign * tr["notional"] / px_then
    vals = {c: sh * end_px[c] for c, sh in net.items() if sh > 1e-9}
    tot = sum(vals.values())
    return {c: v / tot for c, v in vals.items()} if tot else {}


NOCOST = CostModel(commission_rate=0.0, commission_min=0.0, stamp_duty=0.0,
                   transfer_fee=0.0, slippage_bps=0.0, impact_coef=0.0)


def test_a_winner_is_trimmed_back_toward_equal_weight():
    """A doubles while B is flat; the next rebalance must sell some A."""
    a = [10.0] * 5 + [20.0] * 35        # A doubles on day 5
    b = [10.0] * 40
    bars = _two_stock_bars(a, b)
    sig = pd.DataFrame({"date": DATES[:20] * 2,
                        "code": ["A"] * 20 + ["B"] * 20,
                        "score": [1.0] * 20 + [0.9] * 20})
    cfg = BacktestConfig(capital=1_000_000, n_positions=2, rebalance_days=10,
                         max_participation=1.0, costs=NOCOST)
    res = run_backtest(sig, bars, cfg)
    w = _weights(res, bars, 11)
    assert set(w) == {"A", "B"}
    # Without trimming A would sit near 2/3 of the book. Equal weight is 1/2.
    assert w["A"] < 0.58, f"A still at {w['A']:.2f} of the book -- not trimmed"


def test_rebalance_weights_false_reproduces_the_old_drift():
    """Drifting weights must stay reachable as an explicit choice."""
    a = [10.0] * 5 + [20.0] * 35
    b = [10.0] * 40
    bars = _two_stock_bars(a, b)
    sig = pd.DataFrame({"date": DATES[:20] * 2,
                        "code": ["A"] * 20 + ["B"] * 20,
                        "score": [1.0] * 20 + [0.9] * 20})
    cfg = BacktestConfig(capital=1_000_000, n_positions=2, rebalance_days=10,
                         max_participation=1.0, rebalance_weights=False, costs=NOCOST)
    res = run_backtest(sig, bars, cfg)
    w = _weights(res, bars, 11)
    assert w["A"] > 0.60, f"A at {w['A']:.2f} -- drift should be preserved when off"


def test_a_top_up_adds_to_the_position_rather_than_replacing_it():
    """A halves; topping it back up must not discard the shares already held,
    and must fund only the deficit rather than a second full position."""
    a = [20.0] * 5 + [10.0] * 35        # A halves on day 5
    b = [10.0] * 40
    bars = _two_stock_bars(a, b)
    sig = pd.DataFrame({"date": DATES[:20] * 2,
                        "code": ["A"] * 20 + ["B"] * 20,
                        "score": [1.0] * 20 + [0.9] * 20})
    cfg = BacktestConfig(capital=1_000_000, n_positions=2, rebalance_days=10,
                         max_participation=1.0, costs=NOCOST)
    res = run_backtest(sig, bars, cfg)
    w = _weights(res, bars, 11)
    assert 0.42 < w["A"] < 0.58, f"A at {w['A']:.2f} -- top-up mis-sized"


def test_rebalancing_never_spends_more_cash_than_it_has():
    """Trims fund the top-ups, so the order matters; cash must stay >= 0."""
    a = [10.0] * 5 + [30.0] * 35
    b = [10.0] * 5 + [3.0] * 35
    bars = _two_stock_bars(a, b)
    sig = pd.DataFrame({"date": DATES[:20] * 2,
                        "code": ["A"] * 20 + ["B"] * 20,
                        "score": [1.0] * 20 + [0.9] * 20})
    cfg = BacktestConfig(capital=1_000_000, n_positions=2, rebalance_days=10,
                         max_participation=1.0, costs=NOCOST)
    res = run_backtest(sig, bars, cfg)
    assert (res.daily["cash"] >= -1e-6).all(), "went cash-negative while rebalancing"


# --------------------------------------------------------------------------
# Exposure scaling: a risk control that SCALES the book (e.g. volatility
# targeting), as opposed to a binary in/out overlay.

def test_exposure_half_holds_about_half_the_equity_in_stocks():
    bars = _two_stock_bars([10.0] * 40, [10.0] * 40)
    sig = pd.DataFrame({"date": DATES[:20] * 2,
                        "code": ["A"] * 20 + ["B"] * 20,
                        "score": [1.0] * 20 + [0.9] * 20})
    cfg = BacktestConfig(capital=1_000_000, n_positions=2, rebalance_days=10,
                         max_participation=1.0, costs=NOCOST)
    expo = pd.Series(0.5, index=pd.to_datetime(DATES[:20]))
    res = run_backtest(sig, bars, cfg, exposure=expo)
    # Cash on the day after the first rebalance should be about half.
    cash = res.daily["cash"].iloc[1]
    assert 0.45 * 1_000_000 < cash < 0.55 * 1_000_000, f"cash {cash:,.0f}, expected ~500k"


def test_falling_exposure_sells_the_book_down():
    """Scaling down must actually liquidate, not merely stop buying."""
    bars = _two_stock_bars([10.0] * 40, [10.0] * 40)
    sig = pd.DataFrame({"date": DATES[:20] * 2,
                        "code": ["A"] * 20 + ["B"] * 20,
                        "score": [1.0] * 20 + [0.9] * 20})
    expo = pd.Series([1.0] * 10 + [0.2] * 10, index=pd.to_datetime(DATES[:20]))
    cfg = BacktestConfig(capital=1_000_000, n_positions=2, rebalance_days=10,
                         max_participation=1.0, costs=NOCOST)
    res = run_backtest(sig, bars, cfg, exposure=expo)
    sells = res.trades[res.trades["side"] == "sell"]
    assert len(sells) >= 2, "exposure fell to 0.2 and nothing was sold"
    assert res.daily["cash"].iloc[-1] > 0.7 * res.equity.iloc[-1]


def test_exposure_above_one_is_clipped_not_leveraged():
    bars = _two_stock_bars([10.0] * 40, [10.0] * 40)
    sig = pd.DataFrame({"date": DATES[:20] * 2,
                        "code": ["A"] * 20 + ["B"] * 20,
                        "score": [1.0] * 20 + [0.9] * 20})
    cfg = BacktestConfig(capital=1_000_000, n_positions=2, rebalance_days=10,
                         max_participation=1.0, costs=NOCOST)
    res = run_backtest(sig, bars, cfg,
                       exposure=pd.Series(3.0, index=pd.to_datetime(DATES[:20])))
    assert (res.daily["cash"] >= -1e-6).all(), "leverage slipped through"


# ---------------------------------------------------------------------------
# CORPORATE ACTIONS
#
# Valuing a position as shares x RAW price, with raw (unadjusted) prices,
# makes a 4-for-1 split look like a -75% day: the price quarters while the
# stored share count does not quadruple. Dividends become small fake losses.
# ---------------------------------------------------------------------------

def test_a_four_for_one_split_is_not_a_75_percent_loss():
    """The real case: sh.603636 2017-04-07, prev close 96.92 -> close 26.61."""
    recs = []
    for i, d in enumerate(DATES):
        if i < 10:
            recs.append({"date": d, "open": 100.0, "close": 100.0})
        elif i == 10:
            # 4-for-1: preclose is adjusted to 25, so the true return is 0%.
            recs.append({"date": d, "open": 25.0, "close": 25.0, "preclose": 25.0})
        else:
            recs.append({"date": d, "open": 25.0, "close": 25.0})
    bars = make_bars({"A": recs})
    # Signal on EVERY date: the engine emits one equity row per signal date, so
    # a one-row signal makes every return assertion pass vacuously.
    signal = pd.DataFrame({"date": DATES, "code": ["A"] * len(DATES),
                           "score": [1.0] * len(DATES)})
    cfg = BacktestConfig(capital=1_000_000, n_positions=1, rebalance_days=100,
                         costs=CostModel(commission_min=0.0, slippage_bps=0.0, impact_coef=0.0))
    res = run_backtest(signal, bars, cfg)
    total = res.metrics["total_return"]
    assert total > -0.02, f"split read as a real loss: {total:.1%}"


def test_a_cash_dividend_is_not_a_loss():
    """preclose drops by the dividend, so pctChg is already a total return."""
    recs = []
    for i, d in enumerate(DATES):
        if i < 10:
            recs.append({"date": d, "open": 100.0, "close": 100.0})
        elif i == 10:
            recs.append({"date": d, "open": 98.0, "close": 98.0, "preclose": 98.0})
        else:
            recs.append({"date": d, "open": 98.0, "close": 98.0})
    bars = make_bars({"A": recs})
    # Signal on EVERY date: the engine emits one equity row per signal date, so
    # a one-row signal makes every return assertion pass vacuously.
    signal = pd.DataFrame({"date": DATES, "code": ["A"] * len(DATES),
                           "score": [1.0] * len(DATES)})
    cfg = BacktestConfig(capital=1_000_000, n_positions=1, rebalance_days=100,
                         costs=CostModel(commission_min=0.0, slippage_bps=0.0, impact_coef=0.0))
    res = run_backtest(signal, bars, cfg)
    assert res.metrics["total_return"] > -0.01


def test_a_genuine_price_fall_is_still_a_loss():
    """The fix must not launder real losses: preclose == prev close here."""
    recs = []
    for i, d in enumerate(DATES):
        px = 100.0 if i < 10 else 25.0
        recs.append({"date": d, "open": px, "close": px})
    bars = make_bars({"A": recs})            # preclose defaults to prev close = 100
    # Signal on EVERY date: the engine emits one equity row per signal date, so
    # a one-row signal makes every return assertion pass vacuously.
    signal = pd.DataFrame({"date": DATES, "code": ["A"] * len(DATES),
                           "score": [1.0] * len(DATES)})
    cfg = BacktestConfig(capital=1_000_000, n_positions=1, rebalance_days=100,
                         costs=CostModel(commission_min=0.0, slippage_bps=0.0, impact_coef=0.0))
    res = run_backtest(signal, bars, cfg)
    assert res.metrics["total_return"] < -0.70


def test_overnight_and_intraday_ratios_compose_to_the_total_return():
    """Split the day at the open and the two halves must multiply to 1+ret."""
    recs = []
    for i, d in enumerate(DATES):
        if i == 10:
            # gap down to 45 at the open, recover to 50 by the close, on a
            # preclose of 50 -> total return exactly 0.
            recs.append({"date": d, "open": 45.0, "close": 50.0, "preclose": 50.0})
        else:
            recs.append({"date": d, "open": 50.0, "close": 50.0})
    bars = make_bars({"A": recs})
    # Signal on EVERY date: the engine emits one equity row per signal date, so
    # a one-row signal makes every return assertion pass vacuously.
    signal = pd.DataFrame({"date": DATES, "code": ["A"] * len(DATES),
                           "score": [1.0] * len(DATES)})
    cfg = BacktestConfig(capital=1_000_000, n_positions=1, rebalance_days=100,
                         costs=CostModel(commission_min=0.0, slippage_bps=0.0, impact_coef=0.0))
    res = run_backtest(signal, bars, cfg)
    eq = res.equity
    d10, d9 = pd.Timestamp(DATES[10]), pd.Timestamp(DATES[9])
    assert abs(eq.loc[d10] / eq.loc[d9] - 1.0) < 1e-6


def test_a_thin_day_holds_cash_instead_of_concentrating():
    """Few qualifying names must mean a small book, not giant bets.

    An event strategy fires on a handful of names most days. Dividing equity
    by the number of names FOUND would turn two qualifying names into two 50%
    positions.
    """
    bars = make_bars({c: [{"date": d, "open": 10.0, "close": 10.0} for d in DATES]
                      for c in ("A", "B", "C")})
    rows = []
    for d in DATES:
        rows += [{"date": d, "code": "A", "score": 1.0},
                 {"date": d, "code": "B", "score": 0.9}]
    sig = pd.DataFrame(rows)
    cfg = BacktestConfig(capital=1_000_000, n_positions=50, rebalance_days=5,
                         costs=CostModel(commission_min=0.0, slippage_bps=0.0, impact_coef=0.0))
    res = run_backtest(sig, bars, cfg)
    last = res.daily.iloc[-1]
    invested = (last["equity"] - last["cash"]) / last["equity"]
    assert last["n_positions"] == 2
    # 2 names out of a 50-name cap = 4% invested, 96% cash.
    assert 0.02 < invested < 0.06, f"invested {invested:.1%} off a 2-name signal"


def test_a_full_breadth_signal_is_unaffected_by_the_cap_change():
    """Sizing by the cap must be a no-op when the pool always fills it."""
    codes = [f"S{i:02d}" for i in range(10)]
    bars = make_bars({c: [{"date": d, "open": 10.0, "close": 10.0} for d in DATES]
                      for c in codes})
    rows = []
    for d in DATES:
        for i, c in enumerate(codes):
            rows.append({"date": d, "code": c, "score": float(len(codes) - i)})
    sig = pd.DataFrame(rows)
    cfg = BacktestConfig(capital=1_000_000, n_positions=10, rebalance_days=5,
                         costs=CostModel(commission_min=0.0, slippage_bps=0.0, impact_coef=0.0))
    res = run_backtest(sig, bars, cfg)
    last = res.daily.iloc[-1]
    assert last["n_positions"] == 10
    assert (last["equity"] - last["cash"]) / last["equity"] > 0.95


def test_an_event_rule_buys_holds_and_returns_to_cash():
    """The non-ranking shape: buy on an event, hold N days, else sit in cash.

    This is the canonical event strategy, and it pins all four things it needs
    at once -- a complete calendar, timely entry, no daily churn, and cash
    rather than concentration when nothing qualifies. Sizing by the names
    FOUND instead of the breadth cap would turn one live event into a
    100%-of-account position.
    """
    days = pd.bdate_range("2024-01-01", periods=60).strftime("%Y-%m-%d").tolist()
    events, hold = {"A": 10, "B": 30}, 20

    def series(code):
        px, out, ev = 10.0, [], events.get(code)
        for i, d in enumerate(days):
            prev = px
            if ev is not None and ev < i <= ev + hold:
                px = round(px * 1.005, 4)
            out.append({"date": d, "open": prev, "close": px, "preclose": prev})
        return out

    bars = make_bars({c: series(c) for c in ("A", "B", "C")})

    # A row for EVERY (date, code); NaN means "no opinion". Omitting the row
    # instead would drop the date from the equity curve entirely -- 40 rows for
    # 59 trading days -- and inflate the annualisation.
    import numpy as np
    rows = []
    for i, d in enumerate(days):
        for c in ("A", "B", "C"):
            ev = events.get(c)
            live = ev is not None and ev <= i < ev + hold
            rows.append({"date": d, "code": c, "score": 1.0 if live else np.nan})
    sig = pd.DataFrame(rows)

    cfg = BacktestConfig(capital=1_000_000, n_positions=5, rebalance_days=1,
                         rebalance_weights=False,
                         costs=CostModel(commission_min=0.0, slippage_bps=0.0, impact_coef=0.0))
    res = run_backtest(sig, bars, cfg)
    d = res.daily

    # Complete calendar: one row per available trading day, not per event day.
    # `len(d)` is what actually guards that; the `years` assertion below just
    # pins the annualisation convention.
    assert len(d) == len(days) - 1
    # CALENDAR span, not trading-day count / 252. The old assertion encoded
    # `(len(days) - 1) / 252.0`: A-shares trade ~243.6 days a year, so dividing
    # by 252 understates `years` and overstates CAGR. A side effect is that a
    # calendar span cannot be inflated by dropping dates from the MIDDLE of the
    # curve, which is the failure mode the comment above describes.
    span = (pd.Timestamp(days[-1]) - pd.Timestamp(days[1])).days
    assert abs(res.metrics["years"] - span / 365.25) < 1e-9
    # Genuinely flat in cash when nothing qualifies.
    assert (d["n_positions"] == 0).sum() == 19
    assert d.iloc[0]["n_positions"] == 0 and d.iloc[-1]["n_positions"] == 0
    # One position at a time, sized to the cap (1/5), not to the count found.
    held = d[d["n_positions"] > 0]
    assert held["n_positions"].max() == 1
    inv = (held["equity"] - held["cash"]) / held["equity"]
    assert 0.19 < inv.min() and inv.max() < 0.23
    # Entry the day AFTER the event, and four trades for the whole run.
    t = res.trades
    assert len(t) == 4
    assert str(t.iloc[0]["date"].date()) == "2024-01-16"
    assert list(t["side"]) == ["buy", "sell", "buy", "sell"]


# ---------------------------------------------------------------- exit rules

def _two_name_book(aaa_recs=None):
    """AAA and BBB both flat, a signal every day so an exit has a date to act on."""
    bars = make_bars({"AAA": aaa_recs or flat_series("AAA", 10.0),
                      "BBB": flat_series("BBB", 10.0)})
    sig = pd.DataFrame([{"date": d, "code": c, "score": s}
                        for d in DATES for c, s in (("AAA", 2.0), ("BBB", 1.0))])
    return bars, sig


def test_an_exit_flag_sells_off_a_rebalance_day_and_the_cash_sits_idle():
    """The whole point: the calendar-only engine could not express this."""
    bars, sig = _two_name_book()
    cfg = BacktestConfig(capital=100_000.0, n_positions=2, rebalance_days=30,
                         costs=CostModel(slippage_bps=0.0))
    flags = pd.DataFrame([{"date": DATES[5], "code": "AAA"}])   # inside the period
    out = run_backtest(sig, bars, cfg, exit_flags=flags)

    exits = out.trades[out.trades["side"] == "exit"]
    assert len(exits) == 1 and exits.iloc[0]["code"] == "AAA"
    # Sold and NOT replaced before the next rebalance: the freed capital idles.
    after = out.daily.loc[pd.Timestamp(DATES[7]):pd.Timestamp(DATES[20])]
    assert (after["n_positions"] == 1).all()
    assert after["cash"].min() > 0.4 * cfg.capital


def test_an_exit_blocked_by_limit_down_is_not_treated_as_done():
    """A flag that could not fill must stay fillable, never silently consumed."""
    recs = flat_series("AAA", 10.0)
    recs[6]["open_limit_down"] = True          # the execution day for a day-5 signal
    bars, sig = _two_name_book(recs)
    cfg = BacktestConfig(capital=100_000.0, n_positions=2, rebalance_days=30,
                         costs=CostModel(slippage_bps=0.0))

    blocked = run_backtest(sig, bars, cfg,
                           exit_flags=pd.DataFrame([{"date": DATES[5], "code": "AAA"}]))
    assert blocked.trades[blocked.trades["side"] == "exit"].empty

    retried = run_backtest(sig, bars, cfg, exit_flags=pd.DataFrame(
        [{"date": DATES[5], "code": "AAA"}, {"date": DATES[6], "code": "AAA"}]))
    assert len(retried.trades[retried.trades["side"] == "exit"]) == 1


def test_no_exit_flags_reproduces_the_old_curve_exactly():
    """Backwards compatibility: the feature is invisible when unused."""
    bars, sig = _two_name_book()
    cfg = BacktestConfig(capital=100_000.0, n_positions=2, rebalance_days=5)
    a = run_backtest(sig, bars, cfg)
    b = run_backtest(sig, bars, cfg, exit_flags=pd.DataFrame(columns=["date", "code"]))
    pd.testing.assert_series_equal(a.equity, b.equity)


def test_a_held_name_opening_at_limit_up_is_kept_not_sold():
    """An open at limit-up blocks BUYING, not HOLDING.

    The rebalance pool used to drop every name at limit-up, held or not, so a
    held name still ranked first left the target and was sold at that open.
    """
    bars = make_bars({
        "A": [{"date": d, "open": 10.0, "close": 10.0, "open_limit_up": i >= 2}
              for i, d in enumerate(DATES)],
        "B": flat_series("B", 10.0),
    })
    signal = pd.DataFrame([{"date": d, "code": c, "score": s}
                           for d in DATES for c, s in (("A", 2.0), ("B", 1.0))])
    cfg = BacktestConfig(capital=1_000_000, n_positions=1, rebalance_days=5)
    res = run_backtest(signal, bars, cfg)
    sells_a = res.trades[(res.trades.code == "A") & (res.trades.side == "sell")]
    assert sells_a.empty, f"held limit-up name was sold:\n{sells_a}"
    assert set(res.trades.code) == {"A"}, "must not rotate into B"


def test_sharpe_is_the_arithmetic_daily_sharpe():
    """`sharpe` must be the textbook statistic, so it can be compared with outside numbers."""
    import numpy as np
    rng = np.random.default_rng(3)
    path = 10.0 * np.cumprod(1 + rng.normal(0.001, 0.02, len(DATES)))
    bars = make_bars({"A": [{"date": d, "open": round(path[i], 2), "close": round(path[i], 2)}
                            for i, d in enumerate(DATES)]})
    signal = pd.DataFrame({"date": DATES, "code": ["A"] * len(DATES), "score": [1.0] * len(DATES)})
    res = run_backtest(signal, bars, BacktestConfig(capital=1_000_000, n_positions=1, rebalance_days=5))
    r = res.equity.pct_change().dropna()
    assert res.metrics["sharpe"] == pytest.approx(r.mean() * 252 / (r.std() * np.sqrt(252)))
    assert "cagr_over_vol" not in res.metrics
