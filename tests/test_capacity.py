from __future__ import annotations

import pytest

from ashare_bt import BacktestConfig
from ashare_bt.capacity import sweep
from ashare_bt.synthetic import make_market, make_signal


def test_sweep_keeps_every_base_setting_except_capital_and_participation(monkeypatch):
    seen = []
    import ashare_bt.capacity as cap

    real = cap.run_backtest
    monkeypatch.setattr(cap, "run_backtest", lambda s, b, cfg: (seen.append(cfg), real(s, b, cfg))[1])
    bars = make_market(n_stocks=10, n_days=60, seed=0)
    base = BacktestConfig(n_positions=3, lot_size=10, settlement_t0=True, rebalance_weights=False,
                          refill=True, cash_annual_rate=0.02)
    sweep(make_signal(bars), bars, capitals=[1e5], participations=[0.1], base=base)
    cfg = seen[0]
    assert (cfg.capital, cfg.max_participation) == (1e5, 0.1)
    assert (cfg.lot_size, cfg.settlement_t0, cfg.rebalance_weights, cfg.refill, cfg.cash_annual_rate) \
        == (10, True, False, True, 0.02)


def test_sweep_does_not_swallow_an_input_error():
    bars = make_market(n_stocks=10, n_days=60, seed=0).drop(columns=["amount"])
    with pytest.raises(ValueError, match="amount"):
        sweep(make_signal(bars.assign(ret=0.0)), bars, capitals=[1e5], participations=[0.05])
