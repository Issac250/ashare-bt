"""Quickstart: a 20-day momentum book on synthetic A-share bars."""
from ashare_bt import BacktestConfig, run_backtest
from ashare_bt.synthetic import make_market, make_signal

bars = make_market(n_stocks=100, n_days=750, seed=7)
signal = make_signal(bars, kind="momentum", lookback=20)
res = run_backtest(signal, bars, BacktestConfig(capital=1_000_000, n_positions=20, rebalance_days=5))
print(res.summary())
print()
print("trades by side:")
print(res.trades["side"].value_counts().to_string())
