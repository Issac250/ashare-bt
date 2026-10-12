"""用合成行情跑一个 20 日动量组合，每 5 个信号日调仓。"""
from ashare_bt import BacktestConfig, run_backtest
from ashare_bt.synthetic import make_market, make_signal

bars = make_market(n_stocks=100, n_days=750, seed=7)
signal = make_signal(bars, kind="momentum", lookback=20)
res = run_backtest(signal, bars, BacktestConfig(capital=1_000_000, n_positions=20, rebalance_days=5))
print(res.summary())
print()
print("trades by side:")
print(res.trades["side"].value_counts().to_string())
