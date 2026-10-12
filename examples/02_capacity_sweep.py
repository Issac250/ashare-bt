"""同一个动量信号，从 10 万到 1 亿资金各跑一遍。"""
from ashare_bt import BacktestConfig
from ashare_bt.capacity import render, sweep
from ashare_bt.synthetic import make_market, make_signal

bars = make_market(n_stocks=40, n_days=300, seed=11)
signal = make_signal(bars, kind="momentum", lookback=20)
df = sweep(signal, bars, capitals=[1e5, 1e6, 1e7, 1e8], participations=[0.05],
           base=BacktestConfig(n_positions=10, rebalance_days=5))
print(render(df))
