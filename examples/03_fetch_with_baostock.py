"""用 baostock 拉取五只股票的日线，再跑动量回测。

先安装 pip install -e '.[baostock]'，运行时需要联网。
baostock 不支持并发连接，请按顺序请求。

示例固定使用 10% 涨跌停幅度。更换股票或日期范围时，
需要重新核对板块、ST 状态和对应时期的交易规则。
"""
import baostock as bs
import pandas as pd

from ashare_bt import BacktestConfig, run_backtest
from ashare_bt.sizing import limit_prices
from ashare_bt.synthetic import make_signal

CODES = ["sh.600000", "sh.600036", "sh.601318", "sz.000001", "sz.000002"]
FIELDS = "date,code,open,close,preclose,amount,tradestatus,pctChg"
LIMIT_WIDTH = 0.10


def fetch(code: str, start: str, end: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(code, FIELDS, start_date=start, end_date=end,
                                      frequency="d", adjustflag="3")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


bs.login()
try:
    raw = pd.concat([fetch(c, "2022-01-01", "2024-12-31") for c in CODES], ignore_index=True)
finally:
    bs.logout()

for col in ("open", "close", "preclose", "amount", "pctChg"):
    raw[col] = pd.to_numeric(raw[col], errors="coerce")
up, dn = limit_prices(raw["preclose"], LIMIT_WIDTH)
bars = raw.assign(
    ret=raw["pctChg"] / 100,
    tradable=(raw["tradestatus"] == "1") & (raw["amount"] > 0),
    open_limit_up=raw["open"] >= up,
    open_limit_down=raw["open"] <= dn,
).drop(columns=["tradestatus", "pctChg"])

signal = make_signal(bars, kind="momentum", lookback=20)
res = run_backtest(signal, bars, BacktestConfig(capital=200_000, n_positions=2, rebalance_days=5))
print(res.summary())
