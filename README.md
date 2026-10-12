# ashare-bt

[![ci](https://github.com/Issac250/ashare-bt/actions/workflows/ci.yml/badge.svg)](https://github.com/Issac250/ashare-bt/actions/workflows/ci.yml)

**What.** A cross-sectional backtester for China A-shares that only fills the orders the exchange would actually have filled.

**Why.** Most A-share backtests are optimistic in the same few ways: they trade at the close they used to decide, buy stocks that opened locked at limit-up, sell the same day they bought (T+1), ignore the ¥5 minimum commission and 100-share lots, book splits as losses, and let delisted stocks freeze capital. Each of these is a rule here, with a test that fails if it is broken.

**Results.** 94 tests on synthetic panels, one per venue rule: next-open execution, open-based limit gates, T+1, whole lots, participation cap and market impact, historical stamp duty back to 2001, total-return position accounting through splits and dividends, forced liquidation on delisting, and a capital-capacity sweep.

```bash
git clone https://github.com/Issac250/ashare-bt && cd ashare-bt
pip install -e .
python examples/01_quickstart.py
```

---

## 中文说明

### 它解决什么问题

A 股回测最常见的高估，几乎都来自"这笔单子其实成交不了"。下面每一条在本引擎里都是一条规则，各自有测试守着：

| 如果不处理 | 回测会多算什么 |
|---|---|
| 用当天收盘价决策、又按当天收盘价成交 | 用到了成交时还不知道的价格；本引擎在信号日 t 的**下一个交易日开盘**成交 |
| 开盘一字涨停照样买进 | 买到实际排不上队的股票，正好是之后最强的那批 |
| 用**收盘**是否涨停来判断开盘能否买入 | 反过来拒绝了开盘正常、收盘才涨停的股票，同样是未来函数 |
| 当天买当天卖 | 违反 T+1，高估短线策略 |
| 忽略整手和 ¥5 最低佣金 | 小资金下每笔小单的真实成本被严重低估 |
| 用不复权价格乘持股数计算市值 | 送转、拆股被记成暴跌，分红被记成亏损 |
| 退市股按最后价格一直挂着 | 资金被冻结在不存在的股票里，掩盖退市风险 |

### 输入

- `signal`：`[date, code, score]`，分数越高越想买；可选 `w` 列设定相对权重。
- `bars`：`[date, code, open, close, preclose, amount, tradable, open_limit_up, open_limit_down]`（`ret` 可选，引擎不读，`make_signal` 会用）
  - `open_limit_up` / `open_limit_down` 必须描述**开盘**时是否在涨跌停。传入旧字段名 `at_limit_up` 会直接报错，提醒你检查口径。涨跌停价按交易所规则四舍五入到分，可用 `ashare_bt.sizing.limit_prices(preclose, width)` 计算；直接用浮点 `round` 会把 27.95 × 1.1 = 30.745 算成 30.74。
  - `preclose` 必须是除权后的昨收（baostock 的 `preclose` 已经除权），`open / preclose` 就是隔夜的总收益。
  - 涨跌停字段为空时按"在涨跌停"处理：不能买，也不能卖。

### 成交规则

| 规则 | 默认 | 配置 |
|---|---|---|
| 成交时点 | 信号日下一交易日开盘 | — |
| 交收 | T+1 | `settlement_t0=True`（可转债） |
| 每手 | 100 股 | `lot_size=10`（可转债） |
| 涨停开盘 | 不能买入或加仓，已持有的继续持有 | — |
| 跌停开盘、停牌 | 不能卖出，继续持有到下次可以卖出 | — |
| 单票上限 | 20 日成交额中位数的 5% | `max_participation` |
| 佣金 | 万 2.5，最低 ¥5 | `CostModel(commission_rate, commission_min)` |
| 过户费 | 十万分之一，双边 | `transfer_fee` |
| 滑点和冲击 | 5bp + 0.10 × 参与率 | `slippage_bps`, `impact_coef` |
| 印花税 | 固定 0.05%，仅卖出 | `CostModel(historical_stamp_duty=True)` 按成交日取当时税率 |

`historical_stamp_duty=True` 使用的税率表：

| 生效日 | 买方 | 卖方 | 出处 |
|---|---|---|---|
| 2001-11-16 | 0.2% | 0.2% | 财政部、国家税务总局下调 A、B 股交易印花税 |
| 2005-01-24 | 0.1% | 0.1% | 财政部 2005-01-23 公布下调 |
| 2007-05-30 | 0.3% | 0.3% | 财政部上调证券交易印花税 |
| 2008-04-24 | 0.1% | 0.1% | 财政部下调证券交易印花税 |
| 2008-09-19 | 0 | 0.1% | 财政部、国家税务总局改为单边征收（[中新网](https://www.chinanews.com/cj/agyw/news/2008/09-18/1386777.shtml)） |
| 2023-08-28 | 0 | 0.05% | 财政部 税务总局公告 2023 年第 39 号（[新华网](https://www.news.cn/2023-08/28/c_1212260850.htm)） |

历次调整的整理可参考[财联社](https://www.cls.cn/detail/1433377)。早于 2001-11-16 的日期会直接报错，不会静默套用某个税率。

### 指标口径

- `cagr` 按日历时间计算：年数 = 天数 / 365.25。A 股一年约 243 个交易日，用 交易日数 / 252 会高估 CAGR。
- `sharpe` 是日收益的算术均值 × 252 ÷ 年化波动，无风险利率取 0。CAGR / 波动是另一个量，不叫 Sharpe。

### 示例

- `examples/01_quickstart.py`：合成数据上的 20 日动量组合。合成数据是随机游走，本身没有 alpha，跑出来的就是成本拖累，这正是要看的。
- `examples/02_capacity_sweep.py`：同一信号在 10 万到 1 亿资金下的衰减。
- `examples/03_fetch_with_baostock.py`：用 baostock 拉几只真实股票的日线再跑一遍。需要联网。baostock 禁止并发连接，请保持单线程。

### 不做的事

做空、期货、分钟级回测、事件驱动框架。

### License

MIT
