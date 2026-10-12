"""ashare-bt：基于日线行情和选股评分的 A 股组合回测。"""
from .costs import CostModel
from .engine import BacktestConfig, BacktestResult, run_backtest

__all__ = ["BacktestConfig", "BacktestResult", "CostModel", "run_backtest"]
__version__ = "0.1.0"
