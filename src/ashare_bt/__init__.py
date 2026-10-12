"""ashare-bt: a cross-sectional backtester for China A-shares."""
from .costs import CostModel
from .engine import BacktestConfig, BacktestResult, run_backtest

__all__ = ["BacktestConfig", "BacktestResult", "CostModel", "run_backtest"]
__version__ = "0.1.0"
