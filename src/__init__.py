"""
TridentAI: Adaptive Multi-Regime Stock Trading Agent

Core modules:
- utils: Helper functions for data processing
- sentiment_engine: FinBERT-based sentiment analysis
- hyperparameter_tuner: Volatility-regime-based hyperparameter selection
- forecaster: Prophet + Attention LSTM time-series forecasting
- trading_env: Gymnasium environment for RL training
- rl_agent: PPO agent training and loading
"""

from src.utils import (
    extract_close_price,
    to_log_returns,
    robust_standardize,
    compute_volatility_metrics,
    get_regime,
    set_seeds,
)

from src.sentiment_engine import SentimentEngine
from src.hyperparameter_tuner import StockProfiler

__version__ = "1.0.0"
__all__ = [
    'extract_close_price',
    'to_log_returns',
    'robust_standardize',
    'compute_volatility_metrics',
    'get_regime',
    'set_seeds',
    'SentimentEngine',
    'StockProfiler',
]