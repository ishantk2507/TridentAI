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
from src.forecaster import TridentForecaster, AttentionLSTM, Attention
from src.trading_env import TridentTradingEnv
from src.rl_agent import load_or_train_agent, evaluate_agent

__version__ = "1.0.0"
__all__ = [
    # Utilities
    'extract_close_price',
    'to_log_returns',
    'robust_standardize',
    'compute_volatility_metrics',
    'get_regime',
    'set_seeds',
    # Sentiment
    'SentimentEngine',
    # Hyperparameters
    'StockProfiler',
    # Forecasting
    'TridentForecaster',
    'AttentionLSTM',
    'Attention',
    # RL
    'TridentTradingEnv',
    'load_or_train_agent',
    'evaluate_agent',
]