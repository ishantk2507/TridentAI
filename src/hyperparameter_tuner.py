"""
Adaptive hyperparameter selection based on market regime classification.

Uses 6-tier volatility regime classification to automatically tune neural network
and RL algorithm hyperparameters.
"""

import numpy as np
import pandas as pd 
from typing import Dict, Any
from src.utils import extract_close_price, compute_volatility_metrics, get_regime

class StockProfiler:
    """
    Analyzes stock characteristics and returns optimal hyperparameters.
    
    Uses 6-tier volatility regime classification to automatically tune:
    - LSTM hidden dimensions (64→256)
    - Learning rates (0.0002→0.003)
    - Reward thresholds (0.001→0.005)
    - RL training steps (50K→75K)
    
    Philosophy:
    - High volatility → deeper networks, more exploration, smaller rewards
    - Low volatility → simpler networks, less exploration, larger rewards
    
    Example:
        >>> profiler = StockProfiler()
        >>> profile = profiler.analyze(df, 'NVDA')
        >>> hyperparams = profiler.get_hyperparams(df, 'NVDA')
        >>> print(f"Regime: {profile['regime']}, Steps: {hyperparams['total_timesteps']}")
    """

    @staticmethod
    def analyze(df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Compute volatility profile for a stock.
        
        Args:
            df: OHLCV DataFrame from yfinance
            ticker: Stock symbol
        
        Returns:
            dict: Profile with keys:
                - ticker: Stock symbol
                - daily_vol: Daily volatility %
                - annual_vol: Annualized volatility %
                - trend_strength: % positive days
                - price_range: (max-min)/mean
                - vol_drift: Recent vs historical vol change %
                - regime: Tier classification
                - num_samples: Number of trading days
        """
        close = extract_close_price(df, ticker)

        returns = close.pct_change().dropna().values

        vol_metrics = compute_volatility_metrics(returns)
        regime = get_regime(vol_metrics['annual_vol'])

        price_range = (close.max() - close.min()) / close.mean() if close.mean() != 0 else 0

        profile = {
            'ticker':ticker,
            'daily_vol':vol_metrics['daily_vol'],
            'annual_vol':vol_metrics['annual_vol'],
            'trend_strength':vol_metrics['trend_strength'],
            'price_range':price_range,
            'vol_drift':vol_metrics['vol_drift'],
            'regime':regime,
            'num_samples':len(close)
        }

        return profile 
    
    @staticmethod
    def get_hyperparams(df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Get adaptive hyperparameters based on volatility regime.
        
        The 6-tier system scales parameters inversely with stability:
        - Extreme volatility (>80%) → conservative, exploratory RL
        - Low volatility (<12%) → aggressive, exploitation-focused
        
        Args:
            df: OHLCV DataFrame
            ticker: Stock symbol
        
        Returns:
            dict: Hyperparameter configuration with keys:
                - regime: Market regime string
                - hidden_gen: Benchmark LSTM hidden dim
                - hidden_spec: Stock LSTM hidden dim
                - lookback: Sequence length for LSTM
                - epochs: Prophet training iterations
                - lr: Prophet learning rate
                - changepoint_prior: Prophet changepoint detection prior
                - seasonality_mode: 'additive' or 'multiplicative'
                - reward_threshold: Min market move to trigger regime bonus
                - reward_scale: Scaling factor for rewards (100-200)
                - ent_coef: PPO entropy coefficient (exploration)
                - learning_rate: PPO learning rate
                - total_timesteps: RL training steps
        """

        profile = StockProfiler.analyze(df, ticker)
        regime = profile['regime']

        if regime == "extreme_volatility":
            return {
                'regime': regime,
                'hidden_gen': 128,          # Larger networks for chaos
                'hidden_spec': 256,
                'lookback': 40,             # Shorter memory (older data less relevant)
                'epochs': 20,               # Fewer prophet iterations
                'lr': 0.003,                # Aggressive learning
                'changepoint_prior': 0.1,  # More changepoints expected
                'seasonality_mode': 'additive',
                'reward_threshold': 0.005,  # High bar for regime bonus
                'reward_scale': 120,        # Scaled down (volatile)
                'ent_coef': 0.02,           # High entropy (exploration)
                'learning_rate': 0.0003,
                'total_timesteps': 75000,
            }
        
        elif regime == "very_high_volatility":
            return {
                'regime': regime,
                'hidden_gen': 96, 'hidden_spec': 192,
                'lookback': 45, 'epochs': 25, 'lr': 0.0027,
                'changepoint_prior': 0.08, 'seasonality_mode': 'multiplicative',
                'reward_threshold': 0.003, 'reward_scale': 130,
                'ent_coef': 0.018, 'learning_rate': 0.00027, 'total_timesteps': 75000
            }
        
        elif regime == "high_volatility":
            return {
                'regime': regime,
                'hidden_gen': 80, 'hidden_spec': 160,
                'lookback': 50, 'epochs': 27, 'lr': 0.0025,
                'changepoint_prior': 0.06, 'seasonality_mode': 'multiplicative',
                'reward_threshold': 0.0025, 'reward_scale': 140,
                'ent_coef': 0.016, 'learning_rate': 0.00025, 'total_timesteps': 75000
            }
        
        elif regime == "medium_high_volatility":
            return {
                'regime': regime,
                'hidden_gen': 64, 'hidden_spec': 128,
                'lookback': 55, 'epochs': 28, 'lr': 0.0023,
                'changepoint_prior': 0.055, 'seasonality_mode': 'multiplicative',
                'reward_threshold': 0.002, 'reward_scale': 150,
                'ent_coef': 0.012, 'learning_rate': 0.00023, 'total_timesteps': 70000
            }
        
        elif regime == "medium_volatility":
            return {
                'regime': regime,
                'hidden_gen': 64, 'hidden_spec': 128,
                'lookback': 60, 'epochs': 30, 'lr': 0.002,
                'changepoint_prior': 0.05, 'seasonality_mode': 'additive',
                'reward_threshold': 0.0015, 'reward_scale': 160,
                'ent_coef': 0.01, 'learning_rate': 0.0002, 'total_timesteps': 60000
            }
        
        else:
            return {
                'regime': regime,
                'hidden_gen': 64, 'hidden_spec': 128,
                'lookback': 60, 'epochs': 30, 'lr': 0.002,
                'changepoint_prior': 0.03, 'seasonality_mode': 'additive',
                'reward_threshold': 0.001, 'reward_scale': 200,
                'ent_coef': 0.01, 'learning_rate': 0.0002, 'total_timesteps': 50000
            }