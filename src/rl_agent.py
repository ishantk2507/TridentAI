"""
RL agent training and loading using Stable-Baselines3 PPO.

Handles:
- Loading existing trained models (if available)
- Training new models (if needed or forced)
- Environment setup and vectorization
- Model persistence
"""
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

MODEL_DIR = "artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)


def make_trading_env(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    signals: Tuple[np.ndarray, np.ndarray, np.ndarray],
    sentiment_score: float,
    hyperparams: Dict[str, Any],
    training_mode: bool = True
):
    """
    Factory function to create TridentTradingEnv.
    
    Args:
        stock_df: Stock OHLCV DataFrame
        benchmark_df: Benchmark OHLCV DataFrame
        signals: Tuple of (gen_signal, spec_signal, prophet_signal)
        sentiment_score: Sentiment [-1, 1]
        hyperparams: Regime-specific hyperparameters
        training_mode: If True, allow early truncation
    
    Returns:
        TridentTradingEnv instance
    """
    from src.trading_env import TridentTradingEnv
    
    def _make():
        return TridentTradingEnv(
            stock_df=stock_df,
            benchmark_df=benchmark_df,
            signals=signals,
            sentiment_score=sentiment_score,
            hyperparams=hyperparams,
            training_mode=training_mode
        )
    
    return _make


def load_or_train_agent(
    ticker: str,
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    signals: Tuple[np.ndarray, np.ndarray, np.ndarray],
    hyperparams: Dict[str, Any],
    sentiment_score: float = 0.0,
    retrain: bool = False
) -> Tuple[PPO, str]:
    """
    Load existing trained model or train a new one.
    
    Logic:
    1. If retrain=True, force training (discard existing model)
    2. If retrain=False and model exists, load it
    3. If model doesn't exist, train it
    
    Args:
        ticker: Stock symbol (e.g., 'NVDA')
        stock_df: Stock OHLCV DataFrame (training data)
        benchmark_df: Benchmark DataFrame (training data)
        signals: Tuple of forecast signals
        hyperparams: Regime-specific hyperparameters
        sentiment_score: News sentiment [-1, 1]
        retrain: If True, force training new model
    
    Returns:
        Tuple of (trained_model, status_string)
        - status_string is either "Loaded" or "Trained"
    
    Example:
        >>> model, status = load_or_train_agent(
        ...     'NVDA', train_stock, train_benchmark,
        ...     (gen_sig, spec_sig, prophet_sig),
        ...     hyperparams, retrain=False
        ... )
        >>> print(f"Status: {status}")  # "Loaded" or "Trained"
    """
    
    clean_ticker = ticker.replace(".NS", "")
    model_path = os.path.join(MODEL_DIR, f"agent_{clean_ticker}")
    
    # 1. Try to load existing model (if not forcing retrain)
    if not retrain and os.path.exists(f"{model_path}.zip"):
        try:
            logger.info(f"Attempting to load existing model from {model_path}")
            
            # Create dummy environment just for loading
            env_fn = make_trading_env(
                stock_df, benchmark_df, signals, sentiment_score,
                hyperparams, training_mode=False
            )
            train_env = DummyVecEnv([env_fn])
            
            # Load model
            model = PPO.load(model_path, env=train_env, device='cpu')
            
            # Verify model dimensions match environment
            if model.observation_space.shape[0] == 12:
                logger.info(f"Successfully loaded model for {ticker}")
                return model, "Loaded"
            else:
                logger.warning(f"Model dimension mismatch, retraining")
        
        except Exception as e:
            logger.warning(f"Failed to load model: {e}, will train new one")
    
    # 2. Train new model
    logger.info(f"Training new PPO agent for {ticker}")
    
    try:
        # Create environment
        env_fn = make_trading_env(
            stock_df, benchmark_df, signals, sentiment_score,
            hyperparams, training_mode=True
        )
        train_env = DummyVecEnv([env_fn])
        
        # PPO hyperparameters
        learning_rate = hyperparams.get('learning_rate', 0.0002)
        ent_coef = hyperparams.get('ent_coef', 0.01)
        total_timesteps = hyperparams.get('total_timesteps', 60000)
        
        # Create and train PPO agent
        model = PPO(
            'MlpPolicy',
            train_env,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            batch_size=64,
            n_steps=2048,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
            device='cpu'
        )
        
        # Train
        model.learn(total_timesteps=total_timesteps)
        
        # Save model
        model.save(model_path)
        logger.info(f"Trained and saved model to {model_path}")
        
        train_env.close()
        
        return model, "Trained"
    
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


def evaluate_agent(
    model: PPO,
    test_env,
    num_episodes: int = 1,
    deterministic: bool = True
) -> Dict[str, float]:
    """
    Evaluate trained agent on test environment.
    
    Args:
        model: Trained PPO model
        test_env: Test environment
        num_episodes: Number of episodes to run
        deterministic: If True, use policy deterministically; if False, sample
    
    Returns:
        dict with evaluation metrics:
            - mean_return: Average episode return
            - std_return: Std dev of returns
            - max_return: Maximum return
            - min_return: Minimum return
    """
    episode_returns = []
    
    for _ in range(num_episodes):
        obs, _ = test_env.reset()
        done = False
        total_return = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, _ = test_env.step(action)
            total_return += reward
            done = done or truncated
        
        episode_returns.append(total_return)
    
    episode_returns = np.array(episode_returns)
    
    return {
        'mean_return': float(episode_returns.mean()),
        'std_return': float(episode_returns.std()),
        'max_return': float(episode_returns.max()),
        'min_return': float(episode_returns.min()),
    }