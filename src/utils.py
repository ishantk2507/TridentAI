import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import random
import torch 
import os

def extract_close_price(df: pd.DataFrame, ticker :str) -> pd.Series:

    """
    Robustly extract close price from yfinance DataFrame.
    
    Handles MultiIndex columns (multi-ticker downloads), single-ticker DataFrames,
    and edge cases (missing data, wrong format).
    
    Args:
        df: DataFrame from yfinance with OHLCV data
        ticker: Stock symbol (e.g., 'NVDA', 'NVDA.NS')
    
    Returns:
        pd.Series: Close prices as float64
        
    Example:
        >>> df = yf.download('NVDA', start='2020-01-01', end='2024-12-31')
        >>> close = extract_close_price(df, 'NVDA')
    """

    # Try MultiIndex extraction first (multi-ticker downloads)

    if isinstance(df.columns, pd.MultiIndex):
        try:
            close_data = df.xs(ticker, level=1, axis=1)['Close']
        except KeyError:
            close_data = df['Close']
    else:
        close_data = df['Close']


    close_data = close_data.squeeze()
    if isinstance(close_data, pd.DataFrame):
        close_data = close_data.iloc[:,0]

    return close_data.astype(float)


def to_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Convert price series to log returns.
    
    log_return_t = log(price_t / price_{t-1})
    
    Args:
        prices: Array of prices (shape: [T] or [T, 1])
    
    Returns:
        np.ndarray: Log returns (shape: [T-1])
    """

    prices = np.asarray(prices).reshape(-1, 1)
    return np.diff(np.log(prices + 1e-9), axis=0).squeeze()



def robust_standardize(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Standardize data with zero-division protection.
    
    Args:
        data: Array of values
    
    Returns:
        tuple: (standardized_data, mean, std)
        - Returns zeros if std < 1e-9 to prevent division by zero
    """
    mean = np.mean(data)
    std = np.std(data)
    if std < 1e-9:
        return np.zeros_like(data), mean, std
    return (data - mean) / std, mean, std


def compute_volatility_metrics(returns: np.ndarray) -> Dict[str, float]:
    """
    Compute annualized volatility and trend metrics.
    
    Args:
        returns: Array of daily log returns
    
    Returns:
        dict with keys:
            - daily_vol: Daily volatility as percentage
            - annual_vol: Annualized volatility (252 trading days)
            - trend_strength: % of positive return days
            - vol_drift: Recent vol vs historical vol drift
    """

    daily_vol = np.std(returns) * 100
    annual_vol = daily_vol * np.sqrt(252)

    trend_strength = (returns > 0).sum() /len(returns) if len(returns) > 0 else 0.5

    if len(returns) > 60:
        recent_returns = returns[-60:]
    else:
        recent_returns = returns

    recent_vol = np.std(recent_returns) * np.sqrt(252) * 100
    vol_drift = ((recent_vol - annual_vol) / annual_vol * 100) if annual_vol > 0 else 0

    return {
        'daily_vol': daily_vol,
        'annual_vol': annual_vol,
        'trend_strength': trend_strength,
        'vol_drift': vol_drift
    }


def get_regime(annual_vol: float) -> str:
    """
    Classify volatility regime (6-tier system).
    
    Args:
        annual_vol: Annual volatility as percentage
    
    Returns:
        str: One of ['extreme_volatility', 'very_high_volatility', 
                     'high_volatility', 'medium_high_volatility',
                     'medium_volatility', 'low_volatility']
    """

    if annual_vol > 80:
        return "extreme_volatility"
    elif annual_vol > 50:
        return "very_high_volatility"
    elif annual_vol > 35:
        return "high_volatility"
    elif annual_vol > 20:
        return "medium_high_volatility"
    elif annual_vol > 12:
        return "medium_volatility"
    else:
        return "low_volatility"
    

def set_seeds(seed:int = 42) -> None:
    """
    Set random seeds for reproducibility across numpy, torch, python.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
