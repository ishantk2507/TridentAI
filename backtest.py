"""
Historical backtesting pipeline for TridentAI.

Computes Sharpe, Sortino, max drawdown, and compares vs benchmark.

Usage:
    python backtest.py --ticker NVDA --start 2020-01-01 --end 2024-12-31
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import sys
import os
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src import (
    StockProfiler, TridentForecaster, SentimentEngine,
    load_or_train_agent, TridentTradingEnv, set_seeds
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(returns_series, benchmark_returns=None):
    """
    Compute Sharpe, Sortino, max drawdown, win rate.
    
    Args:
        returns_series: pd.Series of daily returns
        benchmark_returns: pd.Series of benchmark returns (for comparison)
    
    Returns:
        dict with keys: sharpe_ratio, sortino_ratio, max_drawdown, annual_return, win_rate
    """
    
    # Sharpe Ratio (252 trading days)
    excess_returns = returns_series
    daily_mean = np.mean(excess_returns)
    daily_std = np.std(excess_returns)
    
    if daily_std > 0:
        sharpe = daily_mean / daily_std * np.sqrt(252)
    else:
        sharpe = 0
    
    # Sortino Ratio (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) > 0:
        downside_std = np.std(downside_returns)
        if downside_std > 0:
            sortino = daily_mean / downside_std * np.sqrt(252)
        else:
            sortino = 0
    else:
        sortino = 0
    
    # Max Drawdown
    cumulative = (1 + excess_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win Rate
    win_rate = (excess_returns > 0).sum() / len(excess_returns) if len(excess_returns) > 0 else 0
    
    # Annual return
    annual_return = (1 + daily_mean) ** 252 - 1
    
    return {
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'annual_return': annual_return,
        'win_rate': win_rate
    }


def backtest_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    benchmark_ticker: str = 'SPY',
    split_pct: float = 0.7
):
    """
    Full backtest pipeline for a single ticker.
    
    Args:
        ticker: Stock symbol (e.g., 'NVDA')
        start_date: Start of backtest (e.g., '2020-01-01')
        end_date: End of backtest (e.g., '2024-12-31')
        benchmark_ticker: Benchmark for comparison (default: SPY)
        split_pct: % of data for training (default: 70%)
    
    Returns:
        dict with all metrics + comparison vs benchmark
    """
    from src.utils import extract_close_price
    
    print(f"\n{'='*70}")
    print(f"BACKTESTING: {ticker} ({start_date} → {end_date})")
    print(f"{'='*70}\n")
    
    # 1. Download data
    print("📊 Downloading data...")
    try:
        benchmark_df = yf.download(
            benchmark_ticker, start=start_date, end=end_date,
            progress=False, auto_adjust=True
        )
        stock_df = yf.download(
            ticker, start=start_date, end=end_date,
            progress=False, auto_adjust=True
        )
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None
    
    if len(stock_df) < 100:
        print(f"❌ Not enough data for {ticker}")
        return None
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # 2. Train/test split
    split_idx = int(len(stock_df) * split_pct)
    train_benchmark = benchmark_df.iloc[:split_idx]
    train_stock = stock_df.iloc[:split_idx]
    test_benchmark = benchmark_df.iloc[split_idx:]
    test_stock = stock_df.iloc[split_idx:]
    
    print(f"📈 Train: {len(train_stock)} days | Test: {len(test_stock)} days\n")
    
    # 3. Analyze volatility & get hyperparams
    print("🔍 Analyzing stock profile...")
    profiler = StockProfiler()
    profile = profiler.analyze(train_stock, ticker)
    hyperparams = profiler.get_hyperparams(train_stock, ticker)
    print(f"   Regime: {profile['regime']}")
    print(f"   Annual Vol: {profile['annual_vol']:.1f}%\n")
    
    # 4. Get sentiment
    print("🧠 Extracting sentiment...")
    try:
        sentiment_engine = SentimentEngine()
        sentiment_score, _ = sentiment_engine.get_sentiment(ticker)
    except Exception as e:
        logger.warning(f"Sentiment extraction failed: {e}")
        sentiment_score = 0.0
    print(f"   Sentiment: {sentiment_score:.2f}\n")
    
    # 5. Train forecaster
    print("📈 Training forecaster...")
    forecaster = TridentForecaster(
        ticker,
        general_ticker=benchmark_ticker,
        hyperparams=hyperparams
    )
    success = forecaster.train(train_benchmark, train_stock)
    if not success:
        print("❌ Forecaster training failed")
        return None
    print("   ✅ Forecaster ready\n")
    
    # 6. Get test signals
    print("🔮 Generating test signals...")
    gen_sig, spec_sig, prophet_sig = forecaster.batch_predict(test_benchmark, test_stock)
    print(f"   Generated {len(gen_sig)} test predictions\n")
    
    # 7. Train RL agent
    print("🤖 Training RL agent...")
    train_signals = forecaster.batch_predict(train_benchmark, train_stock)
    
    model, status = load_or_train_agent(
        ticker,
        train_stock,
        train_benchmark,
        train_signals,
        hyperparams,
        sentiment_score=sentiment_score,
        retrain=True  # Force retrain for clean backtest
    )
    print(f"   {status}\n")
    
    # 8. Run agent on test set
    print("🎯 Running agent on test set...")
    env = TridentTradingEnv(
        test_stock,
        test_benchmark,
        (gen_sig, spec_sig, prophet_sig),
        sentiment_score=sentiment_score,
        hyperparams=hyperparams,
        training_mode=False
    )
    
    obs, _ = env.reset()
    done = False
    portfolio_values = [env.initial_balance]  # Track actual portfolio value
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        # Extract actual portfolio value from environment
        current_price = float(test_stock['Close'].values[min(env.current_step, len(test_stock)-1)])
        portfolio_val = env.balance + (env.shares_held * current_price)
        portfolio_values.append(portfolio_val)
        
        done = done or truncated
    
    # Calculate actual daily returns from portfolio value
    portfolio_values = np.array(portfolio_values)
    agent_returns = pd.Series(np.diff(portfolio_values) / portfolio_values[:-1])
    
    # 9. Get benchmark returns (buy & hold)
    try:
        benchmark_close = extract_close_price(test_benchmark, benchmark_ticker)
    except:
        benchmark_close = test_benchmark['Close'].squeeze()
    
    benchmark_prices = np.asarray(benchmark_close.values).flatten().astype(float)
    
    if len(benchmark_prices) > 1:
        benchmark_returns_arr = np.diff(benchmark_prices) / benchmark_prices[:-1]
        benchmark_returns = pd.Series(benchmark_returns_arr)
    else:
        benchmark_returns = pd.Series([0.0])
    
    # 10. Compute metrics
    print("📊 Computing metrics...\n")
    agent_metrics = compute_metrics(agent_returns)
    benchmark_metrics = compute_metrics(benchmark_returns)
    
    # 11. Print results
    print(f"{'METRIC':<25} {'AGENT':<18} {'BENCHMARK':<18} {'ADVANTAGE':<18}")
    print("-" * 80)
    
    annual_ret_diff = (agent_metrics['annual_return'] - benchmark_metrics['annual_return']) * 100
    sharpe_diff = agent_metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio']
    sortino_diff = agent_metrics['sortino_ratio'] - benchmark_metrics['sortino_ratio']
    max_dd_diff = (benchmark_metrics['max_drawdown'] - agent_metrics['max_drawdown']) * 100
    win_rate_diff = (agent_metrics['win_rate'] - benchmark_metrics['win_rate']) * 100
    
    print(f"{'Annual Return':<25} {agent_metrics['annual_return']*100:>16.2f}% {benchmark_metrics['annual_return']*100:>16.2f}% {annual_ret_diff:>+16.2f}%")
    print(f"{'Sharpe Ratio':<25} {agent_metrics['sharpe_ratio']:>17.2f} {benchmark_metrics['sharpe_ratio']:>17.2f} {sharpe_diff:>+17.2f}")
    print(f"{'Sortino Ratio':<25} {agent_metrics['sortino_ratio']:>17.2f} {benchmark_metrics['sortino_ratio']:>17.2f} {sortino_diff:>+17.2f}")
    print(f"{'Max Drawdown':<25} {agent_metrics['max_drawdown']*100:>16.2f}% {benchmark_metrics['max_drawdown']*100:>16.2f}% {max_dd_diff:>+16.2f}%")
    print(f"{'Win Rate':<25} {agent_metrics['win_rate']*100:>16.2f}% {benchmark_metrics['win_rate']*100:>16.2f}% {win_rate_diff:>+16.2f}%")
    
    print("\n" + "="*80)
    
    return {
        'ticker': ticker,
        'period': f"{start_date} to {end_date}",
        'train_days': len(train_stock),
        'test_days': len(test_stock),
        'regime': profile['regime'],
        'agent_metrics': agent_metrics,
        'benchmark_metrics': benchmark_metrics,
        'outperformance': {
            'annual_return': annual_ret_diff,
            'sharpe': sharpe_diff,
            'sortino': sortino_diff,
            'max_drawdown': max_dd_diff,
            'win_rate': win_rate_diff
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest TridentAI strategy")
    parser.add_argument('--ticker', type=str, default='NVDA', help='Stock ticker')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date')
    parser.add_argument('--benchmark', type=str, default='SPY', help='Benchmark ticker')
    
    args = parser.parse_args()
    
    results = backtest_ticker(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        benchmark_ticker=args.benchmark
    )
    
    if results:
        print("\n✅ Backtest complete!")
    else:
        print("\n❌ Backtest failed!")
        sys.exit(1)