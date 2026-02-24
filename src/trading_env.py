"""
Gymnasium environment for RL-based portfolio optimization.

State: [price, returns, sentiment, 3 forecasts, balance, shares, allocation]
Action: Continuous [0, 1] representing portfolio allocation percentage
Reward: 5-component composite (base profit + sizing quality + churn penalty + holding bonus + wake-up bonus)
"""
import numpy as np
import pandas as pd
from gymnasium import Env, spaces
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TridentTradingEnv(Env):
    """
    Gymnasium environment for adaptive stock trading via RL.
    
    State Space (12 dimensions):
    - current_price: Stock price
    - price_return: Daily log return
    - sentiment: News sentiment [-1, 1]
    - gen_signal: Benchmark LSTM output
    - spec_signal: Stock LSTM output
    - prophet_signal: Prophet trend
    - balance: Cash available
    - shares_held: Position size
    - portfolio_value: Total account value
    - allocation: Current allocation %
    - volatility: Realized volatility
    - drift: Trend strength
    
    Action Space (continuous):
    - allocation [0, 1]: Portfolio allocation to stock (0 = 100% cash, 1 = 100% stock)
    
    Reward (5-component):
    1. Base Profit: portfolio_return * reward_scale
    2. Sizing Quality: Bonus for correct directional bet
    3. Churn Penalty: Discourages overtrading (regime-dependent)
    4. Holding Bonus: Encourages conviction in winners
    5. Wake-up Bonus: Prevents staying 100% cash
    
    Example:
        >>> env = TridentTradingEnv(stock_df, benchmark_df, signals, sentiment=0.5, hyperparams={...})
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()  # Random allocation [0, 1]
        >>> obs, reward, done, truncated, info = env.step(action)
    """
    
    def __init__(
        self,
        stock_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        signals: Tuple[np.ndarray, np.ndarray, np.ndarray],
        sentiment_score: float = 0.0,
        hyperparams: Optional[Dict[str, Any]] = None,
        training_mode: bool = True,
        initial_balance: float = 10000.0,
        transaction_cost_pct: float = 0.001
    ):
        """
        Args:
            stock_df: Stock OHLCV DataFrame
            benchmark_df: Benchmark OHLCV DataFrame
            signals: Tuple of (gen_signal, spec_signal, prophet_signal)
            sentiment_score: Sentiment [-1, 1]
            hyperparams: Regime-specific hyperparameters
            training_mode: If True, use training truncation; if False, run full episode
            initial_balance: Starting cash ($10k by default)
            transaction_cost_pct: Transaction cost as % of trade value (default: 0.1%)
        """
        super().__init__()
        
        self.stock_df = stock_df
        self.benchmark_df = benchmark_df
        self.gen_signal, self.spec_signal, self.prophet_signal = signals
        self.sentiment = sentiment_score
        self.hyperparams = hyperparams or {}
        self.training_mode = training_mode
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        
        # State space: 12 continuous values
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )
        
        # Action space: continuous allocation [0, 1]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Initialize state
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0.0
        self.history = [initial_balance]
        self.prev_action = 0.0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed (for reproducibility)
            options: Additional options dict
        
        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.history = [self.initial_balance]
        self.prev_action = 0.0
        
        obs = self._get_observation()
        
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one trading step.
        
        Args:
            action: Array [allocation] where 0 = cash, 1 = full stock
        
        Returns:
            Tuple of (obs, reward, done, truncated, info)
        """
        # Extract allocation from action
        target_alloc = float(action[0])
        target_alloc = np.clip(target_alloc, 0.0, 1.0)
        
        # Current portfolio state
        price_t = float(self.stock_df['Close'].values[self.current_step])
        total_val = self.balance + (self.shares_held * price_t)
        
        # Rebalance portfolio
        target_value = total_val * target_alloc
        current_stock_value = self.shares_held * price_t
        
        # Calculate shares to buy/sell
        shares_needed = (target_value - current_stock_value) / (price_t + 1e-9)
        trade_value = abs(shares_needed * price_t)
        
        # Apply transaction costs (slippage + commission)
        transaction_cost = trade_value * self.transaction_cost_pct
        
        self.shares_held += shares_needed
        self.balance -= (shares_needed * price_t + transaction_cost)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        truncated = False
        if self.current_step >= len(self.stock_df) - 1:
            done = True
            price_next = price_t
        else:
            done = False
            try:
                price_val = self.stock_df['Close'].values[self.current_step]
                # Handle both scalar and array returns
                if hasattr(price_val, 'item'):
                    price_next = float(price_val.item())
                else:
                    price_next = float(price_val)
            except (IndexError, ValueError):
                done = True
                price_next = price_t
        
        # Training mode: truncate long episodes
        if self.training_mode and (self.current_step - 0 > 300):
            truncated = True
        
        # Calculate new portfolio value
        new_val = self.balance + (self.shares_held * price_next)
        self.history.append(new_val)
        
        # ===== REWARD FUNCTION (5 components) =====
        
        # 1. BASE PROFIT (primary driver)
        port_ret = np.log(new_val / total_val) if total_val > 0 else 0
        reward_scale = self.hyperparams.get('reward_scale', 100)
        base_reward = port_ret * reward_scale
        
        # 2. SIZING QUALITY (directional correctness)
        market_ret = (price_next - price_t) / (price_t + 1e-9)
        sizing_quality = target_alloc if market_ret > 0 else (1.0 - target_alloc)
        
        regime_bonus = 0.0
        threshold = self.hyperparams.get('reward_threshold', 0.002)
        if abs(market_ret) > threshold:
            regime_bonus = sizing_quality * 0.5
        
        # 3. CHURN PENALTY (discourages overtrading)
        current_regime = self.hyperparams.get('regime', 'low_volatility')
        base_penalty = -0.01
        
        if current_regime in ["extreme_volatility", "very_high_volatility"]:
            base_penalty = -0.025  # Higher penalty in chaos
        
        delta_alloc = abs(target_alloc - self.prev_action)
        churn_penalty = base_penalty * delta_alloc
        
        # 4. HOLDING BONUS (encourages conviction in winners)
        holding_bonus = 0.0
        if target_alloc > 0.1 and market_ret > 0:
            holding_bonus = 0.5 * reward_scale * market_ret
        
        # 5. WAKE-UP BONUS (prevents staying 100% cash)
        wake_up_bonus = 0.0
        if target_alloc > 0.05:
            wake_up_bonus = 0.05
        
        # Track action for next step's churn penalty
        self.prev_action = target_alloc
        
        # COMPOSITE REWARD
        reward = base_reward + regime_bonus + churn_penalty + holding_bonus + wake_up_bonus
        
        # Get next observation
        obs = self._get_observation()
        
        return obs, reward, done or truncated, truncated, {}
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state vector).
        
        Returns:
            Array of 12 state values
        """
        # Ensure current_step is within bounds
        if self.current_step >= len(self.stock_df):
            self.current_step = len(self.stock_df) - 1
        
        # Current price
        current_price = float(self.stock_df['Close'].values[self.current_step])
        
        # Daily return
        if self.current_step > 0:
            prev_price = float(self.stock_df['Close'].values[self.current_step - 1])
            price_return = (current_price - prev_price) / (prev_price + 1e-9)
        else:
            price_return = 0.0
        
        # Signals (ensure bounds and handle index overflow)
        sig_idx = min(self.current_step, len(self.gen_signal) - 1)
        gen_sig = float(np.clip(self.gen_signal[sig_idx], -2, 2)) if len(self.gen_signal) > 0 else 0.0
        spec_sig = float(np.clip(self.spec_signal[sig_idx], -2, 2)) if len(self.spec_signal) > 0 else 0.0
        prophet_sig = float(np.clip(self.prophet_signal[sig_idx], -2, 2)) if len(self.prophet_signal) > 0 else 0.0
        
        # Portfolio metrics
        current_price_safe = max(current_price, 1e-9)
        portfolio_value = self.balance + (self.shares_held * current_price_safe)
        allocation = (self.shares_held * current_price_safe) / max(portfolio_value, 1e-9)
        
        # Volatility (rolling std of recent returns)
        if self.current_step >= 20:
            recent_prices = self.stock_df['Close'].values[self.current_step-20:self.current_step+1]
            recent_prices = np.asarray(recent_prices).flatten().astype(float)
            
            if len(recent_prices) > 1:
                recent_returns = np.diff(recent_prices) / recent_prices[:-1]
                volatility = float(np.std(recent_returns))
            else:
                volatility = 0.0
        else:
            volatility = 0.0
        
        # Drift (trend strength: % positive days)
        if self.current_step >= 20:
            recent_prices = self.stock_df['Close'].values[self.current_step-20:self.current_step+1]
            recent_prices = np.asarray(recent_prices).flatten().astype(float)
            
            if len(recent_prices) > 1:
                recent_returns = np.diff(recent_prices) / recent_prices[:-1]
                drift = float((recent_returns > 0).sum() / len(recent_returns))
            else:
                drift = 0.5
        else:
            drift = 0.5
        
        # State vector [12 dimensions]
        obs = np.array([
            current_price / 100.0,              # Normalize price
            price_return,                        # Return [-1, 1]
            self.sentiment,                      # Sentiment [-1, 1]
            gen_sig,                             # Benchmark signal [-2, 2]
            spec_sig,                            # Stock signal [-2, 2]
            prophet_sig,                         # Prophet signal [-2, 2]
            self.balance / self.initial_balance, # Normalized balance
            self.shares_held,                    # Position size
            portfolio_value / self.initial_balance, # Normalized portfolio value
            allocation,                          # Current allocation [0, 1]
            volatility,                          # Volatility [0, 1]
            drift                                # Drift [0, 1]
        ], dtype=np.float32)
        
        return obs
    
    def render(self, mode: str = 'human') -> None:
        """
        Render environment state (optional).
        
        Args:
            mode: Rendering mode ('human' for print)
        """
        if mode == 'human':
            current_price = float(self.stock_df['Close'].values[self.current_step])
            portfolio_value = self.balance + (self.shares_held * current_price)
            
            print(f"Step {self.current_step}: Price=${current_price:.2f}, "
                  f"Portfolio=${portfolio_value:.2f}, "
                  f"Shares={self.shares_held:.2f}")