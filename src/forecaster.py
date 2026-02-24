"""
Adaptive hybrid forecaster combining Prophet + Attention LSTM.

Architecture:
- Prophet: Captures global trend and seasonality
- Attention LSTM (Dual-stream): Learns deep temporal dependencies
  - Benchmark stream: General market patterns (e.g., SPY)
  - Stock-specific stream: Individual stock dynamics
- Attention mechanism: Weights important timesteps
- Hyperparameters: Auto-tune based on volatility regime
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Attention(nn.Module):
    """
    Multi-head attention mechanism for LSTM hidden states.
    
    Learns to weight which timesteps are most important for prediction.
    """
    def __init__(self, hidden_dim:int):
        """
        Args:
            hidden_dim: Dimension of LSTM hidden states
        """
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, hidden_dim]
        
        Returns:
            Tensor of shape [batch_size, hidden_dim] (context vector)
        """
        
        scores = self.attention(x) 

        weights = torch.softmax(scores, dim=1)
    
        context = torch.sum(weights * x, dim =1 )
    
        return context

class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism.
    
    Learns sequential patterns with learned attention over timesteps.
    Outputs: Single prediction per sequence (many-to-one architecture).
    
    Example:
        >>> model = AttentionLSTM(input_dim=1, hidden_dim=64, num_layers=2)
        >>> x = torch.randn(32, 50, 1)  # batch=32, seq_len=50, input=1
        >>> pred = model(x)  # Output: [32, 1]
    """
    
    def __init__(
          self, 
          input_dim: int=1,
          hidden_dim: int=32,
          num_layers: int=2,
          output_dim: int=1,
          dropout: float=0.2
    ) :
        """
        Args:
            input_dim: Input feature dimension (default: 1 for univariate returns)
            hidden_dim: LSTM hidden state dimension (default: 32)
            num_layers: Number of LSTM layers (default: 2)
            output_dim: Output dimension (default: 1 for next-day return)
            dropout: Dropout probability (default: 0.2)
        """
        super(AttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
     
        self.attention = Attention(hidden_dim)
     
        self.fc = nn.Linear(hidden_dim, output_dim)
     
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
        
        Returns:
            Output tensor [batch_size, output_dim]
        """

        lstm_out, (h_n, c_n) = self.lstm(x)
        context = self.attention(lstm_out)
        out = self.fc(context)
        return out

class TridentForecaster:
    """
    Hybrid time-series forecaster: Prophet + Attention LSTM.
    
    Combines:
    1. **Prophet**: Global trend + seasonality (interpretable)
    2. **Benchmark LSTM**: General market patterns (e.g., SPY returns)
    3. **Stock LSTM**: Stock-specific dynamics (e.g., NVDA returns)
    
    All hyperparameters adapt based on market volatility regime.
    
    Attributes:
        ticker: Stock symbol
        general_ticker: Benchmark symbol (e.g., SPY)
        hyperparams: Regime-specific hyperparameters
        lstm_gen: Benchmark LSTM model
        lstm_spec: Stock-specific LSTM model
        prophet_model: Prophet trend model
        scaler_gen, scaler_spec: Feature normalizers
        device: torch.device (CPU or GPU)
    
    Example:
        >>> hyperparams = {'hidden_gen': 64, 'hidden_spec': 128, 'lookback': 50, ...}
        >>> forecaster = TridentForecaster('NVDA', 'SPY', hyperparams)
        >>> success = forecaster.train(gen_df, spec_df)
        >>> gen_sig, spec_sig, prophet_sig = forecaster.batch_predict(gen_test, spec_test)
    """

    def __init__(
        self,
        ticker: str,
        general_ticker: str ="SPY",
        hyperparams: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            ticker: Stock symbol (e.g., 'NVDA')
            general_ticker: Benchmark symbol (e.g., 'SPY')
            hyperparams: Regime-specific hyperparameters dict
        """

        self.ticker = ticker
        self.general_ticker = general_ticker
        self.hyperparams = hyperparams or {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_gen = self.hyperparams.get('hidden_gen', 32)
        hidden_spec = self.hyperparams.get('hidden_spec', 64)
        self.lookback = self.hyperparams.get('lookback', 50)

        self.lstm_gen = AttentionLSTM(
            input_dim= 1, 
            hidden_dim= hidden_gen,
            num_layers= 2,
            output_dim= 1,
        ).to(self.device)

        self.lstm_spec = AttentionLSTM(
            input_dim= 1, 
            hidden_dim= hidden_spec,
            num_layers= 2,
            output_dim= 1,
        ).to(self.device)

        self.scaler_gen = StandardScaler()
        self.scaler_spec = StandardScaler()

        self.prophet_model = None

    
    def get_data(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download OHLCV data for ticker and benchmark.
        
        Args:
            start_date: Start date (e.g., '2020-01-01')
            end_date: End date (e.g., '2024-12-31')
        
        Returns:
            Tuple of (benchmark_df, stock_df)
        """

        import yfinance as yf

        gen_df = yf.download(
            self.general_ticker,
            start=start_date,
            end=end_date,
            progress= False,
            auto_adjust= True
        )

        spec_def = yf.download(
            self.ticker,
            start=start_date,
            end=end_date,
            progress= False,
            auto_adjust= True
        )

        return gen_df, spec_def

    @staticmethod
    def _to_log_returns(prices: np.ndarray) -> np.ndarray:
        """Convert prices to log returns."""
        prices = np.asarray(prices).flatten()
        return np.diff(np.log(prices + 1e-9))
    
    def train_prophet(self, df: pd.DataFrame) -> bool:
        """
        Train Prophet model on close prices.
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            bool: True if training successful
        """
        
        from src.utils import extract_close_price

        try:
            close_data = extract_close_price(df, self.ticker)

            p_df = df.reset_index()
            date_col = 'Date' if 'Date' in p_df.columns else p_df.columns[0]

            p_df = pd.DataFrame({
                 'ds': pd.to_datetime(p_df[date_col]),
                 'y': close_data.values
            })

            changepoint_prior = self.hyperparams.get('changepoint_prior', 0.05)
            seasonlity_mode = self.hyperparams.get('seasonality_mode', 'multiplicative')

            self.prophet_model = Prophet(
                 changepoint_prior_scale=changepoint_prior,
                seasonality_mode=seasonlity_mode,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.95
            )

            with open('NUL', 'w') as f:
                 self.prophet_model.fit(p_df)

            return True
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            return False
    
    def get_prophet_signal(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get Prophet trend signal (normalized).
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            Array of trend signals (standardized)
        """
         
        
        if self.prophet_model is None:
            return np.zeros(len(df))
    
        try:
            
            dates_df = df.reset_index()
            date_col = 'Date' if 'Date' in dates_df.columns else dates_df.columns[0]

            future = pd.DataFrame({'ds': pd.to_datetime(dates_df[date_col])})
            forecast = self.prophet_model.predict(future)

            trend_prices = forecast['yhat'].values
            trend_returns = np.diff(np.log(trend_prices + 1e-9))
            trend_returns = np.insert(trend_returns, 0, 0.0)

            mean_ret = np.mean(trend_returns)
            std_ret = np.std(trend_returns)
            if std_ret < 1e-9:
                return np.zeros_like(trend_returns)
            else:
                return (trend_returns - mean_ret) / std_ret
        
        except Exception as e:
            logger.warning(f"Prophet signal generation failed: {e}")
            return np.zeros(len(df))
    
    def train(self, gen_df: pd.DataFrame, spec_df:pd.DataFrame) -> bool:
        """
        Train Prophet + LSTMs on historical data.
        
        Args:
            gen_df: Benchmark OHLCV DataFrame (e.g., SPY)
            spec_df: Stock OHLCV DataFrame (e.g., NVDA)
        
        Returns:
            bool: True if training successful
        """
        
        from src.utils import extract_close_price
    
        if len(spec_df) < self.lookback + 10:
            logger.error(f"Insufficient data: {len(spec_df)} < {self.lookback + 10}")
            return False

        try:
            
            if not self.train_prophet(spec_df):
                logger.warning("Prophet training failed, continuing without it")
            
                        # 2. Extract and prepare data for LSTMs
            common_len = min(len(gen_df), len(spec_df))
            gen_close = extract_close_price(gen_df, self.general_ticker).values[-common_len:]
            spec_close = extract_close_price(spec_df, self.ticker).values[-common_len:]
            
            # Ensure 2D arrays
            if len(gen_close.shape) == 1:
                gen_close = gen_close.reshape(-1, 1)
            if len(spec_close.shape) == 1:
                spec_close = spec_close.reshape(-1, 1)
            
            # Convert to log returns
            gen_ret = self._to_log_returns(gen_close.flatten()).reshape(-1, 1)
            spec_ret = self._to_log_returns(spec_close.flatten()).reshape(-1, 1)
            
            # Normalize (ensure 2D input for scaler)
            gen_norm = self.scaler_gen.fit_transform(gen_ret)
            spec_norm = self.scaler_spec.fit_transform(spec_ret)

            gen_X, gen_y = self._create_sequences(gen_norm, self.lookback)
            spec_X, spec_y = self._create_sequences(spec_norm, self.lookback)

            if len(gen_X) == 0 or len(spec_X) == 0:
                logger.error("Failed to create sequences")
                return False
            
            self._train_lstm(
                self.lstm_gen , gen_X, gen_y,
                epochs= self.hyperparams.get('epochs', 30),
                batch_size = 32
            )
            self._train_lstm(
                self.lstm_spec , spec_X, spec_y,
                epochs= self.hyperparams.get('epochs', 30),
                batch_size = 32
            )

            logger.info(f"Forecaster training complete for {self.ticker}")
            return True
    
        except Exception as e:
            logger.error(f"Forecaster training failed: {e}")
            return False
    
    @staticmethod
    def _create_sequences(data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows for LSTM training.
        
        Args:
            data: 1D array of returns
            lookback: Sequence length
        
        Returns:
            Tuple of (X, y) where X is [n_samples, lookback, 1] and y is [n_samples, 1]
        """
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback])
        
        return np.array(X), np.array(y)
    
    def _train_lstm(
        self,
        model: AttentionLSTM,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> None:
        """
        Train LSTM model.
        
        Args:
            model: AttentionLSTM instance
            X: Training features [n_samples, lookback, 1]
            y: Training targets [n_samples, 1]
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Convert to torch tensors
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().to(self.device)
        
        # Create dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                # Forward pass
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.6f}")

    def batch_predict(
    self,
    gen_df: pd.DataFrame,
    spec_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions on test/validation set.
        
        Args:
            gen_df: Benchmark test data
            spec_df: Stock test data
        
        Returns:
            Tuple of (gen_signals, spec_signals, prophet_signals)
            - Each is array of shape [n_days]
            - Values are standardized (mean=0, std=1)
        """
        from src.utils import extract_close_price
        
        try:
            # Prepare data
            common_len = min(len(gen_df), len(spec_df))
            gen_close = extract_close_price(gen_df, self.general_ticker).values[-common_len:]
            spec_close = extract_close_price(spec_df, self.ticker).values[-common_len:]
            
            # Ensure 1D for conversion
            gen_close = np.asarray(gen_close).flatten()
            spec_close = np.asarray(spec_close).flatten()
            
            gen_ret = self._to_log_returns(gen_close).reshape(-1, 1)
            spec_ret = self._to_log_returns(spec_close).reshape(-1, 1)
            
            # Normalize using trained scalers (ensure 2D)
            gen_norm = self.scaler_gen.transform(gen_ret)
            spec_norm = self.scaler_spec.transform(spec_ret)
            
            # Create sequences
            gen_X, _ = self._create_sequences(gen_norm, self.lookback)
            spec_X, _ = self._create_sequences(spec_norm, self.lookback)
            
            # Predict
            self.lstm_gen.eval()
            self.lstm_spec.eval()
            
            with torch.no_grad():
                gen_pred = self._lstm_predict(self.lstm_gen, gen_X)
                spec_pred = self._lstm_predict(self.lstm_spec, spec_X)
            
            # Pad to match original length (first lookback predictions are zero)
            gen_sig = np.concatenate([np.zeros(self.lookback), gen_pred])[:common_len]
            spec_sig = np.concatenate([np.zeros(self.lookback), spec_pred])[:common_len]
            
            # Get Prophet signal
            prophet_sig = self.get_prophet_signal(spec_df)[:common_len]
            
            return gen_sig, spec_sig, prophet_sig
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return (
                np.zeros(len(spec_df)),
                np.zeros(len(spec_df)),
                np.zeros(len(spec_df))
            )
        
    def _lstm_predict(self, model: AttentionLSTM, X: np.ndarray) -> np.ndarray:
        """
        Generate LSTM predictions on batch.
        
        Args:
            model: AttentionLSTM instance
            X: Input sequences [n_samples, lookback, 1]
        
        Returns:
            Predictions [n_samples]
        """
        X_tensor = torch.from_numpy(X).float().to(self.device)
        pred = model(X_tensor)
        return pred.cpu().numpy().squeeze()
    
    def get_train_signals(
        self,
        gen_df: pd.DataFrame,
        spec_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Alias for batch_predict (for consistency with training phase).
        """
        return self.batch_predict(gen_df, spec_df)
    
