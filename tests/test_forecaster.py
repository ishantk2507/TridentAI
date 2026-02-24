"""
Unit tests for forecasting components.
"""
import pytest
import numpy as np
import pandas as pd
import torch
from src.forecaster import Attention, AttentionLSTM, TridentForecaster


class TestAttention:
    """Tests for Attention mechanism."""
    
    def test_attention_initialization(self):
        """Test Attention layer initialization."""
        hidden_dim = 64
        attention = Attention(hidden_dim)
        
        assert attention.attention.in_features == hidden_dim
        assert attention.attention.out_features == 1
    
    def test_attention_forward(self):
        """Test Attention forward pass."""
        batch_size = 32
        seq_len = 50
        hidden_dim = 64
        
        attention = Attention(hidden_dim)
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        output = attention(x)
        
        assert output.shape == (batch_size, hidden_dim), f"Output shape mismatch: {output.shape}"


class TestAttentionLSTM:
    """Tests for Attention LSTM model."""
    
    def test_lstm_initialization(self):
        """Test AttentionLSTM initialization."""
        model = AttentionLSTM(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)
        
        assert model.lstm is not None
        assert model.attention is not None
        assert model.fc is not None
    
    def test_lstm_forward(self):
        """Test AttentionLSTM forward pass."""
        batch_size = 16
        seq_len = 50
        
        model = AttentionLSTM(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1)
        x = torch.randn(batch_size, seq_len, 1)
        
        output = model(x)
        
        assert output.shape == (batch_size, 1), f"Output shape mismatch: {output.shape}"
    
    def test_lstm_output_range(self):
        """Test that LSTM outputs are reasonable."""
        model = AttentionLSTM(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)
        x = torch.randn(8, 50, 1)
        
        output = model(x)
        
        # Outputs should be reasonable (not NaN or Inf)
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"


class TestTridentForecaster:
    """Tests for TridentForecaster."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2020-01-01', periods=200)
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(200) * 2 + 100)
        
        df = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 200)
        }, index=dates)
        
        return df
    
    def test_forecaster_initialization(self):
        """Test TridentForecaster initialization."""
        hyperparams = {'hidden_gen': 32, 'hidden_spec': 64, 'lookback': 50}
        forecaster = TridentForecaster('NVDA', 'SPY', hyperparams)
        
        assert forecaster.ticker == 'NVDA'
        assert forecaster.general_ticker == 'SPY'
        assert forecaster.lstm_gen is not None
        assert forecaster.lstm_spec is not None
    
    def test_forecaster_sequences(self, sample_data):
        """Test sequence creation."""
        data = np.random.randn(100, 1)
        lookback = 50
        
        X, y = TridentForecaster._create_sequences(data, lookback)
        
        assert X.shape == (len(data) - lookback, lookback, 1)
        assert y.shape == (len(data) - lookback, 1)
    
    def test_forecaster_log_returns(self):
        """Test log return computation."""
        prices = np.array([100, 101, 102, 101, 100]).reshape(-1, 1)
        
        returns = TridentForecaster._to_log_returns(prices)
        
        assert len(returns) == len(prices) - 1
        assert np.all(np.abs(returns) < 0.02)  # Small changes expected