"""
Unit tests for utility functions in utils.py
"""

import pytest 
import numpy as np 
import pandas as pd
from src.utils import (
    extract_close_price, to_log_returns, robust_standardize,
    compute_volatility_metrics, get_regime,set_seeds
)

class TestDataExtraction:
    """Tests for data extraction functions."""

    def test_extract_close_price_single_column(self):
        """Test close price extraction from simple DataFrame."""
        dates = pd.date_range(start='2020-01-01', periods=10)
        df = pd.DataFrame({
            'Open':np.random.rand(10) * 100,
            'Close':np.random.rand(10) * 100,
            'Volume':np.random.randint(1000, 10000, 10),
        }, index=dates)

        close = extract_close_price(df, 'AAPL')

        assert isinstance(close, pd.Series), "Should return Series"
        assert len(close)==10, "Should have 10 entries"
        assert close.dtype == float or close.dtype == np.float64, "Should be float"

    
    def test_to_log_returns(self):
        """Test log return computation."""
        prices  = np.array([100, 101, 102, 101, 100 ], dtype=float)

        returns = to_log_returns(prices)

        assert len(returns) == len(prices) - 1, "Should have T-1 returns"
        assert isinstance(returns, np.ndarray), "Should return ndarray"

        assert np.all(returns < 0.02), "Returns should be <2% for these prices"
    
    def test_robust_standardize(self):
        """Test robust standardization with zero-varaince protection."""

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        standardized, mean, std = robust_standardize(data)

        assert np.isclose(np.mean(standardized), 0, atol=1e-6), "Mean should be 0"
        assert np.isclose(np.std(standardized), 1, atol=1e-6), "Std should be 1"


    def test_robust_standardize_zero_variance(self):
        """Test robust standardization when variance is zero."""

        data = np.array([5.0, 5.0, 5.0, 5.0])

        standardized, mean, std = robust_standardize(data)

        assert np.allclose(standardized, 0), "Should return zeros for constant data"

        assert std < 1e-9, "Std should be near-zero"


class TestVolatilityMetrics:
    """Tests for volatility and regime classification."""

    def test_volatility_metrics_structure(self):
        """Test that volatility computation returns correct structure."""

        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 252)  # Simulate daily returns for 1 year

        metrics = compute_volatility_metrics(returns)

        assert 'daily_vol' in metrics, "Should contain daily_vol"
        assert 'annual_vol' in metrics, "Should contain annual_vol"
        assert 'trend_strength' in metrics, "Should contain trend_strength"
        assert 'vol_drift' in metrics, "Should contain vol_drift"

        assert 0 <= metrics['trend_strength'] <= 1, "Trend strength should be [0, 1]"

    def test_regime_classification(self):
        """Test 6-tier regime classification."""
        test_cases = [
            (85, "extreme_volatility"),
            (60, "very_high_volatility"),
            (40, "high_volatility"),
            (25, "medium_high_volatility"),
            (15, "medium_volatility"),
            (8, "low_volatility"),
        ]
        
        for vol, expected_regime in test_cases:
            regime = get_regime(vol)
            assert regime == expected_regime, f"Vol {vol}% should be {expected_regime}, got {regime}"

class TestSeeding:

    def test_set_seeds_reproducibility(self):
        """Test that set_seeds produces reproducible random numbers."""

        set_seeds(42)
        array1 = np.random.randn(5)
        
        set_seeds(42)
        array2 = np.random.randn(5)
        
        assert np.allclose(array1, array2), "Same seed should produce same random numbers"


if __name__ == "__main__":
    pytest.main([__file__,"-v"])

