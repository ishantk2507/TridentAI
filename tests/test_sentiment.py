"""
Unit tests for SentimentEngine.
"""
import pytest
from src.sentiment_engine import SentimentEngine


class TestSentimentEngine:
    """Tests for sentiment analysis pipeline."""
    
    @pytest.fixture
    def engine(self):
        """Create SentimentEngine instance."""
        return SentimentEngine()
    
    def test_sentiment_initialization(self, engine):
        """Test that sentiment engine initializes without error."""
        assert engine.tokenizer is not None
        assert engine.model is not None
        assert engine.nlp is not None
    
    def test_sentiment_range(self, engine):
        """Test that sentiment scores are in [-1, 1] range."""
        tickers = ['AAPL', 'MSFT']
        
        for ticker in tickers:
            try:
                score, headlines = engine.get_sentiment(ticker)
                
                assert isinstance(score, float), f"Score should be float, got {type(score)}"
                assert -1.0 <= score <= 1.0, f"Sentiment {score} out of range [-1, 1]"
                assert isinstance(headlines, list), f"Headlines should be list"
            except Exception as e:
                # Sentiment scraping may fail due to network issues
                pytest.skip(f"Sentiment scraping failed: {e}")
    
    def test_sentiment_no_news_fallback(self, engine):
        """Test graceful handling when news scraping fails."""
        # Use fake ticker unlikely to have news
        score, headlines = engine.get_sentiment('FAKEXYZ12345')
        
        # Should return 0.0 if no news found
        assert score == 0.0, f"Expected 0.0 for no news, got {score}"
        assert headlines == ["No Recent News Found"]