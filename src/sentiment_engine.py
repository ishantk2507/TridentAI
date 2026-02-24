"""
Sentiment analysis engine using FinBERT for financial text.

Scrapes financial news and analyzes sentiment with a finance-specific BERT model.
"""

import torch 
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from typing import Tuple, List
import logging 

logger = logging.getLogger(__name__)

class SentimentEngine:
    """
    Analyzes news sentiment using FinBERT (finance-specific BERT model).
    
    This engine:
    1. Scrapes recent headlines from Google News RSS
    2. Runs FinBERT (fine-tuned on financial text) on each headline
    3. Aggregates sentiment with confidence weighting
    4. Returns [-1, 1] sentiment score for use in RL agent
    
    Attributes:
        tokenizer: BertTokenizer for FinBERT
        model: BertForSequenceClassification for sentiment
        nlp: Hugging Face pipeline for inference
        
    Example:
        >>> engine = SentimentEngine()
        >>> sentiment_score, headlines = engine.get_sentiment('NVDA')
        >>> print(f"Sentiment: {sentiment_score:.2f}, Headlines: {len(headlines)}")
        Sentiment: 0.62, Headlines: 5
    """

    def __init__(self):
        """Initialize FinBERT model and tokenizer."""
        model_name = 'yiyanghkust/finbert-tone'

        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise

        device = 0 if torch.cuda.is_available() else -1
        self.nlp = pipeline(
            "sentiment-analysis",
            model = self.model,
            tokenizer = self.tokenizer,
            device = device
        )


    def scrape_news(self, ticker: str, max_headlines: int = 5) -> List[str]:
        """
        Scrape latest financial news headlines from Google News RSS.
        
        Args:
            ticker: Stock symbol (e.g., 'NVDA', 'NVDA.NS')
            max_headlines: Max headlines to fetch (default: 5)
        
        Returns:
            list: Headline strings (empty list if scrape fails)
        """

        search_ticker = ticker.replace(".NS", "")
        query = f"{search_ticker} stock news"
        url = f"https://news.google.com/res/search?q={query}&hl=en-US&gl=US&ceid=US:en"

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.content, features="xml")
            items = soup.find_all('item')

            headlines = [item.title.text for item in items[:max_headlines]]
            return headlines
        except Exception as e:
            logger.warning(f"News Scrapping failed for {ticker}: {e}")
            return []
        
    
    def get_sentiment(self, ticker: str) -> Tuple[float, List[str]]:
        """
        Get aggregated sentiment score for a ticker.
        
        Process:
        1. Scrape 5 recent headlines
        2. Run FinBERT on each headline
        3. Weight scores by model confidence
        4. Aggregate to [-1, 1] range
        
        Args:
            ticker: Stock symbol
        
        Returns:
            tuple: (sentiment_score, headlines)
                - sentiment_score (float): [-1, 1] range, higher = more bullish
                - headlines (list): Raw headlines used for analysis
                
        Example:
            >>> sentiment_score, headlines = engine.get_sentiment('NVDA')
            >>> print(f"Score: {sentiment_score:.2f}")
            >>> for h in headlines:
            ...     print(f"  - {h}")
        """

        headlines = self.scrape_news(ticker)

        if not headlines:
            return 0.0 , ["No Recent News Found"] 
        
        results = self.nlp(headlines)

        score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}

        weighted_scores = [
            score_map[res['label']] * res['score']
            for res in results
        ]

        final_sentiment = sum(weighted_scores) / len(weighted_scores)
        final_sentiment = max(-1, min(1, final_sentiment))

        return final_sentiment, headlines
