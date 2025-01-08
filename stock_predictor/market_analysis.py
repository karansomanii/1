import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import random
import warnings
warnings.filterwarnings('ignore')

class MarketSentimentAnalyzer:
    def __init__(self):
        self.sentiment_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = MinMaxScaler()
        self.market_cache = {}
        self.last_request_time = 0
        self.min_request_interval = 2.0
        
    def calculate_market_sentiment(self, ticker):
        """Calculate overall market sentiment"""
        try:
            # Get market data
            market_data = self._get_market_data(ticker)
            if not market_data:
                return None
            
            # Get global indicators
            global_indicators = self.get_global_indicators()
            
            # Calculate sentiment scores
            technical_sentiment = self._calculate_technical_sentiment(market_data)
            market_sentiment = self._calculate_market_sentiment(global_indicators)
            news_sentiment = self._analyze_news_sentiment(ticker)
            
            # Combine sentiment scores
            sentiment = {
                'technical': technical_sentiment,
                'market': market_sentiment,
                'news': news_sentiment,
                'global_indicators': global_indicators,
                'overall_score': (
                    technical_sentiment * 0.4 +
                    market_sentiment * 0.3 +
                    news_sentiment * 0.3
                )
            }
            
            return sentiment
            
        except Exception as e:
            print(f"Sentiment calculation error: {e}")
            return None
    
    def _get_market_data(self, ticker):
        """Get market data with rate limiting"""
        try:
            current_time = time.time()
            if ticker in self.market_cache:
                cache_time = self.market_cache[ticker]['timestamp']
                if current_time - cache_time < 300:  # 5 minutes cache
                    return self.market_cache[ticker]['data']
            
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last_request)
            
            stock = yf.Ticker(ticker)
            data = stock.history(period="1mo")
            
            self.market_cache[ticker] = {
                'data': data,
                'timestamp': current_time
            }
            
            self.last_request_time = time.time()
            return data
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None
    
    def _calculate_technical_sentiment(self, data):
        """Calculate technical sentiment score"""
        try:
            if data is None or len(data) < 20:
                return 0
            
            closes = data['Close']
            ma20 = closes.rolling(window=20).mean().iloc[-1]
            ma50 = closes.rolling(window=50).mean().iloc[-1]
            current_price = closes.iloc[-1]
            
            # Calculate momentum
            momentum = (current_price - closes.iloc[-20]) / closes.iloc[-20]
            
            # Calculate trend strength
            trend = 1 if current_price > ma20 > ma50 else -1 if current_price < ma20 < ma50 else 0
            
            # Combine factors
            sentiment_score = (
                (trend * 0.5) +
                (momentum * 0.3) +
                ((current_price - ma20) / ma20 * 0.2)
            )
            
            return float(np.clip(sentiment_score, -1, 1))
            
        except Exception as e:
            print(f"Technical sentiment error: {e}")
            return 0
    
    def _calculate_market_sentiment(self, indicators):
        """Calculate market sentiment from global indicators"""
        try:
            if not indicators:
                return 0
            
            # Weight different factors
            weights = {
                'Nifty 50': 0.3,
                'India VIX': -0.2,
                'USD/INR': -0.1,
                'Crude Oil': -0.1,
                'Gold': 0.1,
                'US 10Y Yield': -0.1,
                'S&P 500': 0.1
            }
            
            sentiment_score = 0
            for name, indicator in indicators.items():
                if name in weights:
                    change = indicator.get('change', 0)
                    sentiment_score += change * weights[name]
            
            return float(np.clip(sentiment_score, -1, 1))
            
        except Exception as e:
            print(f"Market sentiment error: {e}")
            return 0
    
    def _analyze_news_sentiment(self, ticker):
        """Analyze news sentiment"""
        try:
            # Simplified news sentiment (random for demo)
            # In a real implementation, you would:
            # 1. Fetch news from reliable sources
            # 2. Use NLP to analyze sentiment
            # 3. Weight recent news more heavily
            return random.uniform(-0.5, 0.5)
            
        except Exception as e:
            print(f"News sentiment error: {e}")
            return 0
