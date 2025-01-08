import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import re
import json
from ..utils.logger import Logger
from ..utils.error_handler import handle_errors

class NewsSentimentAnalyzer:
    def __init__(self):
        self.logger = Logger()
        self.sia = SentimentIntensityAnalyzer()
        self.transformer_sentiment = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        self.news_cache = {}
        self.sentiment_history = defaultdict(list)
        
        # Download required NLTK data
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
        except Exception as e:
            self.logger.error(f"NLTK download error: {e}")
    
    @handle_errors
    def analyze_sentiment(self, ticker, lookback_days=7):
        """Comprehensive news sentiment analysis"""
        try:
            # Fetch news from multiple sources
            news_items = self._fetch_news(ticker, lookback_days)
            
            if not news_items:
                return None
            
            # Analyze sentiment for each news item
            analyzed_news = [
                self._analyze_news_item(item)
                for item in news_items
            ]
            
            # Aggregate sentiment metrics
            aggregated = self._aggregate_sentiment(analyzed_news)
            
            # Analyze sentiment trends
            trends = self._analyze_sentiment_trends(ticker, aggregated)
            
            # Detect significant events
            events = self._detect_significant_events(analyzed_news)
            
            return {
                'current_sentiment': aggregated,
                'sentiment_trends': trends,
                'significant_events': events,
                'recent_news': analyzed_news[:5],  # Top 5 recent news
                'sentiment_score': self._calculate_sentiment_score(aggregated, trends)
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return None
    
    def _fetch_news(self, ticker, lookback_days):
        """Fetch news from multiple sources"""
        try:
            news_items = []
            
            # Financial news APIs (replace with actual API calls)
            sources = [
                self._fetch_moneycontrol(ticker, lookback_days),
                self._fetch_economic_times(ticker, lookback_days),
                self._fetch_reuters(ticker, lookback_days),
                self._fetch_bloomberg(ticker, lookback_days)
            ]
            
            for source_items in sources:
                if source_items:
                    news_items.extend(source_items)
            
            # Sort by timestamp
            news_items.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Cache the results
            self.news_cache[ticker] = {
                'timestamp': datetime.now(),
                'items': news_items
            }
            
            return news_items
            
        except Exception as e:
            self.logger.error(f"News fetch error: {e}")
            return []
    
    def _analyze_news_item(self, item):
        """Analyze individual news item"""
        try:
            # Clean text
            text = self._preprocess_text(item['text'])
            
            # VADER sentiment
            vader_scores = self.sia.polarity_scores(text)
            
            # FinBERT sentiment
            finbert_result = self.transformer_sentiment(text[:512])[0]  # Truncate to max length
            
            # Extract key metrics and topics
            metrics = self._extract_metrics(text)
            topics = self._extract_topics(text)
            
            # Combine sentiment scores
            combined_sentiment = self._combine_sentiment_scores(
                vader_scores['compound'],
                finbert_result['score'] if finbert_result['label'] == 'POSITIVE' else -finbert_result['score']
            )
            
            return {
                'title': item['title'],
                'timestamp': item['timestamp'],
                'source': item['source'],
                'url': item['url'],
                'sentiment': {
                    'score': float(combined_sentiment),
                    'vader': vader_scores,
                    'finbert': {
                        'label': finbert_result['label'],
                        'score': float(finbert_result['score'])
                    }
                },
                'metrics': metrics,
                'topics': topics,
                'impact_score': self._calculate_impact_score(
                    combined_sentiment,
                    item['source'],
                    metrics
                )
            }
            
        except Exception as e:
            self.logger.error(f"News item analysis error: {e}")
            return None
    
    def _aggregate_sentiment(self, analyzed_news):
        """Aggregate sentiment metrics"""
        try:
            if not analyzed_news:
                return None
            
            sentiments = [item['sentiment']['score'] for item in analyzed_news]
            impacts = [item['impact_score'] for item in analyzed_news]
            
            # Calculate weighted sentiment
            weighted_sentiment = np.average(
                sentiments,
                weights=impacts
            )
            
            # Calculate sentiment distribution
            sentiment_dist = {
                'positive': len([s for s in sentiments if s > 0.2]),
                'neutral': len([s for s in sentiments if -0.2 <= s <= 0.2]),
                'negative': len([s for s in sentiments if s < -0.2])
            }
            
            return {
                'weighted_sentiment': float(weighted_sentiment),
                'raw_sentiment': float(np.mean(sentiments)),
                'sentiment_std': float(np.std(sentiments)),
                'distribution': sentiment_dist,
                'confidence': self._calculate_confidence(sentiments, impacts)
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment aggregation error: {e}")
            return None
    
    def _analyze_sentiment_trends(self, ticker, current_sentiment):
        """Analyze sentiment trends"""
        try:
            history = self.sentiment_history[ticker]
            history.append({
                'timestamp': datetime.now(),
                'sentiment': current_sentiment
            })
            
            # Keep only last 30 days
            history = history[-30:]
            self.sentiment_history[ticker] = history
            
            if len(history) < 2:
                return None
            
            # Calculate trend metrics
            sentiments = [h['sentiment']['weighted_sentiment'] for h in history]
            
            return {
                'direction': 'improving' if sentiments[-1] > sentiments[-2] else 'deteriorating',
                'momentum': float(sentiments[-1] - np.mean(sentiments[:-1])),
                'volatility': float(np.std(sentiments)),
                'trend_strength': self._calculate_trend_strength(sentiments)
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment trend analysis error: {e}")
            return None
    
    def _detect_significant_events(self, analyzed_news):
        """Detect significant news events"""
        try:
            significant_events = []
            
            for item in analyzed_news:
                impact = item['impact_score']
                sentiment = item['sentiment']['score']
                
                # Check for significant events
                if abs(impact) > 0.7 or abs(sentiment) > 0.7:
                    significant_events.append({
                        'title': item['title'],
                        'timestamp': item['timestamp'],
                        'impact': float(impact),
                        'sentiment': float(sentiment),
                        'url': item['url']
                    })
            
            return sorted(
                significant_events,
                key=lambda x: abs(x['impact']),
                reverse=True
            )
            
        except Exception as e:
            self.logger.error(f"Significant event detection error: {e}")
            return []
    
    def _calculate_sentiment_score(self, aggregated, trends):
        """Calculate final sentiment score"""
        try:
            if not aggregated or not trends:
                return 0
            
            base_score = aggregated['weighted_sentiment']
            
            # Adjust for trend
            trend_adjustment = trends['momentum'] * 0.3
            
            # Adjust for confidence
            confidence_multiplier = aggregated['confidence']
            
            # Calculate final score
            final_score = (base_score + trend_adjustment) * confidence_multiplier
            
            # Normalize to [-1, 1]
            return float(np.clip(final_score, -1, 1))
            
        except Exception as e:
            self.logger.error(f"Sentiment score calculation error: {e}")
            return 0 