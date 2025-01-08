import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from .ml_patterns import PatternRecognitionML
from .order_flow import OrderFlowAnalyzer
from .options_analysis import OptionsAnalyzer
from .market_microstructure import MarketMicrostructureAnalyzer
from .news_sentiment import NewsSentimentAnalyzer
from ..utils.logger import Logger
from ..utils.error_handler import handle_errors

class IntegratedAnalyzer:
    def __init__(self):
        self.logger = Logger()
        self.pattern_analyzer = PatternRecognitionML()
        self.order_flow_analyzer = OrderFlowAnalyzer()
        self.options_analyzer = OptionsAnalyzer()
        self.microstructure_analyzer = MarketMicrostructureAnalyzer()
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.analysis_weights = self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize component weights for final decision"""
        return {
            'pattern_recognition': 0.25,
            'order_flow': 0.20,
            'options_analysis': 0.20,
            'microstructure': 0.15,
            'sentiment': 0.20
        }
    
    @handle_errors
    def analyze_stock(self, ticker, data):
        """Comprehensive stock analysis using all components"""
        try:
            # Parallel analysis execution
            with ThreadPoolExecutor(max_workers=5) as executor:
                pattern_future = executor.submit(self.pattern_analyzer.predict_pattern, data)
                order_flow_future = executor.submit(self.order_flow_analyzer.analyze_order_flow, ticker)
                options_future = executor.submit(self.options_analyzer.analyze_options_chain, ticker, data['Close'].iloc[-1])
                microstructure_future = executor.submit(self.microstructure_analyzer.analyze_microstructure, ticker)
                sentiment_future = executor.submit(self.sentiment_analyzer.analyze_sentiment, ticker)
            
            # Gather results
            pattern_analysis = pattern_future.result()
            order_flow_analysis = order_flow_future.result()
            options_analysis = options_future.result()
            microstructure_analysis = microstructure_future.result()
            sentiment_analysis = sentiment_future.result()
            
            # Combine analyses
            combined_analysis = self._combine_analyses(
                pattern_analysis,
                order_flow_analysis,
                options_analysis,
                microstructure_analysis,
                sentiment_analysis
            )
            
            # Generate trading signals
            signals = self._generate_trading_signals(combined_analysis)
            
            # Calculate confidence and risk metrics
            confidence = self._calculate_confidence(combined_analysis)
            risk_metrics = self._calculate_risk_metrics(combined_analysis)
            
            return {
                'timestamp': datetime.now(),
                'ticker': ticker,
                'analysis': combined_analysis,
                'signals': signals,
                'confidence': confidence,
                'risk_metrics': risk_metrics,
                'recommendation': self._generate_recommendation(signals, confidence, risk_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Integrated analysis error: {e}")
            return None
    
    def _combine_analyses(self, pattern, order_flow, options, microstructure, sentiment):
        """Combine analyses from all components"""
        try:
            return {
                'technical': {
                    'pattern_recognition': pattern,
                    'trend_strength': self._calculate_trend_strength(pattern),
                    'support_resistance': self._identify_key_levels(pattern)
                },
                'order_flow': {
                    'analysis': order_flow,
                    'buying_pressure': order_flow.get('buying_pressure', 0),
                    'large_orders': order_flow.get('large_orders', [])
                },
                'options': {
                    'analysis': options,
                    'implied_direction': self._interpret_options_sentiment(options),
                    'key_strikes': self._identify_key_strikes(options)
                },
                'microstructure': {
                    'analysis': microstructure,
                    'liquidity_score': microstructure.get('liquidity_score', 0),
                    'efficiency_metrics': microstructure.get('market_efficiency', {})
                },
                'sentiment': {
                    'analysis': sentiment,
                    'score': sentiment.get('sentiment_score', 0) if sentiment else 0,
                    'significant_events': sentiment.get('significant_events', []) if sentiment else []
                }
            }
            
        except Exception as e:
            self.logger.error(f"Analysis combination error: {e}")
            return {}
    
    def _generate_trading_signals(self, combined_analysis):
        """Generate trading signals from combined analysis"""
        try:
            signals = {
                'primary_direction': None,
                'strength': 0,
                'timeframe': None,
                'entry_points': [],
                'exit_points': []
            }
            
            # Calculate directional signals from each component
            pattern_signal = self._get_pattern_signal(combined_analysis['technical'])
            flow_signal = self._get_flow_signal(combined_analysis['order_flow'])
            options_signal = self._get_options_signal(combined_analysis['options'])
            micro_signal = self._get_microstructure_signal(combined_analysis['microstructure'])
            sentiment_signal = self._get_sentiment_signal(combined_analysis['sentiment'])
            
            # Weight and combine signals
            weighted_signal = (
                pattern_signal * self.analysis_weights['pattern_recognition'] +
                flow_signal * self.analysis_weights['order_flow'] +
                options_signal * self.analysis_weights['options_analysis'] +
                micro_signal * self.analysis_weights['microstructure'] +
                sentiment_signal * self.analysis_weights['sentiment']
            )
            
            # Determine primary direction and strength
            signals['primary_direction'] = 'buy' if weighted_signal > 0.1 else 'sell' if weighted_signal < -0.1 else 'neutral'
            signals['strength'] = abs(weighted_signal)
            
            # Determine optimal timeframe
            signals['timeframe'] = self._determine_timeframe(combined_analysis)
            
            # Generate entry and exit points
            signals['entry_points'] = self._generate_entry_points(combined_analysis, signals['primary_direction'])
            signals['exit_points'] = self._generate_exit_points(combined_analysis, signals['primary_direction'])
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return None
    
    def _calculate_confidence(self, combined_analysis):
        """Calculate confidence level in the analysis"""
        try:
            component_confidence = {
                'pattern': self._get_pattern_confidence(combined_analysis['technical']),
                'order_flow': self._get_flow_confidence(combined_analysis['order_flow']),
                'options': self._get_options_confidence(combined_analysis['options']),
                'microstructure': self._get_microstructure_confidence(combined_analysis['microstructure']),
                'sentiment': self._get_sentiment_confidence(combined_analysis['sentiment'])
            }
            
            # Weight confidences
            weighted_confidence = sum(
                conf * self.analysis_weights[comp.replace('pattern', 'pattern_recognition')]
                for comp, conf in component_confidence.items()
            )
            
            # Adjust for agreement between components
            agreement_factor = self._calculate_agreement_factor(component_confidence)
            
            return float(weighted_confidence * agreement_factor)
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {e}")
            return 0.0
    
    def _generate_recommendation(self, signals, confidence, risk_metrics):
        """Generate final trading recommendation"""
        try:
            if signals['primary_direction'] == 'neutral' or confidence < 0.6:
                return {
                    'action': 'HOLD',
                    'reasoning': 'Insufficient confidence or unclear direction'
                }
            
            # Adjust position size based on confidence and risk
            position_size = self._calculate_position_size(confidence, risk_metrics)
            
            return {
                'action': signals['primary_direction'].upper(),
                'confidence': float(confidence),
                'strength': float(signals['strength']),
                'timeframe': signals['timeframe'],
                'position_size': position_size,
                'entry_points': signals['entry_points'],
                'exit_points': signals['exit_points'],
                'stop_loss': self._calculate_stop_loss(signals, risk_metrics),
                'take_profit': self._calculate_take_profit(signals, risk_metrics),
                'risk_reward_ratio': float(risk_metrics['risk_reward_ratio'])
            }
            
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {e}")
            return None 