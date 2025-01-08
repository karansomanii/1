import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class TechnicalAnalyzer:
    def __init__(self):
        self.patterns = {}
        self.support_resistance = {}
        self.indicators = {}
    
    def analyze_price_action(self, data):
        """Complete technical analysis of price action"""
        try:
            if len(data) < 20:  # Minimum required data
                return None
            
            analysis = {
                'trends': self._analyze_trend(data),
                'patterns': self._identify_patterns(data),
                'support_resistance': self._find_support_resistance(data),
                'indicators': self._calculate_indicators(data),
                'volatility': self._analyze_volatility(data)
            }
            
            # Calculate overall technical score
            analysis['technical_score'] = self._calculate_technical_score(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"Technical analysis error: {e}")
            return None
    
    def _analyze_trend(self, data):
        """Analyze price trends using multiple timeframes"""
        try:
            closes = data['Close'].values.flatten()  # Ensure 1D array
            trends = {}
            
            # Multiple timeframe analysis
            for period in [10, 20, 50, 200]:
                if len(closes) < period:
                    continue
                    
                ma = pd.Series(closes).rolling(window=period).mean().iloc[-1]
                current_price = closes[-1]
                
                trends[f'MA{period}'] = {
                    'value': float(ma),
                    'position': 'above' if current_price > ma else 'below',
                    'strength': self._calculate_trend_strength(closes[-period:], period)
                }
            
            # Overall trend determination
            if 'MA20' in trends and 'MA50' in trends:
                short_term = trends['MA20']['position']
                long_term = trends['MA50']['position']
                
                if short_term == long_term == 'above':
                    primary_trend = 'strong_uptrend'
                elif short_term == long_term == 'below':
                    primary_trend = 'strong_downtrend'
                elif short_term == 'above' and long_term == 'below':
                    primary_trend = 'potential_reversal_up'
                else:
                    primary_trend = 'potential_reversal_down'
                
                trends['primary_trend'] = primary_trend
            
            return trends
                
        except Exception as e:
            print(f"Trend analysis error: {e}")
            return {}
    
    def _calculate_indicators(self, data):
        """Calculate technical indicators"""
        try:
            closes = data['Close']
            indicators = {}
            
            # RSI
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = float(100 - (100 / (1 + rs.iloc[-1])))
            
            # MACD
            exp1 = closes.ewm(span=12, adjust=False).mean()
            exp2 = closes.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            indicators['MACD'] = {
                'macd': float(macd.iloc[-1]),
                'signal': float(signal.iloc[-1]),
                'histogram': float(macd.iloc[-1] - signal.iloc[-1])
            }
            
            # Bollinger Bands
            ma20 = closes.rolling(window=20).mean()
            std20 = closes.rolling(window=20).std()
            indicators['BB'] = {
                'upper': float(ma20.iloc[-1] + (std20.iloc[-1] * 2)),
                'middle': float(ma20.iloc[-1]),
                'lower': float(ma20.iloc[-1] - (std20.iloc[-1] * 2))
            }
            
            return indicators
                
        except Exception as e:
            print(f"Indicator calculation error: {e}")
            return {}
    
    def _calculate_trend_strength(self, prices, period):
        """Calculate trend strength"""
        try:
            if len(prices) < period:
                return 0
                
            # Linear regression slope
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            # Normalize slope to -1 to 1 range
            return float(np.clip(slope * period / prices.mean(), -1, 1))
            
        except Exception as e:
            print(f"Trend strength calculation error: {e}")
            return 0
    
    def _identify_patterns(self, data):
        """Identify chart patterns"""
        try:
            patterns = {}
            closes = data['Close'].values.flatten()
            highs = data['High'].values.flatten()
            lows = data['Low'].values.flatten()
            
            # Find peaks and troughs
            peaks, _ = find_peaks(closes, distance=5)
            troughs, _ = find_peaks(-closes, distance=5)
            
            if len(peaks) >= 2:
                last_two_peaks = closes[peaks[-2:]]
                if np.abs(last_two_peaks[0] - last_two_peaks[1]) / last_two_peaks[0] < 0.02:
                    patterns['double_top'] = {
                        'confidence': 0.8,
                        'price_target': float(min(closes[peaks[-2]:]))
                    }
            
            return patterns
                
        except Exception as e:
            print(f"Pattern identification error: {e}")
            return {}
    
    def _find_support_resistance(self, data):
        """Find support and resistance levels"""
        try:
            closes = data['Close'].values.flatten()
            highs = data['High'].values.flatten()
            lows = data['Low'].values.flatten()
            
            support_levels = []
            resistance_levels = []
            
            # Find local minima and maxima
            for i in range(2, len(closes)-2):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    support_levels.append(float(lows[i]))
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    resistance_levels.append(float(highs[i]))
            
            return {
                'support': sorted(list(set(support_levels))),
                'resistance': sorted(list(set(resistance_levels)))
            }
                
        except Exception as e:
            print(f"Support/Resistance error: {e}")
            return {'support': [], 'resistance': []}
    
    def _analyze_volatility(self, data):
        """Analyze volatility patterns"""
        try:
            # Ensure we're working with a DataFrame
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # Calculate returns using pandas operations
            returns = data['Close'].pct_change().dropna()
            
            if len(returns) < 2:
                return {}
            
            volatility = {
                'daily_volatility': float(returns.std() * np.sqrt(252)),
                'weekly_volatility': float(returns.rolling(5).std().iloc[-1] * np.sqrt(52)),
                'monthly_volatility': float(returns.rolling(21).std().iloc[-1] * np.sqrt(12)),
                'atr': self._calculate_atr(data)
            }
            
            return volatility
                
        except Exception as e:
            print(f"Volatility analysis error: {e}")
            return {}
    
    def _calculate_technical_score(self, analysis):
        """Calculate overall technical score"""
        try:
            score = 0
            weights = {
                'trend': 0.4,
                'momentum': 0.3,
                'volatility': 0.3
            }
            
            # Trend score
            if 'trends' in analysis and 'primary_trend' in analysis['trends']:
                trend = analysis['trends']['primary_trend']
                score += weights['trend'] * (
                    1 if trend == 'strong_uptrend'
                    else -1 if trend == 'strong_downtrend'
                    else 0.5 if trend == 'potential_reversal_up'
                    else -0.5
                )
            
            # Momentum score
            if 'indicators' in analysis and 'RSI' in analysis['indicators']:
                rsi = analysis['indicators']['RSI']
                score += weights['momentum'] * (
                    (rsi - 50) / 50  # Normalize RSI to -1 to 1
                )
            
            # Volatility score
            if 'volatility' in analysis and 'daily_volatility' in analysis['volatility']:
                vol = analysis['volatility']['daily_volatility']
                score += weights['volatility'] * (
                    -1 if vol > 0.02  # High volatility negative
                    else 1 if vol < 0.01  # Low volatility positive
                    else 0
                )
            
            return float(np.clip(score, -1, 1))
                
        except Exception as e:
            print(f"Technical score calculation error: {e}")
            return 0
