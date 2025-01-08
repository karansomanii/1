import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import math
from scipy.stats import norm
from ..utils.logger import Logger
from ..utils.error_handler import handle_errors

class OptionsAnalyzer:
    def __init__(self):
        self.logger = Logger()
        self.options_data = {}
        self.greeks = {}
        self.implied_volatility = {}
        self.risk_free_rate = 0.05  # 5% (update periodically)
        
    @handle_errors
    def analyze_options_chain(self, ticker, spot_price):
        """Comprehensive options chain analysis"""
        try:
            # Fetch options chain data
            chain_data = self._fetch_options_chain(ticker)
            if not chain_data:
                return None
            
            # Calculate implied volatility surface
            iv_surface = self._calculate_iv_surface(chain_data, spot_price)
            
            # Calculate options Greeks
            greeks = self._calculate_greeks(chain_data, spot_price)
            
            # Analyze put-call ratio
            pcr = self._analyze_put_call_ratio(chain_data)
            
            # Analyze options flow
            flow = self._analyze_options_flow(ticker)
            
            # Maximum pain analysis
            max_pain = self._calculate_max_pain(chain_data)
            
            # Options sentiment
            sentiment = self._analyze_options_sentiment(chain_data, pcr, flow)
            
            return {
                'iv_surface': iv_surface,
                'greeks': greeks,
                'put_call_ratio': pcr,
                'options_flow': flow,
                'max_pain': max_pain,
                'sentiment': sentiment,
                'unusual_activity': self._detect_unusual_activity(chain_data)
            }
            
        except Exception as e:
            self.logger.error(f"Options chain analysis error: {e}")
            return None
    
    def _fetch_options_chain(self, ticker):
        """Fetch options chain data from NSE"""
        try:
            # Note: In production, replace with actual NSE options data feed
            # This is a placeholder structure
            expiry_dates = self._get_expiry_dates(ticker)
            chain_data = {
                'calls': [],
                'puts': [],
                'expiry_dates': expiry_dates,
                'timestamp': datetime.now()
            }
            
            self.options_data[ticker] = chain_data
            return chain_data
            
        except Exception as e:
            self.logger.error(f"Options chain fetch error: {e}")
            return None
    
    def _calculate_iv_surface(self, chain_data, spot_price):
        """Calculate implied volatility surface"""
        try:
            iv_surface = {
                'calls': {},
                'puts': {}
            }
            
            for option_type in ['calls', 'puts']:
                for option in chain_data[option_type]:
                    strike = option['strike']
                    expiry = option['expiry']
                    price = option['last_price']
                    
                    iv = self._calculate_implied_volatility(
                        option_type,
                        spot_price,
                        strike,
                        expiry,
                        price
                    )
                    
                    if expiry not in iv_surface[option_type]:
                        iv_surface[option_type][expiry] = {}
                    
                    iv_surface[option_type][expiry][strike] = float(iv)
            
            return iv_surface
            
        except Exception as e:
            self.logger.error(f"IV surface calculation error: {e}")
            return None
    
    def _calculate_greeks(self, chain_data, spot_price):
        """Calculate option Greeks"""
        try:
            greeks = {
                'calls': {},
                'puts': {}
            }
            
            for option_type in ['calls', 'puts']:
                for option in chain_data[option_type]:
                    strike = option['strike']
                    expiry = option['expiry']
                    iv = option.get('implied_volatility', 
                                  self._calculate_implied_volatility(
                                      option_type, spot_price, strike, expiry, option['last_price']
                                  ))
                    
                    greeks[option_type][strike] = self._calculate_option_greeks(
                        option_type,
                        spot_price,
                        strike,
                        expiry,
                        iv
                    )
            
            return greeks
            
        except Exception as e:
            self.logger.error(f"Greeks calculation error: {e}")
            return None
    
    def _analyze_options_flow(self, ticker):
        """Analyze options order flow"""
        try:
            flow_data = self._fetch_options_flow(ticker)
            if not flow_data:
                return None
            
            # Analyze large trades
            large_trades = self._analyze_large_options_trades(flow_data)
            
            # Calculate net delta flow
            net_delta = self._calculate_net_delta_flow(flow_data)
            
            # Analyze institutional activity
            institutional = self._analyze_institutional_activity(flow_data)
            
            return {
                'large_trades': large_trades,
                'net_delta': float(net_delta),
                'institutional_activity': institutional,
                'unusual_volume': self._detect_unusual_volume(flow_data)
            }
            
        except Exception as e:
            self.logger.error(f"Options flow analysis error: {e}")
            return None
    
    def _calculate_max_pain(self, chain_data):
        """Calculate options maximum pain point"""
        try:
            strikes = sorted(list(set(
                [opt['strike'] for opt in chain_data['calls']] +
                [opt['strike'] for opt in chain_data['puts']]
            )))
            
            pain_values = {}
            
            for strike in strikes:
                total_pain = 0
                
                # Calculate pain for call options
                for call in chain_data['calls']:
                    if strike < call['strike']:
                        total_pain += (strike - call['strike']) * call['open_interest']
                
                # Calculate pain for put options
                for put in chain_data['puts']:
                    if strike > put['strike']:
                        total_pain += (put['strike'] - strike) * put['open_interest']
                
                pain_values[strike] = total_pain
            
            # Find strike with minimum pain
            max_pain_strike = min(pain_values.items(), key=lambda x: x[1])[0]
            
            return {
                'max_pain_strike': float(max_pain_strike),
                'pain_distribution': {float(k): float(v) for k, v in pain_values.items()}
            }
            
        except Exception as e:
            self.logger.error(f"Max pain calculation error: {e}")
            return None
    
    def _analyze_options_sentiment(self, chain_data, pcr, flow):
        """Analyze overall options sentiment"""
        try:
            sentiment_score = 0
            
            # PCR analysis
            if pcr['total'] > 1.5:
                sentiment_score -= 2  # Bearish
            elif pcr['total'] < 0.7:
                sentiment_score += 2  # Bullish
            
            # Call/Put OI analysis
            call_oi = sum(opt['open_interest'] for opt in chain_data['calls'])
            put_oi = sum(opt['open_interest'] for opt in chain_data['puts'])
            
            if call_oi > put_oi * 1.5:
                sentiment_score += 1
            elif put_oi > call_oi * 1.5:
                sentiment_score -= 1
            
            # Options flow analysis
            if flow and flow['net_delta'] > 0:
                sentiment_score += 1
            elif flow and flow['net_delta'] < 0:
                sentiment_score -= 1
            
            # Institutional activity
            if flow and flow['institutional_activity']['net_bias'] == 'bullish':
                sentiment_score += 2
            elif flow and flow['institutional_activity']['net_bias'] == 'bearish':
                sentiment_score -= 2
            
            return {
                'score': sentiment_score,
                'interpretation': 'bullish' if sentiment_score > 2 else 'bearish' if sentiment_score < -2 else 'neutral',
                'confidence': abs(sentiment_score) / 6  # Normalize to 0-1
            }
            
        except Exception as e:
            self.logger.error(f"Options sentiment analysis error: {e}")
            return None
    
    def _detect_unusual_activity(self, chain_data):
        """Detect unusual options activity"""
        try:
            unusual_activity = []
            
            # Volume/OI ratio threshold
            vol_oi_threshold = 3
            
            for option_type in ['calls', 'puts']:
                for option in chain_data[option_type]:
                    if option['volume'] > option['open_interest'] * vol_oi_threshold:
                        unusual_activity.append({
                            'type': option_type,
                            'strike': float(option['strike']),
                            'expiry': option['expiry'],
                            'volume': int(option['volume']),
                            'open_interest': int(option['open_interest']),
                            'vol_oi_ratio': float(option['volume'] / option['open_interest'])
                        })
            
            return sorted(unusual_activity, key=lambda x: x['vol_oi_ratio'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Unusual activity detection error: {e}")
            return [] 