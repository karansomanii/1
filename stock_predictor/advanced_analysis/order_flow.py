import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import requests
from ..utils.logger import Logger
from ..utils.error_handler import handle_errors

class OrderFlowAnalyzer:
    def __init__(self):
        self.logger = Logger()
        self.tick_data = defaultdict(list)
        self.order_book = defaultdict(dict)
        self.volume_profile = defaultdict(lambda: defaultdict(float))
        self.large_orders = defaultdict(list)
        
    @handle_errors
    def analyze_order_flow(self, ticker):
        """Comprehensive order flow analysis"""
        try:
            # Get order book data
            order_book_data = self._fetch_order_book(ticker)
            
            # Analyze order book imbalance
            imbalance = self._analyze_order_imbalance(order_book_data)
            
            # Analyze volume profile
            volume_analysis = self._analyze_volume_profile(ticker)
            
            # Detect large orders
            large_orders = self._detect_large_orders(ticker)
            
            # Analyze price impact
            price_impact = self._analyze_price_impact(ticker)
            
            return {
                'order_imbalance': imbalance,
                'volume_profile': volume_analysis,
                'large_orders': large_orders,
                'price_impact': price_impact,
                'buying_pressure': self._calculate_buying_pressure(ticker)
            }
            
        except Exception as e:
            self.logger.error(f"Order flow analysis error: {e}")
            return None
    
    def _fetch_order_book(self, ticker):
        """Fetch real-time order book data"""
        try:
            # Note: In production, replace with actual NSE order book data feed
            # This is a placeholder structure
            order_book = {
                'bids': [],
                'asks': [],
                'timestamp': datetime.now()
            }
            
            # Store in memory
            self.order_book[ticker] = order_book
            return order_book
            
        except Exception as e:
            self.logger.error(f"Order book fetch error: {e}")
            return None
    
    def _analyze_order_imbalance(self, order_book):
        """Analyze order book imbalance"""
        try:
            if not order_book:
                return None
                
            total_bid_volume = sum(order['volume'] for order in order_book['bids'])
            total_ask_volume = sum(order['volume'] for order in order_book['asks'])
            
            # Calculate imbalance ratio
            imbalance_ratio = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
            # Calculate bid-ask spread
            best_bid = max(order['price'] for order in order_book['bids'])
            best_ask = min(order['price'] for order in order_book['asks'])
            spread = best_ask - best_bid
            
            return {
                'imbalance_ratio': float(imbalance_ratio),
                'bid_volume': float(total_bid_volume),
                'ask_volume': float(total_ask_volume),
                'spread': float(spread),
                'pressure': 'buying' if imbalance_ratio > 0.1 else 'selling' if imbalance_ratio < -0.1 else 'neutral'
            }
            
        except Exception as e:
            self.logger.error(f"Order imbalance analysis error: {e}")
            return None
    
    def _analyze_volume_profile(self, ticker):
        """Analyze volume profile"""
        try:
            # Get intraday price levels and volumes
            price_levels = defaultdict(float)
            
            # Aggregate volume at each price level
            for tick in self.tick_data[ticker]:
                price_level = round(tick['price'], 2)
                price_levels[price_level] += tick['volume']
            
            # Find POC (Point of Control)
            poc_price = max(price_levels.items(), key=lambda x: x[1])[0]
            
            # Calculate Value Area
            total_volume = sum(price_levels.values())
            value_area_volume = 0
            value_area_prices = []
            
            for price, volume in sorted(price_levels.items(), key=lambda x: x[1], reverse=True):
                value_area_volume += volume
                value_area_prices.append(price)
                if value_area_volume >= total_volume * 0.68:  # 68% of volume
                    break
            
            return {
                'poc': float(poc_price),
                'value_area_high': float(max(value_area_prices)),
                'value_area_low': float(min(value_area_prices)),
                'volume_distribution': dict(price_levels)
            }
            
        except Exception as e:
            self.logger.error(f"Volume profile analysis error: {e}")
            return None
    
    def _detect_large_orders(self, ticker):
        """Detect and analyze large orders"""
        try:
            large_orders = []
            volume_threshold = self._calculate_volume_threshold(ticker)
            
            for tick in self.tick_data[ticker]:
                if tick['volume'] > volume_threshold:
                    large_orders.append({
                        'timestamp': tick['timestamp'],
                        'price': float(tick['price']),
                        'volume': float(tick['volume']),
                        'side': tick['side'],
                        'impact': self._calculate_price_impact(tick)
                    })
            
            return sorted(large_orders, key=lambda x: x['volume'], reverse=True)[:10]
            
        except Exception as e:
            self.logger.error(f"Large order detection error: {e}")
            return []
    
    def _analyze_price_impact(self, ticker):
        """Analyze price impact of orders"""
        try:
            impacts = []
            window_size = 100  # ticks
            
            for i in range(len(self.tick_data[ticker]) - window_size):
                window = self.tick_data[ticker][i:i+window_size]
                
                # Calculate volume-weighted price change
                vwap_start = sum(t['price'] * t['volume'] for t in window[:10]) / sum(t['volume'] for t in window[:10])
                vwap_end = sum(t['price'] * t['volume'] for t in window[-10:]) / sum(t['volume'] for t in window[-10:])
                
                total_volume = sum(t['volume'] for t in window)
                price_change = (vwap_end - vwap_start) / vwap_start
                
                impacts.append({
                    'volume': float(total_volume),
                    'price_change': float(price_change),
                    'impact_ratio': float(price_change / total_volume)
                })
            
            return {
                'average_impact': float(np.mean([i['impact_ratio'] for i in impacts])),
                'max_impact': float(max([i['impact_ratio'] for i in impacts])),
                'recent_impacts': impacts[-5:]
            }
            
        except Exception as e:
            self.logger.error(f"Price impact analysis error: {e}")
            return None
    
    def _calculate_buying_pressure(self, ticker):
        """Calculate buying pressure indicator"""
        try:
            recent_ticks = self.tick_data[ticker][-1000:]  # Last 1000 ticks
            
            buy_volume = sum(t['volume'] for t in recent_ticks if t['side'] == 'buy')
            sell_volume = sum(t['volume'] for t in recent_ticks if t['side'] == 'sell')
            
            # Calculate various pressure metrics
            volume_ratio = buy_volume / sell_volume if sell_volume > 0 else float('inf')
            price_trend = self._calculate_price_trend(recent_ticks)
            
            return {
                'volume_ratio': float(volume_ratio),
                'price_trend': price_trend,
                'intensity': 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.67 else 'medium',
                'direction': 'bullish' if volume_ratio > 1 else 'bearish'
            }
            
        except Exception as e:
            self.logger.error(f"Buying pressure calculation error: {e}")
            return None
    
    def update_tick_data(self, ticker, new_ticks):
        """Update tick data with new information"""
        try:
            self.tick_data[ticker].extend(new_ticks)
            
            # Keep only recent ticks to manage memory
            max_ticks = 100000
            if len(self.tick_data[ticker]) > max_ticks:
                self.tick_data[ticker] = self.tick_data[ticker][-max_ticks:]
                
        except Exception as e:
            self.logger.error(f"Tick data update error: {e}") 