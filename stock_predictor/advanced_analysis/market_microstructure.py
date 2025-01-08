import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import statsmodels.api as sm
from scipy import stats
from ..utils.logger import Logger
from ..utils.error_handler import handle_errors

class MarketMicrostructureAnalyzer:
    def __init__(self):
        self.logger = Logger()
        self.tick_data = defaultdict(list)
        self.trade_impact = defaultdict(list)
        self.liquidity_metrics = defaultdict(dict)
        self.market_impact_models = {}
        
    @handle_errors
    def analyze_microstructure(self, ticker):
        """Comprehensive market microstructure analysis"""
        try:
            # Analyze bid-ask spread dynamics
            spread_analysis = self._analyze_spread_dynamics(ticker)
            
            # Analyze market impact
            impact_analysis = self._analyze_market_impact(ticker)
            
            # Analyze order book depth
            depth_analysis = self._analyze_order_book_depth(ticker)
            
            # Analyze trade size distribution
            size_analysis = self._analyze_trade_size_distribution(ticker)
            
            # Calculate market efficiency metrics
            efficiency_metrics = self._calculate_efficiency_metrics(ticker)
            
            return {
                'spread_analysis': spread_analysis,
                'market_impact': impact_analysis,
                'order_book_depth': depth_analysis,
                'trade_size_distribution': size_analysis,
                'market_efficiency': efficiency_metrics,
                'liquidity_score': self._calculate_liquidity_score(ticker)
            }
            
        except Exception as e:
            self.logger.error(f"Microstructure analysis error: {e}")
            return None
    
    def _analyze_spread_dynamics(self, ticker):
        """Analyze bid-ask spread patterns"""
        try:
            spreads = []
            effective_spreads = []
            realized_spreads = []
            
            for tick in self.tick_data[ticker][-1000:]:  # Last 1000 ticks
                quoted_spread = tick['ask'] - tick['bid']
                mid_price = (tick['ask'] + tick['bid']) / 2
                
                # Calculate effective spread
                if tick['side'] == 'buy':
                    effective_spread = 2 * (tick['price'] - mid_price)
                else:
                    effective_spread = 2 * (mid_price - tick['price'])
                
                spreads.append(quoted_spread)
                effective_spreads.append(effective_spread)
                
                # Calculate realized spread (if future price available)
                if len(self.tick_data[ticker]) > tick['index'] + 5:
                    future_mid = (self.tick_data[ticker][tick['index'] + 5]['ask'] + 
                                self.tick_data[ticker][tick['index'] + 5]['bid']) / 2
                    realized_spread = 2 * (tick['price'] - future_mid) * (1 if tick['side'] == 'buy' else -1)
                    realized_spreads.append(realized_spread)
            
            return {
                'average_quoted_spread': float(np.mean(spreads)),
                'average_effective_spread': float(np.mean(effective_spreads)),
                'average_realized_spread': float(np.mean(realized_spreads)) if realized_spreads else None,
                'spread_volatility': float(np.std(spreads)),
                'spread_percentiles': {
                    '25th': float(np.percentile(spreads, 25)),
                    '50th': float(np.percentile(spreads, 50)),
                    '75th': float(np.percentile(spreads, 75))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Spread dynamics analysis error: {e}")
            return None
    
    def _analyze_market_impact(self, ticker):
        """Analyze price impact of trades"""
        try:
            impacts = []
            permanent_impacts = []
            temporary_impacts = []
            
            for trade in self.trade_impact[ticker]:
                # Calculate temporary impact
                pre_trade_price = trade['pre_trade_price']
                trade_price = trade['price']
                post_trade_price = trade['post_trade_price']
                
                temp_impact = (trade_price - pre_trade_price) / pre_trade_price
                perm_impact = (post_trade_price - pre_trade_price) / pre_trade_price
                
                impacts.append({
                    'size': trade['size'],
                    'temporary_impact': temp_impact,
                    'permanent_impact': perm_impact
                })
            
            # Fit market impact model
            if len(impacts) > 0:
                sizes = np.array([imp['size'] for imp in impacts])
                temp_impacts = np.array([imp['temporary_impact'] for imp in impacts])
                perm_impacts = np.array([imp['permanent_impact'] for imp in impacts])
                
                # Power law fit for temporary impact
                temp_model = sm.OLS(np.log(np.abs(temp_impacts)), sm.add_constant(np.log(sizes))).fit()
                
                # Power law fit for permanent impact
                perm_model = sm.OLS(np.log(np.abs(perm_impacts)), sm.add_constant(np.log(sizes))).fit()
                
                self.market_impact_models[ticker] = {
                    'temporary': temp_model,
                    'permanent': perm_model
                }
            
            return {
                'average_temporary_impact': float(np.mean([imp['temporary_impact'] for imp in impacts])),
                'average_permanent_impact': float(np.mean([imp['permanent_impact'] for imp in impacts])),
                'impact_model_parameters': {
                    'temporary': {
                        'alpha': float(np.exp(temp_model.params[0])),
                        'beta': float(temp_model.params[1])
                    } if len(impacts) > 0 else None,
                    'permanent': {
                        'alpha': float(np.exp(perm_model.params[0])),
                        'beta': float(perm_model.params[1])
                    } if len(impacts) > 0 else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Market impact analysis error: {e}")
            return None
    
    def _analyze_order_book_depth(self, ticker):
        """Analyze order book depth and resilience"""
        try:
            depth_metrics = {
                'bid_depth': [],
                'ask_depth': [],
                'total_depth': [],
                'depth_imbalance': []
            }
            
            # Calculate depth at different price levels
            for level in range(5):  # Top 5 price levels
                bid_depth = sum(order['volume'] for order in self.order_book[ticker]['bids'][:level+1])
                ask_depth = sum(order['volume'] for order in self.order_book[ticker]['asks'][:level+1])
                
                depth_metrics['bid_depth'].append(float(bid_depth))
                depth_metrics['ask_depth'].append(float(ask_depth))
                depth_metrics['total_depth'].append(float(bid_depth + ask_depth))
                depth_metrics['depth_imbalance'].append(
                    float((bid_depth - ask_depth) / (bid_depth + ask_depth))
                )
            
            # Calculate order book resilience
            resilience = self._calculate_order_book_resilience(ticker)
            
            return {
                'depth_metrics': depth_metrics,
                'resilience': resilience,
                'depth_score': self._calculate_depth_score(depth_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Order book depth analysis error: {e}")
            return None
    
    def _analyze_trade_size_distribution(self, ticker):
        """Analyze distribution of trade sizes"""
        try:
            trade_sizes = [trade['volume'] for trade in self.tick_data[ticker]]
            
            # Calculate distribution statistics
            size_stats = {
                'mean': float(np.mean(trade_sizes)),
                'median': float(np.median(trade_sizes)),
                'std': float(np.std(trade_sizes)),
                'skew': float(stats.skew(trade_sizes)),
                'kurtosis': float(stats.kurtosis(trade_sizes))
            }
            
            # Calculate size percentiles
            percentiles = [10, 25, 50, 75, 90]
            size_stats['percentiles'] = {
                f'p{p}': float(np.percentile(trade_sizes, p))
                for p in percentiles
            }
            
            # Identify unusual trade sizes
            threshold = size_stats['mean'] + 2 * size_stats['std']
            large_trades = [size for size in trade_sizes if size > threshold]
            
            size_stats['large_trade_frequency'] = len(large_trades) / len(trade_sizes)
            
            return size_stats
            
        except Exception as e:
            self.logger.error(f"Trade size distribution analysis error: {e}")
            return None
    
    def _calculate_efficiency_metrics(self, ticker):
        """Calculate market efficiency metrics"""
        try:
            prices = [tick['price'] for tick in self.tick_data[ticker]]
            returns = np.diff(np.log(prices))
            
            # Calculate variance ratio
            def variance_ratio(returns, k):
                var_k = np.var(returns[::k].sum(k))
                var_1 = np.var(returns)
                return var_k / (k * var_1)
            
            vr_stats = {
                f'vr_{k}': float(variance_ratio(returns, k))
                for k in [2, 5, 10]
            }
            
            # Calculate autocorrelation
            acf = sm.tsa.acf(returns, nlags=10)
            
            # Calculate Hurst exponent
            hurst = self._calculate_hurst_exponent(returns)
            
            return {
                'variance_ratios': vr_stats,
                'autocorrelation': [float(ac) for ac in acf],
                'hurst_exponent': float(hurst),
                'efficiency_score': self._calculate_efficiency_score(vr_stats, acf, hurst)
            }
            
        except Exception as e:
            self.logger.error(f"Efficiency metrics calculation error: {e}")
            return None
    
    def _calculate_liquidity_score(self, ticker):
        """Calculate overall liquidity score"""
        try:
            metrics = self.liquidity_metrics[ticker]
            
            # Combine various liquidity measures
            spread_score = 1 / (1 + metrics.get('average_spread', float('inf')))
            depth_score = np.log1p(metrics.get('total_depth', 0))
            impact_score = 1 / (1 + metrics.get('average_impact', float('inf')))
            
            # Weighted average of scores
            weights = {
                'spread': 0.4,
                'depth': 0.3,
                'impact': 0.3
            }
            
            liquidity_score = (
                weights['spread'] * spread_score +
                weights['depth'] * depth_score +
                weights['impact'] * impact_score
            )
            
            return float(liquidity_score)
            
        except Exception as e:
            self.logger.error(f"Liquidity score calculation error: {e}")
            return None 