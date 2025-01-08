import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
import yfinance as yf
from .utils.logger import Logger
from .utils.error_handler import handle_errors

class PortfolioManager:
    def __init__(self, predictor, risk_free_rate=0.02):
        self.predictor = predictor
        self.risk_free_rate = risk_free_rate
        self.logger = Logger()
        self.portfolio = {}
        self.positions = {}
        self.performance = {}
        
    @handle_errors
    def optimize_portfolio(self, tickers, initial_weights=None):
        """Optimize portfolio weights using Modern Portfolio Theory"""
        try:
            # Get historical data for all tickers
            data = self._get_portfolio_data(tickers)
            if data is None:
                return None
            
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # Get predictions for each ticker
            predictions = {}
            for ticker in tickers:
                pred = self.predictor.predict_movement(ticker, data[ticker])
                if pred:
                    predictions[ticker] = pred
            
            # Adjust expected returns based on predictions
            exp_returns = self._calculate_expected_returns(returns, predictions)
            
            # Calculate covariance matrix
            cov_matrix = returns.cov() * 252  # Annualized
            
            # Initial guess
            if initial_weights is None:
                initial_weights = np.array([1/len(tickers)] * len(tickers))
            
            # Optimize
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
                {'type': 'ineq', 'fun': lambda x: x},  # non-negative weights
                {'type': 'ineq', 'fun': lambda x: 0.4 - x}  # maximum 40% per position
            )
            
            result = minimize(
                self._portfolio_objective,
                initial_weights,
                args=(exp_returns, cov_matrix),
                method='SLSQP',
                constraints=constraints
            )
            
            if result.success:
                optimized_weights = result.x
                portfolio_metrics = self._calculate_portfolio_metrics(
                    optimized_weights, exp_returns, cov_matrix
                )
                
                return {
                    'weights': dict(zip(tickers, optimized_weights)),
                    'metrics': portfolio_metrics
                }
            else:
                self.logger.error("Portfolio optimization failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Portfolio optimization error: {e}")
            return None
    
    def _get_portfolio_data(self, tickers, period='1y'):
        """Get historical data for portfolio"""
        try:
            data = pd.DataFrame()
            
            for ticker in tickers:
                ticker_data = yf.download(ticker, period=period, progress=False)
                if len(ticker_data) > 0:
                    data[ticker] = ticker_data['Close']
                else:
                    self.logger.error(f"No data found for {ticker}")
                    return None
            
            return data
            
        except Exception as e:
            self.logger.error(f"Portfolio data error: {e}")
            return None
    
    def _calculate_expected_returns(self, historical_returns, predictions):
        """Calculate expected returns using historical data and predictions"""
        try:
            exp_returns = historical_returns.mean() * 252  # Annualized
            
            # Adjust based on predictions
            for ticker, pred in predictions.items():
                confidence = pred['confidence']
                predicted_return = (pred['target_price'] - pred['current_price']) / pred['current_price']
                
                # Blend historical and predicted returns based on confidence
                exp_returns[ticker] = (
                    exp_returns[ticker] * (1 - confidence) +
                    predicted_return * confidence
                )
            
            return exp_returns
            
        except Exception as e:
            self.logger.error(f"Expected returns calculation error: {e}")
            return historical_returns.mean() * 252
    
    def _portfolio_objective(self, weights, exp_returns, cov_matrix):
        """Portfolio optimization objective function"""
        try:
            port_return = np.sum(exp_returns * weights)
            port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Sharpe Ratio (negative for minimization)
            return -(port_return - self.risk_free_rate) / port_risk
            
        except Exception as e:
            self.logger.error(f"Portfolio objective calculation error: {e}")
            return float('inf')
    
    def _calculate_portfolio_metrics(self, weights, exp_returns, cov_matrix):
        """Calculate portfolio performance metrics"""
        try:
            portfolio_return = np.sum(exp_returns * weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
            
            return {
                'expected_return': float(portfolio_return),
                'volatility': float(portfolio_risk),
                'sharpe_ratio': float(sharpe_ratio)
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio metrics calculation error: {e}")
            return None
    
    @handle_errors
    def rebalance_portfolio(self, current_positions, target_weights):
        """Calculate rebalancing trades"""
        try:
            trades = {}
            total_value = sum(
                pos['shares'] * pos['current_price'] 
                for pos in current_positions.values()
            )
            
            for ticker, target_weight in target_weights.items():
                current_value = 0
                if ticker in current_positions:
                    current_value = (
                        current_positions[ticker]['shares'] * 
                        current_positions[ticker]['current_price']
                    )
                
                target_value = total_value * target_weight
                value_difference = target_value - current_value
                
                if abs(value_difference) > total_value * 0.01:  # 1% threshold
                    trades[ticker] = {
                        'action': 'buy' if value_difference > 0 else 'sell',
                        'amount': abs(value_difference)
                    }
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Rebalancing calculation error: {e}")
            return None 