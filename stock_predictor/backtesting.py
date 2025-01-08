import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from tqdm import tqdm
from .utils.logger import Logger
from .utils.error_handler import handle_errors

class Backtester:
    def __init__(self, predictor, initial_capital=100000):
        self.predictor = predictor
        self.initial_capital = initial_capital
        self.logger = Logger()
        self.results = {}
        
    @handle_errors
    def run_backtest(self, ticker, start_date, end_date=None):
        """Run backtest for a specific period"""
        try:
            # Get historical data
            data = self._get_historical_data(ticker, start_date, end_date)
            if data is None:
                return None
            
            # Initialize backtest variables
            capital = self.initial_capital
            positions = []
            trades = []
            equity_curve = [capital]
            
            # Run simulation
            for i in tqdm(range(len(data)-30), desc="Backtesting"):
                # Get data window
                window = data.iloc[i:i+30]
                current_date = window.index[-1]
                
                # Make prediction
                prediction = self.predictor.predict_movement(ticker, window)
                if prediction is None:
                    continue
                
                # Process open positions
                capital, positions = self._process_positions(
                    capital, positions, window.iloc[-1], current_date
                )
                
                # Open new position if conditions met
                if len(positions) == 0:  # Only open if no positions
                    position = self._open_position(
                        prediction, capital, window.iloc[-1], current_date
                    )
                    if position:
                        positions.append(position)
                        trades.append(position)
                
                equity_curve.append(capital)
            
            # Close any remaining positions
            final_price = data.iloc[-1]['Close']
            for position in positions:
                capital += position['size'] * (final_price - position['entry_price'])
            
            # Calculate performance metrics
            results = self._calculate_performance(
                capital, equity_curve, trades, data
            )
            
            self.results[ticker] = results
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            return None
    
    def _get_historical_data(self, ticker, start_date, end_date=None):
        """Get historical data for backtesting"""
        try:
            end_date = end_date or datetime.now()
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if len(data) < 30:  # Minimum required data
                self.logger.error("Insufficient historical data")
                return None
                
            return data
            
        except Exception as e:
            self.logger.error(f"Data download error: {e}")
            return None
    
    def _process_positions(self, capital, positions, current_bar, current_date):
        """Process open positions"""
        remaining_positions = []
        current_price = current_bar['Close']
        
        for position in positions:
            # Check stop loss
            if current_price <= position['stop_loss']:
                capital += position['size'] * (position['stop_loss'] - position['entry_price'])
                position['exit_price'] = position['stop_loss']
                position['exit_date'] = current_date
                position['status'] = 'stopped_out'
                continue
                
            # Check target price
            if current_price >= position['target']:
                capital += position['size'] * (position['target'] - position['entry_price'])
                position['exit_price'] = position['target']
                position['exit_date'] = current_date
                position['status'] = 'target_hit'
                continue
                
            remaining_positions.append(position)
        
        return capital, remaining_positions
    
    def _open_position(self, prediction, capital, current_bar, current_date):
        """Open new position based on prediction"""
        if prediction['confidence'] < 0.6:  # Minimum confidence threshold
            return None
            
        # Calculate position size using Kelly Criterion
        win_prob = prediction['confidence']
        potential_gain = (prediction['target_price'] - current_bar['Close'])
        potential_loss = (current_bar['Close'] - prediction['stop_loss'])
        
        if potential_loss <= 0:
            return None
            
        kelly = win_prob - ((1 - win_prob) / (potential_gain / potential_loss))
        kelly = max(0, min(kelly, 0.2))  # Limit to 20% of capital
        
        position_size = (capital * kelly) / current_bar['Close']
        
        return {
            'entry_date': current_date,
            'entry_price': current_bar['Close'],
            'size': position_size,
            'target': prediction['target_price'],
            'stop_loss': prediction['stop_loss'],
            'confidence': prediction['confidence'],
            'status': 'open'
        }
    
    def _calculate_performance(self, final_capital, equity_curve, trades, data):
        """Calculate backtest performance metrics"""
        try:
            equity_curve = np.array(equity_curve)
            returns = np.diff(equity_curve) / equity_curve[:-1]
            
            # Basic metrics
            total_return = (final_capital - self.initial_capital) / self.initial_capital
            num_trades = len(trades)
            winning_trades = len([t for t in trades if t['status'] == 'target_hit'])
            
            # Advanced metrics
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) if len(returns) > 1 else 0
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            
            # Calculate market comparison
            market_return = (data['Close'][-1] - data['Close'][0]) / data['Close'][0]
            
            results = {
                'initial_capital': self.initial_capital,
                'final_capital': final_capital,
                'total_return': float(total_return),
                'market_return': float(market_return),
                'excess_return': float(total_return - market_return),
                'num_trades': num_trades,
                'win_rate': float(winning_trades / num_trades if num_trades > 0 else 0),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'equity_curve': equity_curve.tolist(),
                'trades': trades
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Performance calculation error: {e}")
            return None
    
    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown"""
        try:
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / peak
            return float(abs(min(drawdown)))
        except Exception as e:
            self.logger.error(f"Drawdown calculation error: {e}")
            return 0 